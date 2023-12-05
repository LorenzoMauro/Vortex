#pragma once
#ifndef SAC_H
#define SAC_H
#include <torch/torch.h>

#include "InputComposer.h"
#include "PathGuidingNetwork.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/Wrappers/KernelTimings.h"
#include "NeuralNetworks/NetworkImplementation.h"
#include "NeuralNetworks/NetworkSettings.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "NeuralNetworks/tools.h"

namespace vtx::network
{
    struct QNetworkImpl : torch::nn::Module {
        QNetworkImpl(int64_t inputDim, int64_t outputDim)
            :
            fc1(torch::nn::Linear(inputDim, 64)),
            fc2(torch::nn::Linear(64, 64)),
            fc3(torch::nn::Linear(64, 64)),
            fc4(torch::nn::Linear(64, 64)),
            fc5(torch::nn::Linear(64, 64)),
            fc6(torch::nn::Linear(64, outputDim))
        {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
            register_module("fc4", fc4);
            register_module("fc5", fc5);
            register_module("fc6", fc6);
        }

        torch::Tensor forward(const torch::Tensor& x) {
            torch::Tensor output = relu(fc1->forward(x));
            output               = relu(fc2->forward(output));
            output               = relu(fc3->forward(output));
            output               = relu(fc4->forward(output));
            output               = relu(fc5->forward(output));
            output               = fc6->forward(output);
            return output;
        }

        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        torch::nn::Linear fc3;
        torch::nn::Linear fc4;
        torch::nn::Linear fc5;
        torch::nn::Linear fc6;
    };
    TORCH_MODULE(QNetwork);

    class Sac : public NetworkImplementation
    {
    public:
		Sac(NetworkSettings* _settings) : device(torch::kCUDA, 0)
        {
            settings = _settings;
            actionDim = 3;
            targetEntropy = torch::tensor(-actionDim).to(device);

            init();
        }

        void init() override {
            gamma = torch::scalar_tensor(settings->sac.gamma).to(device);
            logAlpha = torch::full({ 1 }, settings->sac.logAlphaStart, torch::TensorOptions().device(device).requires_grad(true));

            ic                       = InputComposer(device, &settings->inputSettings);
			const int inputDimension = ic.dimension();
            policyNetwork            = PathGuidingNetwork(inputDimension, device, &settings->pathGuidingSettings, &settings->inputSettings);
            q1Network                = QNetwork(inputDimension + actionDim, 1);
            q2Network                = QNetwork(inputDimension + actionDim, 1);
            q1TargetNetwork          = QNetwork(inputDimension + actionDim, 1);
            q2TargetNetwork          = QNetwork(inputDimension + actionDim, 1);

            q1Network->to(device);
            q2Network->to(device);
            q1TargetNetwork->to(device);
            q2TargetNetwork->to(device);
            logAlpha = logAlpha.to(device);

            policyOptimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(policyNetwork.parameters(), torch::optim::AdamOptions(settings->sac.policyLr)));
            q1Optimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(q1Network->parameters(), torch::optim::AdamOptions(settings->sac.qLr)));
            q2Optimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(q2Network->parameters(), torch::optim::AdamOptions(settings->sac.qLr)));
            alphaOptimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam({ logAlpha }, torch::optim::AdamOptions(settings->sac.alphaLr)));

            //torch::NoGradGuard noGrad;
            copyNetworkParameters(q1Network.ptr(), q1TargetNetwork.ptr());
            copyNetworkParameters(q2Network.ptr(), q2TargetNetwork.ptr());
        };

        void reset() override
        {
            graphs.reset();
            init();
        };

        void train() override {
            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_TRAIN]);
            cudaEventRecord(events.first);
            LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
            shuffleDataset(deviceParams);

			const device::ReplayBufferBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.replayBufferBuffers;

            const auto actionPtr = buffers.actionBuffer.castedPointer<float>();
            const auto rewardPtr = buffers.rewardBuffer.castedPointer<float>();
            const auto donePtr = buffers.doneBuffer.castedPointer<int>();

            for (int i = 0; i < settings->maxTrainingStepPerFrame; i++)
            {
                float* actionBatch = actionPtr + i * settings->batchSize;
                float* rewardBatch = rewardPtr + i * settings->batchSize;
                int* doneBatch = donePtr + i * settings->batchSize;
                
                ic.setFromBuffer(buffers.stateBuffer, settings->batchSize, i);
                torch::Tensor states = ic.getInput();

                ic.setFromBuffer(buffers.nextStatesBuffer, settings->batchSize, i);
                torch::Tensor nextStates = ic.getInput();

                torch::Tensor actions = torch::from_blob(actionBatch, { settings->batchSize, actionDim }, torch::TensorOptions().device(device).dtype(torch::kFloat));
                torch::Tensor rewards = torch::from_blob(rewardBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
                torch::Tensor done = torch::from_blob(doneBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kI32));
                trainStep(states, nextStates, actions, rewards, done);
            }
            cudaEventRecord(events.second);
            //settings->batchSize += 1;
        };

        void inference(const int& depth) override {

            device::InferenceBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceBuffers;
            CUDABuffer inferenceSizeBuffers = buffers.inferenceSize;
            int inferenceSize;
            inferenceSizeBuffers.download(&inferenceSize);

            if (inferenceSize == 0)
            {
                return;
            }

            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_INFER]);
            cudaEventRecord(events.first);

            const auto distributionParametersPtr = buffers.distributionParameters.castedPointer<float>();
            const auto samplesPtr = buffers.samplesBuffer.castedPointer<float>();
            const auto probPtr = buffers.probabilitiesBuffer.castedPointer<float>();
            const auto mixtureWeightPtr = buffers.mixtureWeightBuffer.castedPointer<float>();

            ic.setFromBuffer(buffers.stateBuffer, inferenceSize, 0);
            const torch::Tensor inferenceInput = ic.getInput();

            const torch::Tensor distributionParameterCuda = torch::from_blob(distributionParametersPtr, { inferenceSize, policyNetwork.getMixtureParameterCount() }, at::device(device).dtype(torch::kFloat32));
            const torch::Tensor samplesCuda = torch::from_blob(samplesPtr, { inferenceSize, 3 }, at::device(device).dtype(torch::kFloat32));
            const torch::Tensor probCuda = torch::from_blob(probPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat32));
            const torch::Tensor mixtureWeightCuda = torch::from_blob(mixtureWeightPtr, { inferenceSize, settings->pathGuidingSettings.mixtureSize }, at::device(device).dtype(torch::kFloat32));

            const auto [mixtureParameters, mixtureWeight, sample, prob, c] = policyNetwork.inferenceWithSample(inferenceInput);

            CHECK_TENSOR_ANOMALY(inferenceInput);
            CHECK_TENSOR_ANOMALY(mixtureParameters);
            CHECK_TENSOR_ANOMALY(mixtureWeight);
            CHECK_TENSOR_ANOMALY(sample);
            CHECK_TENSOR_ANOMALY(prob);


            distributionParameterCuda.copy_(mixtureParameters);
            samplesCuda.copy_(sample);
            probCuda.copy_(prob);
            mixtureWeightCuda.copy_(mixtureWeight);

            cudaEventRecord(events.second);
        };

        GraphsData& getGraphs() override {
            return graphs;
        };

    private:
    	
        std::tuple<torch::Tensor, torch::Tensor> policyTargetValue(torch::Tensor& states, const bool useTargetQ) {
            auto [action, p] = policyNetwork.sample(states);
            torch::Tensor logL = torch::log(p + EPS);
            torch::Tensor q1Value;
            torch::Tensor q2Value;

            const torch::Tensor stateActions = torch::cat({ states, action }, /*dim=*/1);

            if (useTargetQ)
            {
                q1Value = q1TargetNetwork->forward(stateActions);
                q2Value = q2TargetNetwork->forward(stateActions);
            }
            else
            {
                q1Value = q1Network->forward(stateActions);
                q2Value = q2Network->forward(stateActions);
            }
            const torch::Tensor minQ = min(q1Value, q2Value);

            torch::Tensor policyTarget = (minQ - torch::exp(logAlpha) * logL);

            PRINT_TENSORS("POLICY TARGET VALUE COMPUTATION", action, logL, minQ, policyTarget);

            return { policyTarget, logL };
        };

        void trainStep(
            torch::Tensor& states,
            torch::Tensor& nextStates,
            torch::Tensor& actions,
            const torch::Tensor& rewards,
            const torch::Tensor& done) {
            ANOMALY_SWITCH;
            PRINT_TENSORS("TRAIN STEP", states, nextStates, actions, rewards, done);

            torch::Tensor q1Loss;
            torch::Tensor q2Loss;
            torch::Tensor q1Val;
            torch::Tensor q2Val;
            {
                const auto [policyTarget, logL] = policyTargetValue(nextStates, true);

                // Compute the target Q-value

                const torch::Tensor targetQ = rewards + (1 - done) * gamma * policyTarget;

                const torch::Tensor stateActions = torch::cat({ states, actions }, /*dim=*/1);
                q1Val = q1Network->forward(stateActions);
                q2Val = q2Network->forward(stateActions);

                // Compute Q-net losses
                q1Loss = 0.5f * torch::nn::functional::mse_loss(q1Val, targetQ.detach());
                q2Loss = 0.5f * torch::nn::functional::mse_loss(q2Val, targetQ.detach());

                PRINT_TENSORS("Q-VALUE COMPUTATION", policyTarget, logL, targetQ, stateActions, q1Val, q2Val, q1Loss, q2Loss);
            }

            q1Optimizer->zero_grad();
            q1Loss.backward();
            q1Optimizer->step();

            q2Optimizer->zero_grad();
            q2Loss.backward();
            q2Optimizer->step();

            torch::Tensor policyLoss;
            torch::Tensor alphaLoss;
            {
                const auto [policyTarget, logL] = policyTargetValue(states, false);
                policyLoss = -policyTarget.mean();
                alphaLoss = -(logAlpha * (logL + targetEntropy).detach()).mean();
                PRINT_TENSORS("POLICY LOSS COMPUTATION", policyTarget, logL, policyLoss, alphaLoss);
            }
            policyOptimizer->zero_grad();
            policyLoss.backward();
            policyOptimizer->step();

            alphaOptimizer->zero_grad();
            alphaLoss.backward();
            alphaOptimizer->step();

            polyakUpdate(q1Network.ptr(), q1TargetNetwork.ptr(), settings->sac.polyakFactor);
            polyakUpdate(q2Network.ptr(), q2TargetNetwork.ptr(), settings->sac.polyakFactor);

            const std::vector<torch::Tensor> hostTensors = downloadTensors(
                {
                    policyLoss.unsqueeze(-1),
                    q1Loss.unsqueeze(-1),
                    q2Loss.unsqueeze(-1),
                    alphaLoss.unsqueeze(-1),
                    rewards.mean().unsqueeze(-1),
                    q1Val.mean().unsqueeze(-1),
                    q2Val.mean().unsqueeze(-1),
                    torch::exp(logAlpha)
                }
            );

            graphs.addData(G_POLICY_LOSS, hostTensors[0].item<float>());
            graphs.addData(G_Q1_LOSS, hostTensors[1].item<float>());
            graphs.addData(G_Q2_LOSS, hostTensors[2].item<float>());
            graphs.addData(G_ALPHA_LOSS, hostTensors[3].item<float>());
            graphs.addData(G_DATASET_REWARDS, hostTensors[4].item<float>());
            graphs.addData(G_Q1_VALUES, hostTensors[5].item<float>());
            graphs.addData(G_Q2_VALUES, hostTensors[6].item<float>());
            graphs.addData(G_ALPHA_VALUES, hostTensors[7].item<float>());
        };

    public:
        int64_t positionEncodingDim;
        int64_t actionDim;
        torch::Tensor logAlpha;
        torch::Tensor gamma;
        torch::Tensor targetEntropy;
        torch::Device device; // This selects the second GPU


        InputComposer ic;
        PathGuidingNetwork policyNetwork;
        QNetwork q1Network = nullptr;
        QNetwork q2Network = nullptr;
        QNetwork q1TargetNetwork = nullptr;
        QNetwork q2TargetNetwork = nullptr;

        std::shared_ptr<torch::optim::Adam> autoEncoderOptimizer;
        std::shared_ptr<torch::optim::Adam> alphaOptimizer;
        std::shared_ptr<torch::optim::Adam> policyOptimizer;
        std::shared_ptr<torch::optim::Adam> q1Optimizer;
        std::shared_ptr<torch::optim::Adam> q2Optimizer;

        GraphsData graphs;
    };
}




#endif
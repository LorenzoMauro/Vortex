#pragma once
#ifndef SAC_H
#define SAC_H
#include <torch/torch.h>
#include "Core/Options.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "NeuralNetworks/Encodings.h"
#include "NeuralNetworks/NetworkImplementation.h"
#include "NeuralNetworks/NetworkSettings.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "NeuralNetworks/ReplayBuffer.h"
#include "NeuralNetworks/Distributions/SphericalGaussian.h"

//#define DEBUG_TENSORS
//#define CHECK_ANOMALY
#include "NeuralNetworks/tools.h"

namespace vtx::network
{
    struct PolicyNetworkImpl : torch::nn::Module {
        PolicyNetworkImpl(int64_t inputDim, int64_t outputDim)
            :
            fc1(torch::nn::Linear(inputDim, 64)),
            fc2(torch::nn::Linear(64, 64)),
            fc3(torch::nn::Linear(64, 64)),
            fc4(torch::nn::Linear(64, 64)),
            fc5(torch::nn::Linear(64, 64)),
            fc6(torch::nn::Linear(64, outputDim + 1))
            //fcMean(torch::nn::Linear(256, outputDim)),
            //fcK(torch::nn::Linear(256 + outputDim, 1))
        {  // 2x outputDim for mean and log covariance

            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
            register_module("fc4", fc4);
            register_module("fc5", fc5);
            register_module("fc6", fc6);
            //register_module("fcMean", fcMean);
            //register_module("fcK", fcK);
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
            x = relu(fc1->forward(x));
            x = relu(fc2->forward(x));
            x = relu(fc3->forward(x));
            x = relu(fc4->forward(x));
            x = relu(fc5->forward(x));
            x = fc6->forward(x);

            torch::Tensor mean = x.narrow(1, 0, x.size(1) - 1);
            mean               = mean / mean.norm(2, 1, true);
            torch::Tensor k    = relu(x.narrow(1, x.size(1) - 1, 1));
            k                  = clamp(k, -1.0f, MAX_CONCENTRATION) + EPS;

            //auto mean = fcMean->forward(x);
            //mean = mean / mean.norm(2, 1, true);
            //
            //x = torch::cat({ x, mean }, /*dim=*/-1);
            //auto k = torch::relu(fcK->forward(x));
            //k = torch::clamp(k, -1.0f, MAX_CONCENTRATION) + EPS;

            return { mean, k };
        }

        std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& state, const torch::Device& device)
        {
            //mean and logCov represent a 2d gaussian distribution for spherical angles in the range 0-1, sample a direction from it
            auto [mean, k] = forward(state);

            const torch::Tensor action = SphericalGaussian::sample(mean, k, device);
            const torch::Tensor logLikelihood = SphericalGaussian::logLikelihood(action, mean, k);

            return { action, logLikelihood };
        }

        std::tuple<torch::Tensor> evaluate(const torch::Tensor& state, const torch::Tensor& action)
        {
            //mean and logCov represent a 2d gaussian distribution for spherical angles in the range 0-1, sample a direction from it
            auto [mean, k] = forward(state);
            const torch::Tensor logLikelihood = SphericalGaussian::logLikelihood(action, mean, k);
            return { logLikelihood };
        }

        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        torch::nn::Linear fc3;
        torch::nn::Linear fc4;
        torch::nn::Linear fc5;
        torch::nn::Linear fc6;
        //torch::nn::Linear fcMean;
        //torch::nn::Linear fcK;

    };
    TORCH_MODULE(PolicyNetwork);

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
		Sac() : device(torch::kCUDA, 0)
		{
			settings                         = SacSettings{};
			settings.batchSize               = getOptions()->batchSize;
			settings.maxTrainingStepPerFrame = getOptions()->maxTrainingStepPerFrame;
			settings.polyakFactor            = getOptions()->polyakFactor;
			settings.logAlphaStart           = getOptions()->logAlphaStart;
			settings.gamma                   = getOptions()->neuralGamma;
			settings.inferenceIterationStart = getOptions()->inferenceIterationStart;
			settings.clearOnInferenceStart   = getOptions()->clearOnInferenceStart;
			settings.doInference             = getOptions()->doInference;
			settings.neuralSampleFraction    = getOptions()->neuralSampleFraction;
			positionEncodingDim              = 12;
			stateDim                         = positionEncodingDim * 3 + 3 + 3;
			actionDim                        = 3;
			targetEntropy                    = torch::tensor(-actionDim).to(device);

			settings.policyLr      = getOptions()->policyLr;
			settings.qLr           = getOptions()->qLr;
			settings.alphaLr       = getOptions()->alphaLr;
			settings.autoencoderLr = getOptions()->autoencoderLr;

			init();
		}

        void init() override
        {
            gamma = torch::scalar_tensor(settings.gamma).to(device);
            logAlpha = torch::full({ 1 }, settings.logAlphaStart, torch::TensorOptions().device(device).requires_grad(true));

            policyNetwork = PolicyNetwork(stateDim, actionDim);
            q1Network = QNetwork(stateDim + actionDim, 1);
            q2Network = QNetwork(stateDim + actionDim, 1);
            q1TargetNetwork = QNetwork(stateDim + actionDim, 1);
            q2TargetNetwork = QNetwork(stateDim + actionDim, 1);

            policyNetwork->to(device);
            q1Network->to(device);
            q2Network->to(device);
            q1TargetNetwork->to(device);
            q2TargetNetwork->to(device);
            logAlpha = logAlpha.to(device);

            policyOptimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(policyNetwork->parameters(), torch::optim::AdamOptions(settings.policyLr)));
            q1Optimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(q1Network->parameters(), torch::optim::AdamOptions(settings.qLr)));
            q2Optimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam(q2Network->parameters(), torch::optim::AdamOptions(settings.qLr)));
            alphaOptimizer = std::make_shared<torch::optim::Adam>(torch::optim::Adam({ logAlpha }, torch::optim::AdamOptions(settings.alphaLr)));

            //torch::NoGradGuard noGrad;
			copyNetworkParameters(q1Network.ptr(), q1TargetNetwork.ptr());
            copyNetworkParameters(q2Network.ptr(), q2TargetNetwork.ptr());
        }

        void reset() override
        {
            graphs.reset({ G_POLICY_LOSS, G_Q1_LOSS, G_Q2_LOSS, G_ALPHA_LOSS, G_DATASET_REWARDS, G_Q1_VALUES, G_Q2_VALUES, G_ALPHA_VALUES });
            init();
        }

        void train() override
        {
            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_TRAIN]);
            cudaEventRecord(events.first);
            const auto deviceParams = UPLOAD_BUFFERS->launchParamsBuffer.castedPointer<LaunchParams>();

            const int replayBufferSize = settings.batchSize * settings.maxTrainingStepPerFrame;
            shuffleReplayBuffer(deviceParams, replayBufferSize);


            device::Buffers::ReplayBufferBuffers& replayBufferBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.replayBufferBuffers;

            const auto actionPtr = replayBufferBuffers.actionBuffer.castedPointer<float>();
            const auto rewardPtr = replayBufferBuffers.rewardBuffer.castedPointer<float>();
            const auto donePtr = replayBufferBuffers.doneBuffer.castedPointer<int>();

            for (int i = 0; i < settings.maxTrainingStepPerFrame; i++)
            {
                float* actionBatch = actionPtr + i * settings.batchSize;
                float* rewardBatch = rewardPtr + i * settings.batchSize;
                int* doneBatch = donePtr + i * settings.batchSize;

                torch::Tensor states = createStateInput(replayBufferBuffers.stateBuffer, settings.batchSize, i);
                torch::Tensor nextStates = createStateInput(replayBufferBuffers.nextStatesBuffer, settings.batchSize, i);
                torch::Tensor actions = torch::from_blob(actionBatch, { settings.batchSize, actionDim }, torch::TensorOptions().device(device).dtype(torch::kFloat));
                torch::Tensor rewards = torch::from_blob(rewardBatch, { settings.batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
                torch::Tensor done = torch::from_blob(doneBatch, { settings.batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kI32));
                trainStep(states, nextStates, actions, rewards, done);
            }
            cudaEventRecord(events.second);
            //settings.batchSize += 1;
        }

        void inference() override
        {

            device::Buffers::InferenceBuffers& inferenceBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers;
            CUDABuffer inferenceSizeBuffers = inferenceBuffers.inferenceSize;
            int inferenceSize;
            inferenceSizeBuffers.download(&inferenceSize);

            if (inferenceSize == 0)
            {
                return;
            }

            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_INFER]);
            cudaEventRecord(events.first);

            const auto meanPtr = inferenceBuffers.meanBuffer.castedPointer<float>();
            const auto concentrationPtr = inferenceBuffers.concentrationBuffer.castedPointer<float>();

            const torch::Tensor inferenceStates = createStateInput(inferenceBuffers.stateBuffer, inferenceSize);
            const torch::Tensor inferenceMean = torch::from_blob(meanPtr, { inferenceSize, actionDim }, at::device(device).dtype(torch::kFloat32));
            const torch::Tensor inferenceLogCov = torch::from_blob(concentrationPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat32));

            const std::vector<torch::Tensor> hostTensors = downloadTensors(
                {
                    inferenceLogCov.mean().unsqueeze(-1),
                }
            );

            graphs.graphs[G_INFERENCE_CONCENTRATION].push_back(hostTensors[0].item<float>());

            auto [mean, k] = policyNetwork->forward(inferenceStates);

            inferenceMean.copy_(mean);
            inferenceLogCov.copy_(k);

            cudaEventRecord(events.second);
        }

        NetworkSettings& getSettings() override
		{
            return settings;
		}

        GraphsData& getGraphs() override
		{
			return graphs;
		}

        void shuffleReplayBuffer(LaunchParams* params, const int replayBufferSize)
        {
            LaunchParams* paramsCopy = params;
            gpuParallelFor(eventNames[N_SHUFFLE_DATASET],
                replayBufferSize,
                [paramsCopy] __device__(int id)
            {
                paramsCopy->networkInterface->randomFillReplayBuffer(id);
            });
        }
    private:
    	torch::Tensor createStateInput(device::Buffers::NetworkStateBuffers& buffers, const int batchSize, const int batchId = 0)
        {
            const auto positionPtr = buffers.positionBuffer.castedPointer<float>() + batchId * batchSize * 3;
            const auto woPtr = buffers.woBuffer.castedPointer<float>() + batchId * batchSize * 3;
            const auto normalPtr = buffers.normalBuffer.castedPointer<float>() + batchId * batchSize * 3;

            float* encodedPositionPtr = encoding::TriangleWaveEncoding::encode(buffers.encodedPositionBuffer, positionPtr, batchSize, 3, positionEncodingDim);
            torch::Tensor encodedPosition = torch::from_blob(encodedPositionPtr, { batchSize, positionEncodingDim * 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor position = torch::from_blob(positionPtr, { batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor wo = torch::from_blob(woPtr, { batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor normal = torch::from_blob(normalPtr, { batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor states = torch::cat({ encodedPosition, wo, normal }, -1);

            PRINT_TENSORS("CREATE STATE INPUT", position, encodedPosition, wo, normal, states);
            return states;
        }

        std::tuple<torch::Tensor, torch::Tensor> policyTargetValue(torch::Tensor& states, bool useTargetQ)
        {
            auto [action, logL] = policyNetwork->sample(states, device);
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
        }

        void trainStep(
            torch::Tensor& states,
            torch::Tensor& nextStates,
            torch::Tensor& actions,
            const torch::Tensor& rewards,
            const torch::Tensor& done)
        {
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

            polyakUpdate(q1Network.ptr(), q1TargetNetwork.ptr(), settings.polyakFactor);
            polyakUpdate(q2Network.ptr(), q2TargetNetwork.ptr(), settings.polyakFactor);

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

            graphs.graphs[G_POLICY_LOSS].push_back(hostTensors[0].item<float>());
            graphs.graphs[G_Q1_LOSS].push_back(hostTensors[1].item<float>());
            graphs.graphs[G_Q2_LOSS].push_back(hostTensors[2].item<float>());
            graphs.graphs[G_ALPHA_LOSS].push_back(hostTensors[3].item<float>());
            graphs.graphs[G_DATASET_REWARDS].push_back(hostTensors[4].item<float>());
            graphs.graphs[G_Q1_VALUES].push_back(hostTensors[5].item<float>());
            graphs.graphs[G_Q2_VALUES].push_back(hostTensors[6].item<float>());
            graphs.graphs[G_ALPHA_VALUES].push_back(hostTensors[7].item<float>());
        }

        

    public:
        int64_t stateDim;  // Dimensionality of state (3D vector for position)
        int64_t positionEncodingDim;
        int64_t actionDim;
        torch::Tensor logAlpha;
        torch::Tensor gamma;
        torch::Tensor targetEntropy;
        torch::Device device; // This selects the second GPU

        PolicyNetwork policyNetwork = nullptr;
        QNetwork q1Network = nullptr;
        QNetwork q2Network = nullptr;
        QNetwork q1TargetNetwork = nullptr;
        QNetwork q2TargetNetwork = nullptr;

        std::shared_ptr<torch::optim::Adam> autoEncoderOptimizer;
        std::shared_ptr<torch::optim::Adam> alphaOptimizer;
        std::shared_ptr<torch::optim::Adam> policyOptimizer;
        std::shared_ptr<torch::optim::Adam> q1Optimizer;
        std::shared_ptr<torch::optim::Adam> q2Optimizer;

        SacSettings settings;
        GraphsData graphs;
    };
}




#endif
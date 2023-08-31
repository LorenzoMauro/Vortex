#include "Npg.h"
#include "InputComposer.h"
#include "Device/Wrappers/KernelTimings.h"
#include "NeuralNetworks/tools.h"
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx::network
{
    Npg::Npg(NetworkSettings* _settings) :
        device(torch::kCUDA, 0)
    {
        settings = _settings;
		Npg::init();
    }

    void Npg::init() {
        ic = InputComposer(device, &settings->inputSettings);
        pgn = PathGuidingNetwork(ic.dimension(), device, &settings->pathGuidingSettings);
        optimizer = std::make_shared<torch::optim::Adam>(pgn.parameters(), torch::optim::AdamOptions(settings->npg.learningRate));
        trainingStep = 0;
    }

    void Npg::reset() {
        graphs.reset();
        init();
    }

    void Npg::train() {
        ANOMALY_SWITCH;
        if(trainingStep >= settings->maxTrainingSteps)
        {
            settings->doTraining = false;
            return;
        }
        const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_TRAIN]);
        cudaEventRecord(events.first);
        const auto deviceParams = UPLOAD_BUFFERS->launchParamsBuffer.castedPointer<LaunchParams>();

        shuffleDataset(deviceParams);

        const device::Buffers::NpgTrainingDataBuffers& buffers = UPLOAD_BUFFERS->networkInterfaceBuffer.npgTrainingDataBuffers;

        const auto incomingDirectionPtr = buffers.incomingDirectionBuffer.castedPointer<float>();
        const auto bsdfPdfPtr = buffers.bsdfProbabilitiesBuffer.castedPointer<float>();
        const auto luminancePtr = buffers.outgoingRadianceBuffer.castedPointer<float>();

        for (int i = 0; i < settings->maxTrainingStepPerFrame; i++)
        {
            trainingStep += 1;
            float* incomingDirectionBatch = incomingDirectionPtr + i * settings->batchSize;
            float* bsdfPtrBatch = bsdfPdfPtr + i * settings->batchSize;
            float* luminanceBatch = luminancePtr + i * settings->batchSize;

            ic.setFromBuffer(buffers.inputBuffer, settings->batchSize, i);
            torch::Tensor inputTensor = ic.getInput();
            torch::Tensor incomingDirection = torch::from_blob(incomingDirectionBatch, { settings->batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor bsdfPdf = torch::from_blob(bsdfPtrBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat));
            torch::Tensor luminance = torch::from_blob(luminanceBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat));

            //PRINT_TENSOR_ALWAYS("Luminance", luminance);
            CHECK_TENSOR_ANOMALY(inputTensor);
            CHECK_TENSOR_ANOMALY(luminance);
            CHECK_TENSOR_ANOMALY(incomingDirection);
            CHECK_TENSOR_ANOMALY(bsdfPdf);

            trainStep(inputTensor, luminance, incomingDirection, bsdfPdf);
        }
        cudaEventRecord(events.second);
    }

    void Npg::inference(const int& depth) {
        device::Buffers::InferenceBuffers& buffers = UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers;
        CUDABuffer inferenceSizeBuffers = buffers.inferenceSize;
        int inferenceSize;
        inferenceSizeBuffers.download(&inferenceSize);

        if (inferenceSize == 0)
        {
            return;
        }

        const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_INFER]);
        cudaEventRecord(events.first);

        ic.setFromBuffer(buffers.stateBuffer, inferenceSize, 0);
        const torch::Tensor inferenceInput = ic.getInput();
        CHECK_TENSOR_ANOMALY(inferenceInput);

        const auto distributionParametersPtr = buffers.distributionParameters.castedPointer<float>();
        const auto samplingFractionPtr = buffers.samplingFractionArrayBuffer.castedPointer<float>();
        const auto mixtureWeightPtr = buffers.mixtureWeightBuffer.castedPointer<float>();

        int                 distributionParamCount = distribution::Mixture::getDistributionParametersCount(settings->pathGuidingSettings.distributionType);
        const torch::Tensor distributionParameterCuda = torch::from_blob(distributionParametersPtr, { inferenceSize, pgn.settings->mixtureSize, distributionParamCount }, at::device(device).dtype(torch::kFloat32));
        const torch::Tensor samplingFractionCuda = torch::from_blob(samplingFractionPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat32));
        const torch::Tensor mixtureWeightCuda = torch::from_blob(mixtureWeightPtr, { inferenceSize, settings->pathGuidingSettings.mixtureSize }, at::device(device).dtype(torch::kFloat32));

        auto [mixtureParameters, mixtureWeight, c] = pgn.inference(inferenceInput);

        if (settings->npg.samplingFractionBlend)
        {
			c = tau() * c;
        }
        CHECK_TENSOR_ANOMALY(c);
        CHECK_TENSOR_ANOMALY(mixtureParameters);
        CHECK_TENSOR_ANOMALY(mixtureWeight);

        distributionParameterCuda.copy_(mixtureParameters);
        samplingFractionCuda.copy_(c);
        mixtureWeightCuda.copy_(mixtureWeight);

        const std::vector<torch::Tensor> hostTensors = downloadTensors(
            {
                samplingFractionCuda.mean().unsqueeze(-1)
            }
        );

        graphs.addData(G_NGP_I_SAMPLING_FRACTION, hostTensors[0].item<float>(), depth);
        distribution::Mixture::setGraphData(settings->pathGuidingSettings.distributionType, pgn.mixtureParameters, pgn.mixtureWeights, graphs, false, depth);
        cudaEventRecord(events.second);
    }

    GraphsData& Npg::getGraphs() {
        return graphs;
    }
    

    void Npg::trainStep(const torch::Tensor& input, const torch::Tensor& luminance, const torch::Tensor& incomingDirection, const torch::Tensor& bsdfProb) {

        PRINT_TENSORS("TRAIN STEP INPUTS", input, luminance, incomingDirection, bsdfProb);

        optimizer->zero_grad();

        const torch::Tensor neuralPdf = pgn.evaluate(input, incomingDirection);
        const torch::Tensor logNeuralPdf = torch::log(neuralPdf + EPS);
        const torch::Tensor c = pgn.getLastRunSamplingFraction();
        PRINT_TENSORS("TRAIN STEP OUTPUTS", neuralPdf, c);
        CHECK_TENSOR_ANOMALY(neuralPdf);
        CHECK_TENSOR_ANOMALY(logNeuralPdf);
        CHECK_TENSOR_ANOMALY(c);

        //const torch::Tensor entropy = -torch::sum(neuralPdf * logNeuralPdf, 1);
        //const float targetEntropy = -logf(3.0f);
        //torch::Tensor entropyLoss = torch::pow(entropy - targetEntropy, 2).mean();
        //CHECK_TENSOR_ANOMALY(entropy);
        //CHECK_TENSOR_ANOMALY(entropyLoss);
        //PRINT_TENSORS("ENTROPY LOSS", entropyLoss);

        const torch::Tensor blendedQ = c * neuralPdf + (1 - c) * bsdfProb;
        CHECK_TENSOR_ANOMALY(blendedQ);

        const torch::Tensor lossQ        = loss(neuralPdf, luminance);
        const torch::Tensor lossBlendedQ = loss(blendedQ, luminance);

        CHECK_TENSOR_ANOMALY(lossQ);
        CHECK_TENSOR_ANOMALY(lossBlendedQ);
        PRINT_TENSORS("DIVERGENCE", lossQ, lossBlendedQ);

        const torch::Tensor loss = lossBlendFactor() * lossQ + (1.0f - lossBlendFactor()) * lossBlendedQ;// +0.8f * entropyLoss;
        CHECK_TENSOR_ANOMALY(loss);
        PRINT_TENSORS("LOSS", loss);

        PRINT_TENSORS("TRAIN STEP OUTPUTS", c, neuralPdf, blendedQ, lossQ, lossBlendedQ, loss);
        loss.backward();
        optimizer->step();

        const std::vector<torch::Tensor> hostTensors = downloadTensors(
            {
                c.mean().unsqueeze(-1),

                luminance.mean().unsqueeze(-1),
                neuralPdf.mean().unsqueeze(-1),
                bsdfProb.mean().unsqueeze(-1),
                blendedQ.mean().unsqueeze(-1),

                lossQ.unsqueeze(-1),
                lossBlendedQ.unsqueeze(-1),
                loss.unsqueeze(-1),
            }
        );
        graphs.addData(G_NGP_T_SAMPLING_FRACTION, hostTensors[0].item<float>());
        graphs.addData(G_NGP_T_TARGET_P, hostTensors[1].item<float>());
        graphs.addData(G_NGP_T_NEURAL_P, hostTensors[2].item<float>());
        graphs.addData(G_NGP_T_BSDF_P, hostTensors[3].item<float>());
        graphs.addData(G_NGP_T_BLENDED_P, hostTensors[4].item<float>());

        graphs.addData(G_NGP_T_LOSS_Q, hostTensors[5].item<float>());
        graphs.addData(G_NGP_T_LOSS_BLENDED_Q, hostTensors[6].item<float>());
        graphs.addData(G_NGP_T_LOSS, hostTensors[7].item<float>());
        graphs.addData(G_NGP_TAU, lossBlendFactor());

        distribution::Mixture::setGraphData(settings->pathGuidingSettings.distributionType, pgn.mixtureParameters, pgn.mixtureWeights, graphs, true);

        PRINT_TENSORS("TRAIN STEP", neuralPdf, blendedQ, luminance, loss);


    }

    float Npg::lossBlendFactor()
    {
        if(settings->npg.constantBlendFactor)
        {
	        return settings->npg.e;
        }
        return std::min(1.0f, pow(0.3333f, 5.0f * tau()));
    }

    float Npg::inferenceSamplingFractionBlend()
    {
        return tau();
    }

    float Npg::tau()
    {
        float tau = std::min(1.0f, (float)trainingStep / ((float)settings->maxTrainingSteps));
        return tau;
    }

    torch::Tensor Npg::loss(const torch::Tensor& neuralProb, const torch::Tensor& targetProb)
    {
        torch::Tensor loss;
        if (settings->npg.lossType == L_KL_DIV)
        {
	        loss =kl_div(torch::log(neuralProb + EPS), targetProb, at::Reduction::None);
        }
        else if(settings->npg.lossType == L_KL_DIV_MC_ESTIMATION)
        {
            loss = -targetProb * torch::log(neuralProb + EPS);
            CHECK_TENSOR_ANOMALY(loss);
        }
        else if (settings->npg.lossType == L_PEARSON_DIV)
        {
            loss = torch::pow(targetProb - neuralProb, 2.0f)/ (neuralProb + EPS);
            CHECK_TENSOR_ANOMALY(loss);
        }
        else if (settings->npg.lossType == L_PEARSON_DIV_MC_ESTIMATION)
        {
	        loss = -torch::pow(targetProb, 2.0f) * torch::log(neuralProb + EPS);
            CHECK_TENSOR_ANOMALY(loss);
        }
        else
        {
            VTX_ERROR("Not implemented");
            loss = torch::zeros({ 1 });
        }

        if(settings->npg.absoluteLoss)
        {
	        loss = torch::abs(loss);
        }
        if(settings->npg.meanLoss)
        {
	        loss = loss.mean();
        }
        else
        {
            loss = loss.sum();
        }
        return loss;
    }

}

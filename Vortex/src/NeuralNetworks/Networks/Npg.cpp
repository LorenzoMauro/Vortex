#include "Npg.h"
#include "InputComposer.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
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
        pgn = PathGuidingNetwork(ic.dimension(), device, &settings->pathGuidingSettings, &settings->inputSettings);
        optimizer = std::make_shared<torch::optim::Adam>(pgn.parameters(), torch::optim::AdamOptions(settings->npg.learningRate).amsgrad(true));
        trainingStep = 0;
    }

    void Npg::reset() {
        graphs.reset();
        init();
    }

    void Npg::train() {
        VTX_INFO("NPG TRAIN STEP");
        CLEAR_TENSOR_DEBUGGER();
        ANOMALY_SWITCH;
        if(trainingStep >= settings->maxTrainingSteps)
        {
            settings->doTraining = false;
            return;
        }
        const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_TRAIN]);
        cudaEventRecord(events.first);
        LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
        //settings->batchSize = (settings->batchSize/tcnn::cpp::batch_size_granularity()) * tcnn::cpp::batch_size_granularity();

        shuffleDataset(deviceParams);

        const device::NpgTrainingDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.npgTrainingDataBuffers;

        const auto incomingDirectionPtr = buffers.incomingDirectionBuffer.castedPointer<float>();
        const auto bsdfPdfPtr = buffers.bsdfProbabilitiesBuffer.castedPointer<float>();
        const auto luminancePtr = buffers.outgoingRadianceBuffer.castedPointer<float>();

        const auto positionPtr = buffers.inputBuffer.positionBuffer.castedPointer<float>();
        const auto woPtr = buffers.inputBuffer.woBuffer.castedPointer<float>();
        const auto normalPtr = buffers.inputBuffer.normalBuffer.castedPointer<float>();
        const auto aabb =onDeviceData->launchParamsData.getHostImage().aabb;
        torch::Tensor minExtents = torch::tensor({ aabb.minX, aabb.minY, aabb.minZ }).to(device).to(torch::kFloat);
        torch::Tensor maxExtents = torch::tensor({ aabb.maxX, aabb.maxY, aabb.maxZ }).to(device).to(torch::kFloat);
        const torch::Tensor deltaExtents = maxExtents - minExtents;

        // Ensure that the extents tensors are of the same type and device as the positions tensor

        for (int i = 0; i < settings->maxTrainingStepPerFrame; i++)
        {
            trainingStep += 1;
            float* bsdfPtrBatch = bsdfPdfPtr + i * settings->batchSize;
            float* luminanceBatch = luminancePtr + i * settings->batchSize;
            float* incomingDirectionBatch = incomingDirectionPtr + i * settings->batchSize * 3;
            float* positionBatchPtr = positionPtr + i * settings->batchSize * 3;
            float* woBatchPtr = woPtr + i * settings->batchSize * 3;
            float* normalBatchPtr = normalPtr + i * settings->batchSize * 3;

            torch::Tensor positionTensor = torch::from_blob(positionBatchPtr, { settings->batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
#ifdef _DEBUG
            if (!deltaExtents.eq(0).any().item<bool>())
            {
                positionTensor = (positionTensor - minExtents) / deltaExtents;
            }
#else
            positionTensor = (positionTensor - minExtents) / deltaExtents;
#endif
            torch::Tensor woTensor = torch::from_blob(woBatchPtr, { settings->batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
            torch::Tensor normalTensor = torch::from_blob(normalBatchPtr, { settings->batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
            torch::Tensor inputTensor = torch::cat({ positionTensor, woTensor, normalTensor }, -1);
            torch::Tensor incomingDirection = torch::from_blob(incomingDirectionBatch, { settings->batchSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
            torch::Tensor bsdfPdf = torch::from_blob(bsdfPtrBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat)).clamp(-65504, 65504)toPrecision;
            torch::Tensor luminance = torch::from_blob(luminanceBatch, { settings->batchSize, 1 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;

        	TRACE_TENSOR(pgn.network->params);
            TRACE_TENSOR(inputTensor);
            TRACE_TENSOR((inputTensor)toPrecision);
            TRACE_TENSOR(luminance);
            TRACE_TENSOR(incomingDirection);
            TRACE_TENSOR(bsdfPdf);

            trainStep(inputTensor, luminance, incomingDirection, bsdfPdf);
        }
        cudaEventRecord(events.second);
    }

    void Npg::inference(const int& depth) {
        CLEAR_TENSOR_DEBUGGER();
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

        //ic.setFromBuffer(buffers.stateBuffer, inferenceSize, 0);
        //const torch::Tensor inferenceInput = ic.getInput();

        const auto          positionPtr     = buffers.stateBuffer.positionBuffer.castedPointer<float>();
        const auto          woPtr           = buffers.stateBuffer.woBuffer.castedPointer<float>();
        const auto          normalPtr       = buffers.stateBuffer.normalBuffer.castedPointer<float>();
        const auto          aabb            = onDeviceData->launchParamsData.getHostImage().aabb;
		const torch::Tensor minExtents      = torch::tensor({ aabb.minX, aabb.minY, aabb.minZ }).to(device).to(torch::kFloat);
		const torch::Tensor maxExtents      = torch::tensor({ aabb.maxX, aabb.maxY, aabb.maxZ }).to(device).to(torch::kFloat);
        torch::Tensor       positionTensor = torch::from_blob(positionPtr, { inferenceSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
        const torch::Tensor deltaExtents   = maxExtents - minExtents;
#ifdef _DEBUG
        if(!deltaExtents.eq(0).any().item<bool>())
        {
            positionTensor = (positionTensor - minExtents) / deltaExtents;
        }
#else
        positionTensor = (positionTensor - minExtents) / deltaExtents;
#endif

        torch::Tensor        woTensor       = torch::from_blob(woPtr, { inferenceSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
        torch::Tensor        normalTensor   = torch::from_blob(normalPtr, { inferenceSize, 3 }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
		const torch ::Tensor inferenceInput = torch::cat({ positionTensor, woTensor, normalTensor }, -1);
        TRACE_TENSOR(inferenceInput);

        const auto distributionParametersPtr = buffers.distributionParameters.castedPointer<float>();
        const auto samplingFractionPtr = buffers.samplingFractionArrayBuffer.castedPointer<float>();
        const auto mixtureWeightPtr = buffers.mixtureWeightBuffer.castedPointer<float>();

        int                 distributionParamCount = distribution::Mixture::getDistributionParametersCount(settings->pathGuidingSettings.distributionType);
        const torch::Tensor distributionParameterCuda = torch::from_blob(distributionParametersPtr, { inferenceSize, pgn.settings->mixtureSize, distributionParamCount }, at::device(device).dtype(torch::kFloat));
        const torch::Tensor samplingFractionCuda = torch::from_blob(samplingFractionPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat));
        const torch::Tensor mixtureWeightCuda = torch::from_blob(mixtureWeightPtr, { inferenceSize, settings->pathGuidingSettings.mixtureSize }, at::device(device).dtype(torch::kFloat));

        auto [mixtureParameters, mixtureWeight, c] = pgn.inference(inferenceInput);

        if (settings->npg.samplingFractionBlend)
        {
			c = tau() * c;
        }
        TRACE_TENSOR(c);
        TRACE_TENSOR(mixtureParameters);
        TRACE_TENSOR(mixtureWeight);

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
        optimizer->zero_grad();

        const torch::Tensor neuralPdf = pgn.evaluate(input, incomingDirection);
        const torch::Tensor logNeuralPdf = torch::log(neuralPdf + EPS);
        const torch::Tensor c = pgn.getLastRunSamplingFraction();
        TRACE_TENSOR(neuralPdf);
        TRACE_TENSOR(logNeuralPdf);
        TRACE_TENSOR(c);

        //const torch::Tensor entropy = -torch::sum(neuralPdf * logNeuralPdf, 1);
        //const float targetEntropy = -logf(3.0f);
        //torch::Tensor entropyLoss = torch::pow(entropy - targetEntropy, 2).mean();
        //CHECK_TENSOR_ANOMALY(entropy);
        //CHECK_TENSOR_ANOMALY(entropyLoss);
        //PRINT_TENSORS("ENTROPY LOSS", entropyLoss);

        const torch::Tensor blendedQ = c * neuralPdf + (1 - c) * bsdfProb;
        TRACE_TENSOR(blendedQ);

        const torch::Tensor lossQ        = loss(neuralPdf, luminance);
        const torch::Tensor lossBlendedQ = loss(blendedQ, luminance);

        TRACE_TENSOR(lossQ);
        TRACE_TENSOR(lossBlendedQ);

        const torch::Tensor loss = lossBlendFactor() * lossQ + (1.0f - lossBlendFactor()) * lossBlendedQ;// +0.8f * entropyLoss;
        TRACE_TENSOR(loss);

        loss.backward();
        if (TensorDebugger::analyzeGradients()) {
        	return; // NAN or INF gradients
        }
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
            torch::Tensor neuralProbLog = torch::log(neuralProb + EPS);
            TRACE_TENSOR(neuralProbLog);
            loss = -targetProb * neuralProbLog;
        }
        else if (settings->npg.lossType == L_PEARSON_DIV)
        {
            loss = torch::pow(targetProb - neuralProb, 2.0f)/ (neuralProb + EPS);
        }
        else if (settings->npg.lossType == L_PEARSON_DIV_MC_ESTIMATION)
        {
	        loss = -torch::pow(targetProb, 2.0f) * torch::log(neuralProb + EPS);
        }
        else
        {
            VTX_ERROR("Not implemented");
            loss = torch::zeros({ 1 });
        }
        TRACE_TENSOR(loss);

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
        TRACE_TENSOR(loss);
        return loss;
    }

}

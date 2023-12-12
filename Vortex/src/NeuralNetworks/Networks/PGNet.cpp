#include "PGNet.h"

#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Device/Wrappers/KernelTimings.h"
#include "NeuralNetworks/tools.h"
#include "NeuralNetworks/Config/EncodingConfig.h"
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx::network
{
#define PRINT_TENSOR_SLICE(tensor) std::cout << #tensor << " :\n" << tensor.slice(0, 0, 10) << std::endl;

	PGNet::PGNet(config::NetworkSettings* _settings) :
		device(torch::kCUDA, 0)
	{
		settings = _settings;
		PGNet::init();
		ntscLuminance = torch::tensor({ 0.299f, 0.587f, 0.114f }, torch::TensorOptions().dtype(torch::kFloat).device(device));
	}
	void PGNet::init()
	{
		constexpr int auxAdditionInputSize = 3; // wi
		int mainInputSize = 9; // position, normal, wo
		constexpr int mainOutputSize = 9; // outRadiance, inRadiance, throughput
		distributionParametersCount = distribution::Mixture::getDistributionParametersCount(settings->distributionType, true);
		mixtureParametersCount = settings->mixtureSize * distributionParametersCount;
		outputDim = mixtureParametersCount + settings->mixtureSize + 1;
		int auxFeatureSize = settings->totAuxInputSize - auxAdditionInputSize;

		if (settings->useAuxiliaryNetwork)
		{
			outputDim += auxFeatureSize;
		}

		auto encodingSettings = torchTcnn::getEncodingSettings(settings->inputSettings);

		int configN = 3;
		if(settings->useMaterialId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->materialIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		if (settings->useInstanceId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->instanceIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		if (settings->useTriangleId)
		{
			encodingSettings["nested"][configN] = torchTcnn::getEncodingSettings(settings->triangleIdEncodingConfig);
			encodingSettings["nested"][configN]["n_dims_to_encode"] = 1;
			mainInputSize++;
			configN++;
		}

		mlpBase = torchTcnn::TcnnModule(mainInputSize, outputDim, encodingSettings, torchTcnn::getNetworkSettings(settings->mainNetSettings), 1337);
		if (settings->useAuxiliaryNetwork)
		{
			auto netSettings = torchTcnn::getNetworkSettings(settings->auxiliaryNetSettings);
			auto auxInputSettings = torchTcnn::getEncodingSettings(auxFeatureSize, settings->auxiliaryInputSettings);
			auxiliaryMlp = torchTcnn::TcnnModule(settings->totAuxInputSize, mainOutputSize, auxInputSettings, netSettings, 1440);
			optimizer = std::make_shared<torch::optim::Adam>(std::vector<torch::optim::OptimizerParamGroup>{mlpBase->parameters(), auxiliaryMlp->parameters()}, torch::optim::AdamOptions(settings->learningRate).amsgrad(true));
		}
		else
		{
			optimizer = std::make_shared<torch::optim::Adam>(mlpBase->parameters(), torch::optim::AdamOptions(settings->learningRate).amsgrad(true));
		}


		trainingStep = 0;
	}

	void PGNet::reset()
	{
		graphs.reset();
		init();
	}

	void PGNet::train()
	{
		CLEAR_TENSOR_DEBUGGER();
		ANOMALY_SWITCH;
		if (trainingStep >= settings->maxTrainingSteps)
		{
			settings->doTraining = false;
			return;
		}
		trainingStep++;
		optimizer->zero_grad();

		auto inputTensors = generateTrainingInputTensors();
		const torch::Tensor& position = inputTensors.position;
		const torch::Tensor& wo = inputTensors.wo;
		const torch::Tensor& normal = inputTensors.normal;
		const torch::Tensor& incomingDirection = inputTensors.incomingDirection;
		torch::Tensor& bsdfProb = inputTensors.bsdfProb;
		const torch::Tensor& outRadianceTarget = inputTensors.outRadiance;

		torch::Tensor input = torch::cat({ position, wo, normal }, -1);
		if(settings->useTriangleId)
		{
			input = torch::cat({ input, inputTensors.triangleId }, -1);
		}
		if(settings->useMaterialId)
		{
			input = torch::cat({ input, inputTensors.materialId }, -1);
		}
		if(settings->useInstanceId)
		{
			input = torch::cat({ input, inputTensors.instanceId }, -1);
		}
		const torch::Tensor rawOutput = mlpBase->forward(input);
		TRACE_TENSOR(rawOutput);
		const auto& [mixtureParameters, mixtureWeights, c] = guidingForward(rawOutput);
		const torch::Tensor neuralProb = distribution::Mixture::prob(incomingDirection, mixtureParameters, mixtureWeights, settings->distributionType);

		torch::Tensor pgLoss = guidingLoss(neuralProb, bsdfProb, outRadianceTarget, c);


		if (settings->useEntropyLoss)
		{
			const torch::Tensor entropyLoss = settings->entropyWeight * guidingEntropyLoss(neuralProb);
			pgLoss += entropyLoss;
			if (settings->plotGraphs)
			{
				graphs.addData(LOSS_PLOT, "Entropy Loss", entropyLoss.mean().item<float>());
			}
		}

		if (settings->useAuxiliaryNetwork)
		{
			const torch::Tensor& inRadianceTarget = inputTensors.inRadiance;
			const torch::Tensor& throughputTarget = inputTensors.throughput;
			const torch::Tensor featureVector = rawOutput.narrow(1, mixtureParametersCount + settings->mixtureSize + 1, settings->totAuxInputSize-3);
			const torch::Tensor auxInput = torch::cat({ featureVector, incomingDirection }, -1);
			auto [incomingRadiance, throughput, outgoingRadiance] = splitAuxiliaryOutput(auxInput);
			const torch::Tensor auxLoss = auxiliaryLoss(incomingRadiance, inRadianceTarget, throughput, throughputTarget, outgoingRadiance, outRadianceTarget);

			pgLoss += (1.0f - lossBlendFactor())*auxLoss;

		}

		pgLoss.backward();
		if (TensorDebugger::analyzeGradients()) {
			return; // NAN or INF gradients
		}
		optimizer->step();

		if (settings->plotGraphs)
		{

			graphs.addData(SAMPLING_FRACTION_PLOT, "Sampling Fraction Train", c.mean().item<float>());
			graphs.addData(SAMPLING_FRACTION_PLOT, "Loss Blend", lossBlendFactor());

			distribution::Mixture::setGraphData(settings->distributionType, mixtureParameters, mixtureWeights, graphs, true);
		}

	}

	void PGNet::inference(const int& depth)
	{
		CLEAR_TENSOR_DEBUGGER();
		torch::NoGradGuard noGrad;
		int inferenceSize;
		const auto& inputTensors = generateInferenceInputTensors(&inferenceSize);
		if (inferenceSize == 0)
		{
			return;
		}
		const torch::Tensor& position = inputTensors.position;
		const torch::Tensor& wo = inputTensors.wo;
		const torch::Tensor& normal = inputTensors.normal;
		torch::Tensor input = torch::cat({ position, wo, normal }, -1);

		if (settings->useTriangleId)
		{
			input = torch::cat({ input, inputTensors.triangleId }, -1);
		}
		if (settings->useMaterialId)
		{
			input = torch::cat({ input, inputTensors.materialId }, -1);
		}
		if (settings->useInstanceId)
		{
			input = torch::cat({ input, inputTensors.instanceId }, -1);
		}
		const torch::Tensor rawOutput = mlpBase->forward(input);
		TRACE_TENSOR(rawOutput);
		auto [mixtureParameters, mixtureWeights, c] = guidingForward(rawOutput);
		if (settings->samplingFractionBlend)
		{
			c = tau() * c;
		}

		TRACE_TENSOR(c);
		TRACE_TENSOR(mixtureParameters);
		//print first 10 batches of mixturePrameters
		TRACE_TENSOR(mixtureWeights);
		if (settings->plotGraphs)
		{

			graphs.addData(SAMPLING_FRACTION_PLOT, "Sampling Fraction Inference", c.mean().item<float>(), depth);
			distribution::Mixture::setGraphData(settings->distributionType, mixtureParameters, mixtureWeights, graphs, false, depth);
		}

		copyOutputToCudaBuffer(mixtureParameters, c, mixtureWeights, inferenceSize);

	}

	void PGNet::copyOutputToCudaBuffer(const torch::Tensor& mixtureParameters, const torch::Tensor& c, const torch::Tensor& mixtureWeights, int inferenceSize)
	{
		const device::InferenceBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceBuffers;
		const auto distributionParametersPtr = buffers.distributionParameters.castedPointer<float>();
		const auto samplingFractionPtr = buffers.samplingFractionArrayBuffer.castedPointer<float>();
		const auto mixtureWeightPtr = buffers.mixtureWeightBuffer.castedPointer<float>();

		int inferenceDistributionParameterCount = distribution::Mixture::getDistributionParametersCount(settings->distributionType);
		const torch::Tensor distributionParameterCuda = torch::from_blob(distributionParametersPtr, { inferenceSize, settings->mixtureSize, inferenceDistributionParameterCount }, at::device(device).dtype(torch::kFloat));
		const torch::Tensor samplingFractionCuda = torch::from_blob(samplingFractionPtr, { inferenceSize, 1 }, at::device(device).dtype(torch::kFloat));
		const torch::Tensor mixtureWeightCuda = torch::from_blob(mixtureWeightPtr, { inferenceSize, settings->mixtureSize }, at::device(device).dtype(torch::kFloat));

		distributionParameterCuda.copy_(mixtureParameters);
		samplingFractionCuda.copy_(c);
		mixtureWeightCuda.copy_(mixtureWeights);

		if (false)
		{
			std::cout << "Distribution Parameter Cuda: " << distributionParameterCuda.slice(0, 0, 10) << std::endl;
			std::cout << "Sampling Fraction Cuda: " << samplingFractionCuda.slice(0, 0, 10) << std::endl;
			std::cout << "Mixture Weight Cuda: " << mixtureWeightCuda.slice(0, 0, 10) << std::endl;

			std::cout << "MixtureParameters: " << mixtureParameters.slice(0, 0, 10) << std::endl;
			std::cout << "Sampling Fraction: " << c.slice(0, 0, 10) << std::endl;
			std::cout << "Mixture Weights: " << mixtureWeights.slice(0, 0, 10) << std::endl;
		}

		TRACE_TENSOR(distributionParameterCuda);
		TRACE_TENSOR(samplingFractionCuda);
		TRACE_TENSOR(mixtureWeightCuda);
	}

	GraphsData& PGNet::getGraphs()
	{
		return graphs;
	}

	bool PGNet::doNormalizePosition()
	{
		if (
			settings->inputSettings.normalizePosition ||
			settings->inputSettings.position.otype == config::EncodingType::Frequency ||
			settings->inputSettings.position.otype == config::EncodingType::Grid ||
			settings->inputSettings.position.otype == config::EncodingType::OneBlob ||
			settings->inputSettings.position.otype == config::EncodingType::SphericalHarmonics ||
			settings->inputSettings.position.otype == config::EncodingType::TriangleWave
			)
		{
			return true;
		}
		return false;
	}

	InputTensors PGNet::generateTrainingInputTensors()
	{
		LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
		shuffleDataset(deviceParams);
		const device::NpgTrainingDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.npgTrainingDataBuffers;


		InputDataPointers inputPointers;
		inputPointers.incomingDirection = buffers.incomingDirectionBuffer.castedPointer<float>();
		inputPointers.bsdfProb = buffers.bsdfProbabilitiesBuffer.castedPointer<float>();
		inputPointers.position = buffers.inputBuffer.positionBuffer.castedPointer<float>();

		if (settings->useTriangleId)
		{
			inputPointers.triangleId = buffers.inputBuffer.triangleIdBuffer.castedPointer<float>();
		}
		if (settings->useMaterialId)
		{
			inputPointers.materialId = buffers.inputBuffer.materialIdBuffer.castedPointer<float>();
		}
		if (settings->useInstanceId)
		{
			inputPointers.instanceId = buffers.inputBuffer.instanceIdBuffer.castedPointer<float>();
		}
		inputPointers.wo = buffers.inputBuffer.woBuffer.castedPointer<float>();
		inputPointers.normal = buffers.inputBuffer.normalBuffer.castedPointer<float>();
		inputPointers.outRadiance = buffers.outgoingRadianceBuffer.castedPointer<float>();
		if(settings->useAuxiliaryNetwork)
		{
			inputPointers.throughput = buffers.trhoughputBuffer.castedPointer<float>();
			inputPointers.inRadiance = buffers.incomingRadianceBuffer.castedPointer<float>();
		}

		const auto          aabb = onDeviceData->launchParamsData.getHostImage().aabb;
		const torch::Tensor minExtents = torch::tensor({ aabb.minX, aabb.minY, aabb.minZ }).to(device).to(torch::kFloat);
		const torch::Tensor maxExtents = torch::tensor({ aabb.maxX, aabb.maxY, aabb.maxZ }).to(device).to(torch::kFloat);
		const torch::Tensor deltaExtents = maxExtents - minExtents;

		InputTensors output = generateInputTensors(settings->batchSize, inputPointers, minExtents, deltaExtents);
		TRACE_TENSOR(output.position);
		TRACE_TENSOR(output.wo);
		TRACE_TENSOR(output.normal);
		TRACE_TENSOR(output.incomingDirection);
		TRACE_TENSOR(output.bsdfProb);
		TRACE_TENSOR(output.outRadiance);
		return output;
	}

	InputTensors PGNet::generateInferenceInputTensors(int* inferenceSize)
	{
		const device::InferenceBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceBuffers;
		CUDABuffer inferenceSizeBuffers = buffers.inferenceSize;
		inferenceSizeBuffers.download(inferenceSize);
		if (*inferenceSize == 0)
		{
			return {};
		}

		InputDataPointers inputPointers;
		inputPointers.position = buffers.stateBuffer.positionBuffer.castedPointer<float>();
		inputPointers.wo = buffers.stateBuffer.woBuffer.castedPointer<float>();
		inputPointers.normal = buffers.stateBuffer.normalBuffer.castedPointer<float>();
		if (settings->useTriangleId)
		{
			inputPointers.triangleId = buffers.stateBuffer.triangleIdBuffer.castedPointer<float>();
		}
		if (settings->useMaterialId)
		{
			inputPointers.materialId = buffers.stateBuffer.materialIdBuffer.castedPointer<float>();
		}
		if (settings->useInstanceId)
		{
			inputPointers.instanceId = buffers.stateBuffer.instanceIdBuffer.castedPointer<float>();
		}

		const auto          aabb = onDeviceData->launchParamsData.getHostImage().aabb;
		const torch::Tensor minExtents = torch::tensor({ aabb.minX, aabb.minY, aabb.minZ }).to(device).to(torch::kFloat);
		const torch::Tensor maxExtents = torch::tensor({ aabb.maxX, aabb.maxY, aabb.maxZ }).to(device).to(torch::kFloat);
		const torch::Tensor deltaExtents = maxExtents - minExtents;


		InputTensors output = generateInputTensors(*inferenceSize, inputPointers, minExtents, deltaExtents);
		TRACE_TENSOR(output.position);
		TRACE_TENSOR(output.wo);
		TRACE_TENSOR(output.normal);
		TRACE_TENSOR(output.hitProp);

		return output;
	}

	torch::Tensor PGNet::pointerToTensor(float* pointer, int batchSize, int dim, float minClamp = 0.0f, float maxClamp = 0.0f)
	{
		if (pointer == nullptr) return {};
		torch::Tensor tensor = torch::from_blob(pointer, { batchSize, dim }, torch::TensorOptions().device(device).dtype(torch::kFloat))toPrecision;
		if (minClamp != 0.0f || maxClamp != 0.0f)
		{
			tensor = torch::clamp(tensor, minClamp, maxClamp);
		}

		return tensor;
	}

	InputTensors PGNet::generateInputTensors(const int batchDim, const InputDataPointers& inputPointers, const torch::Tensor& minExtents, const torch::Tensor& deltaExtents)
	{
		InputTensors tensors;

		tensors.position = pointerToTensor(inputPointers.position, batchDim, 3);
		if (doNormalizePosition())
		{
			//#ifdef _DEBUG
			if (!deltaExtents.eq(0).any().item<bool>())
			{
				tensors.position = (tensors.position - minExtents) / deltaExtents;
			}
			//#else
						//tensors.position = (tensors.position - minExtents) / deltaExtents;
			//#endif
		}
		tensors.wo = pointerToTensor(inputPointers.wo, batchDim, 3);
		tensors.normal = pointerToTensor(inputPointers.normal, batchDim, 3);
		if (settings->inputSettings.normal.otype == config::EncodingType::SphericalHarmonics)
		{
			tensors.wo = (tensors.wo + 1.0f) / 2.0f; // map to [0,1]
		}
		if (settings->inputSettings.normal.otype == config::EncodingType::SphericalHarmonics)
		{
			tensors.normal = (tensors.normal + 1.0f) / 2.0f; // map to [0,1]
		}
		tensors.incomingDirection = pointerToTensor(inputPointers.incomingDirection, batchDim, 3);
		tensors.bsdfProb = pointerToTensor(inputPointers.bsdfProb, batchDim, 1, -65504, 65504);
		tensors.outRadiance = pointerToTensor(inputPointers.outRadiance, batchDim, 3);
		tensors.inRadiance = pointerToTensor(inputPointers.inRadiance, batchDim, 3);
		tensors.throughput = pointerToTensor(inputPointers.throughput, batchDim, 3);
		tensors.triangleId = pointerToTensor(inputPointers.triangleId, batchDim, 1);
		tensors.instanceId = pointerToTensor(inputPointers.instanceId, batchDim, 1);
		tensors.materialId = pointerToTensor(inputPointers.materialId, batchDim, 1);
		return tensors;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::splitGuidingOutput(const torch::Tensor& rawOutput)
	{
		const torch::Tensor rawSamplingFraction = rawOutput.narrow(1, 0, 1);
		torch::Tensor rawMixtureParameters = rawOutput.narrow(1, 1, mixtureParametersCount);
		rawMixtureParameters = rawMixtureParameters.view({ rawOutput.size(0), settings->mixtureSize, distributionParametersCount });
		const torch::Tensor rawMixtureWeights = rawOutput.narrow(1, 1 + mixtureParametersCount, settings->mixtureSize);
		TRACE_TENSOR(rawMixtureParameters);
		TRACE_TENSOR(rawMixtureWeights);
		TRACE_TENSOR(rawSamplingFraction);

		return { rawMixtureParameters, rawMixtureWeights, rawSamplingFraction };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::finalizeGuidingOutput(torch::Tensor& rawMixtureParameters, const torch::Tensor& rawMixtureWeights, const torch::Tensor& rawSamplingFraction)
	{
		torch::Tensor mixtureParameters = distribution::Mixture::finalizeParams(rawMixtureParameters, settings->distributionType);
		torch::Tensor mixtureWeights = softmax(rawMixtureWeights, 1);
		torch::Tensor cTensor = sigmoid(rawSamplingFraction);
		TRACE_TENSOR(mixtureParameters);
		TRACE_TENSOR(mixtureWeights);
		TRACE_TENSOR(cTensor);
		return { mixtureParameters, mixtureWeights, cTensor };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::guidingForward(const torch::Tensor& rawOutput)
	{
		auto [rawMixtureParameters, rawMixtureWeights, rawSamplingFraction] = splitGuidingOutput(rawOutput);
		auto [mixtureParameters, mixtureWeights, cTensor] = finalizeGuidingOutput(rawMixtureParameters, rawMixtureWeights, rawSamplingFraction);
		return { mixtureParameters, mixtureWeights, cTensor };
	}

	torch::Tensor PGNet::guidingLoss(const torch::Tensor& neuralProb, torch::Tensor& bsdfProb, const torch::Tensor& outRadiance, const torch::Tensor& c)
	{
		//const auto mask = bsdfProb > 100.0f; // Replace `threshold` with your threshold value
		//const auto neuralProbNoGrad = neuralProb.detach();
		//const auto neuralProbConditional = torch::where(mask, neuralProbNoGrad, neuralProb);

		//auto bsdfProbClamped = torch::clamp(bsdfProb, 0.0f, 1.0f);

		const torch::Tensor targetLuminance = settings->targetScale*(outRadiance * ntscLuminance).sum(-1, true);
		const torch::Tensor lossQ = divergenceLoss(neuralProb, targetLuminance);
		if(settings->clampBsdfProb)
		{
			bsdfProb = torch::min(bsdfProb, targetLuminance);
		}
		const torch::Tensor blendedQ = c * neuralProb + (1 - c) * bsdfProb;
		torch::Tensor lossBlendedQ = divergenceLoss(blendedQ, targetLuminance);
		if(settings->scaleLossBlendedQ)
		{
			lossBlendedQ  = lossBlendedQ / (bsdfProb + 0.01);
		}

		torch::Tensor loss = lossBlendFactor() * lossQ + (1.0f - lossBlendFactor()) * lossBlendedQ;

		switch (settings->lossReduction)
		{
		case config::SUM:
			loss = loss.sum();
			break;
		case config::MEAN:
			loss = loss.mean();
			break;
		case config::ABS:
			loss = torch::abs(loss);
			break;
		default:
			VTX_ERROR("Loss Reduction Not implemented");
		}

		TRACE_TENSOR(loss);

		if (settings->plotGraphs)
		{
			graphs.addData(LOSS_PLOT, "Path Guiding Loss", loss.unsqueeze(-1).item<float>());
			graphs.addData(LOSS_PLOT, "Loss Q", lossQ.mean().unsqueeze(-1).item<float>());
			graphs.addData(LOSS_PLOT, "Loss Blended Q", lossBlendedQ.mean().unsqueeze(-1).item<float>());

			graphs.addData(PROB_PLOT, "Neural Prob", neuralProb.mean().unsqueeze(-1).item<float>());
			graphs.addData(PROB_PLOT, "Target Prob", targetLuminance.mean().unsqueeze(-1).item<float>());

			graphs.addData(SAMPLING_PROB_PLOT, "BSDF Prob", bsdfProb.mean().unsqueeze(-1).item<float>());
			graphs.addData(SAMPLING_PROB_PLOT, "Blended Prob", blendedQ.mean().unsqueeze(-1).item<float>());
		}


		TRACE_TENSOR(blendedQ);
		TRACE_TENSOR(lossQ);
		TRACE_TENSOR(lossBlendedQ);
		TRACE_TENSOR(loss);
		return loss;
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PGNet::splitAuxiliaryOutput(const torch::Tensor& rawAuxiliaryOutput)
	{
		torch::Tensor incomingRadiance = rawAuxiliaryOutput.narrow(1, 0, 3);
		torch::Tensor throughput = rawAuxiliaryOutput.narrow(1, 3, 3);
		torch::Tensor outgoingRadiance = rawAuxiliaryOutput.narrow(1, 6, 3);
		TRACE_TENSOR(incomingRadiance);
		TRACE_TENSOR(throughput);
		TRACE_TENSOR(outgoingRadiance);
		return { incomingRadiance, throughput, outgoingRadiance };
	}

	torch::Tensor PGNet::relativeL2LossRadiance(const torch::Tensor& radiance, const torch::Tensor& radianceTarget)
	{
		const torch::Tensor scaledTarget = (radianceTarget * settings->radianceTargetScaleFactor).clamp(0.0f, 1e+6f); // GfxExp says it is required to avoid dominance of the epsilon
		const torch::Tensor difference = radiance - scaledTarget;
		const torch::Tensor luminance = (radiance * ntscLuminance).sum(-1, true).detach();
		const torch::Tensor predictionSqPlusEpsilon = torch::pow(luminance, 2) + 0.01f;
		const torch::Tensor relativeL2 = torch::pow(difference, 2) / predictionSqPlusEpsilon;
		const torch::Tensor relativeL2Mean = relativeL2.mean();

		//PRINT_TENSOR_SLICE(radiance);
		//PRINT_TENSOR_SLICE(radianceTarget);
		//PRINT_TENSOR_SLICE(scaledTarget);
		//PRINT_TENSOR_SLICE(difference);
		//PRINT_TENSOR_SLICE(luminance);
		//PRINT_TENSOR_SLICE(predictionSqPlusEpsilon);
		//PRINT_TENSOR_SLICE(relativeL2);
		//PRINT_TENSOR_ALWAYS("relativeL2Mean", relativeL2Mean);

		//PRINT_TENSOR_ALWAYS("Loss Radiance", radiance, radianceTarget, squaredLuminance, relativeL2);

		TRACE_TENSOR(relativeL2);

		return relativeL2Mean;
	}

	torch::Tensor PGNet::relativeL2LossThroughput(const torch::Tensor& throughput, const torch::Tensor& throughputTarget)
	{
		const torch::Tensor scaledTarget = (throughputTarget * settings->throughputTargetScaleFactor).clamp(0.0f, 1e+6f); // GfxExp says it is required to avoid dominance of the epsilon
		const torch::Tensor difference = throughput - scaledTarget;

		const torch::Tensor denominator = torch::pow(throughput, 2).detach() + 0.01f;
		const torch::Tensor relativeL2 = torch::pow(difference, 2) / denominator;
		torch::Tensor relativeL2Mean = relativeL2.mean();

		//PRINT_TENSOR_SLICE(throughputTarget);
		//PRINT_TENSOR_SLICE(scaledTarget);
		//PRINT_TENSOR_SLICE(throughput);
		//PRINT_TENSOR_SLICE(difference);
		//PRINT_TENSOR_SLICE(denominator);
		//PRINT_TENSOR_SLICE(relativeL2);
		//PRINT_TENSOR_ALWAYS("relativeL2Mean", relativeL2Mean);

		TRACE_TENSOR(denominator);
		TRACE_TENSOR(relativeL2);

		return relativeL2Mean;
	}

	torch::Tensor PGNet::auxiliaryLoss(
		const torch::Tensor& inRadiance, const torch::Tensor& inRadianceTarget,
		const torch::Tensor& throughput, const torch::Tensor& throughputTarget,
		const torch::Tensor& outRadiance, const torch::Tensor& outRadianceTarget)

	{
		const float totWeights = settings->inRadianceLossFactor + settings->outRadianceLossFactor + settings->throughputLossFactor;
		const float inRadScale = settings->inRadianceLossFactor; // / totWeights;
		const float outRadScale = settings->outRadianceLossFactor;// / totWeights;
		const float throughputScale = settings->throughputLossFactor;// / totWeights;

		const torch::Tensor lossInRadiance = inRadScale * relativeL2LossRadiance(inRadiance, inRadianceTarget);
		const torch::Tensor lossOutRadiance = outRadScale * relativeL2LossRadiance(outRadiance, outRadianceTarget);
		const torch::Tensor lossThroughput = throughputScale * relativeL2LossThroughput(throughput, throughputTarget);

		torch::Tensor loss = settings->auxiliaryWeight * (lossInRadiance +  lossOutRadiance +  lossThroughput);

		if (settings->plotGraphs)
		{
			graphs.addData("Auxiliary Loss","Iteration","Loss", "Input Radiance loss", lossInRadiance.item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Output Radiance loss", lossOutRadiance.item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Throughput loss", lossThroughput.item<float>());
			graphs.addData("Auxiliary Loss", "Iteration", "Loss", "Total loss", loss.item<float>());
		}
		TRACE_TENSOR(loss);

		return loss;
	}

	torch::Tensor PGNet::guidingEntropyLoss(const torch::Tensor& neuralProb)
	{
		const torch::Tensor entropy = -torch::sum(neuralProb * torch::log(neuralProb + EPS), 1);
		const float targetEntropy = -logf(settings->targetEntropy);
		torch::Tensor entropyLoss = torch::pow(entropy - targetEntropy, 2).mean();
		TRACE_TENSOR(entropy);
		TRACE_TENSOR(entropyLoss);
		return entropyLoss;
	}

	torch::Tensor PGNet::divergenceLoss(const torch::Tensor& neuralProb, const torch::Tensor& targetProb)
	{
		torch::Tensor loss;
		switch (settings->lossType)
		{
		case config::L_KL_DIV:
			loss = kl_div(torch::log(neuralProb + EPS), targetProb, at::Reduction::None);
			break;
		case config::L_KL_DIV_MC_ESTIMATION:
			loss = - targetProb * torch::log(neuralProb + EPS);
			break;
		case config::L_PEARSON_DIV:
			loss = torch::pow(targetProb - neuralProb, 2.0f) / (neuralProb + EPS);
			break;
		case config::L_PEARSON_DIV_MC_ESTIMATION:
			loss = -torch::pow(targetProb, 2.0f) * torch::log(neuralProb + EPS);
			break;
		default:
			VTX_ERROR("Loss Type Not implemented");
			loss = torch::zeros({ 1 });
		}
		TRACE_TENSOR(loss);

		return loss;
	}

	float PGNet::lossBlendFactor()
	{
		if (settings->constantBlendFactor)
		{
			return settings->blendFactor;
		}
		return std::min(1.0f, pow(0.3333f, 5.0f * tau()));
	}

	float PGNet::tau()
	{
		float tau = std::min(1.0f, (float)trainingStep / ((float)settings->maxTrainingSteps));
		return tau;
	}
}

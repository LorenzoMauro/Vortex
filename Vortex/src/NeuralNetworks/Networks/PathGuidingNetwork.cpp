#include "PathGuidingNetwork.h"
#include "NeuralNetworks/tools.h"
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx::network
{

	PathGuidingNetwork::PathGuidingNetwork(const int& _inputDim, const torch::Device& _device, PathGuidingNetworkSettings* _settings)
	{
		settings = _settings;
		device = _device;
		inputDim = _inputDim;
		init();
	}

	void PathGuidingNetwork::init()
	{
		computeOutputDim();
		std::vector<int64_t> networkShape = { inputDim };
		std::vector<ActivationType> activationTypes;

		for (int i = 0; i < settings->numHiddenLayers; i++)
		{
			networkShape.push_back(settings->hiddenDim);
		}

		for (int i = 0; i < settings->numHiddenLayers; i++)
		{
			activationTypes.push_back(AT_RELU);
		}
		networkShape.push_back(outputDim);
		activationTypes.push_back(AT_NONE);
		network = FcNetwork(networkShape, activationTypes);
		network->to(device);
	}

	torch::Tensor PathGuidingNetwork::evaluate(const torch::Tensor& input, const torch::Tensor& sample)
	{
		run(input);
		PRINT_TENSORS("PGN RUN", cTensor, mixtureWeights, mixtureParameters);

		return distribution::Mixture::prob(sample, mixtureParameters, mixtureWeights, settings->distributionType);
	}

	std::tuple<torch::Tensor, torch::Tensor> PathGuidingNetwork::sample(const torch::Tensor& input)
	{
		run(input);

		return distribution::Mixture::sample(mixtureParameters, mixtureWeights, settings->distributionType);
	}

	torch::Tensor& PathGuidingNetwork::getLastRunSamplingFraction()
	{
		return cTensor;
	}

	std::vector<torch::Tensor> PathGuidingNetwork::parameters()
	{
		return network->parameters();
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
		PathGuidingNetwork::inferenceWithSample(const torch::Tensor& input)
	{
		torch::NoGradGuard noGrad;
		auto [sampleTensor, p] = sample(input);
		CHECK_TENSOR_ANOMALY(sampleTensor);
		CHECK_TENSOR_ANOMALY(p);
		return { mixtureParameters, mixtureWeights, sampleTensor, p, cTensor };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PathGuidingNetwork::inference(const torch::Tensor& input)
	{
		torch::NoGradGuard noGrad;
		run(input);
		return { mixtureParameters, mixtureWeights, cTensor };
	}

	int PathGuidingNetwork::getMixtureParameterCount()
	{
		return mixtureParametersCount;
	}

	void PathGuidingNetwork::computeOutputDim()
	{
		outputDim = 0;
		distributionParametersCount = 0;
		mixtureParametersCount = 0;

		distributionParametersCount = distribution::Mixture::getDistributionParametersCount(settings->distributionType, true);

		mixtureParametersCount = settings->mixtureSize * distributionParametersCount;

		outputDim += mixtureParametersCount;
		outputDim += settings->mixtureSize;

		if (settings->produceSamplingFraction)
		{
			outputDim += 1;
		}
	}

	std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> PathGuidingNetwork::run(const torch::Tensor& input) {
		const torch::Tensor output = network->forward(input);
		CHECK_TENSOR_ANOMALY(output);

		torch::Tensor rawMixtureParameters = output.narrow(1, 0, mixtureParametersCount);
		rawMixtureParameters = rawMixtureParameters.view({ input.size(0), settings->mixtureSize, distributionParametersCount });
		mixtureParameters = distribution::Mixture::finalizeParams(rawMixtureParameters, settings->distributionType);
		CHECK_TENSOR_ANOMALY(mixtureParameters);

		const torch::Tensor rawMixtureWeights = output.narrow(1, mixtureParametersCount, settings->mixtureSize);
		mixtureWeights = softmax(rawMixtureWeights, 1);
		CHECK_TENSOR_ANOMALY(mixtureWeights);

		if (settings->produceSamplingFraction)
		{
			const torch::Tensor rawSamplingFraction = output.narrow(1, mixtureParametersCount + settings->mixtureSize, 1);
			cTensor                           = sigmoid(rawSamplingFraction);
			CHECK_TENSOR_ANOMALY(cTensor);
		}

		PRINT_TENSORS("PGN RUN", cTensor, mixtureWeights, mixtureParameters);
		return { mixtureParameters, mixtureWeights, cTensor };
	}


}

﻿#pragma once
#ifndef PATH_GUIDING_NETWORK_TORCH_H
#define PATH_GUIDING_NETWORK_TORCH_H

#include "FullyConnectedNetwork.h"
#include "NeuralNetworks/NetworkSettings.h"
#include "tcnn/TcnnTorchModule.h"
#define TCNN
namespace vtx::network
{
	
	class PathGuidingNetwork
	{
	public:
		PathGuidingNetwork() = default;

		PathGuidingNetwork(
			const int& _inputDim,
			const torch::Device& _device,
			PathGuidingNetworkSettings* _settings,
			InputSettings* _inputSettings);

		void init();

		torch::Tensor evaluate(const torch::Tensor& input, const torch::Tensor& sample);

		std::tuple<torch::Tensor, torch::Tensor> sample(const torch::Tensor& input);

		torch::Tensor& getLastRunSamplingFraction();

		std::vector<torch::Tensor> parameters();

		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> inferenceWithSample(const torch::Tensor& input);

		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> inference(const torch::Tensor& input);

		int getMixtureParameterCount();

		std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> run(const torch::Tensor& input);

	private:

		void computeOutputDim();


	public:
#ifdef TCNN
		torchTcnn::TcnnModule network;
#else
		FcNetwork network;
#endif
		int64_t inputDim;
		int64_t outputDim;

		int distributionParametersCount = 0;
		int mixtureParametersCount = 0;

		torch::Tensor mixtureParameters;
		torch::Tensor mixtureWeights;
		torch::Tensor cTensor;

		torch::Device device = torch::kCPU;

		PathGuidingNetworkSettings* settings = nullptr;
		InputSettings*                         inputSettings;
	};
}

#endif

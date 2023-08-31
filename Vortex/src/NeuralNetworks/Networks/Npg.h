#pragma once
#ifndef NPG_H
#define NPG_H
#include "PathGuidingNetwork.h"
#include "InputComposer.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "NeuralNetworks/NetworkImplementation.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx
{
	struct LaunchParams;
}

namespace vtx::network
{
	class Npg : public NetworkImplementation
	{
	public:
        Npg(NetworkSettings* _settings);

        void init() override;

        void reset() override;
		
        void train() override;

		void inference(const int& depth) override;

		GraphsData& getGraphs() override;

	private:

        void trainStep(
			const torch::Tensor& input,
			const torch::Tensor& luminance,
			const torch::Tensor& incomingDirection,
			const torch::Tensor& bsdfProb);

        float lossBlendFactor();

        float inferenceSamplingFractionBlend();

        float tau();

        torch::Tensor loss(const torch::Tensor& neuralProb, const torch::Tensor& targetProb);

	public:

        torch::Device device;
        GraphsData graphs;
        InputComposer ic;
        PathGuidingNetwork pgn;
        std::shared_ptr<torch::optim::Adam> optimizer;
        int trainingStep;
	};
}

#endif
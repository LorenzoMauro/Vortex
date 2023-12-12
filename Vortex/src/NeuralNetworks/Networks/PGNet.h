#pragma once
#include "NeuralNetworks/NetworkImplementation.h"
#include <torch/torch.h>

#include "Device/UploadCode/UploadBuffers.h"
#include "tcnn/TcnnTorchModule.h"

namespace vtx::network
{
	struct InputTensors
	{
		torch::Tensor position = torch::Tensor();
		torch::Tensor wo = torch::Tensor();
		torch::Tensor normal = torch::Tensor();
		torch::Tensor incomingDirection = torch::Tensor();
		torch::Tensor bsdfProb = torch::Tensor();
		torch::Tensor outRadiance = torch::Tensor();
		torch::Tensor throughput = torch::Tensor();
		torch::Tensor inRadiance = torch::Tensor();
		torch::Tensor instanceId = torch::Tensor();
		torch::Tensor triangleId = torch::Tensor();
		torch::Tensor materialId = torch::Tensor();
	};

	struct InputDataPointers
	{
		float* position          = nullptr;
		float* wo                = nullptr;
		float* normal            = nullptr;
		float* incomingDirection = nullptr;
		float* bsdfProb          = nullptr;
		float* outRadiance       = nullptr;
		float* throughput        = nullptr;
		float* inRadiance        = nullptr;
		float* instanceId		 = nullptr;
		float* materialId		 = nullptr;
		float* triangleId		 = nullptr;
	};

	class PGNet : public NetworkImplementation
	{
	public:
		PGNet(config::NetworkSettings* _settings);

		void init() override;

		void reset() override;

		void train() override;
		//void registerTensorPlot(GraphType graphType, const torch::Tensor& tensor);
		//void addTensorsToPlot();

		void inference(const int& depth) override;
		void copyOutputToCudaBuffer(const torch::Tensor& mixtureParameters, const torch::Tensor& c,
			const torch::Tensor& mixtureWeights, int                     inferenceSize);

		GraphsData& getGraphs() override;
		bool          doNormalizePosition();
		InputTensors  generateTrainingInputTensors();
		InputTensors  generateInferenceInputTensors(int* inferenceSize);
		torch::Tensor pointerToTensor(float* pointer, int batchSize, int dim, float minClamp, float maxClamp);


		InputTensors generateInputTensors(
			int                      batchDim,
			const InputDataPointers& inputPointers,
			const torch::Tensor& minExtents,
			const torch::Tensor& deltaExtents);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> splitGuidingOutput(const torch::Tensor& rawOutput);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> finalizeGuidingOutput(
			torch::Tensor& rawMixtureParameters, const torch::Tensor& rawMixtureWeights,
			const torch::Tensor& rawSamplingFraction);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> guidingForward(const torch::Tensor& rawOutput);
		torch::Tensor guidingLoss(const torch::Tensor& neuralProb, torch::Tensor& bsdfProb,
								  const torch::Tensor& outRadiance, const torch::Tensor& c);
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> splitAuxiliaryOutput(
			const torch::Tensor& rawAuxiliaryOutput);
		torch::Tensor relativeL2LossRadiance(const torch::Tensor& radiance, const torch::Tensor& radianceTarget);
		torch::Tensor relativeL2LossThroughput(const torch::Tensor& throughput, const torch::Tensor& throughputTarget);
		torch::Tensor auxiliaryLoss(const torch::Tensor& inRadiance, const torch::Tensor& inRadianceTarget,
			const torch::Tensor& throughput, const torch::Tensor& throughputTarget,
			const torch::Tensor& outRadiance, const torch::Tensor& outRadianceTarget);
		torch::Tensor guidingEntropyLoss(const torch::Tensor& neuralProb);
		torch::Tensor divergenceLoss(const torch::Tensor& neuralProb, const torch::Tensor& targetProb);
		float         lossBlendFactor();
		float         tau();

	private:
	public:
		std::shared_ptr<torch::optim::Adam> optimizer;
		torch::Device device;


		torchTcnn::TcnnModule mlpBase;
		torchTcnn::TcnnModule auxiliaryMlp;

		int outputDim;

		int trainingStep = 0;

		int mixtureParametersCount;
		int distributionParametersCount;

		GraphsData graphs;

		torch::Tensor ntscLuminance;
		//std::vector<GraphType> plotTypes;
		//std::vector<torch::Tensor> plotTensors;
	};
}

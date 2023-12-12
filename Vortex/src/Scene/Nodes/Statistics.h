#pragma once
#include <memory>

namespace vtx::graph
{
	class Renderer;

	struct Statistics
	{
		void update(const std::shared_ptr<graph::Renderer>& renderNode);

		float totTimeSeconds = 0.0f;
		int samplesPerPixel = 0.0f;
		float sppPerSecond = 0.0f;
		float frameTime = 0.0f;
		float fps = 0.0f;
		float totTimeInternal = 0.0f;
		float internalFps = 0.0f;
		float rendererNoise = 0.0f;
		float rendererTrace = 0.0f;
		float rendererPost = 0.0f;
		float rendererDisplay = 0.0f;
		float waveFrontGenerateRay = 0.0f;
		float waveFrontTrace = 0.0f;
		float waveFrontShade = 0.0f;
		float waveFrontShadow = 0.0f;
		float waveFrontEscaped = 0.0f;
		float waveFrontAccumulate = 0.0f;
		float waveFrontReset = 0.0f;
		float waveFrontFetchQueueSize = 0.0f;
		float neuralShuffleDataset = 0.0f;
		float neuralNetworkTrain = 0.0f;
		float neuralNetworkInfer = 0.0f;
	};
}

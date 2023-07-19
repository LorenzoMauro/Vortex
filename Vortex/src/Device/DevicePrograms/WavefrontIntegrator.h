#include "LaunchParams.h"
#include "NeuralNetworks/NeuralNetwork.h"
#include "Scene/Nodes/RendererSettings.h"

namespace vtx
{
	enum Queue
	{
		Q_RADIANCE_TRACE,
		Q_SHADE,
		Q_ESCAPED,
		Q_ACCUMULATION,
		Q_PIXEL,
		Q_SHADOW_TRACE
	};

	class WaveFrontIntegrator
	{
	public:

		WaveFrontIntegrator(graph::RendererSettings* rendererSettings)
		{
			queueSizeRetrievalBuffer.alloc(sizeof(int));
			queueSizeDevicePtr = queueSizeRetrievalBuffer.castedPointer<int>();
			this->settings = rendererSettings;
		}

		void render();

		void downloadCounters();

		void generatePixelQueue();

		void launchOptixKernel(math::vec2i launchDimension, std::string pipelineName);

		void traceRadianceRays();

		void resetQueue(Queue queue);

		void setCounters();

		void resetCounters();

		void shadeRays();

		void handleShadowTrace();

		void handleEscapedRays();

		void accumulateRays();

		LaunchParams* hostParams;
		LaunchParams* deviceParams;
		int           maxTraceQueueSize;
		CUDABuffer    queueSizeRetrievalBuffer;
		int*          queueSizeDevicePtr;
		int           retrievedQueueSize;
		graph::RendererSettings* settings;
		Counters 	counters;

		CUDABuffer tmpIndicesBuffer;
		int* tmpIndices;
		network::Network network;
	};
}

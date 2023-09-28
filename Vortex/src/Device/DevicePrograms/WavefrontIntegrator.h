#include "LaunchParams.h"
#include "Core/Options.h"
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

		WaveFrontIntegrator(RendererSettings* _rendererSettings)
		{
			queueSizeRetrievalBuffer.alloc(sizeof(int));
			queueSizeDevicePtr = queueSizeRetrievalBuffer.castedPointer<int>();
			rendererSettings = _rendererSettings;
			settings = getOptions()->wavefrontSettings;
		}

		void render();

		void generatePixelQueue();

		void traceRadianceRays();

		void downloadCountersPointers();

		void downloadQueueSize(const int* deviceSize, int& hostSize, int maxSize);

		void shadeRays();

		void handleShadowTrace();

		void handleEscapedRays();

		void accumulateRays();

		const LaunchParams* hostParams;
		const LaunchParams* deviceParams;
		int           maxTraceQueueSize;
		CUDABuffer    queueSizeRetrievalBuffer;
		int*          queueSizeDevicePtr;
		int           retrievedQueueSize;
		RendererSettings* rendererSettings;
		WavefrontSettings settings;
		Counters 	deviceCountersPointers;
		QueueSizes queueSizes;

		CUDABuffer tmpIndicesBuffer;
		int* tmpIndices;
		network::Network network;
	};
}

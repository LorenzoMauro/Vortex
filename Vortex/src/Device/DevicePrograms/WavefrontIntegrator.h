#include "LaunchParams.h"
#include "Scene/Nodes/RendererSettings.h"

namespace vtx
{
	struct KernelTimes
	{
		float genCameraRay     = 0.0f;
		float traceRadianceRay = 0.0f;
		float reset            = 0.0f;
		float shadeRay         = 0.0f;
		float handleEscapedRay = 0.0f;
		float accumulateRay    = 0.0f;
		float fetchQueueSize   = 0.0f;
		float setQueueCounters = 0.0f;

		float totMs()
		{
			return setQueueCounters + genCameraRay + traceRadianceRay + reset + shadeRay + handleEscapedRay + accumulateRay + fetchQueueSize;
		}
	};

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

		WaveFrontIntegrator(graph::RendererSettings* settings);

		KernelTimes& getKernelTime();

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
		KernelTimes   kernelTimes;
		CUDABuffer    queueSizeRetrievalBuffer;
		int*          queueSizeDevicePtr;
		int           retrievedQueueSize;
		graph::RendererSettings* settings;
		Counters 	counters;
	};
}

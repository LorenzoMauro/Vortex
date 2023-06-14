#include "LaunchParams.h"
#include "Core/Timer.h"

namespace vtx
{
	struct KernelTimes
	{
		float genCameraRay           = 0.0f;
		float traceRadianceRay       = 0.0f;
		float reset                  = 0.0f;
		float shadeRay               = 0.0f;
		float handleEscapedRay       = 0.0f;
		float accumulateRay          = 0.0f;
		float fetchQueueSize         = 0.0f;
		float pixelQueue			 = 0.0f;

		float totMs()
		{
			return genCameraRay + traceRadianceRay + reset + shadeRay + handleEscapedRay + accumulateRay + fetchQueueSize + pixelQueue;
		}
	};

	enum Queue
	{
		Q_RADIANCE_TRACE,
		Q_SHADE,
		Q_ESCAPED,
		Q_ACCUMULATION,
		Q_PIXEL
	};

	class WaveFrontIntegrator
	{
	public:

		WaveFrontIntegrator();

		KernelTimes& getKernelTime();

		void cudaShade();

		void render(bool fitKernelSize, int iteration);

		void generatePixelQueue();

		void generateCameraRadianceRays();

		void traceRadianceRays();

		void resetQueue(Queue queue);

		int fetchQueueSize(Queue queue);

		void shadeRays();

		void handleEscapedRays();

		void accumulateRays();

		LaunchParams* hostParams;
		LaunchParams* deviceParams;
		int           numberOfPixels;
		KernelTimes   kernelTimes;
		CUDABuffer    queueSizeRetrievalBuffer;
		int*          queueSizeDevicePtr;
		int           retrievedQueueSize;
		bool           fitKernelSize;
	};
}

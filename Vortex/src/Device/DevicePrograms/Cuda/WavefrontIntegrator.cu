#include "../WavefrontIntegrator.h"
#include "../randomNumberGenerator.h"
#define ARCHITECTURE_OPTIX
#include "../rendererFunctions.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "Device/Wrappers/KernelLaunch.h"
#include "Device/Wrappers/KernelTimings.h"
#include "MDL/CudaLinker.h"
#include "NeuralNetworks/Experiment.h"

namespace vtx
{
	
	void WaveFrontIntegrator::render()
	{
		hostParams = &UPLOAD_DATA->launchParams;
		deviceParams = UPLOAD_BUFFERS->launchParamsBuffer.castedPointer<LaunchParams>();
		maxTraceQueueSize = UPLOAD_DATA->frameBufferData.frameSize.x * UPLOAD_DATA->frameBufferData.frameSize.y;

		generatePixelQueue();

		if (rendererSettings->iteration <= 0 && settings.fitWavefront)
		{
			downloadCountersPointers();
		}

		for (int frameBounces = 0; frameBounces < rendererSettings->maxBounces; frameBounces++)
		{
			traceRadianceRays();
			network.inference(frameBounces);
			shadeRays();
		}
		handleShadowTrace();
		handleEscapedRays();
		accumulateRays();
		network.train();

	}

	void WaveFrontIntegrator::downloadCountersPointers()
	{
		if (settings.fitWavefront)
		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_RETRIEVE_QUEUE_SIZE]);
			
			cudaEventRecord(events.first, nullptr);
			UPLOAD_BUFFERS->workQueueBuffers.countersBuffer.download(&deviceCountersPointers);
			cudaEventRecord(events.second, nullptr);
			CUDA_SYNC_CHECK();
		}
	}

	void WaveFrontIntegrator::downloadQueueSize(const int* deviceSize, int& hostSize, int maxSize)
	{
		if (settings.fitWavefront)
		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_RETRIEVE_QUEUE_SIZE]);
			cudaEventRecord(events.first, nullptr);
			CUDA_CHECK(cudaMemcpy((void*)&hostSize, deviceSize, sizeof(int), cudaMemcpyDeviceToHost));
			cudaEventRecord(events.second, nullptr);
		}
		else
		{
			hostSize = maxSize;
		}
	}

	void WaveFrontIntegrator::generatePixelQueue()
	{
		auto deviceParamsCopy = deviceParams;
		gpuParallelFor(eventNames[K_GEN_CAMERA_RAY],
			maxTraceQueueSize,
			[deviceParamsCopy] __device__(int id)
		{
			wfInitRayEntry(id, deviceParamsCopy);
		});
	}

	void WaveFrontIntegrator::traceRadianceRays()
	{
		downloadQueueSize(deviceCountersPointers.traceQueueCounter, queueSizes.traceQueueCounter, maxTraceQueueSize);
		if (queueSizes.traceQueueCounter == 0)
			return;

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_TRACE_RADIANCE_RAY]);
		cudaEventRecord(events.first, nullptr);

		optix::PipelineOptix::launchOptixKernel(math::vec2i(queueSizes.traceQueueCounter, 1), "wfRadianceTrace");

		cudaEventRecord(events.second, nullptr);
	}

	void WaveFrontIntegrator::shadeRays()
	{
		downloadQueueSize(deviceCountersPointers.shadeQueueCounter, queueSizes.shadeQueueCounter, maxTraceQueueSize);
		if (queueSizes.shadeQueueCounter == 0)
		{
			return;
		}

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_SHADE_RAY]);
		cudaEventRecord(events.first);
		if (settings.optixShade)
		{
			optix::PipelineOptix::launchOptixKernel(math::vec2i{queueSizes.shadeQueueCounter, 1}, "wfShade");
		}
		else {
			const CUfunction& cudaFunction = mdl::getMdlCudaLinker().outKernelFunction;
			constexpr int threadsPerBlockX = 256;
			const int     numBlocksX = (queueSizes.shadeQueueCounter + threadsPerBlockX - 1) / threadsPerBlockX;
			void* params[] = { &deviceParams };

			// Launch the kernel
			const CUresult res = cuLaunchKernel(
				cudaFunction,            // function pointer to kernel
				numBlocksX, 1, 1,        // grid dimensions (only x is non-1)
				threadsPerBlockX, 1, 1,  // block dimensions (only x is non-1)
				0, nullptr,              // shared memory size and stream
				params, nullptr          // kernel parameters and extra
			);
			CU_CHECK(res);
			CUDA_SYNC_CHECK();
		}
		cudaEventRecord(events.second);
	}

	void WaveFrontIntegrator::handleShadowTrace()
	{
		if(rendererSettings->samplingTechnique == S_BSDF)
		{
			return;
		}
		downloadQueueSize(deviceCountersPointers.shadeQueueCounter, queueSizes.shadowQueueCounter, maxTraceQueueSize * (rendererSettings->maxBounces + 1));
		if (queueSizes.shadowQueueCounter == 0)
		{
			VTX_INFO("NO Shadow Queue");
			return;
		}

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[Q_SHADOW_TRACE]);
		cudaEventRecord(events.first, nullptr);

		optix::PipelineOptix::launchOptixKernel(math::vec2i(queueSizes.shadowQueueCounter, 1), "wfShadowTrace");

		cudaEventRecord(events.second, nullptr);
	}

	void WaveFrontIntegrator::handleEscapedRays()
	{
		auto deviceParamsCopy = deviceParams;
		downloadQueueSize(deviceCountersPointers.escapedQueueCounter, queueSizes.escapedQueueCounter, maxTraceQueueSize);
		if (queueSizes.escapedQueueCounter == 0)
		{
			return;
		}
		gpuParallelFor(eventNames[K_HANDLE_ESCAPED_RAY],
			queueSizes.escapedQueueCounter,
			[deviceParamsCopy] __device__(int id)
		{
			wfEscapedEntry(id, deviceParamsCopy);
		});
	}

	void WaveFrontIntegrator::accumulateRays()
	{
		downloadQueueSize(deviceCountersPointers.accumulationQueueCounter, queueSizes.accumulationQueueCounter, maxTraceQueueSize * (rendererSettings->maxBounces + 1));
		if (queueSizes.accumulationQueueCounter == 0)
		{
			return;
		}

		auto deviceParamsCopy = deviceParams;
		gpuParallelFor(eventNames[K_ACCUMULATE_RAY],
			queueSizes.accumulationQueueCounter,
			[deviceParamsCopy] __device__(int id)
		{
			wfAccumulateEntry(id, deviceParamsCopy);
		});
	}
}

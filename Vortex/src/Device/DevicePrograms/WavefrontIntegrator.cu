#include "WavefrontIntegrator.h"
#include "randomNumberGenerator.h"
#define ARCHITECTURE_OPTIX
#include "rendererFunctions.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "Device/Wrappers/dWrapper.h"
#include "MDL/CudaLinker.h"

namespace vtx
{
	enum KernelType
	{
		K_SET_QUEUE_COUNTERS,
		K_GEN_CAMERA_RAY,
		K_TRACE_RADIANCE_RAY,
		K_SHADE_RAY,
		K_HANDLE_ESCAPED_RAY,
		K_ACCUMULATE_RAY,
		K_RETRIEVE_QUEUE_SIZE,
		K_RESET,

		K_COUNT
	};

	inline static const char* KernelNames[] = {
		"setQueueCounters",
		"genCameraRay",
		"traceRadianceRay",
		"shadeRay",
		"handleEscapedRay",
		"accumulateRay",
		"retrieveQueueSize",
		"reset"
	};


	WaveFrontIntegrator::WaveFrontIntegrator(graph::RendererSettings* rendererSettings)
	{
		queueSizeRetrievalBuffer.alloc(sizeof(int));
		queueSizeDevicePtr = queueSizeRetrievalBuffer.castedPointer<int>();
		this->settings = rendererSettings;
	}

	KernelTimes& WaveFrontIntegrator::getKernelTime()
	{
		kernelTimes.setQueueCounters = GetKernelTimeMS(KernelNames[K_SET_QUEUE_COUNTERS]);

		kernelTimes.genCameraRay = GetKernelTimeMS(KernelNames[K_GEN_CAMERA_RAY]);

		kernelTimes.traceRadianceRay = GetKernelTimeMS(KernelNames[K_TRACE_RADIANCE_RAY]);

		kernelTimes.reset = GetKernelTimeMS(KernelNames[K_RESET]);

		kernelTimes.shadeRay = GetKernelTimeMS(KernelNames[K_SHADE_RAY]);

		kernelTimes.handleEscapedRay = GetKernelTimeMS(KernelNames[K_HANDLE_ESCAPED_RAY]);

		kernelTimes.accumulateRay = GetKernelTimeMS(KernelNames[K_ACCUMULATE_RAY]);

		kernelTimes.fetchQueueSize = GetKernelTimeMS(KernelNames[K_RETRIEVE_QUEUE_SIZE]);

		return kernelTimes;
	}


	void WaveFrontIntegrator::launchOptixKernel(math::vec2i launchDimension, std::string pipelineName)
	{
		const optix::State& state = *(optix::getState());
		const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
		const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt(pipelineName);

		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			pipeline, state.stream,
			/*! parameters and SBT */
			UPLOAD_BUFFERS->launchParamsBuffer.dPointer(),
			UPLOAD_BUFFERS->launchParamsBuffer.bytesSize(),
			&sbt,
			/*! dimensions of the launch: */
			launchDimension.x,
			launchDimension.y,
			1
		);
		OPTIX_CHECK(result);
	}

	void WaveFrontIntegrator::render()
	{
		hostParams = &UPLOAD_DATA->launchParams;
		deviceParams = UPLOAD_BUFFERS->launchParamsBuffer.castedPointer<LaunchParams>();
		maxTraceQueueSize = UPLOAD_DATA->settings.maxTraceQueueSize;

	
		if(settings->iteration <= 0)
		{
			setCounters();
		}

		resetCounters();
		generatePixelQueue();

		for (int frameBounces = 0; frameBounces < settings->maxBounces; frameBounces++)
		{
			traceRadianceRays();
			shadeRays();
			downloadCounters();
		}
		handleShadowTrace();
		handleEscapedRays();
		accumulateRays();
	}

	void WaveFrontIntegrator::resetQueue(Queue queue)
	{
		switch (queue)
		{
		case Q_RADIANCE_TRACE:
		{
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->radianceTraceQueue->Reset();
			});
		} break;
		case Q_SHADE:
		{
			const int materialQueuesSize = UPLOAD_DATA->materialDataMap.size;
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->shadeQueue->Reset();
			});
		} break;
		case Q_ESCAPED:
		{
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->escapedQueue->Reset();
			});
		} break;
		case Q_ACCUMULATION:
		{
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->accumulationQueue->Reset();
			});
		} break;
		case Q_SHADOW_TRACE:
		{
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->shadowQueue->Reset();
			});
		} break;
		}
	}

	void WaveFrontIntegrator::setCounters()
	{
		Do(KernelNames[K_SET_QUEUE_COUNTERS], [=, *this] __device__()
		{
			deviceParams->radianceTraceQueue->setCounter(&deviceParams->queueCounters->traceQueueCounter);
			deviceParams->shadeQueue->setCounter(&deviceParams->queueCounters->shadeQueueCounter);
			deviceParams->escapedQueue->setCounter(&deviceParams->queueCounters->escapedQueueCounter);
			deviceParams->accumulationQueue->setCounter(&deviceParams->queueCounters->accumulationQueueCounter);
			deviceParams->shadowQueue->setCounter(&deviceParams->queueCounters->shadowQueueCounter);
		});
	}

	void WaveFrontIntegrator::resetCounters()
	{
		counters.accumulationQueueCounter = maxTraceQueueSize * (settings->maxBounces+1);
		counters.shadowQueueCounter = maxTraceQueueSize * (settings->maxBounces+1);
		counters.shadeQueueCounter = maxTraceQueueSize;
		counters.traceQueueCounter = maxTraceQueueSize;
		counters.escapedQueueCounter = maxTraceQueueSize;
	}

	void WaveFrontIntegrator::downloadCounters()
	{
		if (settings->fitWavefront)
		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_RETRIEVE_QUEUE_SIZE]);
			cudaEventRecord(events.first, nullptr);

			CUDA_CHECK(cudaMemcpy((void*)&counters, UPLOAD_DATA->launchParams.queueCounters, sizeof(Counters), cudaMemcpyDeviceToHost));
			counters.accumulationQueueCounter = counters.shadowQueueCounter + counters.escapedQueueCounter + counters.accumulationQueueCounter;
			cudaEventRecord(events.second, nullptr);

		}
	}

	void WaveFrontIntegrator::generatePixelQueue()
	{
		gpuParallelFor(KernelNames[K_GEN_CAMERA_RAY],
			counters.traceQueueCounter,
			[=, *this] __device__(int id)
		{
			wfInitRayEntry(id, deviceParams);
		});
	}

	void WaveFrontIntegrator::traceRadianceRays()
	{
		if (counters.traceQueueCounter == 0)
			return;

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_TRACE_RADIANCE_RAY]);
		cudaEventRecord(events.first, nullptr);

		launchOptixKernel(math::vec2i(counters.traceQueueCounter, 1), "wfRadianceTrace");

		cudaEventRecord(events.second, nullptr);
	}

	void WaveFrontIntegrator::shadeRays()
	{
		

		if (counters.shadeQueueCounter == 0)
		{
			return;
		}

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_SHADE_RAY]);
		cudaEventRecord(events.first);
		if (settings->optixShade)
		{
			launchOptixKernel(math::vec2i{counters.shadeQueueCounter, 1}, "wfShade");
		}
		else {
			const CUfunction& cudaFunction = mdl::getMdlCudaLinker().outKernelFunction;
			constexpr int threadsPerBlockX = 256;
			const int     numBlocksX = (counters.shadeQueueCounter + threadsPerBlockX - 1) / threadsPerBlockX;
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
		}
		cudaEventRecord(events.second);
	}

	void WaveFrontIntegrator::handleShadowTrace()
	{
		if(counters.shadowQueueCounter ==0)
		{
			VTX_INFO("NO Shadow Queue");
			return;
		}

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[Q_SHADOW_TRACE]);
		cudaEventRecord(events.first, nullptr);

		launchOptixKernel(math::vec2i(counters.shadowQueueCounter, 1), "wfShadowTrace");

		cudaEventRecord(events.second, nullptr);
	}

	void WaveFrontIntegrator::handleEscapedRays()
	{
		if(counters.escapedQueueCounter == 0)
		{
			return;
		}
		gpuParallelFor(KernelNames[K_HANDLE_ESCAPED_RAY],
			counters.escapedQueueCounter,
			[=, *this] __device__(int id)
		{
			wfEscapedEntry(id, deviceParams);
		});
	}

	void WaveFrontIntegrator::accumulateRays()
	{
		if(counters.accumulationQueueCounter == 0)
		{
			return;
		}

		gpuParallelFor(KernelNames[K_ACCUMULATE_RAY],
			counters.accumulationQueueCounter,
			[=, *this] __device__(int id)
		{
			wfAccumulateEntry(id, deviceParams);
		});
	}
}


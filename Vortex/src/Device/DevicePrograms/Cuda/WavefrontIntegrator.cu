#include "../WavefrontIntegrator.h"
#include "../randomNumberGenerator.h"
#define ARCHITECTURE_OPTIX
#include "../rendererFunctions.h"
#include "Device/KernelInfos.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/UploadData.h"
#include "Device/Wrappers/dWrapper.h"
#include "MDL/CudaLinker.h"

namespace vtx
{
	
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
			//CUDA_SYNC_CHECK();
		}

		resetCounters();
		//CUDA_SYNC_CHECK();
		generatePixelQueue();
		//CUDA_SYNC_CHECK();

		for (int frameBounces = 0; frameBounces < settings->maxBounces; frameBounces++)
		{
			traceRadianceRays();
			//CUDA_SYNC_CHECK();
			network.inference();
			//CUDA_SYNC_CHECK();
			shadeRays();
			//CUDA_SYNC_CHECK();
			downloadCounters();
			//CUDA_SYNC_CHECK();
		}
		handleShadowTrace();
		//CUDA_SYNC_CHECK();
		handleEscapedRays();
		//CUDA_SYNC_CHECK();
		accumulateRays();
		//CUDA_SYNC_CHECK();
		network.train();
	}

	void WaveFrontIntegrator::resetQueue(Queue queue)
	{
		auto deviceParamsCopy = deviceParams;
		switch (queue)
		{
		case Q_RADIANCE_TRACE:
		{
			Do(eventNames[K_RESET], [deviceParamsCopy] __device__()
			{
				deviceParamsCopy->radianceTraceQueue->Reset();
			});
		} break;
		case Q_SHADE:
		{
			const int materialQueuesSize = UPLOAD_DATA->materialDataMap.size;
			Do(eventNames[K_RESET], [deviceParamsCopy] __device__()
			{
				deviceParamsCopy->shadeQueue->Reset();
			});
		} break;
		case Q_ESCAPED:
		{
			Do(eventNames[K_RESET], [deviceParamsCopy] __device__()
			{
				deviceParamsCopy->escapedQueue->Reset();
			});
		} break;
		case Q_ACCUMULATION:
		{
			Do(eventNames[K_RESET], [deviceParamsCopy] __device__()
			{
				deviceParamsCopy->accumulationQueue->Reset();
			});
		} break;
		case Q_SHADOW_TRACE:
		{
			Do(eventNames[K_RESET], [deviceParamsCopy] __device__()
			{
				deviceParamsCopy->shadowQueue->Reset();
			});
		} break;
		}
	}

	void WaveFrontIntegrator::setCounters()
	{
		auto deviceParamsCopy = deviceParams;
		Do(eventNames[K_SET_QUEUE_COUNTERS], [deviceParamsCopy] __device__()
		{
			deviceParamsCopy->radianceTraceQueue->setCounter(&deviceParamsCopy->queueCounters->traceQueueCounter);
			deviceParamsCopy->shadeQueue->setCounter(&deviceParamsCopy->queueCounters->shadeQueueCounter);
			deviceParamsCopy->escapedQueue->setCounter(&deviceParamsCopy->queueCounters->escapedQueueCounter);
			deviceParamsCopy->accumulationQueue->setCounter(&deviceParamsCopy->queueCounters->accumulationQueueCounter);
			deviceParamsCopy->shadowQueue->setCounter(&deviceParamsCopy->queueCounters->shadowQueueCounter);
		});
	}

	void WaveFrontIntegrator::resetCounters()
	{
		counters.accumulationQueueCounter = maxTraceQueueSize * (settings->maxBounces + 1);
		counters.shadowQueueCounter = maxTraceQueueSize * (settings->maxBounces + 1);
		counters.shadeQueueCounter = maxTraceQueueSize;
		counters.traceQueueCounter = maxTraceQueueSize;
		counters.escapedQueueCounter = maxTraceQueueSize;

	}

	void WaveFrontIntegrator::downloadCounters()
	{
		if (settings->fitWavefront)
		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_RETRIEVE_QUEUE_SIZE]);
			cudaEventRecord(events.first, nullptr);

			CUDA_CHECK(cudaMemcpy((void*)&counters, UPLOAD_DATA->launchParams.queueCounters, sizeof(Counters), cudaMemcpyDeviceToHost));
			counters.accumulationQueueCounter = counters.shadowQueueCounter + counters.escapedQueueCounter + counters.accumulationQueueCounter;
			cudaEventRecord(events.second, nullptr);

		}
	}

	void WaveFrontIntegrator::generatePixelQueue()
	{
		auto deviceParamsCopy = deviceParams;
		gpuParallelFor(eventNames[K_GEN_CAMERA_RAY],
			counters.traceQueueCounter,
			[deviceParamsCopy] __device__(int id)
		{
			wfInitRayEntry(id, deviceParamsCopy);
		});
	}

	void WaveFrontIntegrator::traceRadianceRays()
	{
		if (counters.traceQueueCounter == 0)
			return;

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_TRACE_RADIANCE_RAY]);
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

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[K_SHADE_RAY]);
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
		if (counters.shadowQueueCounter == 0)
		{
			VTX_INFO("NO Shadow Queue");
			return;
		}

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[Q_SHADOW_TRACE]);
		cudaEventRecord(events.first, nullptr);

		launchOptixKernel(math::vec2i(counters.shadowQueueCounter, 1), "wfShadowTrace");

		cudaEventRecord(events.second, nullptr);
	}

	void WaveFrontIntegrator::handleEscapedRays()
	{
		auto deviceParamsCopy = deviceParams;
		if (counters.escapedQueueCounter == 0)
		{
			return;
		}
		gpuParallelFor(eventNames[K_HANDLE_ESCAPED_RAY],
			counters.escapedQueueCounter,
			[deviceParamsCopy] __device__(int id)
		{
			wfEscapedEntry(id, deviceParamsCopy);
		});
	}

	void WaveFrontIntegrator::accumulateRays()
	{
		if (counters.accumulationQueueCounter == 0)
		{
			return;
		}

		auto deviceParamsCopy = deviceParams;
		gpuParallelFor(eventNames[K_ACCUMULATE_RAY],
			counters.accumulationQueueCounter,
			[deviceParamsCopy] __device__(int id)
		{
			wfAccumulateEntry(id, deviceParamsCopy);
		});
	}
}

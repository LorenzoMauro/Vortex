#include "WavefrontIntegrator.h"
#include "randomNumberGenerator.h"
#include "ToneMapper.h"
#include "Device/OptixWrapper.h"
#include "Device/WorkQueues.h"
#include "Device/UploadCode/UploadData.h"
#include "Device/Wrappers/dWrapper.h"
#include "Device/DevicePrograms/Utils.h"
#include "MDL/CudaLinker.h"

namespace vtx
{
	enum KernelType
	{
		K_PIXEL,
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
		"PixelQueue",
		"genCameraRay",
		"traceRadianceRay",
		"shadeRay",
		"handleEscapedRay",
		"accumulateRay",
		"retrieveQueueSize",
		"reset"
	};

	__forceinline__ __device__ void debugFillBlue(int id, LaunchParams* params)
	{
		FrameBufferData* frameBuffer = &params->frameBuffer;
		frameBuffer->tmRadiance[id] = math::vec3f(0.0f, 0.0f, 1.0f);
		frameBuffer->rawRadiance[id] = math::vec3f(0.0f, 0.0f, 1.0f);
	}

	__forceinline__ __device__ void debugScreenUV(int id, LaunchParams* params)
	{
		FrameBufferData* frameBuffer = &params->frameBuffer;
		int              pixelX = id % (int)frameBuffer->frameSize.x;
		int              pixelY = id / (int)frameBuffer->frameSize.x;
		float            normalizedX = (float)pixelX / (float)frameBuffer->frameSize.x;
		float            normalizedY = (float)pixelY / (float)frameBuffer->frameSize.y;
		math::vec3f      screenUV = math::vec3f(normalizedX, normalizedY, 0.0f);

		frameBuffer->tmRadiance[id] = screenUV;
		frameBuffer->rawRadiance[id] = screenUV;
		/*if (frameBuffer->tmRadiance[id] == screenUV)
		{
			frameBuffer->tmRadiance[id] = math::vec3f(1.0f, 0.0f, 1.0f);
			printf("ID: %d, Pixel: %d, %d, UV: %f, %f\n", id, pixelX, pixelY, normalizedX, normalizedY);
		}
		else
		{
			frameBuffer->tmRadiance[id] = screenUV;
		}

		if( frameBuffer->rawRadiance[id] == screenUV)
		{
			frameBuffer->rawRadiance[id] = math::vec3f(1.0f, 0.0f, 1.0f);
		}
		else
		{
			frameBuffer->rawRadiance[id] = screenUV;
		}*/
	}


	__forceinline__ __device__ void generateCameraRay(PixelWorkItem& px, LaunchParams* params) {

		math::vec2f pixel = math::vec2f((float)(px.pixelId % params->frameBuffer.frameSize.x), (float)(px.pixelId / params->frameBuffer.frameSize.x));
		unsigned seed = tea<4>(px.pixelId, *params->frameID);
		math::vec2f screen {(float)params->frameBuffer.frameSize.x, (float)params->frameBuffer.frameSize.y };
		math::vec2f sample = rng2(seed);
		const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
		const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

		const CameraData camera = params->cameraData;

		math::vec3f origin = camera.position;
		math::vec3f direction = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);

		RayWorkItem rd;
		rd.origin = origin;
		rd.direction = direction;
		rd.seed = seed;
		rd.originPixel = px.pixelId;
		rd.radiance = math::vec3f(0.0f);
		rd.pdf = 0.0f;
		rd.throughput = math::vec3f(1.0f);
		rd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
		rd.depth = 0;
		rd.mediumIor = math::vec3f(1.0f);
		rd.colorsBounceDiffuse = 0.0f;
		rd.colorsTrueNormal = 0.0f;
		rd.colorsShadingNormal = 0.0f;
		rd.colorsTangent = 0.0f;
		rd.colorsOrientation = 0.0f;
		rd.colorsDebugColor1 = 0.0f;
		rd.colorsUv = 0.0f;
		rd.colorsDirectLight = 0.0f;
		rd.firstHitType = mi::neuraylib::BSDF_EVENT_ABSORB;

		params->radianceTraceQueue->Push(rd);

	}

	__forceinline__ __device__ void accumulateRayDevice(int id, LaunchParams* params)
	{
		if(id==0 && params->accumulationQueue->Size()==0)
		{
			printf("Accumulation queue is empty\n");
		}
		if (id >= params->accumulationQueue->Size())
			return;
		const RayWorkItem             prd            = (*params->accumulationQueue)[id];
		const FrameBufferData*        frameBuffer    = &params->frameBuffer;
		const int                     fbIndex        = prd.originPixel;
		const RendererDeviceSettings* settings       = params->settings;
		const bool                    runningAverage = false;

		const int&      iteration = params->settings->iteration;
		const int samples   = iteration >= 0 ? iteration : 0;

		math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		math::vec3f& diffuseIndirectBuffer = frameBuffer->diffuseIndirect[fbIndex];
		math::vec3f& glossyIndirectBuffer = frameBuffer->glossyIndirect[fbIndex];
		math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		/*if (samples == 0)
		{
			directLightBuffer = math::vec3f(0.0f);
			diffuseIndirectBuffer = math::vec3f(0.0f);
			glossyIndirectBuffer = math::vec3f(0.0f);
			transmissionIndirect = math::vec3f(0.0f);
			frameBuffer->rawRadiance[fbIndex] = math::vec3f(0.0f);
		}*/

		const bool isTransmission = (prd.firstHitType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0;
		const bool isDiffuse      = ((prd.firstHitType & mi::neuraylib::BSDF_EVENT_DIFFUSE) != 0) && !isTransmission;
		const bool isGlossy       = ((prd.firstHitType & mi::neuraylib::BSDF_EVENT_GLOSSY) != 0 || (prd.firstHitType & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0) && !isTransmission && !isDiffuse;

		const math::vec3f deltaRadiance        = prd.radiance - prd.colorsDirectLight;
		const math::vec3f transmissionRadiance = isTransmission? deltaRadiance : 0.0f;
		const math::vec3f diffuseRadiance      = isDiffuse? deltaRadiance : 0.0f;
		const math::vec3f glossyRadiance       = isGlossy? deltaRadiance : 0.0f;

		if (!settings->accumulate || samples <= 1)
		{
			utl::replaceNanCheck(prd.radiance, frameBuffer->rawRadiance[fbIndex]);
			utl::replaceNanCheck(prd.colorsDirectLight, directLightBuffer);

			utl::replaceNanCheck(transmissionRadiance, transmissionIndirect);
			utl::replaceNanCheck(diffuseRadiance, diffuseIndirectBuffer);
			utl::replaceNanCheck(glossyRadiance, glossyIndirectBuffer);

			utl::replaceNanCheck(prd.colorsBounceDiffuse, frameBuffer->albedo[fbIndex]);
			utl::replaceNanCheck(prd.colorsShadingNormal, frameBuffer->normal[fbIndex]);
			utl::replaceNanCheck(prd.colorsTrueNormal, frameBuffer->trueNormal[fbIndex]);
			utl::replaceNanCheck(prd.colorsTangent, frameBuffer->tangent[fbIndex]);
			utl::replaceNanCheck(prd.colorsOrientation, frameBuffer->orientation[fbIndex]);
			utl::replaceNanCheck(prd.colorsUv, frameBuffer->uv[fbIndex]);
			utl::replaceNanCheck(prd.colorsDebugColor1, frameBuffer->debugColor1[fbIndex]);
		}
		else
		{
			if (!runningAverage)
			{
				const float kFactor = 1.0f / (float)samples;
				const float jFactor = ((float)samples - 1.0f) * kFactor;
				utl::accumulate(prd.radiance, frameBuffer->rawRadiance[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsDirectLight, directLightBuffer, kFactor, jFactor);

				utl::accumulate(transmissionRadiance, transmissionIndirect, kFactor, jFactor);
				utl::accumulate(diffuseRadiance, diffuseIndirectBuffer, kFactor, jFactor);
				utl::accumulate(glossyRadiance, glossyIndirectBuffer, kFactor, jFactor);

				utl::accumulate(prd.colorsBounceDiffuse, frameBuffer->albedo[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsShadingNormal, frameBuffer->normal[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsTrueNormal, frameBuffer->trueNormal[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsTangent, frameBuffer->tangent[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsOrientation, frameBuffer->orientation[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsUv, frameBuffer->uv[fbIndex], kFactor, jFactor);
				utl::accumulate(prd.colorsDebugColor1, frameBuffer->debugColor1[fbIndex], kFactor, jFactor);

				
			}
			else
			{
				const float fSamples = (float)samples;
				utl::lerpAccumulate(prd.radiance, frameBuffer->rawRadiance[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsDirectLight, directLightBuffer, fSamples);

				utl::lerpAccumulate(transmissionRadiance, transmissionIndirect, fSamples);
				utl::lerpAccumulate(diffuseRadiance, diffuseIndirectBuffer, fSamples);
				utl::lerpAccumulate(glossyRadiance, glossyIndirectBuffer, fSamples);

				utl::lerpAccumulate(prd.colorsBounceDiffuse, frameBuffer->albedo[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsShadingNormal, frameBuffer->normal[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsTrueNormal, frameBuffer->trueNormal[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsTangent, frameBuffer->tangent[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsOrientation, frameBuffer->orientation[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsUv, frameBuffer->uv[fbIndex], fSamples);
				utl::lerpAccumulate(prd.colorsDebugColor1, frameBuffer->debugColor1[fbIndex], fSamples);
			}
		}

		frameBuffer->tmRadiance[fbIndex] = toneMap(params->toneMapperSettings, frameBuffer->rawRadiance[fbIndex]);
		frameBuffer->noiseBuffer[fbIndex].samples++;

		//PixelWorkItem px;
		//px.pixelId = prd.originPixel;
		////params->pixelQueue->Push(px);
		//generateCameraRay(px, params);
	}

	__forceinline__ __device__ void handleEscaped(int id, LaunchParams* params)
	{
		if (id >= params->escapedQueue->Size())
			return;

		RayWorkItem                prd       = (*params->escapedQueue)[id];
		/*! for this simple example, this will remain empty */

		if (params->envLight != nullptr)
		{
			const LightData* envLight = params->envLight;
			EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(envLight->attributes);
			auto texture = attrib.texture;

			math::vec3f dir = math::transformNormal3F(attrib.invTransformation, prd.direction);

			{
				bool computeOriginalUV = false;
				float u, v;
				if (computeOriginalUV)
				{
					u = fmodf(atan2f(dir.y, dir.x) * (float)(0.5 / M_PI) + 0.5f, 1.0f);
					v = acosf(fmax(fminf(-dir.z, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
				}
				else {
					float theta = acosf(-dir.z);
					v = theta / (float)M_PI;
					float phi = atan2f(dir.y, dir.x);// + M_PI / 2.0f; // azimuth angle (theta)
					u = (phi + (float)M_PI) / (float)(2.0f * M_PI);
				}

				const auto x = math::min<unsigned>((unsigned int)(u * (float)texture->dimension.x), texture->dimension.x - 1);
				const auto y = math::min<unsigned>((unsigned int)(v * (float)texture->dimension.y), texture->dimension.y - 1);
				math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
				//emission = emission; *attrib.envIntensity;

				// to incorporate the point light selection probability
				float misWeight = 1.0f;
				// If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
				// then calculate light emission with multiple importance sampling for this implicit light hit as well.
				bool MiSCondition = (params->settings->samplingTechnique == RendererDeviceSettings::S_MIS && (prd.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)));
				if (prd.pdf > 0.0f && MiSCondition)
				{
					float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
					misWeight = utl::heuristic(prd.pdf, envSamplePdf);
					//misWeight = prd.pdf / (prd.pdf + envSamplePdf);
				}
				prd.radiance += prd.throughput * emission * misWeight;

				prd.colorsBounceDiffuse = math::normalize(emission);
				if(prd.depth == 0 || prd.depth == 1)
				{
					prd.colorsDirectLight = prd.radiance;
				}
				if (prd.depth == 0)
				{
					prd.colorsTrueNormal = 0.0f;
					prd.colorsShadingNormal = 0.0f;
				}
			}
		}

		params->accumulationQueue->Push(prd);
		//params->accumulationQueue->addJob(prd, "Accumulation on Miss", true);

	}

	__forceinline__ __device__ void generateCameraRayKernel(int id, LaunchParams* params) {

		/*if (id==0 && params->pixelQueue->Size() == 0)
		{
			printf("Pixel queue is empty\n");
		}

		if (id >= params->pixelQueue->Size())
		{
			return;
		}

		PixelWorkItem  px = (*params->pixelQueue)[id];*/

		//debugScreenUV(px.pixelId, params);


		math::vec2f pixel = math::vec2f((float)(id % params->frameBuffer.frameSize.x), (float)(id / params->frameBuffer.frameSize.x));
		unsigned seed = tea<4>(id, *params->frameID);
		math::vec2f screen {(float)params->frameBuffer.frameSize.x, (float)params->frameBuffer.frameSize.y };
		math::vec2f sample = rng2(seed);
		const math::vec2f fragment = pixel + sample;                    // Jitter the sub-pixel location
		const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].

		const CameraData camera = params->cameraData;

		math::vec3f origin = camera.position;
		math::vec3f direction = math::normalize<float>(camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction);

		RayWorkItem rd;
		rd.origin = origin;
		rd.direction = direction;
		rd.seed = seed;
		rd.originPixel = id;
		rd.radiance = math::vec3f(0.0f);
		rd.pdf = 0.0f;
		rd.throughput = math::vec3f(1.0f);
		rd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
		rd.depth = 0;
		rd.mediumIor = math::vec3f(1.0f);
		rd.colorsBounceDiffuse = 0.0f;
		rd.colorsTrueNormal = 0.0f;
		rd.colorsShadingNormal = 0.0f;
		rd.colorsTangent = 0.0f;
		rd.colorsOrientation = 0.0f;
		rd.colorsDebugColor1 = 0.0f;
		rd.colorsUv = 0.0f;
		rd.colorsDirectLight = 0.0f;
		rd.firstHitType = mi::neuraylib::BSDF_EVENT_ABSORB;
		rd.shadowTrace = false;
		rd.radianceDirect = 0.0f;

		params->radianceTraceQueue->Push(rd);

	}

	__forceinline__ __device__ void initializeRay(int id, LaunchParams* params)
	{
		if(params->settings->iteration == 0)
		{
			FrameBufferData* frameBuffer = &params->frameBuffer;
			frameBuffer->rawRadiance[id] = 0.0f;
			frameBuffer->directLight[id] = 0.0f;
			frameBuffer->diffuseIndirect[id] = 0.0f;
			frameBuffer->glossyIndirect[id] = 0.0f;
			frameBuffer->transmissionIndirect[id] = 0.0f;

			frameBuffer->tmRadiance[id] = 0.0f;
			frameBuffer->tmDirectLight[id] = 0.0f;
			frameBuffer->tmDiffuseIndirect[id] = 0.0f;
			frameBuffer->tmGlossyIndirect[id] = 0.0f;
			frameBuffer->tmTransmissionIndirect[id] = 0.0f;

			frameBuffer->albedo[id] = 0.0f;
			frameBuffer->normal[id] = 0.0f;
			frameBuffer->trueNormal[id] = 0.0f;
			frameBuffer->tangent[id] = 0.0f;
			frameBuffer->orientation[id] = 0.0f;
			frameBuffer->uv[id] = 0.0f;
			frameBuffer->debugColor1[id] = 0.0f;

			frameBuffer->fireflyPass[id] = 0.0f;
			reinterpret_cast<math::vec3f*>(frameBuffer->outputBuffer)[id] = math::vec3f(0.0f);
			frameBuffer->noiseBuffer[id].samples = 0;
		}
		
		generateCameraRayKernel(id, params);
	}

	WaveFrontIntegrator::WaveFrontIntegrator()
	{
		queueSizeRetrievalBuffer.alloc(sizeof(int));
		queueSizeDevicePtr = queueSizeRetrievalBuffer.castedPointer<int>();
	}

	KernelTimes& WaveFrontIntegrator::getKernelTime()
	{
		kernelTimes.genCameraRay = GetKernelTimeMS(KernelNames[K_GEN_CAMERA_RAY]);

		kernelTimes.traceRadianceRay = GetKernelTimeMS(KernelNames[K_TRACE_RADIANCE_RAY]);
		kernelTimes.reset = GetKernelTimeMS(KernelNames[K_RESET]);

		kernelTimes.shadeRay = GetKernelTimeMS(KernelNames[K_SHADE_RAY]);

		kernelTimes.handleEscapedRay = GetKernelTimeMS(KernelNames[K_HANDLE_ESCAPED_RAY]);

		kernelTimes.accumulateRay = GetKernelTimeMS(KernelNames[K_ACCUMULATE_RAY]);

		kernelTimes.fetchQueueSize = GetKernelTimeMS(KernelNames[K_RETRIEVE_QUEUE_SIZE]);

		kernelTimes.pixelQueue = GetKernelTimeMS(KernelNames[K_PIXEL]);

		return kernelTimes;
	}


	void WaveFrontIntegrator::render(bool fitKernelSize, int iteration)
	{
		hostParams = &UPLOAD_DATA->launchParams;
		deviceParams = UPLOAD_BUFFERS->launchParamsBuffer.castedPointer<LaunchParams>();
		numberOfPixels = hostParams->frameBuffer.frameSize.x * hostParams->frameBuffer.frameSize.y;

		this->fitKernelSize = fitKernelSize;

		const int maxBounces = UPLOAD_DATA->settings.maxBounces;

		resetQueue(Q_RADIANCE_TRACE);
		CUDA_SYNC_CHECK();

		generatePixelQueue();
		CUDA_SYNC_CHECK();

		for (int frameBounces = 0; frameBounces <= maxBounces; frameBounces++)
		{
			traceRadianceRays();
			CUDA_SYNC_CHECK();
			resetQueue(Q_RADIANCE_TRACE);

			cudaShade();
			CUDA_SYNC_CHECK();

			//shadeRays();
			//CUDA_SYNC_CHECK();

			resetQueue(Q_SHADE);
			CUDA_SYNC_CHECK();

		}
		handleEscapedRays();
		CUDA_SYNC_CHECK();

		resetQueue(Q_ESCAPED);
		CUDA_SYNC_CHECK();

		accumulateRays();
		CUDA_SYNC_CHECK();

		resetQueue(Q_ACCUMULATION);
		CUDA_SYNC_CHECK();


		//VTX_INFO("\nRESTART LOOP!");
		//resetQueue(Q_RADIANCE_TRACE);
		//resetQueue(Q_PIXEL);
		//generatePixelQueue();
		//generateCameraRadianceRays();
		//resetQueue(Q_PIXEL);
		//const int maxBounces = UPLOAD_DATA->settings.maxBounces;
		//for (int depth = 0; depth < maxBounces; depth++)
		//{
		//	VTX_INFO("DEPTH: {}", depth);
		//	traceRadianceRays();
		//	resetQueue(Q_RADIANCE_TRACE);
		//	shadeRays();
		//	resetQueue(Q_SHADE);
		//	std::cerr << "Press ENTER to continue Bounce Depth..." << std::endl;
		//	std::cin.get();
		//	//__debugbreak();
		//}
		//handleEscapedRays();
		//resetQueue(Q_ESCAPED);
		//accumulateRays();
		//resetQueue(Q_ACCUMULATION);
	}

	void WaveFrontIntegrator::generatePixelQueue()
	{
		gpuParallelFor(KernelNames[K_PIXEL],
			numberOfPixels,
			[=, *this] __device__(int id)
		{
			initializeRay(id, deviceParams);
		});
	}

	void WaveFrontIntegrator::generateCameraRadianceRays()
	{
		gpuParallelFor(KernelNames[K_GEN_CAMERA_RAY],
			numberOfPixels,
			[=, *this] __device__(int id)
		{
			generateCameraRayKernel(id, deviceParams);
		});
	}

	void WaveFrontIntegrator::traceRadianceRays()
	{
		const int traceQueueSize = fetchQueueSize(Q_RADIANCE_TRACE);
		if (traceQueueSize == 0)
			return;

		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_TRACE_RADIANCE_RAY]);
		cudaEventRecord(events.first);

		const optix::State& state = *(optix::getState());
		const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
		const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt("wfRadianceTrace");

		//const int traceQueueSize = gpuDownload(UPLOAD_DATA->launchParams.radianceTraceQueueSize);
		//VTX_INFO("Trace Number: {}", traceQueue.size);

		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			pipeline, state.stream,
			/*! parameters and SBT */
			UPLOAD_BUFFERS->launchParamsBuffer.dPointer(),
			UPLOAD_BUFFERS->launchParamsBuffer.bytesSize(),
			&sbt,
			/*! dimensions of the launch: */
			traceQueueSize,
			1,
			1
		);
		cudaEventRecord(events.second);
		OPTIX_CHECK(result);
		//CUDA_SYNC_CHECK();
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
		case Q_PIXEL:
		{
			Do(KernelNames[K_RESET], [=, *this] __device__()
			{
				deviceParams->pixelQueue->Reset();
			});
		} break;
		}
	}

	int WaveFrontIntegrator::fetchQueueSize(Queue queue)
	{
		if (!fitKernelSize)
		{
			return numberOfPixels;
		} 
		switch (queue)
		{
		case Q_RADIANCE_TRACE:
			{
				Do(KernelNames[K_RETRIEVE_QUEUE_SIZE], [=, *this] __device__()
				{
					*queueSizeDevicePtr = deviceParams->radianceTraceQueue->Size();
				});
			} break;
		case Q_SHADE:
			{
				Do(KernelNames[K_RETRIEVE_QUEUE_SIZE], [=, *this] __device__()
				{
					*queueSizeDevicePtr = deviceParams->shadeQueue->Size();
				});
			} break;
		case Q_ESCAPED:
			{
				Do(KernelNames[K_RETRIEVE_QUEUE_SIZE], [=, *this] __device__()
				{
					*queueSizeDevicePtr = deviceParams->escapedQueue->Size();
				});
			} break;
		case Q_ACCUMULATION:
			{
				Do(KernelNames[K_RETRIEVE_QUEUE_SIZE], [=, *this] __device__()
				{
					*queueSizeDevicePtr = deviceParams->accumulationQueue->Size();
				});
			} break;
		}

		queueSizeRetrievalBuffer.download(&retrievedQueueSize);

		return retrievedQueueSize;
	}

	void WaveFrontIntegrator::cudaShade()
	{
		// Launch kernel
		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_SHADE_RAY]);
		cudaEventRecord(events.first);
		int threadPerBlock = 16;
		const dim3 threadsPerBlock(threadPerBlock, threadPerBlock);
		const dim3 numBlocks((UPLOAD_DATA->launchParams.frameBuffer.frameSize.x + (threadPerBlock-1)) / threadPerBlock,
							(UPLOAD_DATA->launchParams.frameBuffer.frameSize.y + (threadPerBlock-1)) / threadPerBlock);
		void*             params[]     = {&deviceParams};
		const CUfunction& cudaFunction = mdl::getMdlCudaLinker().outKernelFunction;
		const CUresult    res          = cuLaunchKernel(
			cudaFunction,
			numBlocks.x, numBlocks.y, numBlocks.z,
			threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z,
			0, nullptr, params, nullptr);
		CU_CHECK(res);
		cudaEventRecord(events.second);

	}

	void WaveFrontIntegrator::shadeRays()
	{
		const int shadeQueueSize = fetchQueueSize(Q_SHADE);
		if (shadeQueueSize == 0)
		{
			return;
		}
		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(KernelNames[K_SHADE_RAY]);
		cudaEventRecord(events.first);

		const optix::State& state = *(optix::getState());
		const OptixPipeline& pipeline = optix::getRenderingPipeline()->getPipeline();
		const OptixShaderBindingTable& sbt = optix::getRenderingPipeline()->getSbt("wfShade");

		//const int shadeQueueSize = gpuDownload(UPLOAD_DATA->launchParams.shadeQueueSize);

		const auto result = optixLaunch(/*! pipeline we're launching launch: */
			pipeline, state.stream,
			/*! parameters and SBT */
			UPLOAD_BUFFERS->launchParamsBuffer.dPointer(),
			UPLOAD_BUFFERS->launchParamsBuffer.bytesSize(),
			&sbt,
			/*! dimensions of the launch: */
			//UPLOAD_DATA->launchParams.frameBuffer.frameSize.x,
			//UPLOAD_DATA->launchParams.frameBuffer.frameSize.y,
			shadeQueueSize,
			1,
			1
		);
		cudaEventRecord(events.second);
		OPTIX_CHECK(result);
		//CUDA_SYNC_CHECK();
	}

	void WaveFrontIntegrator::handleEscapedRays()
	{
		const int escapedQueueSize = fetchQueueSize(Q_ESCAPED);
		if (escapedQueueSize == 0)
		{
			return;
		}
		//const int escapedQueueSize = gpuDownload(UPLOAD_DATA->launchParams.escapedQueueSize);
		gpuParallelFor(KernelNames[K_HANDLE_ESCAPED_RAY],
			escapedQueueSize,
			[=, *this] __device__(int id)
		{
			handleEscaped(id, deviceParams);
		});
	}

	void WaveFrontIntegrator::accumulateRays()
	{
		const int accumulationQueueSize = fetchQueueSize(Q_ACCUMULATION);

		if (accumulationQueueSize == 0)
		{
			return;
		}

		gpuParallelFor(KernelNames[K_ACCUMULATE_RAY],
			accumulationQueueSize,
			[=, *this] __device__(int id)
		{
			accumulateRayDevice(id, deviceParams);
		});
	}
}


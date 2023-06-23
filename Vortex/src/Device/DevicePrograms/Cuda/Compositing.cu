#include "../CudaKernels.h"
#include "device_launch_parameters.h"
#include "Device/CUDAChecks.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/Wrappers/dWrapper.h"

namespace vtx
{
	__forceinline__ __device__ math::vec3f fireflyRemoval(const math::vec3f* inputBuffer, const int x, const int y, const int width, const int height, const int kernelSize, const NoiseType noiseType, const float threshold)
	{
		// kernelSize is assumed to be an odd number
		const int halfKernel = kernelSize / 2;

		math::vec3f centerValue;
		if (noiseType == LUMINANCE)
		{
			centerValue = inputBuffer[y * width + x];
		}

		math::vec3f sumValue = math::vec3f(0.0f);
		int   count = 0;

		for (int dx = -halfKernel; dx <= halfKernel; ++dx)
		{
			for (int dy = -halfKernel; dy <= halfKernel; ++dy)
			{
				const int nx = x + dx;
				const int ny = y + dy;

				// exclude the center pixel
				if ((dx != 0 || dy != 0) && nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					sumValue += inputBuffer[ny * width + nx];
					count++;
				}
			}
		}

		// Calculate average luminance in the kernel (excluding center)
		float avgLuminance = utl::luminance(sumValue / count);

		// If the center pixel's luminance is higher than average plus threshold
		// replace it with average, otherwise keep the original
		if (utl::luminance(centerValue) > avgLuminance*threshold)
		{
			return sumValue / count;
		}

		return inputBuffer[y * width + x];
	}


	__global__ void fireFlyPass(LaunchParams* launchParams, int kernelSize, NoiseType noiseType, float threshold)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		FrameBufferData* frameBuffer = &launchParams->frameBuffer;
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		if (x >= frameSize.x || y >= frameSize.y) return;

		const uint32_t fbIndex = x + y * frameSize.x;

		/*const math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		const math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		const math::vec3f diffuse = fireflyRemoval(frameBuffer->diffuseIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f glossy = fireflyRemoval(frameBuffer->glossyIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f transmission = fireflyRemoval(frameBuffer->transmissionIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);*/

		const math::vec3f filteredRadiance = fireflyRemoval(frameBuffer->hdriRadiance, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);

		frameBuffer->fireflyPass[fbIndex] = filteredRadiance;
	}

	__forceinline__ __device__ void prepareOutput(math::vec3f* inputBuffer, LaunchParams* params, int pixelId, bool normalizeBySamples, bool dotoneMap)
	{
		math::vec3f output3f = inputBuffer[pixelId];
		if(normalizeBySamples)
		{
			output3f /= params->frameBuffer.samples[pixelId];
		}
		if(dotoneMap)
		{
			output3f = toneMap(params->toneMapperSettings, output3f);
		}
		reinterpret_cast<math::vec4f*>(params->frameBuffer.outputBuffer)[pixelId] = math::vec4f(output3f, 1.0f);
	}

	__forceinline__ __device__ void toneMapRadiance(const int id, const LaunchParams* params)
	{
		params->frameBuffer.hdriRadiance[id] = params->frameBuffer.radianceAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.normalNormalized[id] = params->frameBuffer.normalAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.albedoNormalized[id] = params->frameBuffer.albedoAccumulator[id] / params->frameBuffer.samples[id];
		params->frameBuffer.tmRadiance[id]= toneMap(params->toneMapperSettings, params->frameBuffer.hdriRadiance[id]);
	}

	__global__ void outputSelector(LaunchParams* launchParams, math::vec3f* beauty) {

		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		const FrameBufferData* frameBuffer = &launchParams->frameBuffer;
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		if (x >= frameSize.x || y >= frameSize.y) return;

		const uint32_t fbIndex = x + y * frameSize.x;
		const RendererDeviceSettings* settings     = launchParams->settings;
		const auto                    outputBuffer = reinterpret_cast<math::vec4f*>(frameBuffer->outputBuffer);

		math::vec3f* input = nullptr;
		bool normalizeBySamples = true;
		bool dotoneMap = true;
		switch (settings->displayBuffer)
		{

		case(RendererDeviceSettings::DisplayBuffer::FB_BEAUTY):
		{
				if(beauty!= nullptr)
				{
					input = beauty;
					dotoneMap = true;
				}
				else
				{
					input = frameBuffer->tmRadiance;
					dotoneMap = false;
				}
				normalizeBySamples = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_NOISY):
		{
			input = frameBuffer->tmRadiance;
			dotoneMap = false;
			normalizeBySamples = false;
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_DIFFUSE):
		{
				input = frameBuffer->albedoNormalized;
				dotoneMap = false;
				normalizeBySamples = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_ORIENTATION):
		{
				input = frameBuffer->orientation;
				dotoneMap = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_TRUE_NORMAL):
		{
				input = frameBuffer->trueNormal;
				dotoneMap = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_SHADING_NORMAL):
		{
				input = frameBuffer->normalNormalized;
				dotoneMap = false;
				normalizeBySamples = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_TANGENT):
		{
				input = frameBuffer->tangent;
				dotoneMap = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_UV):
		{
				input = frameBuffer->uv;
				dotoneMap = false;
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_NOISE):
		{
			math:: vec3f value = floatToScientificRGB(frameBuffer->noiseBuffer[fbIndex].noiseAbsolute);
			outputBuffer[fbIndex] = math::vec4f(value, 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_SAMPLES):
		{
			//const int maxPossibleSample = (launchParams->frameBuffer.frameSize.x * launchParams->frameBuffer.frameSize.y) * (launchParams->settings->iteration - launchParams->settings->minAdaptiveSamples) + launchParams->settings->iteration;
			//const int samplesDeltaRange = maxPossibleSample - launchParams->settings->minAdaptiveSamples;
			//const int samplesDelta = frameBuffer->samples[fbIndex] - launchParams->settings->minAdaptiveSamples;
			//float sampleMetric = (float)samplesDelta / (float)samplesDeltaRange;
			float sampleMetric = (float)(frameBuffer->samples[fbIndex] - launchParams->settings->minAdaptiveSamples)/ (float)(launchParams->settings->iteration- launchParams->settings->minAdaptiveSamples);
			sampleMetric *= 0.01f;
			sampleMetric = toneMap(launchParams->toneMapperSettings, math::vec3f(sampleMetric)).x;
			//sampleMetric = toneMap(launchParams->toneMapperSettings, math::vec3f(launchParams->frameBuffer.noiseBuffer[fbIndex].adaptiveSamples)).x;

			math::vec3f value = floatToScientificRGB(sampleMetric);
			//math::vec3f value = floatToScientificRGB(ACESFitted(frameBuffer->noiseBuffer[fbIndex].adaptiveSamples).x);
			outputBuffer[fbIndex] = math::vec4f(value, 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_1):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->debugColor1[fbIndex], 1.0f);
			dotoneMap = false;
		}
		break;
		}

		if(input!=nullptr)
		{
			prepareOutput(input, launchParams, fbIndex, normalizeBySamples, dotoneMap);
		}
	}

	void removeFireflies(LaunchParams* launchParams, int kernelSize, float threshold, int width, int height)
	{
		dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
		// Calculate the number of blocks needed in each dimension
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		fireFlyPass << <numBlocks, threadsPerBlock >> > (launchParams, kernelSize, LUMINANCE, threshold);
		//CUDA_CHECK(cudaDeviceSynchronize());
	}

	void switchOutput(LaunchParams* launchParams, int width, int height, math::vec3f* beauty)
	{
		dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
		// Calculate the number of blocks needed in each dimension
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		outputSelector <<<numBlocks, threadsPerBlock>>>(launchParams, beauty);
		//CUDA_CHECK(cudaDeviceSynchronize());
	}

	void toneMapRadianceKernel(const LaunchParams* launchParams, const int width, const int height, const char* name)
	{
		gpuParallelFor(name,
			width * height,
			[=] __device__(const int id)
		{
			toneMapRadiance(id, launchParams);
		});
	}
}

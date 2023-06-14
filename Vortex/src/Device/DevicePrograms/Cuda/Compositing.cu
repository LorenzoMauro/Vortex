#include "../CudaKernels.h"
#include "device_launch_parameters.h"
#include "Device/CUDAChecks.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/DevicePrograms/LaunchParams.h"

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

		const FrameBufferData* frameBuffer = &launchParams->frameBuffer;
		const math::vec2ui& frameSize = frameBuffer->frameSize;
		if (x >= frameSize.x || y >= frameSize.y) return;

		const uint32_t fbIndex = x + y * frameSize.x;

		const math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		const math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		const math::vec3f diffuse = fireflyRemoval(frameBuffer->diffuseIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f glossy = fireflyRemoval(frameBuffer->glossyIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);
		const math::vec3f transmission = fireflyRemoval(frameBuffer->transmissionIndirect, x, y, frameSize.x, frameSize.y, kernelSize, noiseType, threshold);

		frameBuffer->fireflyPass[fbIndex] = directLightBuffer + diffuse + glossy + transmission;
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

		const math::vec3f& tmRadianceBuffer = frameBuffer->tmRadiance[fbIndex];
		const math::vec3f& directLightBuffer = frameBuffer->directLight[fbIndex];
		const math::vec3f& diffuseIndirectBuffer = frameBuffer->diffuseIndirect[fbIndex];
		const math::vec3f& glossyIndirectBuffer = frameBuffer->glossyIndirect[fbIndex];
		const math::vec3f& transmissionIndirect = frameBuffer->transmissionIndirect[fbIndex];

		switch (settings->displayBuffer)
		{

		case(RendererDeviceSettings::DisplayBuffer::FB_BEAUTY):
		{
				if(beauty!= nullptr)
				{
					outputBuffer[fbIndex] = math::vec4f(toneMap(launchParams->toneMapperSettings, beauty[fbIndex]), 1.0f);
				}
				else
				{
					outputBuffer[fbIndex] = math::vec4f(tmRadianceBuffer, 1.0f);
				}
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_NOISY):
		{
			outputBuffer[fbIndex] = math::vec4f(tmRadianceBuffer, 1.0f);
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_DIRECT_LIGHT):
		{
			outputBuffer[fbIndex] = math::vec4f(toneMap(launchParams->toneMapperSettings,directLightBuffer), 1.0f);
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_DIFFUSE_INDIRECT):
		{
			outputBuffer[fbIndex] = math::vec4f(toneMap(launchParams->toneMapperSettings, diffuseIndirectBuffer), 1.0f);
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_GLOSSY_INDIRECT):
		{
			outputBuffer[fbIndex] = math::vec4f(toneMap(launchParams->toneMapperSettings, glossyIndirectBuffer), 1.0f);
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_TRANSMISSION_INDIRECT):
		{
			outputBuffer[fbIndex] = math::vec4f(toneMap(launchParams->toneMapperSettings, transmissionIndirect), 1.0f);
		}
		break;

		case(RendererDeviceSettings::DisplayBuffer::FB_DIFFUSE):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->albedo[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_ORIENTATION):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->orientation[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_TRUE_NORMAL):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->trueNormal[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_SHADING_NORMAL):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->normal[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_TANGENT):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->tangent[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_UV):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->uv[fbIndex], 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_NOISE):
		{
			outputBuffer[fbIndex] = math::vec4f(floatToScientificRGB(frameBuffer->noiseBuffer[fbIndex].noise), 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_SAMPLES):
		{
			int samples;
			if (settings->adaptiveSampling)
			{
				samples = frameBuffer->noiseBuffer[fbIndex].samples;
			}
			else
			{
				samples = launchParams->settings->iteration;
			}
			float samplesMetric;
			if (launchParams->settings->iteration > settings->minAdaptiveSamples)
			{
				samplesMetric = (float)(samples - settings->minAdaptiveSamples) / (float)(launchParams->settings->iteration - settings->minAdaptiveSamples);
			}
			else
			{
				samplesMetric = 0.5f;
			}
			outputBuffer[fbIndex] = math::vec4f(floatToScientificRGB(ACESFitted(samplesMetric).x), 1.0f);
		}
		break;
		case(RendererDeviceSettings::DisplayBuffer::FB_DEBUG_1):
		{
			outputBuffer[fbIndex] = math::vec4f(frameBuffer->debugColor1[fbIndex], 1.0f);
		}
		break;
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
}

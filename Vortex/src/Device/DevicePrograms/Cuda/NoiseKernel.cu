#include "../CudaKernels.h"
#include "device_launch_parameters.h"
#include "Device/CUDAChecks.h"
#include "Device/DevicePrograms/nvccUtils.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Scene/Nodes/RendererSettings.h"
#include <curand_kernel.h>

namespace vtx
{

	__forceinline__ __host__ __device__ float hue(const math::vec3f& rgb)
	{
		float max = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
		float min = fminf(rgb.x, fminf(rgb.y, rgb.z));

		if (max == min)
		{
			return 0.0f;  // it's gray
		}

		float hue;
		if (max == rgb.x)
		{
			hue = (rgb.y - rgb.z) / (max - min);
		}
		else if (max == rgb.y)
		{
			hue = 2.0f + (rgb.z - rgb.x) / (max - min);
		}
		else
		{
			hue = 4.0f + (rgb.x - rgb.y) / (max - min);
		}

		hue *= 60.0f;
		if (hue < 0.0f)
		{
			hue += 360.0f;
		}

		return hue / 360.0f;  // Normalize to [0, 1]
	}

	__forceinline__ __host__ __device__ float luminance(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{0.30f, 0.59f, 0.11f};
		return dot(rgb, ntscLuminance);
	}

	__forceinline__ __host__ __device__ float color(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };
		return dot(rgb, ntscLuminance);
	}

	__forceinline__ __device__ float getNoise(const math::vec3f* buffer, const int x, const int y, const int width, const int height, const int kernelSize, const NoiseType noiseType, bool normalize = true)
	{
		// kernelSize is assumed to be an odd number
		const int halfKernel = kernelSize / 2;
		float centerValue;
		if (noiseType == HUE)
		{
			centerValue = hue(buffer[y * width + x]);
		}
		else if(noiseType == LUMINANCE)
		{
			centerValue = luminance(buffer[y * width + x]);
		}

		float maxDiff = FLT_MIN;  // New variable to keep track of maximum difference
		float maxValue = FLT_MIN;  // New variable to keep track of maximum difference
		float minValue = FLT_MAX;  // New variable to keep track of maximum difference
		int count = 0;
		for (int dx = -halfKernel; dx <= halfKernel; ++dx)
		{
			for (int dy = -halfKernel; dy <= halfKernel; ++dy)
			{
				const int nx = x + dx;
				const int ny = y + dy;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					float diff;
					float value;
					if(noiseType == HUE)
					{
						value = hue(buffer[ny * width + nx]);
						diff = fabsf(value - centerValue);
						count++;
					}
					else if (noiseType == LUMINANCE)
					{
						value = luminance(buffer[ny * width + nx]);
						diff = fabsf(value - centerValue);
						count++;
					}
					else
					{
						value = math::length<float>(buffer[ny * width + nx] - 0.0f);
						diff = math::length<float>(buffer[ny * width + nx] - buffer[y * width + x]);
						count++;
					}
					maxDiff = fmaxf(maxDiff, diff);  // Update maxDiff if this difference is larger
					maxValue = fmaxf(maxValue, value);  // Update maxDiff if this difference is larger
					minValue = fminf(minValue, value);  // Update maxDiff if this difference is larger
				}
			}
		}
		//if (maxDiff < 0.01f) return 0.0f;
		if (normalize)
		{
			float kernelNoise = (maxDiff - minValue) / (maxValue - minValue);
			return kernelNoise;
		}
			return maxDiff;
	}

	__global__ void computeNoise(const LaunchParams* params, float* noiseSum, const int kernelSize, const float albedoNormalInfluence)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		const unsigned& width = params->frameBuffer.frameSize.x;
		const unsigned& height = params->frameBuffer.frameSize.x;
		const math::vec3f* radianceBuffer = params->frameBuffer.tmRadiance;
		const math::vec3f* albedoBuffer = params->frameBuffer.albedoNormalized;
		const math::vec3f* normalBuffer = params->frameBuffer.normalNormalized;
		NoiseData* noiseBuffer = params->frameBuffer.noiseBuffer;

		if (x >= width || y >= height) return;

		bool isFirstMiss = params->frameBuffer.noiseBuffer[y * width + x].adaptiveSamples == -1;

		if(isFirstMiss)
		{
			noiseBuffer[y * width + x].noiseAbsolute = 0.0f;
			return;
		}
		const float radianceNoiseNormalize = getNoise(radianceBuffer, x, y, width, height, kernelSize, LUMINANCE);
		const float albedoNoiseNormalize = getNoise(albedoBuffer, x, y, width, height, kernelSize+2, COLOR);
		const float normalNoiseNormalize = getNoise(normalBuffer, x, y, width, height, kernelSize+2, HUE, false);

		float maxAuxiliaryNoise = fmaxf(albedoNoiseNormalize, normalNoiseNormalize*2);
		float noise = radianceNoiseNormalize - albedoNormalInfluence * maxAuxiliaryNoise;
		noise = fmaxf(0.0f, sqrtf(noise));
		cuAtomicAdd(noiseSum, noise);
		noiseBuffer[y * width + x].noiseAbsolute = noise;
	}

	__global__ void noiseToSamples(const LaunchParams* params, const float* noiseSum, int* remainingSamples, int* allocatedSamples)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		const unsigned& width = params->frameBuffer.frameSize.x;
		const unsigned& height = params->frameBuffer.frameSize.y;
		NoiseData* noiseBuffer = params->frameBuffer.noiseBuffer;
		if (x >= width || y >= height) return;


		if (params->frameBuffer.noiseBuffer[y * width + x].adaptiveSamples == -1)
		{
			noiseBuffer[y * width + x].normalizedNoise = 0.0f;
			return;
		}

		curandState_t state;
		const float noise = noiseBuffer[y * width + x].noiseAbsolute;
		curand_init(x, y, (int)noise + params->settings->iteration, &state);
		noiseBuffer[y * width + x].normalizedNoise = noise / (*noiseSum);
		noiseBuffer[y * width + x].adaptiveSamples = 0;

		int totAvailableSamples = width * height;
		float allocatedOnThis = (float)totAvailableSamples * noiseBuffer[y * width + x].normalizedNoise;
		int allocatedInt = (int)roundf(allocatedOnThis);

		if (allocatedInt == 0)
		{
			float rnd = curand_uniform(&state);
			if (rnd > 0.5f)
			{
				allocatedInt = 1;
				int remainingReturn = cuAtomicSub(remainingSamples, allocatedInt);
				noiseBuffer[y * width + x].adaptiveSamples = allocatedInt;
				cuAtomicAdd(allocatedSamples, allocatedInt);
			}
		}
		else
		{
			noiseBuffer[y * width + x].adaptiveSamples = 10; // to be elaborated at next step

		}
	}
	
	__global__ void distributeSamples(const LaunchParams* params, const float* noiseSum, int* remainingSamples, int*allocatedSamples)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;
		const unsigned& width = params->frameBuffer.frameSize.x;
		const unsigned& height = params->frameBuffer.frameSize.y;
		NoiseData* noiseBuffer = params->frameBuffer.noiseBuffer;
		if (x >= width || y >= height) return;


		if (params->frameBuffer.noiseBuffer[y * width + x].adaptiveSamples <= 1)
		{
			return;
		}

		curandState_t state;
		const float noise = noiseBuffer[y * width + x].noiseAbsolute;
		curand_init(x, y, (int)noise + params->settings->iteration, &state);
		noiseBuffer[y * width + x].normalizedNoise = noise / (*noiseSum);
		noiseBuffer[y * width + x].adaptiveSamples = 0;

		int totAvailableSamples = *remainingSamples;
		float allocatedOnThis = (float)totAvailableSamples * noiseBuffer[y * width + x].normalizedNoise;
		int allocatedInt = (int)(allocatedOnThis);

		int remainingReturn = cuAtomicSub(remainingSamples, allocatedInt);
		noiseBuffer[y * width + x].adaptiveSamples = allocatedInt;
		cuAtomicAdd(allocatedSamples, allocatedInt);
	}


	__global__ void rangeKernel(const math::vec3f* inputBuffer, math::vec2f* blockRange, const int size, const NoiseType noiseType)
	{
		// Calculate the index for this thread
		const int index = blockIdx.x * blockDim.x + threadIdx.x;

		// Calculate the stride for this thread
		const int stride = blockDim.x * gridDim.x;

		math::vec2f localRange{FLT_MAX, -FLT_MAX};

		for (int i = index; i < size; i += stride)
		{
			float value;
			switch (noiseType)
			{
				case LUMINANCE:
					value = luminance(inputBuffer[i]);
					break;
			case COLOR:
					value = math::length<float>(inputBuffer[i] - math::vec3f(0.0f));
					break;
			}

			localRange.x = (localRange.x > value) ? value : localRange.x;
			localRange.y = (localRange.y < value) ? value : localRange.y;
		}

		__shared__ math::vec2f rangeShared[256];
		rangeShared[threadIdx.x] = localRange;

		// Synchronize to ensure all threads have written to shared memory
		cuSynchThread();
		// Only continue for the first thread of the block
		if (threadIdx.x != 0) return;

		// Sum up the local sums
		math::vec2f range;
		range.x = FLT_MAX;
		range.y = -FLT_MAX;

		for (int i = 0; i < blockDim.x; ++i)
		{
			range.x = (range.x > rangeShared[i].x) ? rangeShared[i].x : range.x;
			range.y = (range.y < rangeShared[i].y) ? rangeShared[i].y : range.y;
		}

		blockRange[blockIdx.x] = range;
	}

	__device__ float atomicMinFloat(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = cuAtomicCas(address_as_i, assumed, cuda_FLOAT_AS_INT(::fminf(val, cuda_INT_AS_FLOAT(assumed))));
		} while (assumed != old);
		return cuda_INT_AS_FLOAT(old);
	}

	__device__ float atomicMaxFloat(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = cuAtomicCas(address_as_i, assumed, cuda_FLOAT_AS_INT(::fmaxf(val, cuda_INT_AS_FLOAT(assumed))));
		} while (assumed != old);
		return cuda_INT_AS_FLOAT(old);
	}

	__global__ void totalRange(const math::vec2f* blockRanges, math::vec2f* globalRange, const int size) {
		const int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= size) return;

		const math::vec2f range = blockRanges[index];

		atomicMinFloat(&globalRange[0].x, range.x);
		atomicMaxFloat(&globalRange[0].y, range.y);
	}

	void noiseComputation(const LaunchParams* deviceParams, const graph::RendererSettings& settings, const int& rendererNodeId)
	{

		/*const CUDABuffer& tmRadianceBuffer = GET_BUFFER(device::Buffers::FrameBufferBuffers, rendererNodeId, tmRadiance);
		const auto*       tmRadiance       = tmRadianceBuffer.castedPointer<math::vec3f>();
		const CUDABuffer & albedoNormalizedBuffer = GET_BUFFER(device::Buffers::FrameBufferBuffers, rendererNodeId, albedoNormalized);
		const auto*        albedoNormalized       = albedoNormalizedBuffer.castedPointer<math::vec3f>();
		const CUDABuffer & normalNormalizedBuffer = GET_BUFFER(device::Buffers::FrameBufferBuffers, rendererNodeId, normalNormalized);
		const auto*        normalNormalized       = normalNormalizedBuffer.castedPointer<math::vec3f>();*/

		const CUDABuffer& radianceRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, radianceRangeBuffer);
		auto* radianceRange = radianceRangeBuffer.castedPointer<math::vec2f>();

		const CUDABuffer & albedoRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, albedoRangeBuffer);
		auto* albedoRange = albedoRangeBuffer.castedPointer<math::vec2f>();

		const CUDABuffer & normalRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, normalRangeBuffer);
		auto* normalRange = normalRangeBuffer.castedPointer<math::vec2f>();

		// Initialize the global ranges to the maximum and minimum possible values
		/*math::vec2f initRange;
		initRange.x = FLT_MAX;
		initRange.y = -FLT_MAX;
		CUDABuffer& globalRadianceRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, globalRadianceRangeBuffer);
		globalRadianceRangeBuffer.upload<math::vec2f>(initRange);
		auto* globalRadianceRange = globalRadianceRangeBuffer.castedPointer<math::vec2f>();
		CUDABuffer& globalAlbedoRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, globalAlbedoRangeBuffer);
		globalAlbedoRangeBuffer.upload<math::vec2f>(initRange);
		auto* globalAlbedoRange = globalAlbedoRangeBuffer.castedPointer<math::vec2f>();
		CUDABuffer& globalNormalRangeBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, globalNormalRangeBuffer);
		globalNormalRangeBuffer.upload<math::vec2f>(initRange);
		auto* globalNormalRange = globalNormalRangeBuffer.castedPointer<math::vec2f>();*/

		CUDABuffer& noiseSumBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, noiseSumBuffer);
		noiseSumBuffer.upload<float>(0.0f);
		auto* noiseSum = noiseSumBuffer.castedPointer<float>();

		const int width = UPLOAD_DATA->frameBufferData.frameSize.x;
		const int height = UPLOAD_DATA->frameBufferData.frameSize.y;

		CUDABuffer& remainingSamplesBuffer = GET_BUFFER(device::Buffers::NoiseComputationBuffers, rendererNodeId, remainingSamplesBuffer);
		remainingSamplesBuffer.upload<int>(width*height);
		auto* remainingSamples = remainingSamplesBuffer.castedPointer<int>();

		//Fetch Radiance, Albedo and Normal Value Range
		//{
		//	const int size = width * height;
		//	constexpr int threadsPerBlock = 256;
		//	const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
		//	// Launch the kernel to compute the block sums
		//	//CUDA_CHECK(cudaDeviceSynchronize());
		//	rangeKernel<<<numBlocks, threadsPerBlock>>>(tmRadiance, radianceRange, size, LUMINANCE);
		//	rangeKernel<<<numBlocks, threadsPerBlock>>>(albedoNormalized, albedoRange, size, COLOR);
		//	rangeKernel<<<numBlocks, threadsPerBlock>>>(normalNormalized, normalRange, size, COLOR);
		//	// Calculate the number of threads per block
		//	int threadsPerBlockTot = numBlocks > 1024 ? 1024 : numBlocks;
		//	totalRange<<<1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float)>>>(radianceRange, globalRadianceRange, numBlocks);
		//	totalRange<<<1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float)>>>(albedoRange, globalAlbedoRange, numBlocks);
		//	totalRange<<<1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float)>>>(normalRange, globalNormalRange, numBlocks);
		//}

		{
			dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
			// Calculate the number of blocks needed in each dimension
			dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

			computeNoise<<<numBlocks, threadsPerBlock>>> (deviceParams, noiseSum, settings.noiseKernelSize, settings.albedoNormalNoiseInfluence);
		}

		{
			dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
			// Calculate the number of blocks needed in each dimension
			dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);


			CUDABuffer allocatedSampleBuffer;
			allocatedSampleBuffer.upload(0);
			int* allocatedSample = allocatedSampleBuffer.castedPointer<int>();

			noiseToSamples <<<numBlocks, threadsPerBlock>>> (deviceParams, noiseSum, remainingSamples, allocatedSample);
			distributeSamples <<<numBlocks, threadsPerBlock>>> (deviceParams,noiseSum, remainingSamples, allocatedSample);

			int allocated;
			int remaining;
			allocatedSampleBuffer.download(&allocated);
			remainingSamplesBuffer.download(&remaining);
		}
	}
}

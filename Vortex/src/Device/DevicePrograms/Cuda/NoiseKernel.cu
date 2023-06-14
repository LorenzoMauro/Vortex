#include "../CudaKernels.h"
#include "device_launch_parameters.h"
#include "Device/CUDAChecks.h"
#include "Device/DevicePrograms/nvccUtils.h"

namespace vtx
{

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

	__forceinline__ __device__ float getNoise(const math::vec3f* buffer, const int x, const int y, const int width, const int height, const int kernelSize, const NoiseType noiseType)
	{
		float sum = 0.0f;
		int   count = 0;

		bool first = true;
		// kernelSize is assumed to be an odd number
		const int halfKernel = kernelSize / 2;
		float centerValue;
		if(noiseType == LUMINANCE)
		{
			centerValue = luminance(buffer[y * width + x]);
		}

		float maxDiff = 0.0f;  // New variable to keep track of maximum difference

		for (int dx = -halfKernel; dx <= halfKernel; ++dx)
		{
			for (int dy = -halfKernel; dy <= halfKernel; ++dy)
			{
				const int nx = x + dx;
				const int ny = y + dy;

				if (nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					float diff;
					if (noiseType == LUMINANCE)
					{
						const float pixelNoise = luminance(buffer[ny * width + nx]);
						diff = fabsf(pixelNoise - centerValue);
					}
					else
					{
						diff = math::length<float>(buffer[ny * width + nx] - buffer[y * width + x]);
					}
					maxDiff = fmaxf(maxDiff, diff);  // Update maxDiff if this difference is larger
				}
			}
		}

		return maxDiff;
	}

	__global__ void computeNoise(const math::vec3f* radianceBuffer, const math::vec3f* albedoBuffer, const math::vec3f* normalBuffer, NoiseData* noiseBuffer, const int width,
								 const int          height, const int      kernelSize)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height) return;

		const int   albedoReduction  = 2;
		const int   albedoKernelSize = (kernelSize - albedoReduction > 0) ? (kernelSize - albedoReduction) : kernelSize;
		const int   normalKernelSize = kernelSize + 4;
		float       radianceNoise    = getNoise(radianceBuffer, x, y, width, height, kernelSize, LUMINANCE);
		float albedoNoise      = getNoise(albedoBuffer, x, y, width, height, kernelSize, COLOR);
		float normalNoise      = getNoise(normalBuffer, x, y, width, height, kernelSize, COLOR);


		noiseBuffer[y * width + x].radianceNoise = radianceNoise;
		noiseBuffer[y * width + x].albedoNoise = albedoNoise;
		noiseBuffer[y * width + x].normalNoise = normalNoise;
	}

	__global__ void normalizeNoise(NoiseData* noiseBuffer, math::vec2f* globalRadianceRange, math::vec2f* globalAlbedoRange, math::vec2f *globalNormalRange, const int width, const int height, float albedoNormalInfluence)
	{
		const int x = threadIdx.x + blockIdx.x * blockDim.x;
		const int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= width || y >= height) return;

		float radianceNoise = noiseBuffer[y * width + x].radianceNoise;
		float albedoNoise = noiseBuffer[y * width + x].albedoNoise;
		float normalNoise = noiseBuffer[y * width + x].normalNoise;

		if(globalRadianceRange[0].y <= 0.0f)
		{
			radianceNoise = 0.0f;
		}
		else
		{
			radianceNoise = (radianceNoise - globalRadianceRange[0].x) / (globalRadianceRange[0].y - globalRadianceRange[0].x);
		}
		if (globalAlbedoRange[0].y <= 0.0f)
		{
			albedoNoise = 0.0f;
		}
		else
		{
			albedoNoise = (albedoNoise - globalAlbedoRange[0].x) / (globalAlbedoRange[0].y - globalAlbedoRange[0].x);
		}
		if (globalNormalRange[0].y <= 0.0f)
		{
			normalNoise = 0.0f;
		}
		else
		{
			normalNoise = (normalNoise - globalNormalRange[0].x) / (globalNormalRange[0].y - globalNormalRange[0].x);
		}

		//printf("Radiance Value %.2f Min %.2f Max %.2f\n", radianceNoise, globalRadianceRange[0].x, globalRadianceRange[0].y);

		float removalNoise = (albedoNoise > normalNoise) ? albedoNoise : normalNoise;
		removalNoise = (radianceNoise < removalNoise) ? 0.0f : removalNoise;

		float outputNoise = radianceNoise - albedoNormalInfluence * removalNoise;
		outputNoise = (outputNoise > 1.0f) ? 1.0f : outputNoise;
		outputNoise = (outputNoise < 0.0f) ? 0.0f : outputNoise;

		noiseBuffer[y * width + x].noise = outputNoise;
	}

	__global__ void rangeKernel(NoiseData* noiseBuffer, math::vec2f* blockRange, const int size, const NoiseDataType type)
	{
		// Calculate the index for this thread
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		// Calculate the stride for this thread
		int stride = blockDim.x * gridDim.x;

		math::vec2f localRange;
		localRange.x = FLT_MAX;
		localRange.y = -FLT_MAX;

		for (int i = index; i < size; i += stride)
		{
			float value;
			switch (type)
			{
				case RADIANCE:
					value = noiseBuffer[i].radianceNoise;
					break;
				case ALBEDO:
					value = noiseBuffer[i].albedoNoise;
					break;
				case NORMAL:
					value = noiseBuffer[i].normalNoise;
					break;
			}

			localRange.x = (localRange.x > value) ? value : localRange.x;
			localRange.y = (localRange.y < value) ? value : localRange.y;
		}

		__shared__ math::vec2f rangeShared[256];
		rangeShared[threadIdx.x] = localRange;

		// Synchronize to ensure all threads have written to shared memory
		cuda_SYNCTHREADS();
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
			old = cuda_ATOMICCAS(address_as_i, assumed, cuda_FLOAT_AS_INT(::fminf(val, cuda_INT_AS_FLOAT(assumed))));
		} while (assumed != old);
		return cuda_INT_AS_FLOAT(old);
	}

	__device__ float atomicMaxFloat(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = cuda_ATOMICCAS(address_as_i, assumed, cuda_FLOAT_AS_INT(::fmaxf(val, cuda_INT_AS_FLOAT(assumed))));
		} while (assumed != old);
		return cuda_INT_AS_FLOAT(old);
	}

	__global__ void totalRange(math::vec2f* blockRanges, math::vec2f* globalRange, const int size) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= size) return;

		math::vec2f range = blockRanges[index];

		atomicMinFloat(&globalRange[0].x, range.x);
		atomicMaxFloat(&globalRange[0].y, range.y);
	}

	void noiseComputation(NoiseData*         noiseBuffer,
						  const math::vec3f* radianceBuffer, const math::vec3f* albedoBuffer, const math::vec3f* normalBuffer,
						  math::vec2f*       radianceRange, math::vec2f*        albedoRange, math::vec2f*         normalRange,
						  const int          width, const int                   height, const int                noiseKernelSize, const float albedoNormalInfluence)
	{
		math::vec2f* globalRadianceRange, * globalAlbedoRange, * globalNormalRange;
		{
			dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
			// Calculate the number of blocks needed in each dimension
			dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

			computeNoise<<<numBlocks, threadsPerBlock>>>(radianceBuffer, albedoBuffer, normalBuffer, noiseBuffer, width, height, noiseKernelSize);
			CUDA_CHECK(cudaDeviceSynchronize());
		}

		{
			const int size = width * height;
			constexpr int threadsPerBlock = 256;
			const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

			// Launch the kernel to compute the block sums
			CUDA_CHECK(cudaDeviceSynchronize());
			rangeKernel<<<numBlocks, threadsPerBlock>>>(noiseBuffer, radianceRange, size, RADIANCE);
			rangeKernel<<<numBlocks, threadsPerBlock>>>(noiseBuffer, albedoRange, size, ALBEDO);
			rangeKernel<<<numBlocks, threadsPerBlock>>>(noiseBuffer, normalRange, size, NORMAL);

			// Calculate the number of threads per block
			int threadsPerBlockTot = numBlocks > 1024 ? 1024 : numBlocks;

			// Allocate device memory for global ranges
			CUDA_CHECK(cudaMalloc(&globalRadianceRange, sizeof(math::vec2f)));
			CUDA_CHECK(cudaMalloc(&globalAlbedoRange, sizeof(math::vec2f)));
			CUDA_CHECK(cudaMalloc(&globalNormalRange, sizeof(math::vec2f)));

			// Initialize the global ranges to the maximum and minimum possible values
			math::vec2f initRange;
			initRange.x = FLT_MAX;
			initRange.y = -FLT_MAX;

			CUDA_CHECK(cudaMemcpy(globalRadianceRange, &initRange, sizeof(math::vec2f), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(globalAlbedoRange, &initRange, sizeof(math::vec2f), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(globalNormalRange, &initRange, sizeof(math::vec2f), cudaMemcpyHostToDevice));


			CUDA_CHECK(cudaDeviceSynchronize());
			totalRange << <1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float) >> > (radianceRange, globalRadianceRange, numBlocks);
			totalRange << <1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float) >> > (albedoRange, globalAlbedoRange, numBlocks);
			totalRange << <1, threadsPerBlockTot, threadsPerBlockTot * 2 * sizeof(float) >> > (normalRange, globalNormalRange, numBlocks);
			// Now globalRadianceRange[0], globalAlbedoRange[0], and globalNormalRange[0] contains the total range of the noise values
		}

		{
			dim3 threadsPerBlock(16, 16);  // a common choice for 2D data
			// Calculate the number of blocks needed in each dimension
			dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
						   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

			CUDA_CHECK(cudaDeviceSynchronize());
			normalizeNoise<<<numBlocks, threadsPerBlock>>>(noiseBuffer, globalRadianceRange, globalAlbedoRange, globalNormalRange, width, height, albedoNormalInfluence);
		}
	}
}

#pragma once
#include "Device/DevicePrograms/nvccUtils.h"
#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Device/DevicePrograms/Utils.h"
#include "Device/UploadCode/CUDABuffer.h"
#include "Device/Wrappers/KernelLaunch.h"

namespace vtx::cuda{

	__forceinline__ __device__ void computeMapeDevice(const math::vec3f* target, const math::vec3f* input, math::vec2f* mape, const int& id)
	{
		constexpr float e = 0.01f;
		const float rMape = fabsf(input[id].x - target[id].x) / (target[id].x + e);
		const float gMape = fabsf(input[id].y - target[id].y) / (target[id].y + e);
		const float bMape = fabsf(input[id].z - target[id].z) / (target[id].z + e);
		float newValue = (rMape + gMape + bMape) / 3.0f;
		float& mapeValue = (*mape).x;
		float& count = (*mape).y;
		float addedMapeValue = cuAtomicAdd(&mapeValue, newValue);
		float addedCountValue = cuAtomicAdd(&count, 1);
	}

	__forceinline__ __device__ void computeMseDevice(const math::vec3f* target, const math::vec3f* input, math::vec2f* mse, const int& id)
	{
		const math::vec3f delta = input[id] - target[id];
		const float rMse = delta.x * delta.x;
		const float gMse = delta.y * delta.y;
		const float bMse = delta.z * delta.z;
		float newValue = (rMse + gMse + bMse) / 3.0f;
		float& mseValue = (*mse).x;
		float& count = (*mse).y;
		float addedMseValue = cuAtomicAdd(&mseValue, newValue);
		float addedCountValue = cuAtomicAdd(&count, 1);
	}

	float computeMape(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height)
	{
		CUDABuffer mapeBuffer;
		mapeBuffer.resize(sizeof(math::vec2f));
		mapeBuffer.upload(math::vec2f{0.0f, 0.0f});
		auto* mape = mapeBuffer.castedPointer<math::vec2f>();

		const int size = width * height;
		gpuParallelFor(eventNames[K_MAPE],
			size,
			[groundTruth, input, mape] __device__(const int id)
		{
			computeMapeDevice(groundTruth, input, mape, id);
		});

		math::vec2f hostMape;
		mapeBuffer.download(&hostMape);
		const float mapeValue = hostMape.x / hostMape.y;
		mapeBuffer.free();

		return mapeValue;
	}


	float computeMse(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height)
	{
		CUDABuffer mseBuffer;
		mseBuffer.resize(sizeof(math::vec2f));
		mseBuffer.upload(math::vec2f{0.0f, 0.0f});
		auto* mse = mseBuffer.castedPointer<math::vec2f>();

		const int size = width * height;
		gpuParallelFor(eventNames[K_MSE],
			size,
			[groundTruth, input, mse] __device__(const int id)
		{
			computeMseDevice(groundTruth, input, mse, id);
		});

		math::vec2f hostMse;
		mseBuffer.download(&hostMse);
		const float mseValue = hostMse.x / hostMse.y;
		mseBuffer.free();

		return mseValue;
	}

	__forceinline__ __device__ void encodeTriangleWave(
		const unsigned id,
		const unsigned nFrequencies,
		const unsigned nFeatures,
		const float* input,
		float* output
	)
	{
		const unsigned elementIdx = id / (nFrequencies * nFeatures);
		const unsigned featureIdx = (id / nFrequencies) % nFeatures;
		const unsigned frequencyIdx = id % nFrequencies;

		const unsigned inputIdx = elementIdx * nFeatures + featureIdx;
		const unsigned outputIdx = id;
		const int log2Frequency = frequencyIdx;

		const float inputVal = input[inputIdx];
		const float x = scalbnf(inputVal, log2Frequency - 1);

		// Small log2_frequency-based phase shift to help disambiguate locations
		const float val = x + (float)log2Frequency * 0.25f;
		const float result = fabsf(val - floorf(val) - 0.5f) * 4 - 1;

		output[outputIdx] = result;
	}

	float* triangleWaveEncoding(CUDABuffer& outputBuffer, float* input, const int inputFeatures, const int outputFeatures, const int nFrequencies)
	{
		const int outputBufferSize = inputFeatures * outputFeatures * nFrequencies;

		if (const size_t memSize = outputBufferSize * sizeof(float); outputBuffer.dPointer() == NULL || outputBuffer.bytesSize() != memSize)
		{
			outputBuffer.resize(outputBufferSize * sizeof(float));
		}

		auto output = outputBuffer.castedPointer<float>();
		int nFrequencyCopy = nFrequencies;
		int nFeatureCopy = outputFeatures;
		float* inputCopy = input;

		gpuParallelFor(eventNames[N_TRIANGLE_WAVE_ENCODING],
			outputBufferSize,
			[nFrequencyCopy, nFeatureCopy, inputCopy, output] __device__(const int id)
		{
			encodeTriangleWave(id, nFrequencyCopy, nFeatureCopy, inputCopy, output);
		});

		return output;
	}


	__forceinline__ __device__ float edgeTransform(const float x, const float curvature, const float scale) {
		if (x > 0.0f && x < 1.0f) {
			//float value = fminf(1.0f, scale * expf((-powf((x - 1.0f), 2.0f)) / curvature));
			float tX = x - 1.0f;
			float value = scale * expf(-(tX * tX) / curvature);
			return value;
		}
		else {
			return 0.0f;
		}
	}

	__forceinline__ __device__ void overlayEdgeKernel(
		const int id,
		const int width,
		const float* edgeData,
		math::vec4f* image,
		const float curvature,
		const float scale
	) {
		const int x = id % width;
		const int y = id / width;

		// Define the orange color
		const math::vec3f orange = math::vec3f(1.0f, 0.5f, 0.0f);
		float edgeValue = edgeData[y * width + x];
		
		edgeValue = edgeTransform(edgeValue, curvature, scale);

		image[y * width + x] = math::vec4f(utl::lerp<math::vec3f>(math::vec3f(image[y * width + x]), orange, edgeValue), 1.0f);
		//image[y * width + x] = math::vec4f(edgeValue, 1.0f);
	}

	void overlayEdgesOnImage(
		float*       d_edgeData,
		math::vec4f* d_image,
		int          width,
		const int    height,
		const float  curvature,
		const float  scale
	)
	{
		const int dataSize = width * height;
		gpuParallelFor("OverlayEdges",
			dataSize,
			[d_edgeData, d_image, width, curvature, scale] __device__(const int id)
		{
			overlayEdgeKernel(id, width, d_edgeData, d_image, curvature, scale);
		});
	}

	__forceinline__ __device__ void selectionEdgeDevice(
		const int id,
		const int width,
		const int height,
		const float* data,
		float* output,
		const float value,
		const int thickness
	)
	{
		const int x = id % width;
		const int y = id / width;

		int matchCount = 0;      // Count of neighboring pixels matching the target value.
		int nonMatchCount = 0;   // Count of neighboring pixels not matching the target value.
		int totalNeighbors = 0;  // Total neighboring pixels within the thickness.

		if(gdt::isEqual(data[id], value))
		{
			output[id] = 0.0f;
		}
		for (int dx = -thickness; dx <= thickness; dx++)
		{
			for (int dy = -thickness; dy <= thickness; dy++)
			{
				// Skip the center pixel.
				//if (dx == 0 && dy == 0) continue;

				const int nx = x + dx;
				const int ny = y + dy;
				if (nx >= 0 && nx < width && ny >= 0 && ny < height)
				{
					float neighborValue = data[ny * width + nx];

					if (gdt::isEqual(neighborValue, value))
					{
						matchCount++;
					}
					else
					{
						nonMatchCount++;
					}

					totalNeighbors++;
				}
			}
		}

		// Compute the edge value based on the ratio of match and non-match counts.
		if (matchCount == nonMatchCount)
		{
			output[id] = 1.0f;
		}
		else
		{
			float ratio = static_cast<float>(abs(matchCount - nonMatchCount)) / static_cast<float>(totalNeighbors);
			output[id] = 1.0f - ratio;
		}
	}

	float* selectionEdge(
		float* d_data,
		int    width,
		int    height,
		float  value,
		int    thickness
	)
	{
		const int  dataSize = width * height;
		CUDABuffer cudaBuffer;
		auto*      output = cudaBuffer.alloc<float>(dataSize);
		gpuParallelFor("CustomEdge",
					   dataSize,
					   [output, width, height, d_data, value, thickness] __device__(const int id)
					   {
			selectionEdgeDevice(id, width, height, d_data, output, value, thickness);
					   });
		return output;
	}

	void overlaySelectionEdge(
		float* gBuffer,
		math::vec4f* outputImage,
		int    width,
		int    height,
		float value,
		const float curvature,
		const float scale,
		CUDABuffer* cudaBuffer = nullptr
	)
	{
		const int  dataSize = width * height;
		float* edgeMap = nullptr;
		bool deleteBuffer = false;
		if (cudaBuffer == nullptr)
		{
			cudaBuffer = new CUDABuffer();
			deleteBuffer = true;
		}
		edgeMap = cudaBuffer->alloc<float>(dataSize);
		gpuParallelFor("Overlay Selection Edge",
			dataSize,
		[edgeMap, width, height, gBuffer, value, outputImage, curvature, scale] __device__(const int id)
		{
			selectionEdgeDevice(id, width, height, gBuffer, edgeMap, value, 1);
			overlayEdgeKernel(id, width, edgeMap, outputImage, curvature, scale);
		});
		if (deleteBuffer)
		{
			cudaBuffer->free();
			delete cudaBuffer;
		}
	}
}

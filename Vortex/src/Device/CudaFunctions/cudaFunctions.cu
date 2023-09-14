#pragma once
#include "Device/DevicePrograms/nvccUtils.h"
#include "Core/Math.h"
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
}

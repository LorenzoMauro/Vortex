#pragma once
#ifndef ENCODING_H
#define ENCODING_H

#include "cuda_runtime.h"
#include "Device/KernelInfos.h"
#include "Device/UploadCode/CUDABuffer.h"
#include "Device/Wrappers/dWrapper.h"

namespace vtx::encoding
{
	__forceinline__ __device__ void encodeTriangleWave(
		const unsigned id,
		const unsigned nFrequencies,
		const unsigned nFeatures,
		const float* input,
		float* output
	)
	{
		//if(id==0)
		//{
		//	printf(
		//		"nFrequencies: %d\n"
		//		"nFeatures: %d\n"
		//		"input pointer %p\n"
		//		"output pointer %p\n",
		//		nFrequencies,
		//		nFeatures,
		//		input,
		//		output
		//	);
		//}
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

		//printf(
		//	"id: %d, elementIdx: %d featureIdx: %d frequencyIdx: %d\n"
		//	"inputIdx: %d outputIdx: %d\n"
		//	"log2Frequency: %d\n"
		//	"Input val: %f\n"
		//	"x: %f\n"
		//	"val: %f\n"
		//	"result: %f\n",
		//	id,
		//	elementIdx,
		//	featureIdx,
		//	frequencyIdx,
		//	inputIdx,
		//	outputIdx,
		//	log2Frequency,
		//	inputVal,
		//	x,
		//	val,
		//	result
		//);
	}

	class TriangleWaveEncoding
	{
	public:
		static float* encode(CUDABuffer& outputBuffer, float* input, const int inputSize, int featureSize, int nFrequencies)
		{
			const int outputBufferSize = inputSize * featureSize * nFrequencies;

			if(const size_t memSize = outputBufferSize * sizeof(float); outputBuffer.dPointer() == NULL || outputBuffer.bytesSize() != memSize)
			{
				outputBuffer.resize(outputBufferSize * sizeof(float));
			}

			auto output = outputBuffer.castedPointer<float>();
			int nFrequencyCopy = nFrequencies;
			int nFeatureCopy = featureSize;
			float* inputCopy = input;

			//VTX_INFO(
			//	"InputSize {}\nOutputBufferSize {}\nnFeature {}\nnFrequency {}\ninput {}\noutput{}\n", inputSize, outputBufferSize, nFrequencyCopy, nFeatureCopy, (void*)inputCopy, (void*)output
			//);
			gpuParallelFor(eventNames[N_TRIANGLE_WAVE_ENCODING],
				outputBufferSize,
				[nFrequencyCopy, nFeatureCopy, inputCopy, output] __device__(int id)
			{
				encodeTriangleWave(id, nFrequencyCopy, nFeatureCopy, inputCopy, output);
			});

			//CUDA_SYNC_CHECK();

			return output;
		}
	};
}


#endif

#pragma once
#ifndef NPG_TRAINING_DATA_H
#define NPG_TRAINING_DATA_H
#include "Core/Math.h"

namespace vtx
{
	namespace device
	{
		struct NpgTrainingDataBuffers;
	}

	struct NetworkInput;

	struct NpgTrainingData
	{

		static NpgTrainingData* upload(device::NpgTrainingDataBuffers& buffers, const int& maxDatasetSize);

		static NpgTrainingData* getPreviouslyUploaded(const device::NpgTrainingDataBuffers& buffers);

	private:
		NpgTrainingData(device::NpgTrainingDataBuffers& buffers, const int& maxDatasetSize);
	public:

		NetworkInput* inputs;
		float* luminance;
		math::vec3f* incomingDirection;
		float* bsdfProbabilities;
		int nAlloc;
	};
}

#endif
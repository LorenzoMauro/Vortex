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
		math::vec3f*  outRadiance;
		math::vec3f*  incomingDirection;
		math::vec3f*  inRadiance;
		math::vec3f*  throughput;
		float*        bsdfProbabilities;
		int           nAlloc;
		float*           overallProb;
	};
}

#endif
#pragma once
#ifndef NPG_TRAINING_DATA_H
#define NPG_TRAINING_DATA_H
#include "Core/Math.h"

namespace vtx
{
	struct NetworkInput;

	struct NpgTrainingData
	{

		static NpgTrainingData* upload(const int& maxDatasetSize);

		static NpgTrainingData* getPreviouslyUploaded();

	private:
		NpgTrainingData(const int& maxDatasetSize);
	public:

		NetworkInput* inputs;
		float* luminance;
		math::vec3f* incomingDirection;
		float* bsdfProbabilities;
		int nAlloc;
	};
}

#endif
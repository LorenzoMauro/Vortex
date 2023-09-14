#pragma once
#include "Core/Math.h"

namespace vtx::cuda
{
	float computeMape(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height);

	float* triangleWaveEncoding(CUDABuffer& outputBuffer, float* input, const int inputFeatures, const int outputFeatures, const int nFrequencies);

}
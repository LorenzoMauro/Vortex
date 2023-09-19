#pragma once
#include "Core/Math.h"

namespace vtx::cuda
{
	float computeMape(math::vec3f* groundTruth, math::vec3f* input, const int& width, const int& height);

	float* triangleWaveEncoding(CUDABuffer& outputBuffer, float* input, const int inputFeatures, const int outputFeatures, const int nFrequencies);

	void overlayEdgesOnImage(
		float* d_edgeData,
		math::vec4f* d_image,
		int width,
		int height,
		const float curvature,
		const float scale
	);

	float* selectionEdge(
		float* d_data,
		int    width,
		int    height,
		float  value,
		int    thickness
	);

	void overlaySelectionEdge(
		float* gBuffer,
		math::vec4f* outputImage,
		int    width,
		int    height,
		float value,
		const float curvature,
		const float scale,
		CUDABuffer* cudaBuffer = nullptr
	);
}
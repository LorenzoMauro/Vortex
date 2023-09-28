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

	void overlaySelectionEdge(
		float* gBuffer,
		math::vec4f* outputImage,
		int    width,
		int    height,
		std::vector<float> selectedIds,
		const float curvature,
		const float scale,
		CUDABuffer* edgeMapBuffer = nullptr,
		CUDABuffer* valuesBuffer = nullptr
	);
}
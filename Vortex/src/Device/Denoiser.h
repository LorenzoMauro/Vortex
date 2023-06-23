#pragma once
#include "Core/Math.h"
#include "UploadCode/CUDABuffer.h"

namespace vtx::optix {

	class OptixDenoiserWrapper
	{
	public:

		void setInputs(CUDABuffer& iRadiance, CUDABuffer& iAlbedo, CUDABuffer& iNormal);
		void shutDown();
		void resize(const unsigned cWidth, const unsigned cHeight);

		math::vec3f* denoise(float blend) const;

		CUDABuffer scratch;
		CUDABuffer intensity;
		CUDABuffer state;
		OptixDenoiser denoiser = nullptr;
		unsigned width;
		unsigned height;
		CUDABuffer output;
		std::vector<OptixImage2D> inputLayers;
		OptixImage2D outputLayer;
		OptixImage2D commonImageSettings;

		CUDABuffer* radiance = nullptr;
		CUDABuffer* albedo = nullptr;
		CUDABuffer* normal = nullptr;
	};

}


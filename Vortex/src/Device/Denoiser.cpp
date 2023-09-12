#include "Denoiser.h"
#include "OptixWrapper.h"
#undef min
#undef max

namespace vtx::optix
{

	void OptixDenoiserWrapper::shutDown()
	{
		if (denoiser)
		{
			OPTIX_CHECK(optixDenoiserDestroy(denoiser));
		}
		scratch.free();
		intensity.free();
		state.free();
	}
	void OptixDenoiserWrapper::setInputs(CUDABuffer& iRadiance, CUDABuffer& iAlbedo, CUDABuffer& iNormal)
	{
		inputLayers.clear();
		radiance = &iRadiance;
		albedo = &iAlbedo;
		normal = &iNormal;

		// -------------------------------------------------------
		commonImageSettings.data = radiance->dPointer();
		inputLayers.push_back(commonImageSettings);

		commonImageSettings.data = albedo->dPointer();
		inputLayers.push_back(commonImageSettings);

		commonImageSettings.data = normal->dPointer();
		inputLayers.push_back(commonImageSettings);
	}

	void OptixDenoiserWrapper::resize(const unsigned cWidth, const unsigned cHeight)
	{
		width = cWidth;
		height = cHeight;

		/// Width of the image (in pixels)
		commonImageSettings.width = width;
		/// Height of the image (in pixels)
		commonImageSettings.height = height;
		/// Stride between subsequent rows of the image (in bytes).
		commonImageSettings.rowStrideInBytes = width * sizeof(math::vec3f);
		/// Stride between subsequent pixels of the image (in bytes).
		/// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
		commonImageSettings.pixelStrideInBytes = sizeof(math::vec3f);
		/// Pixel format.
		commonImageSettings.format = OPTIX_PIXEL_FORMAT_FLOAT3;

		if (denoiser) {
			OPTIX_CHECK(optixDenoiserDestroy(denoiser));
		}
		// ------------------------------------------------------------------
		// create the denoiser:
		OptixDenoiserOptions denoiserOptions = {};
		denoiserOptions.guideAlbedo = 1;
		denoiserOptions.guideNormal = 1;
		OptixResult result = optixDenoiserCreate(optix::getState()->optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser);
		OPTIX_CHECK(result);
		// .. then compute and allocate memory resources for the denoiser
		OptixDenoiserSizes denoiserReturnSizes;
		result = optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserReturnSizes);
		OPTIX_CHECK(result);

		scratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
		state.resize(denoiserReturnSizes.stateSizeInBytes);
		intensity.resize(sizeof(float));
		// ------------------------------------------------------------------
		// resize our cuda frame buffer
		output.resize(width * height * sizeof(math::vec3f));

		// -------------------------------------------------------
		outputLayer = commonImageSettings;
		outputLayer.data = output.dPointer();
		// -------------------------------------------------------

		// ------------------------------------------------------------------
		result = optixDenoiserSetup(denoiser,
			nullptr,
			width,
			height,
			state.dPointer(),
			state.bytesSize(),
			scratch.dPointer(),
			scratch.bytesSize());
		OPTIX_CHECK(result);
	}

	math::vec3f* OptixDenoiserWrapper::denoise(float blend) const
	{
		bool isReady = (radiance != nullptr && albedo != nullptr && normal != nullptr);
		isReady &= (width > 0 && height > 0);
		isReady &= (denoiser != nullptr);
		isReady &= ((void*)output.dPointer() != nullptr);
		isReady &= ((void*)scratch.dPointer() != nullptr);
		isReady &= ((void*)state.dPointer() != nullptr);
		VTX_ASSERT_CLOSE(isReady, "Trying To call Denoiser but not all data has been set up!");

		OptixDenoiserParams denoiserParams;
		denoiserParams.hdrIntensity = intensity.dPointer();
		denoiserParams.blendFactor = blend;

		OptixResult result;
		result = optixDenoiserComputeIntensity(
			denoiser,
			nullptr,
			inputLayers.data(),
			intensity.dPointer(),
			scratch.dPointer(),
			scratch.bytesSize());
		OPTIX_CHECK(result);

		OptixDenoiserGuideLayer denoiserGuideLayer = {};
		denoiserGuideLayer.albedo = inputLayers[1];
		denoiserGuideLayer.normal = inputLayers[2];

		OptixDenoiserLayer denoiserLayer = {};
		denoiserLayer.input = inputLayers[0];
		denoiserLayer.output = outputLayer;

		result = optixDenoiserInvoke(
			denoiser,
			/*stream*/0,
			&denoiserParams,
			state.dPointer(),
			state.bytesSize(),
			&denoiserGuideLayer,
			&denoiserLayer, 1,
			/*inputOffsetX*/0,
			/*inputOffsetY*/0,
			scratch.dPointer(),
			scratch.bytesSize());

		OPTIX_CHECK(result);

		return output.castedPointer<math::vec3f>();
	}

}
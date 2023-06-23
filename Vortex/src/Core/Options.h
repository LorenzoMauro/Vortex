#pragma once
#include "optix.h"
#include <string>
#include <vector>
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx
{
	enum MdlCallType
	{
		MDL_DIRECT_CALL,
		MDL_INLINE,
		MDL_CUDA
	};
	struct Options
	{
		bool        initialized = false;
		int         width;
		int         height;
		std::string windowName;
		std::string dataFolder;
		std::string imGuiIniFile;
		std::string dllPath;
		float       clearColor[4];
		bool isDebug;
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Rendering Settings /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////

		bool         runOnSeparateThread;
		uint32_t                                  maxBounces;
		uint32_t                                  maxSamples;
		bool                                      accumulate;
		bool									  useRussianRoulette;
		RendererDeviceSettings::SamplingTechnique samplingTechnique;
		RendererDeviceSettings::DisplayBuffer     displayBuffer;
		float                                     maxClamp;
		float                                     minClamp;

		int										  useWavefront;
		bool									  fitWavefront;
		bool										optixShade;
		bool         parallelShade;
		float         longPathPercentage;
		bool			useLongPathKernel;

		int   noiseKernelSize;
		int   adaptiveSampling;
		int   minAdaptiveSamples;
		int   minPixelSamples;
		int   maxPixelSamples;
		float albedoNormalNoiseInfluence;
		float noiseCutOff;

		int fireflyKernelSize;
		float fireflyThreshold;
		bool removeFireflies;

		bool enableDenoiser;
		int denoiserStart;
		float denoiserBlend;

		math::vec3f whitePoint;
		math::vec3f colorBalance;
		float       burnHighlights;
		float       crushBlacks;
		float       saturation;
		float       gamma;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		int         OptixVersion;
		int         deviceID;
		int         maxDcDepth;
		int         maxTraversableGraphDepth;
		std::string LaunchParamName;
		bool        enableCache;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		std::vector<std::string> mdlSearchPath;
		int                      numTextureSpaces;
		int                      numTextureResults;
		bool                     enable_derivatives;
		const char*              mdlOptLevel;
		MdlCallType				 mdlCallType;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Gui Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		float       nodeWidth;
		std::string fontPath;
	};

	Options* getOptions();
}

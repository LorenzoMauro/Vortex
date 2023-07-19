#pragma once
#include <string>
#include <vector>
#include "Core/Math.h"

namespace vtx
{
	enum MdlCallType
	{
		MDL_DIRECT_CALL,
		MDL_INLINE,
		MDL_CUDA
	};

	enum SamplingTechnique;
	enum DisplayBuffer;

	namespace network
	{
		enum NetworkType;
	}

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
		SamplingTechnique samplingTechnique;
		DisplayBuffer     displayBuffer;
		float                                     maxClamp;
		float                                     minClamp;

		int										  useWavefront;
		bool									  fitWavefront;
		bool										optixShade;
		bool         parallelShade;
		float         longPathPercentage;
		bool			useLongPathKernel;

		bool         useNetwork;

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
		/////////////////// Neural Network Options /////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		int batchSize;
		int maxTrainingStepPerFrame;
		float polyakFactor;
		float logAlphaStart;
		bool doInference;
		int inferenceIterationStart;
		bool clearOnInferenceStart;
		float neuralGamma;
		float         neuralSampleFraction;

		float policyLr;
		float qLr ;
		float alphaLr ;
		float autoencoderLr ;

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
		float                nodeWidth;
		std::string          fontPath;
		network::NetworkType networkType;
	};

	Options* getOptions();
}

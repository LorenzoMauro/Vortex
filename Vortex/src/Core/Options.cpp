#include "Options.h"
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx {

	static Options options;

	void startOptions()
	{

		options.width = 2100;
		options.height = 900;
		options.windowName = "Vortex";
		options.dataFolder = "E:/Dev/VortexOptix/data/";
		options.imGuiIniFile = options.dataFolder + "ImGui.ini";
		options.dllPath = "./lib/";
		options.clearColor[4] = (0.45f, 0.55f, 0.60f, 1.00f);
#ifdef NDEBUG
		options.isDebug = false;
#else
		options.isDebug = true;
#endif
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Rendering Settings /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////

		options.runOnSeparateThread = false;
		options.maxBounces = 10;
		options.maxSamples = 100000;
		options.accumulate = true;
		options.useRussianRoulette = true;
		options.samplingTechnique = SamplingTechnique::S_BSDF
		;
		options.displayBuffer = DisplayBuffer::FB_BEAUTY;
		options.maxClamp = 0.01f;
		options.minClamp = 1000.0f;
		options.noiseKernelSize = 3;

		options.adaptiveSampling = false;
		options.minAdaptiveSamples = 100;
		options.minPixelSamples = 1;
		options.maxPixelSamples = 1;
		options.albedoNormalNoiseInfluence = 1.0f;
		options.noiseCutOff = 0.1f;
		options.fireflyKernelSize = 3;
		options.fireflyThreshold = 2.0f;
		options.removeFireflies = false;

		options.useWavefront = true;
		options.optixShade = false;
		options.parallelShade = false;
		options.fitWavefront = false;
		options.longPathPercentage = 0.25f;
		options.useLongPathKernel = false;

		options.useNetwork = true;

		options.enableDenoiser = false;
		options.denoiserStart = 10;
		options.denoiserBlend = 0.1f;

		options.whitePoint = { 1.0f, 1.0f, 1.0f };
		options.colorBalance = { 1.0f, 1.0f, 1.0f };
		options.burnHighlights = 0.0f;
		options.crushBlacks = 1.0f;
		options.saturation = 1.0f;
		options.gamma = 2.2f; // Typical gamma value for sRGB color space

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Neural Network Options /////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.networkType = network::NetworkType::NT_SAC;
		options.batchSize   = 100000;

		options.polyakFactor = 0.00001f;
		options.maxTrainingStepPerFrame = 1;
		options.inferenceIterationStart = 0;
		options.clearOnInferenceStart = true;
		options.logAlphaStart = 0.0f;
		options.neuralGamma = 0.99f;
		options.doInference = false;
		options.neuralSampleFraction = 0.9f;

		float errorPopLr = 1.0f;
		float goodLr = 0.00001f;
		options.policyLr = goodLr;
		options.qLr = options.policyLr;
		options.alphaLr = options.policyLr;

		options.autoencoderLr = 0.0001f;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.OptixVersion = OPTIX_VERSION;
		options.deviceID = 0;
		options.maxDcDepth = 2;
		options.maxTraversableGraphDepth = 2;
		options.LaunchParamName = "optixLaunchParams";
		options.enableCache = true;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.mdlSearchPath = { options.dataFolder, "./" , "E:/Dev/OptixTut/bin/Debug-windows-x86_64/OptixApp/mdl" };
		options.numTextureSpaces = 1; //should be set to 1 for performance reasons If you do not use the hair BSDF.
		options.numTextureResults = 16;
		options.enable_derivatives = false;
		options.mdlOptLevel = "2";
		options.initialized = true;
		options.mdlCallType = MDL_CUDA;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Gui Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.nodeWidth = 300;
		options.fontPath = options.dataFolder + "fonts/LucidaGrande.ttf";
	}
	Options* getOptions()
	{
		if(!options.initialized)
		{
			startOptions();
		}
		return &options;
	}

}

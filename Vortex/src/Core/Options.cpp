#include "Options.h"

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

		options.maxBounces = 10;
		options.maxSamples = 10000;
		options.accumulate = true;
		options.samplingTechnique = RendererDeviceSettings::SamplingTechnique::S_MIS;
		options.displayBuffer = RendererDeviceSettings::DisplayBuffer::FB_NOISY;
		options.maxClamp = 0.01f;
		options.minClamp = 1000.0f;

		options.noiseKernelSize = 3;
		options.adaptiveSampling = true;
		options.minAdaptiveSamples = 100;
		options.minPixelSamples = 1;
		options.maxPixelSamples = 200;
		options.albedoNormalNoiseInfluence = 1.0f;
		options.noiseCutOff = 0.00f;


		options.whitePoint = { 1.0f, 1.0f, 1.0f };
		options.colorBalance = { 1.0f, 1.0f, 1.0f };
		options.burnHighlights = 0.0f;
		options.crushBlacks = 1.0f;
		options.saturation = 1.0f;
		options.gamma = 2.2f; // Typical gamma value for sRGB color space

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
		options.directCallable = false;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Gui Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.nodeWidth = 300;
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
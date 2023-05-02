#pragma once
#include "optix.h"
#include <string>
#include <vector>
#include "Device/DevicePrograms/LaunchParams.h"

namespace vtx {
	struct Options {
		int         width         = 2100;
		int         height        = 900;
		std::string windowName    = "Vortex";
		std::string dataFolder	= "E:/Dev/VortexOptix/data/";
		std::string imGuiIniFile  = dataFolder + "ImGui.ini";
		std::string dllPath       = "./lib/";
		float       clearColor[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
		#ifdef NDEBUG
		bool isDebug = false;
		#else
		bool isDebug = true;
		#endif
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Rendering Settings /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////

		uint32_t                                  maxBounces        = 5;
		uint32_t                                  maxSamples        = 500;
		bool                                      accumulate        = true;
		RendererDeviceSettings::SamplingTechnique samplingTechnique = RendererDeviceSettings::SamplingTechnique::S_MIS;
		RendererDeviceSettings::DisplayBuffer     displayBuffer     = RendererDeviceSettings::DisplayBuffer::FB_NOISY;
		float                                     maxClamp          = 0.01f;
		float                                     minClamp          = 1000.0f;
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		int         OptixVersion             = OPTIX_VERSION;
		int         deviceID                 = 0;
		int         maxDcDepth               = 2;
		int         maxTraversableGraphDepth = 2;
		std::string LaunchParamName          = "optixLaunchParams";
		bool        enableCache              = true;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		std::vector<std::string> mdlSearchPath      = { dataFolder, "./" , "E:/Dev/OptixTut/bin/Debug-windows-x86_64/OptixApp/mdl"};
		int                      numTextureSpaces   = 1; //should be set to 1 for performance reasons If you do not use the hair BSDF.
		int                      numTextureResults  = 16;
		bool                     enable_derivatives = false;
		const char*              mdlOptLevel        = "2";
	};

	Options* getOptions();
}

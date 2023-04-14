#pragma once
#include "optix.h"
#include <string>
#include <vector>

namespace vtx {
	struct Options {
		int width = 2100;
		int height = 900;
		std::string WindowName = "Vortex";
		std::string ImGuiIniFile = "./data/ImGui.ini";
		std::string dll_path = "./lib/";
		float ClearColor[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
#ifdef NDEBUG
		bool isDebug = false;
#else
		bool isDebug = true;
#endif

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		int OptixVersion = OPTIX_VERSION;
		int deviceID = 0;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		std::vector<std::string> mdlSearchPath = { "./data/", "./" };
		int numTextureSpaces = 1;//should be set to 1 for performance reasons If you do not use the hair BSDF.
		int numTextureResults = 16;
		bool enable_derivatives = true;

	};

	Options* getOptions();
}


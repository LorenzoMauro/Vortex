#pragma once
#include "optix.h"
#include <string>

struct Options {
	int width = 2100;
	int height = 900;
	std::string WindowName = "Vortex";
	std::string ImGuiIniFile = "./data/ImGui.ini";
	float ClearColor[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
#ifdef NDEBUG
	bool isDebug = false;
#else
	bool isDebug = true;
#endif
	int OptixVersion = OPTIX_VERSION;
};

static Options options;

#pragma once


struct Options {
	int width = 2100;
	int height = 900;
	char* WindowName = "Vortex";
	char* ImGuiIniFile = "./data/ImGui.ini";
	float ClearColor[4] = { 0.45f, 0.55f, 0.60f, 1.00f };
};

static Options g_option;
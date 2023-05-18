#pragma once
#include <string>

#include "GLFW/glfw3.h"


namespace vtx {
	void Init_ImGui(GLFWwindow* window);

    void SetAppStyle();

	void shutDownImGui();
	
	void ImGuiRenderStart();

	void ImGuiDraw(GLFWwindow* window);

	std::string labelPrefix(const char* const label);
}
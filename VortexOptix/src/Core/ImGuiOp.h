#pragma once
#include "GLFW/glfw3.h"


namespace vtx {
	void Init_ImGui(GLFWwindow* window);

    void SetAppStyle();

	void End_ImGui();
	
	void ImGuiRenderStart();

	void ImGuiDraw(GLFWwindow* window);
}
#include "Input.h"
#include "Core/Window.h"
#include "imgui.h"
#include "Core/Application.h"

namespace vtx {


	static GLFWwindow* mainWindowHandle = nullptr;
	static GLFWwindow* windowHandle = nullptr;

	void Input::SetWindowHandle(GLFWwindow* _window) {
		windowHandle = _window;
	}

	//void Input::ResetWindowHandle() {
	//	mainWindowHandle = Window::Get()->GetWindowHandle();
	//	windowHandle = mainWindowHandle;
	//}
	bool Input::IsKeyDown(KeyCode keycode)
	{
		bool IsKeyDown = false;
		//ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
		//for (int n = 0; n < platform_io.Viewports.Size; n++)
		//{
		//	GLFWwindow* windowHandle = (GLFWwindow*)platform_io.Viewports[n]->PlatformHandle;

		//	bool imguiState = ImGui::IsKeyDown(MapInputToImGui(keycode));
		//	//GLFWwindow* windowHandle = Window::Get()->GetWindowHandle();
		//	int state = glfwGetKey(windowHandle, (int)keycode);

		//	//if (state == GLFW_PRESS || state == GLFW_REPEAT || imguiState) {
		//	if (state == GLFW_PRESS || state == GLFW_REPEAT) {
		//		IsKeyDown = true;
		//		break;
		//	}
		//}

		int state = glfwGetKey(windowHandle, (int)keycode);
		if (state == GLFW_PRESS || state == GLFW_REPEAT) {
			IsKeyDown = true;
		}

		return IsKeyDown;
	}

	bool Input::IsMouseButtonDown(MouseButton button)
	{
		bool IsMouseButtonDown = false;
		//ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
		//for (int n = 0; n < platform_io.Viewports.Size; n++)
		//{
		//	GLFWwindow* windowHandle = (GLFWwindow*)platform_io.Viewports[n]->PlatformHandle;
		//	bool imguiState = ImGui::IsKeyDown(MapInputToImGui(button));
		//	//GLFWwindow* windowHandle = Window::Get()->GetWindowHandle();
		//	int state = glfwGetMouseButton(windowHandle, (int)button);

		//	//if (state == GLFW_PRESS || imguiState) {
		//	if (state == GLFW_PRESS) {
		//		IsMouseButtonDown = true;
		//		break;
		//	}
		//}
		int state = glfwGetMouseButton(windowHandle, (int)button);
		if (state == GLFW_PRESS) {
			IsMouseButtonDown = true;
		}
		return IsMouseButtonDown;
	}

	math::vec2f Input::GetMousePosition()
	{
		//GLFWwindow* windowHandle = Window::Get()->GetWindowHandle();
		double x, y;
		glfwGetCursorPos(windowHandle, &x, &y);
		return { (float)x, (float)y };
	}

	void Input::SetCursorMode(CursorMode mode)
	{
		ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
		//for (int n = 0; n < platform_io.Viewports.Size; n++)
		//{
		//	GLFWwindow* wd = (GLFWwindow*)platform_io.Viewports[n]->PlatformHandle;
		//	//GLFWwindow* windowHandle = Window::Get()->GetWindowHandle();
		//	glfwSetInputMode(wd, GLFW_CURSOR, GLFW_CURSOR_NORMAL + (int)mode);
		//}
		glfwSetInputMode(windowHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL + (int)mode);
	}

	float Input::MouseWheel()
	{
		return ImGui::GetIO().MouseWheel;
	}


}
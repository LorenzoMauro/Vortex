#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"
#include "Options.h"
#include "Log.h"

namespace vtx {

	static void glfw_error_callback(int error, const char* description)
	{
		VTX_ERROR("Glfw Error {}: {}", error, description);
	}

	class Application {
	public:
		void Init() {
			InitWindow();
		};
		void InitWindow() {
			glfwSetErrorCallback(glfw_error_callback);
			if (!glfwInit())
			{
				VTX_ERROR("GLFW: Could not initalize GLFW!");
				return;
			}
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

			m_Window = glfwCreateWindow(g_option.width, g_option.height, g_option.WindowName, NULL, NULL);
			glfwMakeContextCurrent(m_Window);
			if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
			{
				VTX_ERROR("Failed to create GLFW window");
			}
			glfwSwapInterval(1); // Enable vsync
			// Initialize the window
			glfwSetWindowUserPointer(m_Window, this);
			//glfwSetFramebufferSizeCallback(m_Window, FramebufferResizeCallback);
		}
		void Run() {
			glfwPollEvents();
			glfwSwapBuffers(m_Window);
		};
	public:
		GLFWwindow* m_Window;
	};
}
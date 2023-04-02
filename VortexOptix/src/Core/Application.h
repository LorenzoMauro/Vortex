#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"
#include "Options.h"
#include "Log.h"
#include "ImGuiOp.h"
#include "Layers/GuiLayer.h"
#include "Layers/AppLayer.h"
#include "Layers/ViewportLayer.h"
#include "Scene/Scene.h"

namespace vtx {

	static void glfw_error_callback(int error, const char* description)
	{
		VTX_ERROR("Glfw Error {}: {}", error, description);
	}

	class Application {
	public:
		void Init() {
			InitWindow();
			Init_ImGui(m_Window);
			CreateLayer<AppLayer>();
			CreateLayer<ViewportLayer>(&m_renderer);
			m_scene.Start();
			m_renderer.ElaborateScene(m_scene.rootNode);
		};
		void ShutDown() {
			End_ImGui();
			glfwDestroyWindow(m_Window);
			glfwTerminate();
			VTX_INFO("GLFW DESTROYED");
		}
		void InitWindow() {
			glfwSetErrorCallback(glfw_error_callback);
			if (!glfwInit())
			{
				VTX_ERROR("GLFW: Could not initalize GLFW!");
				return;
			}
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

			m_Window = glfwCreateWindow(options.width, options.height, options.WindowName.c_str(), NULL, NULL);
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
			ImGuiRenderStart();
			//////////////////////////
			int layerCount = m_LayerStack.size();
			for (int i = 0; i < layerCount; i++) {
				auto& layer = m_LayerStack[i];
				layer->OnUIRender();
				layerCount = m_LayerStack.size();
			}
			ImGuiDraw(m_Window);
			glfwSwapBuffers(m_Window);
		};

		template<typename T, typename... Args>
		void CreateLayer(Args&&... args) {
			static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not subclass of Layer!");
			m_LayerStack.emplace_back(std::make_shared<T>(std::forward<Args>(args)...))->OnAttach();
		}

		void PushLayer(const std::shared_ptr<Layer>& layer) {
			m_LayerStack.emplace_back(layer);
			layer->OnAttach();
		}
		
	public:
		GLFWwindow* m_Window;
		std::vector<std::shared_ptr<Layer>> 	    m_LayerStack;
		Renderer									m_renderer;
		Scene										m_scene;

	};
}
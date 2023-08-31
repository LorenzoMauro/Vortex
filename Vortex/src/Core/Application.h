#pragma once
#include <GLFW/glfw3.h>
#include "Options.h"
#include "Log.h"
#include "Layers/GuiLayer.h"
#include "Layers/AppLayer.h"
#include "Scene/Scene.h"

namespace vtx {

	static void glfwErrorCallback(int error, const char* description)
	{
		VTX_ERROR("Glfw Error {}: {}", error, description);
	}

	class Application {
	public:
		void init();

		void shutDown();

		void initWindow();
		
		void run();

		template<typename T, typename... Args>
		void createLayer(Args&&... args) {
			static_assert(std::is_base_of_v<Layer, T>, "Pushed type is not subclass of Layer!");
			layerStack.emplace_back(std::make_shared<T>(std::forward<Args>(args)...))->OnAttach();
		}

		void pushLayer(const std::shared_ptr<Layer>& layer) {
			layerStack.emplace_back(layer);
			layer->OnAttach();
		}
		
	public:
		GLFWwindow*                         window;
		std::vector<std::shared_ptr<Layer>> layerStack;
		float                               timeStep      = 0.0f;
		float                               frameTime     = 0.0f;
		float                               lastFrameTime = 0.0f;
	};
}
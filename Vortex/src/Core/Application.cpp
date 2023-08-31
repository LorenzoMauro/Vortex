#include "Application.h"
#include "ShutDownOperations.h"
#include "glad/glad.h"
#include "ImGuiOp.h"
#include "Device/OptixWrapper.h"
#include "Device/PipelineConfiguration.h"
#include "Layers/MaterialEditorLayer.h"
#include "Layers/ViewportLayer.h"
#include "Layers/ExperimentsLayer.h"

namespace vtx
{
	bool isWindowMinimized(GLFWwindow* window) {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		return (width == 0 || height == 0);
	}

	void Application::init() {
		initWindow();
		Input::SetWindowHandle(window);
		Init_ImGui(window);

		optix::init();
		pipelineConfiguration();
		mdl::init();

		createLayer<AppLayer>();

		ops::startUpOperations();
		graph::Scene::getScene()->renderer->setWindow(window);
		createLayer<MaterialEditorLayer>();
		createLayer<ViewportLayer>();
		createLayer<ExperimentsLayer>();
	}

	void Application::shutDown() {
		//Last run!
		shutDownOperations();
		glfwDestroyWindow(window);
		glfwTerminate();
		VTX_INFO("GLFW DESTROYED");
	}

	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		// Here you can set your new window size. You could, for example, update
		// your options object. Without knowing the structure of your options
		// object or your application, here's a hypothetical example:

		getOptions()->width = width;
		getOptions()->height = height;

		// You may also need to update your OpenGL viewport size here:
		glViewport(0, 0, width, height);
	}

	void Application::initWindow() {
		glfwSetErrorCallback(glfwErrorCallback);
		if (!glfwInit())
		{
			VTX_ERROR("GLFW: Could not initalize GLFW!");
			return;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);  // Set the window to start maximized

		// Get the primary monitor
		GLFWmonitor* monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode = glfwGetVideoMode(monitor);

		getOptions()->width = mode->width;
		getOptions()->height = mode->height;

		window = glfwCreateWindow(getOptions()->width, getOptions()->height, getOptions()->windowName.c_str(), nullptr, nullptr);
		glfwMakeContextCurrent(window);
		if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
		{
			VTX_ERROR("Failed to create GLFW window");
		}
		glfwSwapInterval(0); // Enable vsync
		// Initialize the window
		glfwSetWindowUserPointer(window, this);
		glfwSetScrollCallback(window, 
			[](GLFWwindow* window, double xOffset, double yOffset)
			{
				ImGuiIO& io = ImGui::GetIO();
				io.MouseWheelH += (float)xOffset;
				io.MouseWheel += (float)yOffset;
			}
		);
		//glfwSetFramebufferSizeCallback(m_Window, FramebufferResizeCallback);
	}

	void Application::run() {
		glfwPollEvents();
		//////////////////////////
		for (auto& layer : layerStack)
			layer->OnUpdate(timeStep);

		int layerCount = layerStack.size();
		if(!isWindowMinimized(window))
		{
			ImGuiRenderStart();
			for (int i = 0; i < layerCount; i++) {
				auto& layer = layerStack[i];
				layer->OnUIRender();
				layerCount = layerStack.size();
			}
			ImGuiDraw(window);
			glfwSwapBuffers(window);
		}
		

		auto time = static_cast<float>(glfwGetTime());
		frameTime = time - lastFrameTime;
		timeStep = std::min(frameTime, 0.0333f);
		lastFrameTime = time;

	}
}


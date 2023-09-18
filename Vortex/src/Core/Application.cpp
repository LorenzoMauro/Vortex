#include "Application.h"
#include "ShutDownOperations.h"
#include "glad/glad.h"
#include "ImGuiOp.h"
#include "LoadingSaving.h"
#include "Device/OptixWrapper.h"
#include "Device/PipelineConfiguration.h"
#include "Gui/AppWindow.h"
#include "Gui/ShaderGraphWindow.h"
#include "Gui/ViewportWindow.h"
#include "Gui/ExperimentsWindow.h"
#include "Gui/SceneHierarchyWindow.h"
#include "Gui/GraphWindow.h"
#include "Gui/PropertiesWindow.h"
#include "Serialization/Serializer.h"

namespace vtx
{
	static Application* appInstance = nullptr;

	bool isWindowMinimized(GLFWwindow* window) {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		return (width == 0 || height == 0);
	}

	void Application::init() {
		appInstance = this;
		initWindow();
		Input::SetWindowHandle(glfwWindow);
		Init_ImGui(glfwWindow);

		optix::init();
		pipelineConfiguration();
		mdl::init();
		windowManager = std::make_shared<WindowManager>();
		windowManager->createWindow<AppWindow>();

		//ops::startUpOperations();
		const std::shared_ptr<graph::Scene> scene = graph::Scene::getScene();
		scene->renderer = ops::createNode<graph::Renderer>();
		graph::Scene::getScene()->renderer->setWindow(glfwWindow);
		windowManager->createWindow<SceneHierarchyWindow>();
		windowManager->createWindow<ViewportWindow>();
		windowManager->createWindow<PropertiesWindow>();
		windowManager->createWindow<GraphWindow>();
		windowManager->createWindow<ShaderGraphWindow>();
		windowManager->createWindow<ExperimentsWindow>();
	}

	void Application::shutDown() {
		//Last run!
		shutDownOperations();
		glfwDestroyWindow(glfwWindow);
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

		glfwWindow = glfwCreateWindow(getOptions()->width, getOptions()->height, getOptions()->windowName.c_str(), nullptr, nullptr);
		glfwMakeContextCurrent(glfwWindow);
		if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
		{
			VTX_ERROR("Failed to create GLFW window");
		}
		glfwSwapInterval(0); // Enable vsync
		// Initialize the window
		glfwSetWindowUserPointer(glfwWindow, this);
		glfwSetScrollCallback(glfwWindow, 
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
		windowManager->updateWindows(timeStep);

		if(!isWindowMinimized(glfwWindow))
		{
			ImGuiRenderStart();
			if(iteration >3) // it seems the minimum frame to render the initial gui
			{
				LoadingSaving::get().performLoadSave();
			}
			windowManager->renderWindows();
			ImGuiDraw(glfwWindow);
			glfwSwapBuffers(glfwWindow);
		}
		windowManager->removeClosedWindows();
		const auto time = static_cast<float>(glfwGetTime());
		frameTime = time - lastFrameTime;
		timeStep = std::min(frameTime, 0.0333f);
		lastFrameTime = time;
		iteration += 1;
	}
	void Application::setStartUpFile(const std::string& filePath)
	{
		LoadingSaving::get().setManualLoad(filePath);
	}
	Application* Application::get()
	{
		return appInstance;
	}
}


#include "Application.h"

int main() {
	vtx::Log::Init();
	try {
		vtx::Application app;
		app.Init();
		while (!glfwWindowShouldClose(app.m_Window))
		{
			app.Run();
		}
		app.ShutDown();
	}
	catch (const std::exception& e) {
		VTX_ERROR("Error in main");
		return EXIT_FAILURE;
	}
}
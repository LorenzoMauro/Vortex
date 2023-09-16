#include "Application.h"

int main() {
	vtx::Log::Init();
	try {
		vtx::Application app = vtx::Application();
		app.init();
		while (!glfwWindowShouldClose(app.glfwWindow))
		{
			app.run();
		}
		app.shutDown();
	}
	catch (const std::exception& e) {
		VTX_ERROR("Error in main: {}", e.what());
		return EXIT_FAILURE;
	}
}
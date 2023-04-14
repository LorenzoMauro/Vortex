#include "Application.h"

int main() {
	vtx::Log::Init();
	try {
		vtx::Application app;
		app.init();
		while (!glfwWindowShouldClose(app.window))
		{
			app.run();
		}
		app.shutDown();
	}
	catch (const std::exception& e) {
		VTX_ERROR("Error in main");
		return EXIT_FAILURE;
	}
}
#include "Application.h"

int main(int argc, char** argv) {
	vtx::Log::Init();

	

	try {
		vtx::Application app = vtx::Application();
		app.init();
		if (argc > 1) { // If there's more than one argument
			std::string arg = argv[1];
			app.setStartUpFile(arg);
		}
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
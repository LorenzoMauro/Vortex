#include "Application.h"

int main(const int argc, char** argv) {
	vtx::Log::Init();
	vtx::Application app = vtx::Application();
	app.init();

	if (argc > 1) {
		std::string arg = argv[1];
		app.setStartUpFile(arg);
	}

	try {
		while (!glfwWindowShouldClose(app.glfwWindow)) {
			app.run();
		}
	}
	catch (const std::exception& e) {
		VTX_ERROR("Error: {}", e.what());
	}
	app.shutDown();

	return 0;
}
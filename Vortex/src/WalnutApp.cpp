#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Utils.h"
#include "EngineUI.h"
#include "AppStyle.h"

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "Vortex";

	Walnut::Application* app = new Walnut::Application(spec);

	AppUiStyle();
	
	app->PushLayer<EngineUI>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}
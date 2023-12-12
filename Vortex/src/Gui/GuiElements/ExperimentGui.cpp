#include "Gui/GuiProvider.h"
#include "NeuralNetworks/Experiment.h"
#include "Core/CustomImGui/CustomImGui.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(Experiment& experiment)
	{
		ImGui::PushID(experiment.name.c_str());
		vtxImGui::halfSpaceWidget("Name: ", vtxImGui::booleanText, "%s", experiment.name.c_str());
		if(!experiment.mape.empty())
		{
			vtxImGui::halfSpaceWidget("MAPE: ", vtxImGui::booleanText, "%f", experiment.mape.back());
			vtxImGui::halfSpaceWidget("Average MAPE: ", vtxImGui::booleanText, "%f", experiment.averageMape);
		}

		if (ImGui::CollapsingHeader("Renderer Settings"))
		{
			RendererSettings dummyRS = experiment.rendererSettings;
			drawEditGui(dummyRS);
		}

		network::config::NetworkSettings dummyNS = experiment.networkSettings;
		drawEditGui(dummyNS);

		drawDisplayGui(experiment.statistics);

		vtxImGui::halfSpaceCheckbox("Store", &(experiment.storeExperiment));
		vtxImGui::halfSpaceCheckbox("Display", &(experiment.displayExperiment));

		ImGui::PopID();
		return false;
	}
}

#include "Gui/GuiProvider.h"

#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Camera.h"

namespace vtx::gui {

	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Camera>& camera)
	{
		const float availableWidth = ImGui::GetContentRegionAvail().x;

		bool updated = false;
		ImGui::PushID(camera->getID());
		if (ImGui::CollapsingHeader("Camera"))
		{
			ImGui::Indent();
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(camera->getID()).c_str());
			ImGui::Separator();

			const math::vec3f oldPosition = camera->position;

			if(vtxImGui::halfSpaceWidget("Position", vtxImGui::vectorGui, (float*)&camera->position, false))
			{
				const math::vec3f delta = camera->position - oldPosition;
				camera->transform->translate(delta);
				camera->isUpdated = true;
			}
			vtxImGui::halfSpaceWidget("vertical", vtxImGui::vectorGui, (float*)&camera->vertical, true);
			vtxImGui::halfSpaceWidget("horizontal", vtxImGui::vectorGui, (float*)&camera->horizontal, true);

			if (vtxImGui::halfSpaceWidget("Fov", ImGui::DragFloat, (hiddenLabel + "_Fov").c_str(), &camera->fovY, 0.01, 0, 180, "%.3f", 0))
			{
				camera->isUpdated = true;
			}
			vtxImGui::halfSpaceWidget("Lock Camera", ImGui::Checkbox,(hiddenLabel + std::to_string(camera->getID()) + "Lock").c_str(), &camera->lockCamera);




			ImGui::PopItemWidth();
			ImGui::Unindent();
		}
		ImGui::PopID();
		if (updated)
		{
			
		}

		return updated;
	}
}

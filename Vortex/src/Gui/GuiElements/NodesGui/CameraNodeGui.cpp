#include "CameraNodeGui.h"

#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Camera.h"

namespace vtx::gui {

	bool vtx::gui::cameraNodeGui(const std::shared_ptr<graph::Camera>& cameraNode)
	{
		const float availableWidth = ImGui::GetContentRegionAvail().x;

		bool updated = false;
		ImGui::PushID(cameraNode->getID());
		if (ImGui::CollapsingHeader("Camera"))
		{
			ImGui::Indent();
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(cameraNode->getID()).c_str());
			ImGui::Separator();

			const math::vec3f oldPosition = cameraNode->position;

			if(vtxImGui::halfSpaceWidget("Position", vtxImGui::vectorGui, (float*)&cameraNode->position, false))
			{
				const math::vec3f delta = cameraNode->position - oldPosition;
				cameraNode->transform->translate(delta);
				cameraNode->isUpdated = true;
			}
			vtxImGui::halfSpaceWidget("vertical", vtxImGui::vectorGui, (float*)&cameraNode->vertical, true);
			vtxImGui::halfSpaceWidget("horizontal", vtxImGui::vectorGui, (float*)&cameraNode->horizontal, true);

			if (vtxImGui::halfSpaceWidget("Fov", ImGui::DragFloat, (hiddenLabel + "_Fov").c_str(), &cameraNode->fovY, 0.01, 0, 180, "%.3f", 0))
			{
				cameraNode->isUpdated = true;
			}
			vtxImGui::halfSpaceWidget("Lock Camera", ImGui::Checkbox,(hiddenLabel + std::to_string(cameraNode->getID()) + "Lock").c_str(), &cameraNode->lockCamera);




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

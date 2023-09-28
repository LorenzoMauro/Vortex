#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Transform.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Transform>& transformNode)
	{
		bool updated = false;
		ImGui::PushID(transformNode->getUID());
		if(ImGui::CollapsingHeader("Transform"))
		{
			ImGui::Indent();
			const float availableWidth = ImGui::GetContentRegionAvail().x;
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(transformNode->getUID()).c_str());
			ImGui::Separator();
			updated |= vtxImGui::halfSpaceWidget("Scale", vtxImGui::vectorGui, (float*)&transformNode->scaleVector, false);
			math::vec3f degreeRotation = transformNode->eulerAngles * 180.0f / M_PI;
			if(vtxImGui::halfSpaceWidget("Rotation", vtxImGui::vectorGui, (float*)&degreeRotation, false))
			{
				transformNode->eulerAngles = degreeRotation * M_PI / 180.0f;
				updated = true;
			}
			updated |= vtxImGui::halfSpaceWidget("Translation", vtxImGui::vectorGui, (float*)&transformNode->translation, false);
			ImGui::Separator();
			vtxImGui::halfSpaceWidget("Affine", vtxImGui::affineGui, transformNode->affineTransform);
			ImGui::PopItemWidth();
			ImGui::Unindent();
		}
		ImGui::PopID();
		if(updated)
		{
			transformNode->updateFromVectors();
		}

		return updated;
	}
}

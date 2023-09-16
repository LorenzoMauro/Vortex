#include "TransformNodeGui.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Transform.h"

namespace vtx::gui
{
	bool transformNodeGui(const std::shared_ptr<graph::Transform>& transformNode)
	{
		const float availableWidth = ImGui::GetContentRegionAvail().x;

		bool updated = false;
		ImGui::PushID(transformNode->getID());
		if(ImGui::CollapsingHeader("Transform"))
		{
			ImGui::Indent();
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(transformNode->getID()).c_str());
			ImGui::Separator();
			updated |= vtxImGui::halfSpaceWidget("Scale", vtxImGui::vectorGui, (float*)&transformNode->scaleVector, false);
			updated |= vtxImGui::halfSpaceWidget("Rotation", vtxImGui::vectorGui, (float*)&transformNode->eulerAngles, false);
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
			transformNode->isUpdated = updated;
		}

		return updated;
	}
}

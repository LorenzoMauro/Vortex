#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Scene/Graph.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Gui/GuiVisitor.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Mesh>& mesh)
	{
		bool changed = false;
		ImGui::PushID(mesh->getUID());
		if (ImGui::CollapsingHeader(mesh->name.c_str()))
		{
			ImGui::Indent();
			const float availableWidth = ImGui::GetContentRegionAvail().x;
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(mesh->getUID()).c_str());
			vtxImGui::halfSpaceWidget("Number Of Vertices:", ImGui::Text, std::to_string(mesh->vertices.size()).c_str());
			vtxImGui::halfSpaceWidget("Number Of Faces:", ImGui::Text, std::to_string((int)(mesh->indices.size()/3)).c_str());
			ImGui::Unindent();
		}
		ImGui::PopID();
		return changed;
	}
}


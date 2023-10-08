#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Scene/Graph.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Gui/GuiVisitor.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Group>& group)
	{
		bool changed = false;
		ImGui::PushID(group->getUID());
		if (ImGui::CollapsingHeader(group->name.c_str()))
		{
			ImGui::Indent();
			const float availableWidth = ImGui::GetContentRegionAvail().x;
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(group->getUID()).c_str());
			ImGui::Separator();
			changed |= drawEditGui(group->transform);
			if(const auto& children = group->getChildren(); children.size()>0)
			{
				if (ImGui::CollapsingHeader("Group Children"))
				{
					GuiVisitor visitor;
					ImGui::Indent();
					for (const auto& child : group->getChildren())
					{
						if (child->getType() != graph::NT_TRANSFORM)
						{
							child->accept(visitor);
						}
					}
					ImGui::Unindent();
				}
			}
			ImGui::Unindent();
		}
		ImGui::PopID();
		return changed;
	}
}

#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Scene/Graph.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Gui/GuiVisitor.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Instance>& instance)
	{
		bool changed = false;
		ImGui::PushID(instance->getUID());
		if (ImGui::CollapsingHeader(instance->name.c_str()))
		{
			ImGui::Indent();
			const float availableWidth = ImGui::GetContentRegionAvail().x;
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(instance->getUID()).c_str());
			ImGui::Separator();
			changed |= drawEditGui(instance->transform);
			GuiVisitor visitor;
			if(const auto& child = instance->getChild(); child)
			{
				child->accept(visitor);
			}
			for (const auto& materialSlot : instance->getMaterialSlots())
			{
				if (ImGui::CollapsingHeader(("Material Slot: " + std::to_string(materialSlot.slotIndex)).c_str()))
				{
					ImGui::Indent();
					changed |= drawEditGui(materialSlot.material);
					ImGui::Unindent();
				}
			}
			ImGui::Unindent();
		}
		ImGui::PopID();
		return changed;
	}
}


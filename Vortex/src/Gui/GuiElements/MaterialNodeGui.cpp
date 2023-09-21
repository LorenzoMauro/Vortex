#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Material.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Material>& material)
	{
        bool changed = false;

        if (ImGui::CollapsingHeader("Material"))
        {
            ImGui::Indent();
            const float availableWidth = ImGui::GetContentRegionAvail().x;
            ImGui::PushItemWidth(availableWidth);
            vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(material->getID()).c_str());
            ImGui::Separator();
            changed |= drawEditGui(material->materialGraph);
        }

        if (changed)
        {
            material->isUpdated = true;
            ops::restartRender();
        }

        return changed;
	}
}

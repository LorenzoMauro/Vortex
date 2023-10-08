#include "Gui/GuiProvider.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Material.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Material>& material)
	{
        bool changed = false;
        ImGui::PushID(material->getUID());
        std::string name = material->name;
        if(material->materialGraph)
        {
	        name = material->materialGraph->name;
        }
        if (ImGui::CollapsingHeader(name.c_str()))
        {
            ImGui::Indent();
            const float availableWidth = ImGui::GetContentRegionAvail().x;
            ImGui::PushItemWidth(availableWidth);
            vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(material->getUID()).c_str());
            ImGui::Separator();
            changed |= drawEditGui(material->materialGraph);
        }

        if (changed)
        {
            material->state.updateOnDevice = true;
        }
        ImGui::PopID();
        return changed;
	}
}

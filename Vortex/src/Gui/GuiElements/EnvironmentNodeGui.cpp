#pragma once
#include <memory>
#include "Core/CustomImGui/CustomImGui.h"
#include "Gui/GuiProvider.h"
#include "Scene/Nodes/EnvironmentLight.h"

bool vtx::gui::GuiProvider::drawEditGui(const std::shared_ptr<graph::EnvironmentLight>& envLight)
{
	bool changed = false;
	ImGui::PushID(envLight->getUID());
	if (ImGui::CollapsingHeader(envLight->name.c_str()))
	{
		ImGui::Indent();
		const float availableWidth = ImGui::GetContentRegionAvail().x;
		ImGui::PushItemWidth(availableWidth);
		vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(envLight->getUID()).c_str());
		ImGui::Separator();
		envLight->state.updateOnDevice = vtxImGui::halfSpaceDragFloat("Scale Luminosity", &envLight->scaleLuminosity, 0.1f, 0.0f, 100.0f);
		envLight->state.updateOnDevice |= drawEditGui(envLight->transform);
		ImGui::Unindent();
	}
	ImGui::PopID();
	return changed;
}

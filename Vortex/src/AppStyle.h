#pragma once
#include "Walnut/Application.h"

void AppUiStyle() {
	ImVec4* colors = ImGui::GetStyle().Colors;
	colors[ImGuiCol_Text] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_Border] = ImVec4(0.31f, 0.30f, 0.30f, 0.50f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.29f, 0.29f, 0.29f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.51f, 0.51f, 0.51f, 0.54f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.87f, 0.87f, 0.87f, 0.54f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.79f, 0.79f, 0.79f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.88f, 0.50f, 0.24f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.98f, 0.55f, 0.26f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.29f, 0.29f, 0.29f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.68f, 0.31f, 0.07f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.98f, 0.43f, 0.06f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.26f, 0.26f, 0.26f, 0.31f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.88f, 0.00f, 1.00f, 0.20f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.97f, 0.00f, 1.00f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.97f, 0.00f, 1.00f, 0.95f);
	colors[ImGuiCol_Tab] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
	colors[ImGuiCol_TabActive] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.31f, 0.31f, 0.31f, 0.70f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.27f, 0.59f, 1.00f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

	ImGui::GetStyle().Alpha = 1.0f;
	
	ImGui::GetStyle().WindowPadding = ImVec2(20.0f, 4.0f);
	ImGui::GetStyle().FramePadding = ImVec2(4.0f, 2.0f);
	ImGui::GetStyle().CellPadding = ImVec2(4.0f, 2.0f);
	ImGui::GetStyle().ItemSpacing = ImVec2(11.0f, 4.0f);
	ImGui::GetStyle().ItemInnerSpacing = ImVec2(4.0f, 4.0f);
	ImGui::GetStyle().IndentSpacing = 21.0f;
	ImGui::GetStyle().ScrollbarSize = 16.0f;
	ImGui::GetStyle().GrabMinSize = 4.0f;

	ImGui::GetStyle().WindowBorderSize = 1;
	ImGui::GetStyle().ChildBorderSize = 0;
	ImGui::GetStyle().PopupBorderSize = 1;
	ImGui::GetStyle().FrameBorderSize = 1;
	ImGui::GetStyle().TabBorderSize = 0;

	ImGui::GetStyle().WindowRounding = 5;
	ImGui::GetStyle().ChildRounding = 5;
	ImGui::GetStyle().FrameRounding = 5;
	ImGui::GetStyle().PopupRounding = 5;
	ImGui::GetStyle().ScrollbarRounding = 5;
	ImGui::GetStyle().GrabRounding = 5;
	ImGui::GetStyle().TabRounding = 5;



	//ImGui::GetStyle().WindowTitleAlign = ;
	ImGui::GetStyle().WindowMenuButtonPosition = 1;
	ImGui::GetStyle().ColorButtonPosition = 0;
	//ImGui::GetStyle().ButtonTextAlign = ;
}

#include "CustomImGui.h"

#include <iostream>
#include <ostream>
#include <vector>

#include "imgui_internal.h"
#include "Core/Log.h"

const char* vtx::vtxImGui::empty_format_callback() {
    return "";
}

bool vtx::vtxImGui::SliderFloat(const char* label, float* value, float min, float max, const char* format) {
    // Get the current style
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32(255, 255, 255, 0)); // Hide the slider grab

    // Set the format callback to our custom callback
    bool value_changed = ImGui::SliderFloat(("##" + std::string(label)).c_str(), value, min, max, empty_format_callback());

    ImGui::PopStyleColor(); // Restore the slider grab color

    // Get the slider's rectangle
    ImVec2 pos = ImGui::GetItemRectMin();
    ImVec2 size = ImGui::GetItemRectSize();

    // Draw a rectangle on the left part of the slider, filled with the desired color
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float x = pos.x + size.x * (*value - min) / (max - min);
    ImU32 color = ImGui::GetColorU32(ImGuiCol_SliderGrabActive); // Change to the ImGui color variable you want
    draw_list->AddRectFilled(pos, ImVec2(x, pos.y + size.y), color, style.FrameRounding); // Add rounding to match the ImGui style

    // Display the text and the value over the slider
    ImGui::SetCursorScreenPos({ pos.x + style.ItemInnerSpacing.x, pos.y }); // Add ImGui style padding
    ImGui::TextUnformatted(label);

    ImGui::SetCursorScreenPos({ pos.x + size.x - ImGui::CalcTextSize(std::to_string(*value).c_str()).x - 5, pos.y });
    ImGui::Text(format, *value);

    return value_changed;
}

bool vtx::vtxImGui::ClippedText(const char* label)
{
	float totalItemWidth = ImGui::CalcItemWidth();

    std::string labelStr = label;
    // Adjust the label to fit the halfItemWidth
    float       labelWidth;
    float       spaceWidth    = ImGui::CalcTextSize(" ").x;
	const float ellipsisWidth = ImGui::CalcTextSize("...").x;

    std::vector<float> positionalWidth;
    float width = 0;
    for(const auto character: labelStr)
    {
        width += ImGui::CalcTextSize(std::string(1, character).c_str()).x;
        positionalWidth.push_back(width);
	}

    std::string newLabel;
    if(positionalWidth.back() > totalItemWidth)
    {
        size_t currentPosition = positionalWidth.size() - 1;
        while (positionalWidth[currentPosition] + ellipsisWidth > totalItemWidth)
        {
            currentPosition--;
		}

        newLabel = labelStr.substr(0, currentPosition) + "...";
	}

    if(!newLabel.empty())
    {
    	labelStr = newLabel;
		labelWidth = ImGui::CalcTextSize(labelStr.c_str()).x;
    }

    while (labelWidth < totalItemWidth && !labelStr.empty()) {
        // If the text is too short, append spaces to it
        labelStr += ' ';
        labelWidth = ImGui::CalcTextSize(labelStr.c_str()).x;
    }

    // Draw the label
    ImGui::TextUnformatted(labelStr.c_str());
}

bool vtx::vtxImGui::colorPicker(const char* label, float* col)
{
    std::string labelStr = label;
    bool valueChanged = false;
    float totalItemWidth = ImGui::CalcItemWidth();

    ImVec4 color(col[0], col[1], col[2], 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, color); // Set the color for the button
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color); // Set the hover color for the button
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, color); // Set the active color for the button
    bool clicked = ImGui::Button((labelStr + "_ColorButton").c_str(), ImVec2(totalItemWidth, ImGui::GetFrameHeight())); // Draw the button in the second column
    ImGui::PopStyleColor(3); // Restore the color

    // Create a color picker when the button is clicked
    if (clicked) {
        ImGui::OpenPopup((labelStr + "_ColorPicker").c_str());
    }

    if (ImGui::BeginPopup((labelStr + "_ColorPicker").c_str())) {
        valueChanged = ImGui::ColorPicker3((labelStr + "_ColorPicker").c_str(), col);
        ImGui::EndPopup();
    }

    return valueChanged;
}
bool vtx::vtxImGui::ColorEdit3NoInputs(const char* label, float* col) {

    bool valueChanged = false;
    ImGuiStyle& style = ImGui::GetStyle();

    // Start a new ImGui ID Scope
    ImGui::PushID(label);

    // Calculate the size of the text and button
    float totalItemWidth = ImGui::CalcItemWidth();
    float halfItemWidth = totalItemWidth * 0.5f; // Subtract the ItemInnerSpacing

    ImGui::PushItemWidth(halfItemWidth - style.ItemSpacing.x); // Set the width of the next widget to 200

    ClippedText(label); // Draw the label in the first column

    ImGui::PopItemWidth(); // Restore the width

    // After rendering the clipped text
    ImVec2 textSize = ImGui::GetItemRectSize();
    float remainingWidth = totalItemWidth - textSize.x - style.ItemSpacing.x;

    ImGui::SameLine(); // Draw the button on the same line as the label
    ImGui::PushItemWidth(remainingWidth); // Set the width of the next widget to 200
    valueChanged = colorPicker(("##" + std::string(label)).c_str(), col);
    ImGui::PopItemWidth(); // Restore the width

    // End ImGui ID Scope
    ImGui::PopID();


    return valueChanged;
}

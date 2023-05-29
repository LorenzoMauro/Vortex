#pragma once
#include <string>

#include "imgui.h"


namespace vtx::vtxImGui
{
    const char* empty_format_callback();

    bool SliderFloat(const char* label, float* value, float min, float max, const char* format="%.3f");

    bool ClippedText(const char* label);

    bool colorPicker(const char* label, float* col);

    bool ColorEdit3NoInputs(const char* label, float* col);

    template<typename Func, typename ...Args>
    bool HalfSpaceWidget(const char* label, Func&& widgetFunc, Args&&... args) {
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
        valueChanged = widgetFunc(std::forward<Args>(args)...);
        ImGui::PopItemWidth(); // Restore the width

        // End ImGui ID Scope
        ImGui::PopID();

        return valueChanged;
    }

}

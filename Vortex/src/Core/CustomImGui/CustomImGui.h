#pragma once
#include <string>
#include "imgui.h"
#include <type_traits>
#include "Core/Math.h"

static const std::string hiddenLabel = "##hidden";

typedef bool (*ComboFuncType)(const char*, int*, const char* const [], int, int);

namespace vtx::vtxImGui
{
    const char* emptyFormatCallback();

    bool sliderFloat(const char* label, float* value, const float min, const float max, const char* format="%.3f");

    bool clippedText(const char* label);

    bool colorPicker(const char* label, float* col);

    bool colorEdit3NoInputs(const char* label, float* col);

    bool booleanText(const char* fmt, ...);

    void affineGui(math::affine3f& affine);

    bool vectorGui(float* data, bool disableEdit = false);

    void pushHalfSpaceWidgetFraction(const float fraction);
    void popHalfSpaceWidgetFraction();
    float getHalfWidgetFraction();

	template<typename Func, typename ...Args>
    bool halfSpaceWidget(const char* label, Func&& widgetFunc, Args&&... args) {
        bool              valueChanged = false;
		const ImGuiStyle& style        = ImGui::GetStyle();

        // Start a new ImGui ID Scope
        ImGui::PushID(label);

        const float labelFraction = getHalfWidgetFraction();

        // Calculate the size of the text and button
		const float totalItemWidth = ImGui::CalcItemWidth();

        const float labelWidth = totalItemWidth * labelFraction;
        const float widgetWidth = totalItemWidth * (1.0f - labelFraction);

        ImGui::PushItemWidth(labelWidth - style.ItemSpacing.x); // Set the width of the next widget to 200

		const float cursorPosX = ImGui::GetCursorPosX();

        clippedText(label); // Draw the label in the first column

        ImGui::PopItemWidth(); // Restore the width

        ImGui::SameLine(); // Draw the button on the same line as the label
        ImGui::SetCursorPosX(cursorPosX + labelWidth); // Position in window coordinates
        ImGui::PushItemWidth(widgetWidth - style.ItemSpacing.x); // Set the width of the next widget to 200
        if constexpr (std::is_same_v<bool, std::invoke_result_t<Func, Args...>>) {
            valueChanged = widgetFunc(std::forward<Args>(args)...);
        }
        else {
            widgetFunc(std::forward<Args>(args)...);
        }
        ImGui::PopItemWidth(); // Restore the width

        // End ImGui ID Scope
        ImGui::PopID();

        return valueChanged;
    }

    void DrawRowsBackground(int row_count, float line_height, float x1, float x2, float y_offset, ImU32 col_even, ImU32 col_odd);

    void childWindowResizerButton(float& percentage, const float& resizerSize, bool isHorizontalSplit);
}

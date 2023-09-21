#include "CustomImGui.h"

#include <stack>
#include <vector>
#include "Core/Log.h"

namespace vtx::vtxImGui
{
    const char* emptyFormatCallback() {
        return "";
    }

    bool sliderFloat(const char* label, float* value, const float min, const float max, const char* format) {
        // Get the current style
        ImGuiStyle& style = ImGui::GetStyle();
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32(255, 255, 255, 0)); // Hide the slider grab

        // Set the format callback to our custom callback
        bool value_changed = ImGui::SliderFloat(("##" + std::string(label)).c_str(), value, min, max, emptyFormatCallback());

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

    void fakeButton(const char* label, const ImVec2& size_arg = ImVec2(0,0))
    {
        ImVec4 regularColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

        // Set the hovered and active colors to the regular color
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, regularColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, regularColor);

        ImGui::Button(label, size_arg);
        
        ImGui::PopStyleColor(2);
    }

    bool clippedText(const char* label)
    {
        float totalItemWidth = ImGui::CalcItemWidth();

        std::string labelStr = label;
        // Adjust the label to fit the halfItemWidth
        float       labelWidth = ImGui::CalcTextSize(labelStr.c_str()).x;
        float       spaceWidth = ImGui::CalcTextSize(" ").x;
        const float ellipsisWidth = ImGui::CalcTextSize("...").x;

        std::vector<float> positionalWidth = {};
        float width = 0;
        for (const auto character : labelStr)
        {
            width += ImGui::CalcTextSize(std::string(1, character).c_str()).x;
            positionalWidth.push_back(width);
        }

        std::string newLabel;
        if (!positionalWidth.empty() && positionalWidth.back() > totalItemWidth)
        {
            size_t currentPosition = positionalWidth.size() - 1;
            while (currentPosition > 0 && positionalWidth[currentPosition] + ellipsisWidth > totalItemWidth)
            {
                currentPosition--;
            }

            newLabel = labelStr.substr(0, currentPosition) + "...";
        }

        if (!newLabel.empty())
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

        return false;
    }

    bool colorPicker(const char* label, float* col)
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

	bool colorEdit3NoInputs(const char* label, float* col) {

        bool valueChanged = false;
        ImGuiStyle& style = ImGui::GetStyle();

        // Start a new ImGui ID Scope
        ImGui::PushID(label);

        // Calculate the size of the text and button
        float totalItemWidth = ImGui::CalcItemWidth();
        float halfItemWidth = totalItemWidth * 0.5f; // Subtract the ItemInnerSpacing

        ImGui::PushItemWidth(halfItemWidth - style.ItemSpacing.x); // Set the width of the next widget to 200

        clippedText(label); // Draw the label in the first column

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

    bool booleanText(const char* fmt, ...)
    {

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);
        return false;
    }


    void affineGui(math::affine3f& affine)
    {
        float availableWidth = ImGui::CalcItemWidth() - ImGui::GetStyle().ItemSpacing.x * 3;
        float elementWidth = availableWidth / 4.0f;//
        ImGui::BeginGroup();
        ImGui::PushItemWidth(elementWidth);
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                fakeButton(std::to_string(affine[row][col]).c_str(), { elementWidth, 0 });
                if (col < 3)
                {
                    ImGui::SameLine();
                }
            }
        }
        ImGui::PopItemWidth();
        ImGui::EndGroup();
    }

    bool customVectorInput(const char* label, float& value, const ImU32 labelColor, const bool disableEdit)
    {
        float availableWidth = ImGui::CalcItemWidth();

        // Start grouping to treat both the label and the input as a single item
        ImGui::BeginGroup();

        ImVec2 p = ImGui::GetCursorScreenPos();
        float rounding = ImGui::GetStyle().FrameRounding;

        // Rounded rectangle for the label
        ImGui::PushStyleColor(ImGuiCol_Button, labelColor);
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
        fakeButton(label);
        float buttonSize = ImGui::GetItemRectSize().x;
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

        ImGui::SameLine();

        float remaininWidth = availableWidth - buttonSize - ImGui::GetStyle().ItemSpacing.x;
        ImGui::PushItemWidth(remaininWidth);
        // Float input for the value
        bool updated = false;
        if(!disableEdit)
        {
             updated = ImGui::InputFloat(("##" + std::string(label)).c_str(), &value, 0.0f, 0.0f, "%.3f");
        }
        else
        {
            fakeButton((std::to_string(value)).c_str(), {remaininWidth, 0});
		}
        ImGui::PopItemWidth();

        // End the grouping
        ImGui::EndGroup();

        return updated;
    }

    bool vectorGui(float* data, const bool disableEdit)
    {
        bool updated = false;

		const float availableWidth = ImGui::CalcItemWidth() - ImGui::GetStyle().ItemSpacing.x * 2;

		const float perComponentWidth = availableWidth / 3.0f;

        ImGui::PushItemWidth(perComponentWidth);

        // X Component in red
        updated |= customVectorInput("X :", data[0], IM_COL32(200, 0, 0, 255), disableEdit);
        ImGui::SameLine();

        // Y Component in green
        updated |= customVectorInput("Y :", data[1], IM_COL32(0, 200, 20, 255), disableEdit);
        ImGui::SameLine();

        // Z Component in blue
        updated |= customVectorInput("Z :", data[2], IM_COL32(0, 0, 200, 255), disableEdit);

        ImGui::PopItemWidth();
        return updated;
    }

    static float             halfWidgetFraction = 0.3f;
    static std::stack<float> halfWidgetFractionStack = std::stack<float>();

    void pushHalfSpaceWidgetFraction(const float fraction)
    {
        halfWidgetFractionStack.push(halfWidgetFraction);
    }

    void popHalfSpaceWidgetFraction()
    {
        if (!halfWidgetFractionStack.empty())
        {
			halfWidgetFractionStack.pop();
		}
    }

    float getHalfWidgetFraction()
    {
        if (halfWidgetFractionStack.empty())
        {
            halfWidgetFractionStack.push(halfWidgetFraction);
        }
        return halfWidgetFractionStack.top();
    }

    void DrawRowsBackground(const int row_count, const float line_height, const float x1, const float x2, const float y_offset, const ImU32 col_even, const ImU32 col_odd)
    {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float y0 = ImGui::GetCursorScreenPos().y + (float)(int)y_offset;

        int row_display_start;
        int row_display_end;
        ImGui::CalcListClipping(row_count, line_height, &row_display_start, &row_display_end);
        for (int row_n = row_display_start; row_n < row_display_end; row_n++)
        {
            ImU32 col = (row_n & 1) ? col_odd : col_even;
            if ((col & IM_COL32_A_MASK) == 0)
                continue;
            float y1 = y0 + (line_height * row_n);
            float y2 = y1 + line_height;
            draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), col);
        }
    }

    void childWindowResizerButton(float& percentage, const float& resizerSize, const bool isHorizontalSplit)
    {
        const ImVec2 buttonSize = isHorizontalSplit ? ImVec2(resizerSize, ImGui::GetContentRegionAvail().y) : ImVec2(ImGui::GetContentRegionAvail().x, resizerSize);
        if (ImGui::Button("##resizer", buttonSize))
        {
            isHorizontalSplit ? ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW) : ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        if (ImGui::IsItemHovered())
        {
            isHorizontalSplit ? ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW) : ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
        {
            const float delta = isHorizontalSplit ? ImGui::GetMouseDragDelta(0).x : ImGui::GetMouseDragDelta(0).y;
            const float deltaPercentage = isHorizontalSplit ? delta / ImGui::GetWindowWidth() : delta / ImGui::GetWindowHeight();
            percentage -= deltaPercentage;
            percentage = std::clamp(percentage, 0.0f, 1.0f);
            ImGui::ResetMouseDragDelta(0);
        }
    }

}
#include "GuiWindow.h"
#include <imgui_internal.h>
#include "Core/Log.h"
#include "Core/CustomImGui/CustomImGui.h"

namespace vtx
{
	void Window::prepareChildWindow()
	{
		if(!isBorderLess)
		{
			ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(childPaddingWidth, childPaddingHeight));
			ImGui::PushStyleColor(ImGuiCol_Border, { 0.0, 0.0, 0.0, 0.0 });
		}
	}

	void Window::endChildWindowPrep()
	{
		if (!isBorderLess)
		{
			ImGui::PopStyleVar();
			ImGui::PopStyleVar();
			ImGui::PopStyleColor();
		}
	}
	void Window::OnUIRender()
	{
		preRender();
		if (!createWindow)
		{
			return;
		}
		if (ImGui::Begin(name.c_str(), &isOpen, windowFlags))
		{
			auto contentRegionAvail = ImGui::GetContentRegionAvail();
			auto itemSpacing = ImGui::GetStyle().ItemSpacing;
			renderMenuBar();

			prepareChildWindow();
			float mainContentWidth;
			float sideBarContentWidth;

			if (useToolbar)
			{
				// Main content space: window width - toolbar width - resizer width
				// The available space needs to account for the item spacing between: main, resizer and toolbar
				const float itemSpacingTotal     = itemSpacing.x * 2;
				const float actualSpaceAvailable = contentRegionAvail.x - itemSpacingTotal;
				mainContentWidth           = actualSpaceAvailable * (1.0f - toolbarPercentage);
				sideBarContentWidth        = actualSpaceAvailable * toolbarPercentage;
			}
			else
			{
				mainContentWidth = contentRegionAvail.x;
				sideBarContentWidth = 0.0f;
			}

			// Main content block

			if(ImGui::BeginChild("MainContent", ImVec2(mainContentWidth, 0), true))
			{
				if (useStripedBackground)
				{
					drawStripedBackground();
				}
				// Top padding
				const float availableWidth = ImGui::GetContentRegionAvail().x;
				ImGui::PushItemWidth(availableWidth); // Set the width of the next widget to 200
				endChildWindowPrep();
				renderMainContent();   // Placeholder function for the main content
				ImGui::PopItemWidth();
			}
			
			ImGui::EndChild();


			if (useToolbar)
			{

				ImGui::SameLine();
				vtxImGui::childWindowResizerButton(toolbarPercentage, resizerSize, true);

				// Draggable handle for resizing
				ImGui::SameLine();
				prepareChildWindow();

				ImGui::SameLine();
				if (ImGui::BeginChild("Sidebar", ImVec2(sideBarContentWidth, 0), true))
				{
					const float availableWidth = ImGui::GetContentRegionAvail().x;
					ImGui::PushItemWidth(availableWidth);
					endChildWindowPrep();

					// Reset the cursor to the start of the button for the next widget
					renderToolBar();  // Call renderToolBar outside of the Sidebar child
					ImGui::PopItemWidth();

				}

				ImGui::EndChild();  // End the Sidebar child here
				
			}

		}
		ImGui::End();  // End main window

	}
	inline void Window::drawStripedBackground()
	{
		float x1 = ImGui::GetCurrentWindow()->WorkRect.Min.x;
		float x2 = ImGui::GetCurrentWindow()->WorkRect.Max.x;
		float item_spacing_y = ImGui::GetStyle().ItemSpacing.y;
		float item_offset_y = -item_spacing_y * 0.5f;
		float line_height = ImGui::GetTextLineHeight() + item_spacing_y;
		vtxImGui::DrawRowsBackground(50, line_height, x1, x2, item_offset_y, ImGui::GetColorU32(ImGuiCol_Header), ImGui::GetColorU32(ImGuiCol_ChildBg));
	}
	void Window::setWindowManager(const std::shared_ptr<vtx::WindowManager>& _windowManager)
	{
		windowManager = _windowManager;
	}

	void WindowManager::addWindow(const std::shared_ptr<Window>& window)
	{
		const std::type_index typeIdx = typeid(*window);

		if (windowMap.find(typeIdx) != windowMap.end())
		{
			return;
		}

		windows.push_back(window);
		windowMap[typeIdx] = window;
		window->OnAttach();
	}
	
	void WindowManager::removeWindow(const std::shared_ptr<Window>& window)
	{
		const auto it = std::find(windows.begin(), windows.end(), window);
		if (it == windows.end())
		{
			return;
		}
		window->OnDetach();
		const std::type_index typeIdx = typeid(*window);
		windowMap.erase(typeIdx);
		windows.erase(it);
	}
	
	void WindowManager::updateWindows(const float timeStep) const
	{
		for (const auto& window : windows) {
			window->OnUpdate(timeStep);
		}
	}
	
	void WindowManager::renderWindows() const
	{
		int windowCount = windows.size();
		for (int i = 0; i < windowCount; i++) {
			auto& window = windows[i];
			window->OnUIRender();
			windowCount = windows.size();  // Update the count in case a window was added/removed.
		}
	}
	
	void WindowManager::removeClosedWindows()
	{
		for (const std::shared_ptr<Window>& window : windows)
		{
			if (!window->isOpen)
			{
				removeWindow(window);
			}
		}
	}
}


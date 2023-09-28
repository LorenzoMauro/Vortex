#include "GraphWindow.h"
#include "Scene/Scene.h"

namespace vtx
{
	GraphWindow::GraphWindow()
	{
		name = "Node Graph";
		useToolbar = false;
		renderCustomMenuBar = true;
		renderer = graph::Scene::getScene()->renderer;
		isBorderLess = true;
		nodeEditor.flags = ImNodesStyleFlags_VerticalLayout;
		nodeEditor.rootNode = renderer;
		nodeEditor.nodesWithColor = true;
	}

	void GraphWindow::mainContent()
	{
		nodeEditor.draw();
		graph::Scene::getScene()->setSelected(nodeEditor.getSelected());
	}
	void GraphWindow::menuBarContent()
	{
		ImGui::BeginGroup();
		{
			const float windowWidth = ImGui::GetWindowWidth();
			ImGui::SetCursorPosY(ImGui::GetStyle().WindowPadding.y);

			float xPos = windowWidth;
			float arrangeButtonWidth = ImGui::CalcTextSize("Arrange Nodes").x + 20.0f;
			ImGui::PushItemWidth(arrangeButtonWidth);
			ImGui::SameLine();
			xPos = xPos - arrangeButtonWidth - ImGui::GetStyle().WindowPadding.x;
			ImGui::SetCursorPosX(xPos);
			if (ImGui::Button("Arrange Nodes"))
			{
				nodeEditor.arrangeNodes(gui::NodeEditor::LayoutDirection::Vertical);
			}

			ImGui::SameLine();
			xPos = xPos - arrangeButtonWidth - ImGui::GetStyle().ItemSpacing.x * 2.0f;
			ImGui::SetCursorPosX(xPos);

			float nodePadding[2] = { nodeEditor.treeWidthPadding, nodeEditor.treeDepthPadding };
			if (ImGui::DragFloat2("##padding", nodePadding, 1.0f, 0.0f, 120.0f))
			{
				nodeEditor.treeWidthPadding = nodePadding[0];
				nodeEditor.treeDepthPadding = nodePadding[1];
				nodeEditor.arrangeNodes(gui::NodeEditor::LayoutDirection::Vertical);
			}
			ImGui::PopItemWidth();
		}
		ImGui::EndGroup();
		menuBarHeight = ImGui::GetItemRectSize().y;

	}
}


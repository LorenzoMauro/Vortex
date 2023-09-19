#include "SceneHierarchyWindow.h"

#include <imgui_internal.h>

#include "Scene/Graph.h"
#include "Scene/Scene.h"

namespace vtx
{
	SceneHierarchyWindow::SceneHierarchyWindow()
	{
		name = "Scene Hierarchy";
		useToolbar = false;
		renderer = graph::Scene::getScene()->renderer;
        useStripedBackground = true;
	}
    void SceneHierarchyWindow::displayNode(const std::shared_ptr<graph::Node>& node) {
        // Check if the node is null
        if (!node) return;

        // Flags for TreeNodeEx
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_OpenOnArrow;
        int node_clicked = -1;

        const auto& children = node->getChildren();
        // Check if the node is a leaf (i.e., it has no children)
        if (children.empty()) {
            flags |= ImGuiTreeNodeFlags_Leaf;
        }

        // Check if the node is selected
        if (selection_mask == node->getID()) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
            flags |= ImGuiTreeNodeFlags_Selected;
        }

        // Draw the node using TreeNodeEx. If it returns true, it means the node is expanded.
        const bool nodeOpen = ImGui::TreeNodeEx(reinterpret_cast<void*>(node->getID()), flags, "%s", node->name.c_str());
        if (selection_mask == node->getID())
        {
			ImGui::PopStyleColor();
        }

        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (selection_mask == node->getID())
            {
                selection_mask = -1;
            }
            else
            {
				node_clicked = node->getID();
            }
        }

        if (nodeOpen) {
            // Loop through and display child nodes
            for (const std::shared_ptr<graph::Node>& child : children) {
                displayNode(child);
            }
            ImGui::TreePop();
        }

        if (node_clicked != -1) {
            if (ImGui::GetIO().KeyCtrl) {
                // CTRL+click to toggle (deselect if it's already selected)
                selection_mask = (selection_mask == node_clicked) ? -1 : node_clicked;
            }
            else {
                selection_mask = node_clicked; // Click to single-select
            }
        }

    }

	void SceneHierarchyWindow::renderMainContent()
	{
        // Now proceed with your node rendering as usual.
        displayNode(renderer);
        if(selection_mask == -1)
        {
	        windowManager->selectedNodes["SceneHierarchyWindow"] = {};
		}
        else
        {
	        windowManager->selectedNodes["SceneHierarchyWindow"] = {(vtxID)selection_mask};
        }
	}
}

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

        std::set<vtxID> selected = graph::Scene::getScene()->getSelected();
        bool isSelected = selected.find(node->getUID()) != selected.end();
        // Flags for TreeNodeEx
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth | ImGuiTreeNodeFlags_OpenOnArrow;
        int node_clicked = -1;

        const auto& children = node->getChildren();
        // Check if the node is a leaf (i.e., it has no children)
        if (children.empty()) {
            flags |= ImGuiTreeNodeFlags_Leaf;
        }

        // Check if the node is selected
        if (isSelected) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive));
            flags |= ImGuiTreeNodeFlags_Selected;
        }

        // Draw the node using TreeNodeEx. If it returns true, it means the node is expanded.
        const bool nodeOpen = ImGui::TreeNodeEx(reinterpret_cast<void*>(node->getUID()), flags, "%s", node->name.c_str());
        if (isSelected)
        {
			ImGui::PopStyleColor();
        }

        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            if (isSelected)
            {
                graph::Scene::getScene()->removeNodesToSelection({node->getUID()});
            }
            else
            {
                graph::Scene::getScene()->addNodesToSelection({node->getUID()});
            }
        }

        if (nodeOpen) {
            // Loop through and display child nodes
            for (const std::shared_ptr<graph::Node>& child : children) {
                displayNode(child);
            }
            ImGui::TreePop();
        }

    }

	void SceneHierarchyWindow::mainContent()
	{
        // Now proceed with your node rendering as usual.
        displayNode(renderer);
	}
}

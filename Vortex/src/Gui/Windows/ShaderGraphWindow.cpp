#include "ShaderGraphWindow.h"

#include "imgui.h"
#include "Scene/Nodes/Material.h"
#include "imnodes.h"

namespace vtx {
    ShaderGraphWindow::ShaderGraphWindow()
    {
        name = "Shader Graph";
        useToolbar = false;
        isBorderLess = true;
        renderCustomMenuBar = true;
    }

    void ShaderGraphWindow::mainContent() {
        bool rearrangeNodes = false;
        if(materialOpenedChanged)
        {
            // Swap Node info so we can keep the previous arrangement
            nodeInfoByMaterial[previousMaterial] = nodeEditor.nodes;
            editorContextByMaterial[previousMaterial] = nodeEditor.sceneGraphContext;
            if( nodeInfoByMaterial.find(materialOpened) != nodeInfoByMaterial.end())
            {
	            nodeEditor.nodes = nodeInfoByMaterial[materialOpened];
            }
            if ( editorContextByMaterial.find(materialOpened) != editorContextByMaterial.end())
            {
	            nodeEditor.sceneGraphContext = editorContextByMaterial[materialOpened];
			}
			else
			{
				nodeEditor.sceneGraphContext = ImNodes::EditorContextCreate();
				editorContextByMaterial[materialOpened] = nodeEditor.sceneGraphContext;
                rearrangeNodes = true;
			}
            const std::shared_ptr<graph::Material> material = graph::SIM::get()->getNode<graph::Material>(materialOpened);
            const std::shared_ptr<graph::Node> node = std::reinterpret_pointer_cast<graph::Node>(material->materialGraph);
            nodeEditor.rootNode = node;
            materialOpenedChanged = false;
        }
        bool isMaterialUpdated = nodeEditor.draw();
        if (rearrangeNodes)
        {
	        nodeEditor.arrangeNodes(gui::NodeEditor::LayoutDirection::Horizontal);
		}
        std::vector<vtxID> selectedNodes(nodeEditor.selectedNodes.size());
        for (int i = 0; i < nodeEditor.selectedNodes.size(); ++i)
            selectedNodes[i] = (vtxID)nodeEditor.selectedNodes[i];

        windowManager->selectedNodes["SceneGraphWindow"] = selectedNodes;


        if(isMaterialUpdated)
        {
            graph::SIM::get()->getNode<graph::Material>(materialOpened)->isUpdated = true;
            ops::restartRender();
        }
    }
    void ShaderGraphWindow::materialSelector()
    {
        const std::vector<vtxID> materialIds = graph::SIM::get()->getAllNodeIdByType<graph::Material>(graph::NT_MATERIAL);
        if(materialOpened == 0 && !materialIds.empty())
        {
	        materialOpened = materialIds[0];
            const std::shared_ptr<graph::Material>& selectedMaterial = graph::SIM::get()->getNode<graph::Material>(materialOpened);
            openedMaterialName = selectedMaterial->name;
            materialOpenedChanged = true;
        }
        if(ImGui::BeginCombo("Materials :", openedMaterialName.c_str()))
        {
	        for (const vtxID materialId : materialIds)
			{
		        const std::shared_ptr<graph::Material>& material = graph::SIM::get()->getNode<graph::Material>(
					materialId);
				const bool isSelected = material->getID() == materialOpened;
				if (ImGui::Selectable(material->name.c_str(), isSelected))
				{
                    previousMaterial = materialOpened;
					materialOpened = material->getID();
                    openedMaterialName = material->name;
                    materialOpenedChanged = true;
                }
				if (isSelected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
        }

    }
    void ShaderGraphWindow::menuBarContent()
    {
        ImGui::BeginGroup();
        {
            const float windowWidth = ImGui::GetWindowWidth();
            ImGui::SetCursorPosY(ImGui::GetStyle().WindowPadding.y);
            const float oneThird = windowWidth / 3.0f;
            ImGui::PushItemWidth(oneThird);

            const float cursorPosX = (windowWidth - oneThird) / 2.0f;

            ImGui::SetCursorPosX(cursorPosX);
            materialSelector(); // Your combo function

            ImGui::PopItemWidth();

            float xPos = windowWidth;
            float arrangeButtonWidth = ImGui::CalcTextSize("Arrange Nodes").x + 20.0f;
            ImGui::PushItemWidth(arrangeButtonWidth);
            ImGui::SameLine();
            xPos = xPos - arrangeButtonWidth - ImGui::GetStyle().WindowPadding.x;
            ImGui::SetCursorPosX(xPos);
            if (ImGui::Button("Arrange Nodes"))
            {
                nodeEditor.arrangeNodes(gui::NodeEditor::LayoutDirection::Horizontal);
            }

            ImGui::SameLine();
            xPos = xPos - arrangeButtonWidth - ImGui::GetStyle().ItemSpacing.x*2.0f;
            ImGui::SetCursorPosX(xPos);

            float nodePadding[2] = { nodeEditor.treeWidthPadding, nodeEditor.treeDepthPadding };
            if (ImGui::DragFloat2("##padding", nodePadding, 1.0f, 0.0f, 120.0f))
            {
                nodeEditor.treeWidthPadding = nodePadding[0];
                nodeEditor.treeDepthPadding = nodePadding[1];
                nodeEditor.arrangeNodes(gui::NodeEditor::LayoutDirection::Horizontal);
            }
            ImGui::PopItemWidth();
        }
        ImGui::EndGroup();
        menuBarHeight = ImGui::GetItemRectSize().y;
    }
}

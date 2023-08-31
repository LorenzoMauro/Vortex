#pragma once
#include "imnodes.h"
#include "Scene/Node.h"
#include "Core/Math.h"
#include "Core/Options.h"
#include "NodesGui/GuiVisitor.h"

namespace vtx::gui
{
    struct LinkInfo
    {
        vtxID linkId;
        vtxID inputSocketId;
        vtxID childOutputSocketId;
    };

    struct NodeInfo
    {
        std::shared_ptr<graph::Node> node;
        math::vec2f size{0, 0};
        math::vec2i pos{ 0, 0 };
        int width = -1;
        int depth = -1;
		int         overallWidth;
		bool widthRemapped = false;
        std::vector<LinkInfo> links;
        bool verticalLayout = false;

		std::string title;
		vtxID       id;
	};

    class NodeGraphUi
    {
    public:
		static int hashInputSocket(vtxID nodeId, int socketPosition)
		{
			return nodeId * 1000 + socketPosition;
		}

		static int hashLink(vtxID parentId, vtxID childId)
		{
			return parentId * 1000 + childId;
		}

        void submitNode(const std::shared_ptr<graph::Node>& node, int depth, int width, int overallWidth)
        {
			if (nodes.find(node->getID()) != nodes.end())
			{
				return;
			}

			NodeInfo nodeInfo;
			nodeInfo.node = node;
			nodeInfo.title = node->name;
			nodeInfo.id = node->getID();
			nodeInfo.depth = depth;
			nodeInfo.width = width;
			nodeInfo.overallWidth = overallWidth;

			const std::vector<std::shared_ptr<graph::Node>> children          = node->getChildren();
			const int                                       numberOfChildren = children.size();
			for (int i = 0; i < numberOfChildren; i++)
			{
				LinkInfo linkInfo;
				linkInfo.inputSocketId = hashInputSocket(nodeInfo.id, i);
				linkInfo.linkId = hashLink(nodeInfo.id, children[i]->getID());

				linkInfo.childOutputSocketId = children[i]->getID();
				nodeInfo.links.push_back(linkInfo);
			}

			nodes[node->getID()] = nodeInfo;
			depthFirstTraversal.push_back(node->getID());
        }

		void drawNodes(NodeInfo& nodeInfo)
		{
			const int numberOfInputSockets = nodeInfo.links.size();
			float nodeWidth = spacing * std::max(numberOfInputSockets, 1);
			const float titleWidth = ImGui::CalcTextSize(nodeInfo.title.c_str()).x;
			nodeWidth = std::max(nodeWidth, titleWidth);
			const float localSpacing = nodeWidth / std::max(numberOfInputSockets, 1);
			const float leftTitlePad = (nodeWidth - titleWidth) / 2;

			ImNodes::BeginNode(nodeInfo.id);

			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
			nodeInfo.size.x = std::max(nodeInfo.size.x, getOptions()->nodeWidth); // Enforce a minimum size
			ImGui::PushItemWidth(nodeInfo.size.x); // Set the width of the next widget to 200

			ImNodes::BeginNodeTitleBar();
			ImGui::Dummy(ImVec2(leftTitlePad, 0));
			ImGui::SameLine();
			ImGui::TextUnformatted((nodeInfo.title).c_str());
			ImNodes::EndNodeTitleBar();


			//{
			//	ImGui::TextUnformatted((std::to_string(nodeInfo.depth)).c_str());
			//	ImGui::TextUnformatted((std::to_string(nodeInfo.width)).c_str());
			//	ImGui::TextUnformatted((std::to_string(nodeInfo.overallWidth)).c_str());
			//}

			ImNodes::BeginOutputAttribute(nodeInfo.id);
			ImGui::Dummy(ImVec2(nodeWidth, 0)); // Placeholder for input
			ImNodes::EndOutputAttribute();

			for (int idx = 0; idx < numberOfInputSockets; idx++)
			{
				ImNodes::BeginInputAttribute(nodeInfo.links[idx].inputSocketId);
				ImGui::Dummy(ImVec2(localSpacing, 0)); // Placeholder for input
				ImNodes::EndInputAttribute();

				if (idx < numberOfInputSockets - 1)
				{
					ImGui::SameLine();
				}


			}

			ImGui::PopStyleVar();
			ImNodes::EndNode();
			ImGui::PopItemWidth();
		}



		static void drawLinks(const NodeInfo& nodeInfo)
		{
			for (const auto& linkInfo : nodeInfo.links)
			{
				ImNodes::Link(linkInfo.linkId, linkInfo.inputSocketId, linkInfo.childOutputSocketId);
			}
		}

		void arrangeNodes()
		{
			std::map<int, std::vector<vtxID>> depthToNodes;
			std::map<int, ImVec2> depthToBoundingBox; // Bounding box for each depth

			float horizzontalPadding = 30;

			// Collect nodes per depth and calculate bounding box
			for (size_t i = 0; i < depthFirstTraversal.size(); ++i) {
				vtxID nodeId = depthFirstTraversal[i];
				int depth = nodes[nodeId].depth;
				depthToNodes[depth].push_back(nodeId);
				ImVec2 nodeDim = ImNodes::GetNodeDimensions(nodeId);
				nodes[nodeId].size = { nodeDim.x, nodeDim.y };
				depthToBoundingBox[depth].x += nodeDim.x; // Width
				if (i < depthFirstTraversal.size() - 1)
				{
					depthToBoundingBox[depth].x += horizzontalPadding;
				}
				depthToBoundingBox[depth].y = std::max(depthToBoundingBox[depth].y, nodeDim.y); // Height
			}

			// Get the window's width
			float windowWidth = ImGui::GetWindowWidth();

			// Vertical position accumulator
			float currentY = 0;

			// Arrange nodes
			for (int depth = 0; depth < depthToNodes.size(); ++depth)
			{
				ImVec2 boundingBox = depthToBoundingBox[depth];
				float currentX = (windowWidth - boundingBox.x) / 2.0f; // Start position for this row

				for (size_t i = 0; i < depthToNodes[depth].size(); ++i)
				{
					vtxID nodeId = depthToNodes[depth][i];
					ImVec2 nodeDim = { nodes[nodeId].size.x, nodes[nodeId].size.y };
					//ImVec2 currentPos = { currentX + nodeDim.x / 2.0f, currentY + nodeDim.y / 2.0f };
					ImVec2 currentPos = { currentX, currentY};
					currentX += nodeDim.x+ horizzontalPadding; // Move to the next position considering the node's width
					ImNodes::SetNodeGridSpacePos(nodeId, currentPos);
				}

				// Increment vertical position by the bounding box height, plus any desired vertical spacing
				currentY += boundingBox.y + 3*horizzontalPadding;
			}
		}



		void draw()
		{
			ImGui::Begin("Scene Graph");
			ImNodes::EditorContextSet(sceneGraphContext);
			ImNodes::BeginNodeEditor();
			ImNodes::PushAttributeFlag(ImNodesStyleFlags_VerticalLayout);
			for (auto& nodeInfo : nodes)
			{
				drawNodes(nodeInfo.second);
			}
			for (auto& nodeInfo : nodes)
			{
				drawLinks(nodeInfo.second);
			}

			if (isFirstTime)
			{
				arrangeNodes();
				isFirstTime = false;
			}

			
			ImNodes::EndNodeEditor();
			ImNodes::PopAttributeFlag();
			ImGui::End();

			ImGui::Begin("Settings");

			int numberOfSelectedNodes = ImNodes::NumSelectedNodes();
			std::vector<int> selectedNodes;
			if (numberOfSelectedNodes!=0)
			{
				selectedNodes.resize(numberOfSelectedNodes);
				ImNodes::GetSelectedNodes(selectedNodes.data());

				for (int idx : selectedNodes)
				{
					vtxID nodeId = idx;

					std::shared_ptr<graph::Node> node = nodes[nodeId].node;

					ImGui::TextUnformatted(node->name.c_str());
					ImGui::Separator();
					ImGui::TextUnformatted(graph::nodeNames[node->getType()].c_str());

					node->accept(guiVisitor);
				}
			}

			//if(ImNodes::IsEditorHovered())
			//{
			//	int* nodeID = nullptr;
			//	ImNodes::IsNodeHovered(nodeID);
			//
			//	if(nodeID)
			//	{
			//		if (ImGui::Button("Right Click Me")) {
			//
			//			if (ImGui::BeginPopupContextItem())
			//			{
			//				ImGui::Text("Right Click");
			//				ImGui::EndPopup();
			//			}
			//		}
			//	}
			//}

			ImGui::End();

		}
        std::map<vtxID, NodeInfo>    nodes;
		ImNodesEditorContext* sceneGraphContext = ImNodes::EditorContextCreate();
		int spacing = 40;
		std::vector<vtxID> depthFirstTraversal;
		bool isFirstTime = true;

		GuiVisitor guiVisitor;
    };

    void arrangeNodes(std::map<vtxID, NodeInfo>& nodeInfoMap, bool remapPosition = true);

    void createLinksAndCalculateWidthAndDepth(std::map<vtxID, NodeInfo>& nodeInfoMap);
}

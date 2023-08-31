#include "NodeEditorWrapper.h"
#include "imnodes.h"

namespace vtx::gui
{
    void arrangeNodes(std::map<vtxID, NodeInfo>& nodeInfoMap, bool remapPosition)
    {
		// Find max depth to position nodes from right to left
        int maxDepth = 0;
        for (const auto& [id, nodeInfo] : nodeInfoMap)
        {
            if (nodeInfo.depth > maxDepth)
                maxDepth = nodeInfo.depth;
        }

        // Create a map for each depth, to map old widths to new, continuous widths
        // Also, keep track of the max node dimensions at each depth level
        std::map<int, std::map<int, int>> widthRemapping;
        std::map<int, ImVec2> maxNodeDimensionsAtDepth;
        for (const auto& [id, nodeInfo] : nodeInfoMap)
        {
            if(remapPosition)
            {
                widthRemapping[nodeInfo.depth][nodeInfo.width] = 0;
            }
            ImVec2  nodeDim = ImNodes::GetNodeDimensions(id);
            ImVec2& maxDimAtDepth = maxNodeDimensionsAtDepth[nodeInfo.depth];
            maxDimAtDepth.x = std::max(maxDimAtDepth.x, nodeDim.x);
            maxDimAtDepth.y = std::max(maxDimAtDepth.y, nodeDim.y);
        }

        if(remapPosition)
        {
        	// Generate new widths
            for (auto& [depth, widthMap] : widthRemapping)
            {
                int newWidth = 0;
                for (auto& [oldWidth, _] : widthMap)
                {
                    widthMap[oldWidth] = newWidth++;
                }
            }
        }
        

        // Arrange nodes using new widths, considering node dimensions
        for (auto& [id, nodeInfo] : nodeInfoMap)
        {
			constexpr float padding           = 50.0f;
            int width;
            if (remapPosition)
            {
	            width = widthRemapping[nodeInfo.depth][nodeInfo.width];
			}
			else
			{
				width = nodeInfo.width;
			}
			const ImVec2    maxDimAtDepth     = maxNodeDimensionsAtDepth[nodeInfo.depth];
			const float     horizontalSpacing = maxDimAtDepth.x + padding;
			const float     verticalSpacing   = maxDimAtDepth.y + padding;
            ImVec2          pos((maxDepth - nodeInfo.depth) * horizontalSpacing, width * verticalSpacing);
            ImNodes::SetNodeGridSpacePos(id, pos);
        }
    }

    int calculateDepth(vtxID nodeId, std::map<vtxID, vtxID>& childToParent, std::map<vtxID, NodeInfo>& nodeInfoMap)
    {
        // If the node's depth is already calculated, return it
        if (nodeInfoMap[nodeId].depth != -1) return nodeInfoMap[nodeId].depth;

        // If the node is not a child, its depth is 0
        if (childToParent.find(nodeId) == childToParent.end()) {
            nodeInfoMap[nodeId].depth = 0;
            return 0;
        }

        // Otherwise, calculate the parent's depth recursively and increment it by 1
		const vtxID parent        = childToParent[nodeId];
		const int   parentDepth   = calculateDepth(parent, childToParent, nodeInfoMap);
        nodeInfoMap[nodeId].depth = parentDepth + 1;
        return parentDepth + 1;
	}

    void remapWidthsByDepth(std::map<vtxID, vtxID>& childToParent, std::map<vtxID, NodeInfo>& nodeInfoMap) {
        for (auto& [childId, parentId] : childToParent) {

            NodeInfo& childNodeInfo = nodeInfoMap[childId];
            NodeInfo& parentNodeInfo = nodeInfoMap[parentId];
            childNodeInfo.width += parentNodeInfo.depth;
        }
    }

    void remapWidth(vtxID nodeId, std::map<vtxID, vtxID>& childToParent, std::map<vtxID, NodeInfo>& nodeInfoMap)
    {
        if (nodeInfoMap[nodeId].widthRemapped) return;
		// If the node is not a child, its depth is 0
		if (childToParent.find(nodeId) == childToParent.end())
		{
			nodeInfoMap[nodeId].width = 0;
            nodeInfoMap[nodeId].widthRemapped = true;
			return;
		}
		// Otherwise, calculate the parent's depth recursively and increment it by 1
		const vtxID parent = childToParent[nodeId];
        remapWidth(parent, childToParent, nodeInfoMap);
		nodeInfoMap[nodeId].width += nodeInfoMap[parent].width;
        nodeInfoMap[nodeId].widthRemapped = true;
    }

    void createLinksAndCalculateWidthAndDepth(std::map<vtxID, NodeInfo>& nodeInfoMap)
    {
        int depth = 0;
        int width = 0;

        std::map<vtxID, vtxID> childToParent;

        for(auto& [nodeId, nodeInfo] : nodeInfoMap)
        {
            width = 0;
	        for(auto& linkInfo : nodeInfo.links)
	        {
                childToParent[linkInfo.childOutputSocketId] = nodeId;
		        ImNodes::Link(linkInfo.linkId, linkInfo.inputSocketId, linkInfo.childOutputSocketId);
                if(nodeInfoMap.find(linkInfo.childOutputSocketId) != nodeInfoMap.end())
                {
	                NodeInfo& outputNodeInfo = nodeInfoMap[linkInfo.childOutputSocketId];
					outputNodeInfo.width = width;
					++width;
                }
			}
        }

        // Calculate depths
        for (auto& [nodeId, nodeInfo] : nodeInfoMap) {
            calculateDepth(nodeId, childToParent, nodeInfoMap);
            remapWidth(nodeId, childToParent, nodeInfoMap);
        }
    }
}

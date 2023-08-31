#include "Traversal.h"
#include "Graph.h"

namespace vtx
{
	void NodeVisitor::visitBegin(const std::shared_ptr<graph::Node>& node)
	{
		if (collectWidthsAndDepths)
		{
			if(depthStack.empty())
			{
				depthStack.push(0);
			}
			else
			{
				depthStack.push(depthStack.top() + 1);
			}
			if (widthStack.empty())
			{
				widthStack.push(0);
			}
			
			// Replace the value for the node's ID with the current depth and width

			int parentWidth = 0;
			if(!parentPath.empty())
			{
				auto& [_, pW, n] = nodesDepthsAndWidths[parentPath.back()];
				parentWidth = pW;
			}
			nodesDepthsAndWidths[node->getID()] = { depthStack.top(), widthStack.top(), parentWidth + widthStack.top()};// , currentWidth + parent
			int& currentWidth = widthStack.top();
			currentWidth += 1;
			widthStack.push(0);

			nodesParents[node->getID()] = parentPath;
			parentPath.push_back(node->getID());
		}

		if (collectTransforms)
		{
			bool foundTransform = false;
			for (const std::shared_ptr<graph::Node> child : node->getChildren())
			{
				if (std::shared_ptr<graph::Transform> transform = child->as<graph::Transform>(); transform != nullptr)
				{
					tmpTransforms.push(currentTransform);
					foundTransform = true;
					currentTransform = currentTransform * transform->affineTransform;
					transform->globalTransform = currentTransform;
					break;
				}
			}
			resetTransforms.push(foundTransform);
		}
	}

	void NodeVisitor::visitEnd(std::shared_ptr<graph::Node> node)
	{
		if (collectWidthsAndDepths)
		{
			depthStack.pop();
			widthStack.pop();
			parentPath.pop_back();
		}
		if (collectTransforms)
		{
			if (resetTransforms.top())
			{
				currentTransform = tmpTransforms.top();
				tmpTransforms.pop();

			}
			resetTransforms.pop();
		}
	}
}

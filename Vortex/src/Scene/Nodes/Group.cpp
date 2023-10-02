#include "Group.h"
#include <memory>
#include "Scene/Traversal.h"
#include "Scene/Utility/Operations.h"

namespace vtx::graph
{
	Group::Group() : Node(NT_GROUP)
	{
		transform = ops::createNode<Transform>();
	}

	Group::~Group()
	{
	}

	std::vector<std::shared_ptr<Node>> Group::getChildren() const
	{
		std::vector<std::shared_ptr<Node>> rC;
		rC.push_back(transform);
		rC.insert(rC.end(), children.begin(), children.end());
		return rC;
	}

	void Group::addChild(const std::shared_ptr<Node>& child) {
		const NodeType newChildType = child->getType();
		if(newChildType == NT_GROUP || newChildType == NT_INSTANCE || newChildType == NT_TRANSFORM)
		{
			children.push_back(child);
		}
		else
		{
			VTX_ERROR("Cannot add node of type {} to group", nodeNames[child->getType()]);
		}
	}

	void Group::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Group>());
	}
}


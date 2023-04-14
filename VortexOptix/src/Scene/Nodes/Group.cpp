#include "Group.h"
#include <memory>
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Group::Group() : Node(NT_GROUP) {}

	std::vector<std::shared_ptr<Node>>& Group::getChildren() {
		return children;
	}

	void Group::addChild(const std::shared_ptr<Node>& child) {
		children.push_back(child);
	}

	void Group::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(visitors)
		for (const auto& child : children) {
			child->traverse(orderedVisitors);
		}
	}

	void Group::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Group>());
	}
}


#pragma once
#include "Transform.h"
#include "Scene/Node.h"

namespace vtx::graph
{
	class Light;

	class Group : public Node{
	public:
		Group();

		std::vector<std::shared_ptr<Node>>& getChildren();

		void addChild(const std::shared_ptr<Node>& child);

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

	public:
		std::shared_ptr<Transform>		   transform;
	private:
		std::vector<std::shared_ptr<Node>> children;
	};
}

#pragma once
#include "Transform.h"
#include "Scene/Node.h"

namespace vtx::graph
{
	class Group : public Node{
	public:
		Group();

		~Group() override;
		std::vector<std::shared_ptr<Node>> getChildren() const override;

		void addChild(const std::shared_ptr<Node>& child);

		std::shared_ptr<Transform> transform;
	protected:
		void accept(NodeVisitor& visitor) override;
		std::vector<std::shared_ptr<Node>> children;
	};
}

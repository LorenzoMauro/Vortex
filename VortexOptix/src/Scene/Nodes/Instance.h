#pragma once
#include "Scene/Node.h"
#include "Material.h"
#include "Transform.h"

namespace vtx::graph
{

	class Instance : public Node {
	public:

		Instance();

		std::shared_ptr<Node> getChild();

		void setChild(std::shared_ptr<Node> _child);

		std::shared_ptr<Transform> getTransform();

		void setTransform(std::shared_ptr<Transform>& _transform);

		std::vector<std::shared_ptr<Material>>& getMaterials();

		void addmaterial(std::shared_ptr<Material> _material);

		void RemoveMaterial(vtxID matID);

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	private:
		std::shared_ptr<Node>       child;
		std::shared_ptr<Transform>  transform;
		std::vector<std::shared_ptr<Material>> materials;

	};

}

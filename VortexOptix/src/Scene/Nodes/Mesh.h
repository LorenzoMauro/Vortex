#pragma once
#include "Scene/Node.h"
#include "Scene/DataStructs/VertexAttribute.h"

namespace vtx::graph
{

	class Mesh : public Node {
	public:
		Mesh() : Node(NT_MESH) {}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	public:
		std::vector<VertexAttributes> vertices;
		std::vector<vtxID> indices; // indices for triangles (every 3 indices define a triangle)
	};

}

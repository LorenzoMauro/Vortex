#pragma once
#include "Scene/Node.h"

namespace vtx::graph
{

	struct VertexAttributes {
		math::vec3f position;
		math::vec3f normal;
		math::vec3f tangent;
		math::vec3f texCoord;
		int         instanceMaterialIndex;
	};

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

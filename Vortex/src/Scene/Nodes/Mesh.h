#pragma once
#include "Scene/Node.h"
#include "Scene/DataStructs/VertexAttribute.h"

namespace vtx::graph
{

	struct MeshStatus
	{
		bool hasTangents = false;
		bool hasNormals = false;
		bool hasFaceAttributes = false;
	};

	class Mesh : public Node {
	public:
		Mesh();

		~Mesh();

		std::vector<std::shared_ptr<Node>> getChildren() const override;;
	protected:
		void accept(NodeVisitor& visitor) override;
	public:
		std::vector<VertexAttributes> vertices;
		std::vector<vtxID>            indices; // indices for triangles (every 3 indices define a triangle)
		std::vector<FaceAttributes>   faceAttributes;
		MeshStatus                    status;
	};

}

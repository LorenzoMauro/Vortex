#pragma once
#include "Scene/Node.h"

namespace vtx::graph
{

	class MeshLight : public Node
	{
	public:
		MeshLight() : Node(NT_MESH_LIGHT) {}

		void init() override;

		std::vector<std::shared_ptr<Node>> getChildren() const override;
	protected:

		void accept(NodeVisitor& visitor) override;
	private:
		void computeAreaCdf();
	public:
		//std::shared_ptr<LightAttributes> attributes;

		std::shared_ptr<graph::Mesh>		mesh;
		std::shared_ptr<graph::Material>	material;
		unsigned int						materialRelativeIndex;
		vtxID								parentInstanceId;

		std::vector<float>					cdfAreas;
		std::vector<unsigned int>			actualTriangleIndices;
		float								area;
		bool								isValid = false;

	};
}
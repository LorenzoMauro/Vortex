#include "Mesh.h"
#include "Scene/SIM.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Mesh::Mesh() : Node(NT_MESH)
	{
		state.updateOnDevice = true;
		typeID = SIM::get()->getTypeId<Mesh>();
	}
	Mesh::~Mesh()
	{
		SIM::get()->releaseTypeId<Mesh>(typeID);
	}

	std::vector<std::shared_ptr<Node>> Mesh::getChildren() const
	{
		return {};
	}
	void Mesh::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Mesh>());
	}
}

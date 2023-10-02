#include "Mesh.h"
#include "Scene/SceneIndexManager.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Mesh::Mesh() : Node(NT_MESH)
	{
		state.updateOnDevice = true;
	}
	Mesh::~Mesh()
	{
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

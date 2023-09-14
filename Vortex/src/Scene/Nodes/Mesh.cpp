#include "Mesh.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	void Mesh::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Mesh>());
	}
}

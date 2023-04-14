#include "Mesh.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	void Mesh::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(visitors)
	}
	
	void Mesh::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Mesh>());
	}
}

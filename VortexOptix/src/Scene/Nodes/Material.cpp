#include "Material.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	void Material::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(visitors);
		shader->traverse(orderedVisitors);
	}

	void Material::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Material>());
	}
}


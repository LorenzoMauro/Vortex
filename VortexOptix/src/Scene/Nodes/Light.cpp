#include "Light.h"
#include "Scene/Traversal.h"


namespace vtx::graph
{
	void Light::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(orderedVisitors);
	}

	void Light::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Light>());
	}

}


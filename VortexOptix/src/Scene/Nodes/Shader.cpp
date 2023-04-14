#include "Shader.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	void Shader::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		ACCEPT(orderedVisitors);
	}

	void Shader::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Shader>());
	}

}

#include "Texture.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"

namespace vtx::graph
{
	void Texture::init()
	{
		mdl::fetchTextureData(sharedFromBase<Texture>());

		isInitialized= true;
	}
	void Texture::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if(!isInitialized)
		{
			init();
		}
		ACCEPT(Texture,orderedVisitors);
	}

	//void Texture::accept(std::shared_ptr<NodeVisitor> visitor)
	//{
	//	visitor->visit(sharedFromBase<Texture>());
	//}
}

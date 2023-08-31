#include "Texture.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"

namespace vtx::graph
{
	void Texture::init()
	{
		if(!isInitialized)
		{
			if(loadFromFile)
			{
				mdl::loadFromFile(as<Texture>());
				loadFromFile = false;
			}
			mdl::fetchTextureData(sharedFromBase<Texture>());
			isInitialized= true;
		}
	}

	void Texture::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Texture>());
	}
}

#include "Texture.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "Scene/SceneIndexManager.h"

namespace vtx::graph
{
	Texture::Texture() :
		Node(NT_MDL_TEXTURE),
		shape(ITarget_code::Texture_shape_invalid)
	{
	}
	Texture::Texture(const char* _databaseName, const ITarget_code::Texture_shape _shape) :
		Node(NT_MDL_TEXTURE),
		databaseName(_databaseName),
		shape(_shape)
	{
	}
	Texture::Texture(const std::string& _filePath) :
		Node(NT_MDL_TEXTURE)
	{
		filePath = _filePath;
		loadFromFile = true;
	}
	Texture::~Texture()
	{
		for (const auto& ptr : imageLayersPointers)
		{
			free(const_cast<void*>(ptr));
		}
	}
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

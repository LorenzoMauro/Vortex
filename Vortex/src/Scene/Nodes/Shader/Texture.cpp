#include "Texture.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "Scene/SIM.h"

namespace vtx::graph
{
	Texture::Texture() :
		Node(NT_MDL_TEXTURE),
		shape(ITarget_code::Texture_shape_invalid)
	{
		typeID = SIM::get()->getTypeId<Texture>();
	}
	Texture::Texture(const char* _databaseName, const ITarget_code::Texture_shape _shape) :
		Node(NT_MDL_TEXTURE),
		databaseName(_databaseName),
		shape(_shape)
	{
		typeID = SIM::get()->getTypeId<Texture>();
	}
	Texture::Texture(std::string filePath) :
		Node(NT_MDL_TEXTURE)
	{
		this->filePath = filePath;
		loadFromFile = true;
		typeID = SIM::get()->getTypeId<Texture>();
	}
	Texture::~Texture()
	{
		for (const auto& ptr : imageLayersPointers)
		{
			free(const_cast<void*>(ptr));
		}
		SIM::get()->releaseTypeId<Texture>(typeID);
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

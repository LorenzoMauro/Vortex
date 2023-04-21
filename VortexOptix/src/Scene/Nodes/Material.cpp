#include "Material.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"

namespace vtx::graph
{
	void Material::init()
	{
		const auto& targetCode = shader->getTargetCodeHandle();
		const auto& materialDbName = shader->getMaterialDbName();
		mdl::setMaterialParameters(materialDbName, targetCode, argBlock, params, mapEnumTypes);
		isInitialized = true;
	}

	std::shared_ptr<graph::Shader> Material::getShader()
	{
		return shader;
	}

	size_t Material::getArgumentBlockSize()
	{
		if (!argBlock)
		{
			return 0;
		}
		return argBlock->get_size();
	}

	char* Material::getArgumentBlockData()
	{
		if (!argBlock)
		{
			return nullptr;
		}
		return argBlock->get_data();
	}

	void Material::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		shader->traverse(orderedVisitors);
		if(!isInitialized)
		{
			init();
		}
		ACCEPT(visitors);
	}

	void Material::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Material>());
	}
}


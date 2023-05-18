#include "Material.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "MDL/ShaderVisitor.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::graph
{
	void Material::init()
	{
		const auto& targetCode = shader->getTargetCodeHandle();
		const auto& materialDbName = shader->getMaterialDbName();
		mdl::setMaterialParameters(materialDbName, targetCode, argBlock, params, mapEnumTypes);

		std::string enable_emission = "enable_emission";
		std::string enable_opacity = "enable_opacity";
		bool* enableEmission;
		bool* enableOpacity;
		for(auto param : params)
		{
			std::string name = param.displayName();

			if(name.find(enable_emission)!=std::string::npos)
			{
				enableEmission = &param.data<bool>();
			}
			else if (name.find(enable_opacity)!=std::string::npos)
			{
				enableOpacity = &param.data<bool>();
			}
		}
		if(enableEmission)
		{
			shader->config.emissivityToggle = enableEmission;

		}
		if(enableOpacity)
		{
			shader->config.opacityToggle = enableOpacity;
		}
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
		if(materialGraph)
		{
			if(materialGraph->isUpdated)
			{
				auto materialCasted = std::dynamic_pointer_cast<graph::shader::Material>(materialGraph);
				auto [moduleName, functionName] =mdl::createNewFunctionInModule(materialGraph);
				shader->name = functionName;
				shader->path = moduleName;
			}
		}
		shader->traverse(orderedVisitors);
		if(!isInitialized)
		{
			init();
		}
		ACCEPT(Material,visitors);
	}

	/*void Material::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Material>());
	}*/
}


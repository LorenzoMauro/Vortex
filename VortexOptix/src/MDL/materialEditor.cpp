#include "materialEditor.h"

#include "ShaderOperations.h"

namespace vtx::graph
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;

	void createNewModule(const std::string& moduleName, const std::string& materialName)
	{
		const std::string				diffuseTexture = "E:/Dev/VortexOptix/data/Textures/xboibga_2K_Albedo.jpg";
		const std::string				roughnessTexture = "E:/Dev/VortexOptix/data/Textures/xboibga_2K_Roughness.jpg";

		{

			mdl::ModuleCreationParameters moduleInfo{};

			{
				mdl::FunctionInfo diffuseTextureConstant = mdl::getTextureConstant(diffuseTexture);

				mdl::FunctionArguments diffuseTextureFileArguments;
				diffuseTextureFileArguments["texture"] = diffuseTextureConstant;
				mdl::FunctionInfo diffuseTextureFile = mdl::createFunction(moduleInfo, mdl::MDL_FILE_TEXTURE, diffuseTextureFileArguments);

				mdl::FunctionArguments diffuseTintArguments;
				diffuseTintArguments["s"] = diffuseTextureFile;
				mdl::FunctionInfo diffuseTextureTint = mdl::createFunction(moduleInfo, mdl::MDL_TEXTURE_RETURN_TINT, diffuseTintArguments);

				mdl::FunctionInfo roughnessTextureConstant = mdl::getTextureConstant(roughnessTexture);
				
				mdl::FunctionArguments roughnessTextureFileArguments;
				roughnessTextureFileArguments["texture"] = roughnessTextureConstant;
				mdl::FunctionInfo roughnessTextureFile = mdl::createFunction(moduleInfo, mdl::MDL_FILE_TEXTURE, roughnessTextureFileArguments);
				
				mdl::FunctionArguments roughnessMonoArguments;
				roughnessMonoArguments["s"] = roughnessTextureFile;
				mdl::FunctionInfo roughnessTextureMono = mdl::createFunction(moduleInfo, mdl::MDL_TEXTURE_RETURN_MONO, roughnessMonoArguments);

				mdl::FunctionArguments brdfArguments;
				brdfArguments["tint"] = diffuseTextureTint;
				brdfArguments["roughness"] = roughnessTextureMono;
				mdl::FunctionInfo diffuseReflection = mdl::createFunction(moduleInfo, mdl::MDL_DIFFUSE_REFLECTION, brdfArguments);

				mdl::FunctionArguments surfaceArguments;
				surfaceArguments["scattering"] = diffuseReflection;
				mdl::FunctionInfo materialSurface = mdl::createFunction(moduleInfo, mdl::MDL_MATERIAL_SURFACE, surfaceArguments);

				mdl::FunctionArguments materialArguments;
				materialArguments["surface"] = materialSurface;
				mdl::FunctionInfo materialFunction = mdl::createFunction(moduleInfo, mdl::MDL_MATERIAL, materialArguments);

				moduleInfo.moduleName = moduleName;
				moduleInfo.functionName = materialName;
				moduleInfo.body = materialFunction.expression;
			}

			mdl::createNewFunctionInModule(moduleInfo);
		}

	}

}

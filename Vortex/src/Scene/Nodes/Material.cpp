#include "Material.h"

#include "MDL/CudaLinker.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "MDL/ShaderVisitor.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"
#include "Shader/BsdfMeasurement.h"
#include "Shader/LightProfile.h"
#include "Shader/Texture.h"

namespace vtx::graph
{
	void Material::init()
	{
		mdl::compileMaterial(materialGraph->name, getMaterialDbName());

		config = mdl::determineShaderConfiguration(getMaterialDbName());
		targetCode = mdl::createTargetCode(getMaterialDbName(), config, getID());

		mdl::getMdlCudaLinker().submitTargetCode(targetCode, name);

		createPrograms();
		textures = mdl::createTextureResources(targetCode);
		bsdfMeasurements = mdl::createBsdfMeasurementResources(targetCode);
		lightProfiles = mdl::createLightProfileResources(targetCode);

		dispatchParameters(mdl::getArgumentBlockData(getMaterialDbName(), materialGraph->functionInfo.signature, targetCode, argBlock, mapEnumTypes));

		isUpdated = true;
		isInitialized = true;
	}

	void Material::dispatchParameters(std::vector<shader::ParameterInfo> params)
	{
		for(auto param : params)
		{
			std::string& argumentName = param.argumentName;

			std::vector<std::string> potentialShaderGraphPath = utl::splitString(argumentName, ".");

			shader::ParameterInfo* parameterInfo = nullptr;
			std::shared_ptr<graph::shader::ShaderNode> shaderNode = materialGraph;
			bool foundSocket = true;

			// basically we are traversing the shader graph to find the socket that matches the argument name
			for( auto& shaderSocketName : potentialShaderGraphPath)
			{
				if (shaderNode->sockets.count(shaderSocketName) > 0) {
					// shaderSocketName is in the map.
					auto& socket = shaderNode->sockets[shaderSocketName];
					parameterInfo = &socket.parameterInfo;
					if (socket.node)
					{
						shaderNode = socket.node;
					}
				}
				else
				{
					foundSocket = false;
					break;
				}
			}
			// if we found the socket that matches the argument name, we can set the parameter info
			if(foundSocket)
			{
				parameterInfo->kind = param.kind;
				parameterInfo->arrayElemKind = param.arrayElemKind;
				parameterInfo->arraySize = param.arraySize;
				parameterInfo->arrayPitch = param.arrayPitch;
				parameterInfo->dataPtr = param.dataPtr;
				parameterInfo->enumInfo = param.enumInfo;
			}
		}
	}

	std::string Material::getMaterialDbName()
	{
		if(materialDbName.empty())
		{
			name = materialGraph->name;
			materialDbName = "compiledMaterial_" + name + "_" + std::to_string(getID());

		}
		return materialDbName;
	}

	FunctionNames& Material::getFunctionNames(const bool cuda)
	{
		const std::string suffix = std::to_string(getID());
		fNames                   = FunctionNames(suffix, cuda);

		return fNames;
	}

	std::vector<std::shared_ptr<Node>> Material::getChildren() const
	{
		std::vector<std::shared_ptr<Node>> children;
		if (materialGraph)
		{
			children.push_back(materialGraph);
		}
		for (const auto& texture : textures)
		{
			children.push_back(texture);
		}
		for (const auto& lightProfile : lightProfiles)
		{
			children.push_back(lightProfile);
		}
		for (const auto& bsdfMeasurement : bsdfMeasurements)
		{
			children.push_back(bsdfMeasurement);
		}
		return children;
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

	//void Material::traverseChildren(NodeVisitor& visitor)
	//{
	//	//if(materialGraph)
	//	//{
	//	//	if(materialGraph->isUpdated)
	//	//	{
	//	//		auto materialCasted = std::dynamic_pointer_cast<graph::shader::Material>(materialGraph);
	//	//		//mdl::createShaderGraphFunctionCalls(materialGraph);
	//	//	}
	//	//}
	//	if (materialGraph)
	//	{
	//		materialGraph->traverse(visitor);
	//	}
	//
	//	if(!isInitialized)
	//	{
	//		init();
	//	}
	//	for (const auto& texture : textures)
	//	{
	//		texture->traverse(visitor);
	//	}
	//	for (const auto& lightProfile : lightProfiles)
	//	{
	//		lightProfile->traverse(visitor);
	//	}
	//	for (const auto& bsdfMeasurement : bsdfMeasurements)
	//	{
	//		bsdfMeasurement->traverse(visitor);
	//	}
	//}

	void Material:: accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Material>());
	}

	void Material::createPrograms()
	{
		const auto module = std::make_shared<optix::ModuleOptix>();
		module->name = name;
		module->code = utl::replaceFunctionNameInPTX(targetCode->get_code(), "__replace__EvaluateMaterial", "unused_" + std::to_string(getID())); ;
		if (getOptions()->mdlCallType == MDL_DIRECT_CALL)
		{
			const auto fNames = graph::FunctionNames(std::to_string(getID()));

			devicePrograms.pgInit = optix::createDcProgram(module, fNames.init);

			if (!config.isThinWalledConstant)
			{
				devicePrograms.pgThinWalled = optix::createDcProgram(module, (fNames.thinWalled));
			}

			if (config.isSurfaceBsdfValid)
			{
				devicePrograms.pgSurfaceScatteringSample = optix::createDcProgram(module, fNames.surfaceScattering + "_sample");
				devicePrograms.pgSurfaceScatteringEval = optix::createDcProgram(module, fNames.surfaceScattering + "_evaluate");
				devicePrograms.pgSurfaceScatteringAuxiliary = optix::createDcProgram(module, fNames.surfaceScattering + "_auxiliary");
			}

			if (config.isBackfaceBsdfValid)
			{
				devicePrograms.pgBackfaceScatteringSample = optix::createDcProgram(module, fNames.backfaceScattering + "_sample");
				devicePrograms.pgBackfaceScatteringEval = optix::createDcProgram(module, fNames.backfaceScattering + "_evaluate");
				devicePrograms.pgBackfaceScatteringAuxiliary = optix::createDcProgram(module, fNames.backfaceScattering + "_auxiliary");
			}

			if (config.isSurfaceEdfValid)
			{
				devicePrograms.pgSurfaceEmissionEval = optix::createDcProgram(module, fNames.surfaceEmissionEmission + "_evaluate");

				if (!config.isSurfaceIntensityConstant)
				{
					devicePrograms.pgSurfaceEmissionIntensity = optix::createDcProgram(module, fNames.surfaceEmissionIntensity);
				}

				if (!config.isSurfaceIntensityModeConstant)
				{
					devicePrograms.pgSurfaceEmissionIntensityMode = optix::createDcProgram(module, fNames.surfaceEmissionMode);
				}
			}

			if (config.isBackfaceEdfValid)
			{
				if (config.useBackfaceEdf)
				{
					devicePrograms.pgBackfaceEmissionEval = optix::createDcProgram(module, fNames.backfaceEmissionEmission + "_evaluate");
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionEval = devicePrograms.pgSurfaceEmissionEval;
				}

				if (config.useBackfaceIntensity)
				{
					if (!config.isBackfaceIntensityConstant)
					{
						devicePrograms.pgBackfaceEmissionIntensity = optix::createDcProgram(module, fNames.backfaceEmissionIntensity);
					}
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionIntensity = devicePrograms.pgSurfaceEmissionIntensity;
				}

				if (config.useBackfaceIntensityMode)
				{
					if (!config.isBackfaceIntensityModeConstant)
					{
						devicePrograms.pgBackfaceEmissionIntensityMode = optix::createDcProgram(module, fNames.backfaceEmissionMode);
					}
				}
				else // Surface and backface expressions were identical. Reuse the code of the surface expression.
				{
					devicePrograms.pgBackfaceEmissionIntensityMode = devicePrograms.pgSurfaceEmissionIntensityMode;
				}
			}

			if (!config.isIorConstant)
			{
				devicePrograms.pgIor = optix::createDcProgram(module, fNames.ior);
			}

			if (!config.isAbsorptionCoefficientConstant)
			{
				devicePrograms.pgVolumeAbsorptionCoefficient = optix::createDcProgram(module, fNames.volumeAbsorptionCoefficient);
			}

			if (config.isVdfValid)
			{
				// The MDL SDK doesn't generate code for the volume.scattering expression.
				// Means volume scattering must be implemented by the renderer and only the parameter expresssions can be generated.

				// The volume scattering coefficient and direction bias are only used when there is a valid VDF. 
				if (!config.isScatteringCoefficientConstant)
				{
					devicePrograms.pgVolumeScatteringCoefficient = optix::createDcProgram(module, fNames.volumeScatteringCoefficient);
				}

				if (!config.isDirectionalBiasConstant)
				{
					devicePrograms.pgVolumeDirectionalBias = optix::createDcProgram(module, fNames.volumeDirectionalBias);
				}

				// volume.scattering.emission_intensity not implemented.
			}

			if (config.useCutoutOpacity)
			{
				//devicePrograms.hasOpacity = true;

				if (!config.isCutoutOpacityConstant)
				{
					devicePrograms.pgGeometryCutoutOpacity = optix::createDcProgram(module, fNames.geometryCutoutOpacity);
				}
			}

			if (config.isHairBsdfValid)
			{
				devicePrograms.pgHairSample = optix::createDcProgram(module, (fNames.hairBsdf + "_sample"));
				devicePrograms.pgHairEval = optix::createDcProgram(module, (fNames.hairBsdf + "_evaluate"));
			}
		}
		else if(getOptions()->mdlCallType== MDL_INLINE || getOptions()->mdlCallType == MDL_CUDA)
		{
			devicePrograms.pgEvaluateMaterial = optix::createDcProgram(module, "__direct_callable__EvaluateMaterial", getID(), {"Default", "wfShade", "wfRadianceTrace"});
			mdl::getMdlCudaLinker().submitTargetCode(targetCode, name);
		}
	}

	const Configuration& Material::getConfiguration()
	{
		if (!isInitialized)
		{
			init();
		}
		return config;
	}


	const DevicePrograms& Material::getPrograms()
	{
		if (!isInitialized)
		{
			init();
		}
		return devicePrograms;
	}


	std::vector<std::shared_ptr<graph::Texture>>  Material::getTextures()
	{
		if (!isInitialized)
		{
			init();
		}
		return textures;
	}

	std::vector<std::shared_ptr<graph::BsdfMeasurement>> Material::getBsdfs()
	{
		if (!isInitialized)
		{
			init();
		}
		return bsdfMeasurements;
	}

	std::vector<std::shared_ptr<graph::LightProfile>> Material::getLightProfiles()
	{
		if (!isInitialized)
		{
			init();
		}
		return lightProfiles;
	}

	bool Material::isThinWalled()
	{
		if (!config.isThinWalledConstant || (config.isThinWalledConstant && config.isThinWalled)) {
			return true;
		}
		return false;
	}

	bool Material::useEmission()
	{
		if (!isInitialized)
		{
			init();
		}
		const bool thinWalled = isThinWalled();

		bool isSurfaceEmissive = false;
		if (config.isSurfaceEdfValid) {
			if (!config.isSurfaceIntensityConstant) {
				isSurfaceEmissive = true;
			}
			else if (config.surfaceIntensity != math::vec3f(0.0f)) {
				isSurfaceEmissive = true;
			}
		}


		bool isBackfaceEmissive = false;
		if (config.isBackfaceEdfValid) {
			if (!config.isBackfaceIntensityConstant) {
				isBackfaceEmissive = true;
			}
			else if (config.backfaceIntensity != math::vec3f(0.0f)) {
				isBackfaceEmissive = true;
			}
		}
		isBackfaceEmissive    = (thinWalled && isBackfaceEmissive);
		const bool isEmissive = isSurfaceEmissive || isBackfaceEmissive; // To be emissive on the backface it needs to be isThinWalled

		if (config.emissivityToggle) //The material has a enable emission option
		{
			if (*(config.emissivityToggle))
			{
				return isEmissive;
			}
			return false;
		}
		return isEmissive;
	}

	bool Material::useOpacity()
	{
		if (!isInitialized)
		{
			init();
		}
		if (config.opacityToggle) //The material has a enable emission option
		{
			if (*(config.opacityToggle))
			{
				return config.useCutoutOpacity;
			}
			return false;
		}
		return config.useCutoutOpacity;
	};

	void processMaterial(std::shared_ptr<Material>& material)
	{
		if (!material->isInitialized)
		{
			mdl::ShaderVisitor visitor;
			material->materialGraph->traverse(visitor);
			mdl::compileMaterial(material->materialGraph->name, material->getMaterialDbName());
			material->config = mdl::determineShaderConfiguration(material->getMaterialDbName());

			Handle<ILink_unit> linkUnit;
			mdl::addMaterialToLinkUnit(material->getMaterialDbName(), material->config, material->getID(), linkUnit);
			VTX_INFO("Creating target code for shader {}", material->getMaterialDbName());
			Handle<ITarget_code const> targetCode = mdl::generateTargetCode(linkUnit);

			material->textures = mdl::createTextureResources(targetCode);
			material->bsdfMeasurements = mdl::createBsdfMeasurementResources(targetCode);
			material->lightProfiles = mdl::createLightProfileResources(targetCode);
			material->targetCode = targetCode;
			material->createPrograms();
			material->dispatchParameters(mdl::getArgumentBlockData(material->getMaterialDbName(), material->materialGraph->functionInfo.signature, material->targetCode, material->argBlock, material->mapEnumTypes, 0));
			material->isInitialized = true;
			material->isUpdated = true;

		}
	}

	void computeMaterialsMultiThreadCode()
	{
		std::vector<std::shared_ptr<Material>> materials = SIM::get()->getAllNodeOfType<graph::Material>(NT_MATERIAL);

		std::vector<std::thread> threads;

		for (std::shared_ptr<Material>& material : materials)
		{
			threads.push_back(std::thread(processMaterial, std::ref(material)));
		}

		for (std::thread& thread : threads)
		{
			if (thread.joinable())
			{
				thread.join();
			}
		}
	}
}


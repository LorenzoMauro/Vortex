#include "Shader.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"
#include "ShaderFlags.h"

namespace vtx::graph
{

	const char* Shader::getTargetCode()
	{
		if(!isInitialized)
		{
			init();
		}
		return targetCode->get_code();
	}

	Handle<ITarget_code const> Shader::getTargetCodeHandle()
	{
		if (!isInitialized)
		{
			init();
		}
		return targetCode;
	}

	void Shader::init()
	{
		mdl::compileMaterial(path, name, &materialDbName);
		config = mdl::determineShaderConfiguration(materialDbName);
		targetCode = mdl::createTargetCode(materialDbName, config, getID());
		createPrograms();
		createChildResources();
		isInitialized = true;
	}

	void Shader::createChildResources()
	{
		// TODO We have to store the textures, light profiles and bsdf indices since these will be refencered by the mdl lookup functions
		for (Size i = 1, n = targetCode->get_texture_count(); i < n; ++i) {
			auto texture = std::make_shared<graph::Texture>(targetCode->get_texture(i), targetCode->get_texture_shape(i));
			texture->mdlIndex = i;
			textures.emplace_back(texture);
		}

		if (targetCode->get_light_profile_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_light_profile_count(); i < n; ++i)
			{
				auto lightProfile = std::make_shared<graph::LightProfile>(targetCode->get_light_profile(i));
				lightProfile->mdlIndex = i;
				lightProfiles.emplace_back(lightProfile);
			}
		}

		if (targetCode->get_bsdf_measurement_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_bsdf_measurement_count(); i < n; ++i)
			{
				auto bsdfMeasurement = std::make_shared<graph::BsdfMeasurement>(targetCode->get_bsdf_measurement(i));
				bsdfMeasurement->mdlIndex = i;
				bsdfMeasurements.emplace_back(bsdfMeasurement);
			}
		}
	}

	const Shader::Configuration& Shader::getConfiguration()
	{
		if (!isInitialized)
		{
			init();
		}
		return config;
	}

	void Shader::createPrograms()
	{
		auto module = std::make_shared<optix::ModuleOptix>();
		module->name = name;
		module->code = targetCode->get_code();

		const auto fNames = graph::Shader::FunctionNames(std::to_string(getID()));

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

		if (config.isEmissive)
		{
			devicePrograms.isEmissive = true;
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
			devicePrograms.hasOpacity = true;

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

	const Shader::DevicePrograms& Shader::getPrograms()
	{
		if (!isInitialized)
		{
			init();
		}
		return devicePrograms;
	}

	void Shader::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if (!isInitialized)
		{
			init();
		}
		for (auto& texture: textures)
		{
			texture->traverse(orderedVisitors);
		}
		for (auto& lightProfile : lightProfiles)
		{
			lightProfile->traverse(orderedVisitors);
		}
		for (auto& bsdfMeasurement : bsdfMeasurements)
		{
			bsdfMeasurement->traverse(orderedVisitors);
		}
		ACCEPT(orderedVisitors);
	}

	void Shader::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Shader>());
	}

	std::string Shader::getMaterialDbName()
	{
		if (!isInitialized)
		{
			init();
		}
		return materialDbName;
	}

	bool Shader::isEmissive()
	{
		if(!isInitialized)
		{
			init();
		}
		return config.isEmissive;
	}

	std::vector<std::shared_ptr<graph::Texture>>  Shader::getTextures()
	{
		if(!isInitialized)
		{
			init();
		}
		return textures;
	}

	std::vector<std::shared_ptr<graph::BsdfMeasurement>> Shader::getBsdfs()
	{
		if (!isInitialized)
		{
			init();
		}
		return bsdfMeasurements;
	}

	std::vector<std::shared_ptr<graph::LightProfile>> Shader::getLightProfiles()
	{
		if (!isInitialized)
		{
			init();
		}
		return lightProfiles;
	}

}

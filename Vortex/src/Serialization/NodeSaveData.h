#pragma once
#include <string>
#include "Scene/Graph.h"

namespace vtx::serializer
{
	struct BaseNodeSaveData
	{
		BaseNodeSaveData() = default;

		BaseNodeSaveData(const std::shared_ptr<graph::Node>& node)
			: UID(node->getUID())
			, TID(node->getTypeID())
			, name(node->name)
			, type(node->getType())
		{}

		vtxID           UID;
		vtxID           TID;
		std::string     name;
		graph::NodeType type;
	};

	struct MeshNodeSaveData
	{
		MeshNodeSaveData() = default;
		MeshNodeSaveData(const std::shared_ptr<graph::Mesh>& node)
			: base(node)
			, vertices(node->vertices)
			, indices(node->indices)
			, faceAttributes(node->faceAttributes)
			, status(node->status)
		{}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode = ops::createNode<graph::Mesh>();
			oldToNewUIDMap[base.UID] = restoredNode->getUID();
			restoredNode->name = base.name;
			restoredNode->vertices = vertices;
			restoredNode->indices = indices;
			restoredNode->faceAttributes = faceAttributes;
			restoredNode->status = status;
		}

		BaseNodeSaveData base;
		std::vector<graph::VertexAttributes> vertices;
		std::vector<vtxID>                   indices;
		std::vector<graph::FaceAttributes>   faceAttributes;
		graph::MeshStatus				     status;

		std::shared_ptr<graph::Mesh> restoredNode = nullptr;
	};

	struct TransformNodeSaveData
	{
		TransformNodeSaveData() = default;
		TransformNodeSaveData(const std::shared_ptr<graph::Transform>& node)
			: base(node)
			, affineTransform(node->affineTransform)
		{}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode = ops::createNode<graph::Transform>();
			oldToNewUIDMap[base.UID] = restoredNode->getUID();
			restoredNode->name = base.name;
			restoredNode->affineTransform = affineTransform;
		}

		BaseNodeSaveData base;
		math::affine3f   affineTransform;

		std::shared_ptr<graph::Transform> restoredNode = nullptr;
	};

	struct MaterialNodeSaveData
	{
		MaterialNodeSaveData() = default;
		MaterialNodeSaveData(const std::shared_ptr<graph::Material>& node)
			: base(node),
			materialGraphUID(node->materialGraph->getUID()),
			materialDbName(node->materialDbName),
			path(node->path),
			materialCallName(node->materialCallName),
			useAsLight(node->useAsLight)
		{}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode = ops::createNode<graph::Material>();
			oldToNewUIDMap[base.UID] = restoredNode->getUID();
			restoredNode->name = base.name;
			restoredNode->materialDbName = materialDbName;
			restoredNode->path = path;
			restoredNode->materialCallName = materialCallName;
			restoredNode->useAsLight = useAsLight;
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode->materialGraph = graph::Scene::getSim()->getNode<graph::shader::ShaderNode>(oldToNewUIDMap[materialGraphUID]);
		}

		BaseNodeSaveData base;
		vtxID            materialGraphUID;
		std::string      materialDbName;
		std::string      path;
		std::string      materialCallName;
		bool             useAsLight;

		std::shared_ptr<graph::Material> restoredNode = nullptr;
	};

	struct MaterialSlotSaveData
	{
		MaterialSlotSaveData() = default;
		MaterialSlotSaveData(const graph::MaterialSlot& materialSlot)
			:
			materialUID(materialSlot.material->getUID()),
			slotIndex(materialSlot.slotIndex)
		{}

		vtxID materialUID;
		int   slotIndex;
	};

	struct InstanceNodeSaveData
	{
		InstanceNodeSaveData() = default;
		InstanceNodeSaveData(const std::shared_ptr<graph::Instance>& node)
			: base(node),
			transformUID(node->transform->getUID()),
			childUID(node->getChild()->getUID())
		{
			for (const graph::MaterialSlot& materialSlot : node->getMaterialSlots())
			{
				materialSlots.emplace_back(materialSlot);
			}
		}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode = ops::createNode<graph::Instance>();
			oldToNewUIDMap[base.UID] = restoreNode->getUID();
			restoreNode->name = base.name;
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode->transform = graph::Scene::getSim()->getNode<graph::Transform>(oldToNewUIDMap[transformUID]);
			restoreNode->setChild(graph::Scene::getSim()->getNode<graph::Node>(oldToNewUIDMap[childUID]));
			for (MaterialSlotSaveData& materialSlot : materialSlots)
			{
				restoreNode->addMaterial(graph::Scene::getSim()->getNode<graph::Material>(oldToNewUIDMap[materialSlot.materialUID]), materialSlot.slotIndex);
			}
		}

		BaseNodeSaveData base;
		vtxID            transformUID;
		vtxID            childUID;
		std::vector<MaterialSlotSaveData> materialSlots;

		std::shared_ptr<graph::Instance> restoreNode = nullptr;
	};

	struct GroupNodeSaveData
	{
		GroupNodeSaveData() = default;

		GroupNodeSaveData(const std::shared_ptr<graph::Group>& node)
			: base(node),
			transformUID(node->transform->getUID())
		{
			for (const std::shared_ptr<graph::Node>& child : node->getChildren())
			{
				childUIDs.emplace_back(child->getUID());
			}
		}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode = ops::createNode<graph::Group>();
			oldToNewUIDMap[base.UID] = restoreNode->getUID();
			restoreNode->name = base.name;
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode->transform = graph::Scene::getSim()->getNode<graph::Transform>(oldToNewUIDMap[transformUID]);
			for (vtxID childUID : childUIDs)
			{
				restoreNode->addChild(graph::Scene::getSim()->getNode<graph::Node>(oldToNewUIDMap[childUID]));
			}
		}

		BaseNodeSaveData base;
		vtxID            transformUID;
		std::vector<vtxID> childUIDs;

		std::shared_ptr<graph::Group> restoreNode = nullptr;
	};

	struct EnvironmentLightNodeSaveData
	{
		EnvironmentLightNodeSaveData() = default;
		EnvironmentLightNodeSaveData(const std::shared_ptr<graph::EnvironmentLight>& node)
			: base(node),
			transformUID(node->transform->getUID()),
			texturePath(node->envTexture->filePath)
		{}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode = ops::createNode<graph::EnvironmentLight>();
			oldToNewUIDMap[base.UID] = restoreNode->getUID();
			restoreNode->name = base.name;

			restoreNode->envTexture = ops::createNode<graph::Texture>(texturePath);
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode->transform = graph::Scene::getSim()->getNode<graph::Transform>(oldToNewUIDMap[transformUID]);
		}

		BaseNodeSaveData base;
		vtxID            transformUID;
		std::string      texturePath;

		std::shared_ptr<graph::EnvironmentLight> restoreNode = nullptr;

	};

	struct CameraNodeSaveData
	{
		CameraNodeSaveData() = default;
		CameraNodeSaveData(const std::shared_ptr<graph::Camera>& node)
			: base(node),
			transformUID(node->transform->getUID()),
			fov(node->fovY)
		{}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode = ops::createNode<graph::Camera>();
			oldToNewUIDMap[base.UID] = restoreNode->getUID();
			restoreNode->name = base.name;
			restoreNode->fovY = fov;
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoreNode->transform = graph::Scene::getSim()->getNode<graph::Transform>(oldToNewUIDMap[transformUID]);
			restoreNode->updateDirections();
		}

		BaseNodeSaveData base;
		vtxID            transformUID;
		float            fov;
		std::shared_ptr<graph::Camera> restoreNode = nullptr;
	};

	struct ExperimentSaveData
	{
		ExperimentSaveData() = default;
		ExperimentSaveData(const Experiment& experiment)
			: name(experiment.name)
			, mape(experiment.mape)
			, rendererSettings(experiment.rendererSettings)
			, networkSettings(experiment.networkSettings)
			, wavefrontSettings(experiment.wavefrontSettings)
		{}

		Experiment restore() const
		{
			Experiment experiment;
			experiment.name = name;
			experiment.mape = mape;
			experiment.rendererSettings = rendererSettings;
			experiment.networkSettings = networkSettings;
			experiment.wavefrontSettings = wavefrontSettings;
			return experiment;
		}

		std::string              name;
		std::vector<float>       mape;
		RendererSettings         rendererSettings;
		network::NetworkSettings networkSettings;
		WavefrontSettings        wavefrontSettings;
	};

	struct ExperimentManagerSaveData
	{
		ExperimentManagerSaveData() = default;
		ExperimentManagerSaveData(ExperimentsManager& experimentManager)
			: currentExperiment(experimentManager.currentExperiment)
			, currentExperimentStep(experimentManager.currentExperimentStep)
			, width(experimentManager.width)
			, height(experimentManager.height)
			, isGroundTruthReady(experimentManager.isGroundTruthReady)
			, maxSamples(experimentManager.maxSamples)
		{
			for (const Experiment& experiment : experimentManager.experiments)
			{
				experiments.emplace_back(experiment);
			}
			if(experimentManager.isGroundTruthReady)
			{
				groundTruthImage = std::vector<math::vec3f>(experimentManager.width * experimentManager.height);
				experimentManager.groundTruthBuffer.download(groundTruthImage.data());
			}
			else
			{
				groundTruthImage = {{0.0f, 0.0f, 0.0f}};
			}
		}

		ExperimentsManager restore() const
		{
			ExperimentsManager experimentManager;
			experimentManager.currentExperiment = currentExperiment;
			experimentManager.currentExperimentStep = currentExperimentStep;
			experimentManager.width = width;
			experimentManager.height = height;
			experimentManager.isGroundTruthReady = isGroundTruthReady;
			experimentManager.maxSamples = maxSamples;
			for (const ExperimentSaveData& experiment : experiments)
			{
				experimentManager.experiments.emplace_back(experiment.restore());
			}
			if(experimentManager.isGroundTruthReady)
			{
				experimentManager.groundTruthBuffer.upload(groundTruthImage.data());
			}
			return experimentManager;
		}

		int currentExperiment;
		int currentExperimentStep;
		int width;
		int height;
		bool isGroundTruthReady;
		int maxSamples;
		std::vector<ExperimentSaveData> experiments;
		std::vector<math::vec3f> groundTruthImage;
	};

	struct RendererNodeSaveData
	{
		RendererNodeSaveData() = default;
		RendererNodeSaveData(const std::shared_ptr<graph::Renderer>& node)
			: base(node),
			rendererSettings(node->settings),
			wavefrontSettings(node->waveFrontIntegrator.settings),
			networkSettings(node->waveFrontIntegrator.network.settings),
			cameraUID(node->camera->getUID()),
			sceneRootUID(node->sceneRoot->getUID()),
			environmentLightUID(0),
			experimentManagerSaveData(node->waveFrontIntegrator.network.experimentManager)
		{
			if(node->environmentLight)
			{
				environmentLightUID = node->environmentLight->getUID();
			}
		}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode = ops::createNode<graph::Renderer>();
			oldToNewUIDMap[base.UID] = restoredNode->getUID();
			restoredNode->name = base.name;
			restoredNode->settings = rendererSettings;
			restoredNode->waveFrontIntegrator.settings = wavefrontSettings;
			restoredNode->waveFrontIntegrator.network.settings = networkSettings;
			restoredNode->waveFrontIntegrator.network.experimentManager = experimentManagerSaveData.restore();
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			restoredNode->camera = graph::Scene::getSim()->getNode<graph::Camera>(oldToNewUIDMap[cameraUID]);
			restoredNode->sceneRoot = graph::Scene::getSim()->getNode<graph::Group>(oldToNewUIDMap[sceneRootUID]);
			if(environmentLightUID != 0)
			{
				restoredNode->environmentLight = graph::Scene::getSim()->getNode<graph::EnvironmentLight>(oldToNewUIDMap[environmentLightUID]);
			}
		}

		BaseNodeSaveData         base;
		RendererSettings         rendererSettings;
		WavefrontSettings        wavefrontSettings;
		network::NetworkSettings networkSettings;
		vtxID                    cameraUID;
		vtxID                    sceneRootUID;
		vtxID                    environmentLightUID;
		ExperimentManagerSaveData experimentManagerSaveData;

		std::shared_ptr<graph::Renderer> restoredNode;
	};

	struct ShaderSocketSaveData
	{
		ShaderSocketSaveData() = default;

		ShaderSocketSaveData(const graph::shader::ShaderNodeSocket& shaderNodeSocket, std::string socketName)
			: socketNodeInputUID(0)
			, socketName(std::move(socketName))
			, paramKind(shaderNodeSocket.parameterInfo.kind)
			, arrayParamKind(shaderNodeSocket.parameterInfo.arrayElemKind)
			, arraySize(shaderNodeSocket.parameterInfo.arraySize)
		{
			if (shaderNodeSocket.node)
			{
				socketNodeInputUID = shaderNodeSocket.node->getUID();
			}

			const graph::shader::ParameterInfo paramInfo = shaderNodeSocket.parameterInfo;
			switch (paramKind)
			{
			case graph::shader::PK_FLOAT:
			{
				dataBuffer.resize(sizeof(float));
				const float data = paramInfo.data<float>();
				memcpy(dataBuffer.data(), &data, sizeof(float));
			}
			break;
			case graph::shader::PK_FLOAT2:
			{
				dataBuffer.resize(sizeof(float) * 2);
				const float* ptr = &paramInfo.data<float>();
				memcpy(dataBuffer.data(), ptr, sizeof(float) * 2);
			}
			break;
			case graph::shader::PK_FLOAT3:
			{
				dataBuffer.resize(sizeof(float) * 3);
				const float* ptr = &paramInfo.data<float>();
				memcpy(dataBuffer.data(), ptr, sizeof(float) * 3);
			}
			break;
			case graph::shader::PK_COLOR:
			{
				dataBuffer.resize(sizeof(float) * 3);
				const float* ptr = &paramInfo.data<float>();
				memcpy(dataBuffer.data(), ptr, sizeof(float) * 3);
			}
			break;
			case graph::shader::PK_BOOL:
			{
				dataBuffer.resize(sizeof(bool));
				const bool data = paramInfo.data<bool>();
				memcpy(dataBuffer.data(), &data, sizeof(bool));
			}
			break;
			case graph::shader::PK_INT:
			{
				dataBuffer.resize(sizeof(int));
				const int data = paramInfo.data<int>();
				memcpy(dataBuffer.data(), &data, sizeof(int));
			}
			break;
			case graph::shader::PK_ARRAY:
			{
				YAML::Node arrayNode;
				const char* ptr = &paramInfo.data<char>();
				const uint64_t arrayPitch = paramInfo.arrayPitch;

				if(arrayParamKind == graph::shader::PK_FLOAT ||
					arrayParamKind == graph::shader::PK_BOOL ||
					arrayParamKind == graph::shader::PK_INT )
				{
					arrayElemSize = 1;
				}
				else if (arrayParamKind == graph::shader::PK_FLOAT2 )
				{
					arrayElemSize = 2;
				}
				else if (arrayParamKind == graph::shader::PK_FLOAT3 ||
					arrayParamKind == graph::shader::PK_COLOR )
				{
					arrayElemSize = 3;
				}

				if (arrayParamKind == graph::shader::PK_FLOAT ||
					arrayParamKind == graph::shader::PK_FLOAT2 ||
					arrayParamKind == graph::shader::PK_FLOAT3 ||
					arrayParamKind == graph::shader::PK_COLOR)
				{
					std::vector<float> vec = getArray<float>(arraySize, arrayElemSize, ptr, arrayPitch);
					dataBuffer.resize(arraySize * arrayElemSize * sizeof(float));
					memcpy(dataBuffer.data(), vec.data(), arraySize * arrayElemSize * sizeof(float));
				}
				else if (arrayParamKind == graph::shader::PK_BOOL)
				{
					std::vector<bool> vec = getArray<bool>(arraySize, arrayElemSize, ptr, arrayPitch);
					for(size_t i=0; i<vec.size(); ++i)
					{
						dataBuffer.push_back((char)vec[i]);
					}
				}
				else if (arrayParamKind == graph::shader::PK_INT)
				{
					std::vector<int> vec = getArray<int>(arraySize, arrayElemSize, ptr, arrayPitch);
					dataBuffer.resize(arraySize * arrayElemSize * sizeof(float));
					memcpy(dataBuffer.data(), vec.data(), arraySize * arrayElemSize * sizeof(float));
				}
				break;
			}
			case graph::shader::PK_TEXTURE:
				{
					const std::string path = mdl::getTexturePathFromExpr(shaderNodeSocket.directExpression);
					dataBuffer.resize(path.size());
					memcpy(dataBuffer.data(), path.data(), path.size());
					break;
				}
			default:
			{
				VTX_INFO("Socket kind not supported: {}", (int)paramKind);
			}
			break;
			}
		}

		mi::base::Handle<mi::neuraylib::IExpression> restoreDirectExpression()
		{
			switch (paramKind)
			{
			case graph::shader::PK_FLOAT:
			{
				const float data = *reinterpret_cast<float*>(dataBuffer.data());
				return mdl::createConstantFloat(data);
			}
			case graph::shader::PK_COLOR:
			{
				const float* ptr = reinterpret_cast<float*>(dataBuffer.data());
				const math::vec3f color = { ptr[0], ptr[1], ptr[2] };
				return mdl::createConstantColor(color);
			}
			case graph::shader::PK_INT:
			{
				const int data = *reinterpret_cast<int*>(dataBuffer.data());
				return mdl::createConstantInt(data);
			}
			case graph::shader::PK_UNKNOWN:
			case graph::shader::PK_FLOAT2:
			case graph::shader::PK_FLOAT3:
			case graph::shader::PK_ARRAY:
			case graph::shader::PK_BOOL:
			case graph::shader::PK_ENUM:
			case graph::shader::PK_STRING:
			case graph::shader::PK_TEXTURE:
			case graph::shader::PK_LIGHT_PROFILE:
			case graph::shader::PK_BSDF_MEASUREMENT:
			default:
			{
				return mi::base::Handle<mi::neuraylib::IExpression>(nullptr);
			}
			}
		}

		vtxID socketNodeInputUID;
		std::string socketName;
		std::vector<char> dataBuffer;
		graph::shader::ParamKind paramKind;
		graph::shader::ParamKind arrayParamKind;
		uint64_t arraySize;
		int arrayElemSize;

		template<typename T>
		std::vector<T> getArray(uint64_t _arraySize, uint64_t _arrayElemSize,const char* ptr, uint64_t _arrayPitch)
		{
			std::vector<T> vec;
			for (uint64_t i = 0; i < _arraySize; ++i)
			{
				for (uint64_t j = 0; j < _arrayElemSize; ++j)
				{
					const T* fPtr = reinterpret_cast<const T*>(ptr) + j;
					vec.push_back(*fPtr);
				}
				ptr += _arrayPitch;
			}

			return vec;
		}
	};

	struct BaseShaderNodeSaveData
	{
		BaseShaderNodeSaveData() = default;
		BaseShaderNodeSaveData(const std::shared_ptr<graph::shader::ShaderNode>& node)
			: base(node),
			functionInfo(node->functionInfo)
		{
			for (const auto& [socketName, socket] : node->sockets)
			{
				ShaderSocketSaveData socketSaveData(socket, socketName);
				shaderSocketSaveData.insert({ socketName, socketSaveData });
			}
			const graph::NodeType type = node->getType();
			if (type == graph::NT_SHADER_COLOR_TEXTURE)
			{
				texturePath = node->as<graph::shader::ColorTexture>()->texturePath;
			}
			else if (type == graph::NT_SHADER_MONO_TEXTURE)
			{
				texturePath = node->as<graph::shader::MonoTexture>()->texturePath;
			}
			else if(type == graph::NT_SHADER_BUMP_TEXTURE)
			{
				texturePath = node->as<graph::shader::BumpTexture>()->texturePath;
			}
			else if (type == graph::NT_SHADER_NORMAL_TEXTURE)
			{
				texturePath = node->as<graph::shader::NormalTexture>()->texturePath;
			}
			else if (type == graph::NT_GET_CHANNEL)
			{
				channel = node->as<graph::shader::GetChannel>()->channel;
			}
		}

		void restore(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			switch(base.type)
			{
			case graph::NT_SHADER_DF: 
			{
				const std::shared_ptr<graph::shader::DiffuseReflection> node = ops::createNode<graph::shader::DiffuseReflection>();
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_MATERIAL: 
			{
				const std::shared_ptr<graph::shader::Material> node = ops::createNode<graph::shader::Material>();
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_SURFACE: 
			{
				const std::shared_ptr<graph::shader::MaterialSurface> node = ops::createNode<graph::shader::MaterialSurface>();
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_IMPORTED: 
			{
				const std::shared_ptr<graph::shader::ImportedNode> node = ops::createNode<graph::shader::ImportedNode>(functionInfo);
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_COORDINATE: 
			{
				const std::shared_ptr<graph::shader::TextureTransform> node = ops::createNode<graph::shader::TextureTransform>();
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_NORMAL_TEXTURE: 
			{
				const std::shared_ptr<graph::shader::NormalTexture> node = ops::createNode<graph::shader::NormalTexture>(texturePath);
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_MONO_TEXTURE: 
			{
				const std::shared_ptr<graph::shader::MonoTexture> node = ops::createNode<graph::shader::MonoTexture>(texturePath);
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_COLOR_TEXTURE: 
			{
				const std::shared_ptr<graph::shader::ColorTexture> node = ops::createNode<graph::shader::ColorTexture>(texturePath);
				restoredNode = node;
				break;
			}
			case graph::NT_SHADER_BUMP_TEXTURE: 
			{
				const std::shared_ptr<graph::shader::BumpTexture> node = ops::createNode<graph::shader::BumpTexture>(texturePath);
				restoredNode = node;
				break;
			}
			case graph::NT_NORMAL_MIX: 
			{
				const std::shared_ptr<graph::shader::NormalMix> node = ops::createNode<graph::shader::NormalMix>();
				restoredNode = node;
				break;
			}
			case graph::NT_GET_CHANNEL: 
			{
				const std::shared_ptr<graph::shader::GetChannel> node = ops::createNode<graph::shader::GetChannel>(channel);
				restoredNode = node;
				break;
			}
			case graph::NT_PRINCIPLED_MATERIAL: 
			{
				const std::shared_ptr<graph::shader::PrincipledMaterial> node = ops::createNode<graph::shader::PrincipledMaterial>();
				restoredNode = node;
				break;
			}
			default: ;
			}
			restoredNode->name = base.name;
			oldToNewUIDMap[base.UID] = restoredNode->getUID();
			restoredNode->functionInfo = functionInfo;
			for (const auto& [socketName, socket]: restoredNode->sockets)
			{
				if (shaderSocketSaveData.find(socketName) != shaderSocketSaveData.end())
				{
					ShaderSocketSaveData& socketSaveData = shaderSocketSaveData[socketName];
					mi::base::Handle<mi::neuraylib::IExpression> expr = socketSaveData.restoreDirectExpression();
					if (expr.is_valid_interface())
					{
						restoredNode->setSocketValue(socketName, expr);
					}
				}
			}
		}

		void link(std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			const std::shared_ptr<graph::SceneIndexManager> sim = graph::Scene::getSim();
			for (const auto& [socketName, socket]: restoredNode->sockets)
			{
				if (shaderSocketSaveData.find(socketName) != shaderSocketSaveData.end())
				{
					ShaderSocketSaveData& socketSaveData = shaderSocketSaveData[socketName];
					if (socketSaveData.socketNodeInputUID != 0)
					{
						const vtxID newSocketNodeInputUID = oldToNewUIDMap[socketSaveData.socketNodeInputUID];
						restoredNode->connectInput(socketName, sim->getNode<graph::shader::ShaderNode>(newSocketNodeInputUID));
					}
				}
			}
		}
		BaseNodeSaveData base;
		std::map<std::string, ShaderSocketSaveData> shaderSocketSaveData;
		mdl::MdlFunctionInfo functionInfo;
		std::string texturePath;
		int channel = 0;

		std::shared_ptr<graph::shader::ShaderNode> restoredNode;
	};

	struct GraphSaveData
	{
		GraphSaveData() = default;

		void prepareSaveData()
		{
			VTX_INFO("Constructing GraphSaveData");
			prepareSaveDataByNodeType<graph::Transform>(graph::NT_TRANSFORM, transforms);
			prepareSaveDataByNodeType<graph::Mesh>(graph::NT_MESH, meshes);
			prepareSaveDataByNodeType<graph::Material>(graph::NT_MATERIAL, materials);
			prepareSaveDataByNodeType<graph::Instance>(graph::NT_INSTANCE, instances);
			prepareSaveDataByNodeType<graph::Group>(graph::NT_GROUP, groups);
			prepareSaveDataByNodeType<graph::EnvironmentLight>(graph::NT_ENV_LIGHT, environmentLights);
			prepareSaveDataByNodeType<graph::Camera>(graph::NT_CAMERA, cameras);
			prepareSaveDataByNodeType<graph::Renderer>(graph::NT_RENDERER, renderers);
			prepareSaveDataByNodeType<graph::shader::DiffuseReflection>(graph::NT_SHADER_DF, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::MaterialSurface>(graph::NT_SHADER_SURFACE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::Material>(graph::NT_SHADER_MATERIAL, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::ImportedNode>(graph::NT_SHADER_IMPORTED, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::PrincipledMaterial>(graph::NT_PRINCIPLED_MATERIAL, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::ColorTexture>(graph::NT_SHADER_COLOR_TEXTURE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::MonoTexture>(graph::NT_SHADER_MONO_TEXTURE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::NormalTexture>(graph::NT_SHADER_NORMAL_TEXTURE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::BumpTexture>(graph::NT_SHADER_BUMP_TEXTURE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::TextureTransform>(graph::NT_SHADER_COORDINATE, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::NormalMix>(graph::NT_NORMAL_MIX, shaderNodes);
			prepareSaveDataByNodeType<graph::shader::GetChannel>(graph::NT_GET_CHANNEL, shaderNodes);

			const graph::Scene* scene = graph::Scene::get();
			activeRendererUID = scene->renderer->getUID();
			activeCameraUID = scene->renderer->camera->getUID();
			sceneRootUID = scene->renderer->sceneRoot->getUID();
			VTX_INFO("Finished constructing GraphSaveData");
		}

		template<typename NodeType, typename SaveDataStruct>
		void prepareSaveDataByNodeType(graph::NodeType type, std::vector<SaveDataStruct>& saveDataStructs)
		{
			const std::vector<std::shared_ptr<NodeType>> allNodes = graph::Scene::getSim()->getAllNodeOfType<NodeType>(type);
			saveDataStructs.reserve(allNodes.size());
			for (const auto& node : allNodes)
			{
				SaveDataStruct saveData(node);
				saveDataStructs.push_back(saveData);
			}
		}

		template<typename SaveDataStruct>
		void restoreSaveData(std::vector<SaveDataStruct>& saveDataStructs, std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			for (SaveDataStruct& saveData : saveDataStructs)
			{
				saveData.restore(oldToNewUIDMap);
			}
		}
		template<typename SaveDataStruct>
		void linkNodes(std::vector<SaveDataStruct>& saveDataStructs, std::map<vtxID, vtxID>& oldToNewUIDMap)
		{
			for (SaveDataStruct& saveData : saveDataStructs)
			{
				saveData.link(oldToNewUIDMap);
			}
		}

		std::tuple<std::shared_ptr<graph::Renderer>, std::shared_ptr<graph::Group>> restoreShaderGraph()
		{
			VTX_INFO("Restoring shader graph");
			std::map<vtxID, vtxID> oldToNewUIDMap;
			restoreSaveData(transforms, oldToNewUIDMap);
			restoreSaveData(meshes, oldToNewUIDMap);
			restoreSaveData(materials, oldToNewUIDMap);
			restoreSaveData(instances, oldToNewUIDMap);
			restoreSaveData(groups, oldToNewUIDMap);
			restoreSaveData(environmentLights, oldToNewUIDMap);
			restoreSaveData(cameras, oldToNewUIDMap);
			restoreSaveData(renderers, oldToNewUIDMap);
			restoreSaveData(shaderNodes, oldToNewUIDMap);
			VTX_INFO("Linking shader graph");
			//linkNodes(transforms, oldToNewUIDMap);
			//linkNodes(meshes, oldToNewUIDMap);
			linkNodes(materials, oldToNewUIDMap);
			linkNodes(instances, oldToNewUIDMap);
			linkNodes(groups, oldToNewUIDMap);
			//linkNodes(environmentLights, oldToNewUIDMap);
			linkNodes(cameras, oldToNewUIDMap);
			linkNodes(renderers, oldToNewUIDMap);
			linkNodes(shaderNodes, oldToNewUIDMap);

			std::shared_ptr<graph::Renderer> renderer = graph::Scene::getSim()->getNode<graph::Renderer>(oldToNewUIDMap[activeRendererUID]);
			std::shared_ptr<graph::Group> sceneRoot = graph::Scene::getSim()->getNode<graph::Group>(oldToNewUIDMap[sceneRootUID]);

			return std::make_tuple(renderer, sceneRoot);
		}


		std::vector<MeshNodeSaveData> meshes;
		std::vector<TransformNodeSaveData> transforms;
		std::vector<MaterialNodeSaveData> materials;
		std::vector<InstanceNodeSaveData> instances;
		std::vector<GroupNodeSaveData> groups;
		std::vector<EnvironmentLightNodeSaveData> environmentLights;
		std::vector<CameraNodeSaveData> cameras;
		std::vector<RendererNodeSaveData> renderers;
		std::vector<BaseShaderNodeSaveData> shaderNodes;

		vtxID activeRendererUID;
		vtxID activeCameraUID;
		vtxID sceneRootUID;
	};
}

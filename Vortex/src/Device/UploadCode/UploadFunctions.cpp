#include "UploadFunctions.h"
#include "UploadData.h"
#include "Device/CUDAChecks.h"
#include "Scene/Nodes/Instance.h"
#include "Scene/Nodes/Renderer.h"
#include <cudaGL.h>

#include "Device/WorkQueues.h"
#include "Device/DevicePrograms/CudaKernels.h"
#include "MDL/CudaLinker.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Shader/Texture.h"

namespace vtx::device
{
	std::tuple<InstanceData, InstanceData*> createInstanceData(std::shared_ptr<graph::Instance> instanceNode, const math::affine3f& transform)
	{
		CUDA_SYNC_CHECK();

		const vtxID& instanceId = instanceNode->getID();
		VTX_INFO("Device Visitor: Creating Instance {} data", instanceId);

		CudaMap<vtxID, std::tuple<GeometryData, GeometryData*>>& geometryDataMap = UPLOAD_DATA->geometryDataMap;
		CudaMap<vtxID, std::tuple<MaterialData, MaterialData*>>& materialDataMap = UPLOAD_DATA->materialDataMap;
		CudaMap<vtxID, std::tuple<LightData, LightData*, bool>>& lightDataMap    = UPLOAD_DATA->lightDataMap;

		InstanceData instanceData{};
		instanceData.instanceId = instanceId;
		instanceData.transform  = transform;


		const vtxID& meshId = instanceNode->getChild()->getID();
		if (geometryDataMap.contains(meshId))
		{
			instanceData.geometryData = std::get<1>(geometryDataMap[meshId]);
		}
		else
		{
			VTX_ERROR("Requesting Mesh device Data of Mesh Node {} but not found!", meshId);
		}

		instanceData.hasEmission = false;
		instanceData.hasOpacity  = false;
		std::vector<InstanceData::SlotIds> materialSlots;
		for (const graph::MaterialSlot& materialSlot : instanceNode->getMaterialSlots())
		{
			InstanceData::SlotIds slotIds{nullptr,nullptr};

			if (vtxID materialId = materialSlot.material->getID(); materialDataMap.contains(materialId))
			{
				slotIds.material = std::get<1>(materialDataMap[materialId]);

				if (vtxID lightId = materialSlot.meshLight->getID(); lightDataMap.contains(lightId)) {
					instanceData.hasEmission = materialSlot.material->useAsLight;
					slotIds.meshLight = std::get<1>(lightDataMap[lightId]);
				}

				if(!instanceData.hasOpacity)
				{
					auto materialNode = graph::SIM::getNode<graph::Material>(materialId);
					if (bool hasOpacity = materialNode->useOpacity())
					{
						instanceData.hasOpacity = true;
					}
				}
			}
			else
			{
				VTX_ERROR("Requesting Material device Data of Material Node {} but not found!", materialId);
			}

			materialSlots.push_back(slotIds);
		}
		if (!materialSlots.empty())
		{
			CUDABuffer& materialSlotsBuffer = GET_BUFFER(Buffers::InstanceBuffers, instanceId, materialSlotsBuffer);
			materialSlotsBuffer.upload(materialSlots);
			instanceData.materialSlots = materialSlotsBuffer.castedPointer<InstanceData::SlotIds>();
			instanceData.numberOfSlots   = materialSlots.size();
		}
		else
		{
			instanceData.materialSlots = nullptr;
		}

		CUDABuffer& instanceDataBuffer = GET_BUFFER(Buffers::InstanceBuffers, instanceId, instanceDataBuffer);
		instanceDataBuffer.upload(instanceData);

		return { instanceData, instanceDataBuffer.castedPointer<InstanceData>() };
	}

	std::tuple<LightData, LightData*> createLightData(std::shared_ptr<graph::Light> lightNode)
	{
		LightData lightData{};
		if (lightNode->attributes->lightType == L_MESH)
		{
			MeshLightAttributesData meshLightData{};

			const auto& attributes = std::dynamic_pointer_cast<graph::MeshLightAttributes>(lightNode->attributes);

			CUDABuffer& areaCdfBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), areaCdfBuffer);
			areaCdfBuffer.upload(attributes->cdfAreas);

			CUDABuffer& actualTriangleIndices = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), actualTriangleIndices);
			actualTriangleIndices.upload(attributes->actualTriangleIndices);

			vtxID meshId = attributes->mesh->getID();
			vtxID materialId = attributes->material->getID();
			GeometryData* geometryData = std::get<1>(UPLOAD_DATA->geometryDataMap[meshId]);
			MaterialData* materialData = std::get<1>(UPLOAD_DATA->materialDataMap[materialId]);

			meshLightData.instanceId            = attributes->parentInstanceId;
			meshLightData.geometryData			= geometryData;
			meshLightData.materialId            = materialData;
			meshLightData.cdfArea               = areaCdfBuffer.castedPointer<float>();
			meshLightData.actualTriangleIndices = actualTriangleIndices.castedPointer<uint32_t>();
			meshLightData.size                  = attributes->cdfAreas.size();
			meshLightData.totalArea             = attributes->area;

			CUDABuffer& attributeBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), attributeBuffer);
			attributeBuffer.upload(meshLightData);

			lightData.type       = L_MESH;
			lightData.attributes = attributeBuffer.dPointer();
		}
		else if(lightNode->attributes->lightType == L_ENV)
		{
			EnvLightAttributesData envLightData{};

			const auto& attributes = std::dynamic_pointer_cast<graph::EvnLightAttributes>(lightNode->attributes);

			//CUDABuffer& cdfUBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), cdfUBuffer);
			//cdfUBuffer.upload(attributes->cdfU);
			//
			//CUDABuffer& cdfVBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), cdfVBuffer);
			//cdfVBuffer.upload(attributes->cdfV);

			CUDABuffer& aliasBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), aliasBuffer);
			aliasBuffer.upload(attributes->aliasMap);

			TextureData* texture = std::get<1>(UPLOAD_DATA->textureDataMap[attributes->envTexture->getID()]);
			envLightData.texture = texture;
			//envLightData.invIntegral = attributes->invIntegral;
			//envLightData.cdfU = cdfUBuffer.castedPointer<float>();
			//envLightData.cdfV = cdfVBuffer.castedPointer<float>();
			envLightData.transformation = attributes->transform->transformationAttribute.affineTransform;
			envLightData.invTransformation = math::affine3f(envLightData.transformation.l.inverse(), envLightData.transformation.p);
			envLightData.aliasMap = aliasBuffer.castedPointer<AliasData>();



			CUDABuffer& attributeBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), attributeBuffer);
			attributeBuffer.upload(envLightData);

			lightData.type = L_ENV;
			lightData.attributes = attributeBuffer.dPointer();
		}

		CUDABuffer& lightDataBuffer = GET_BUFFER(Buffers::LightBuffers, lightNode->getID(), lightDataBuffer);
		lightDataBuffer.upload(lightData);
		return { lightData, lightDataBuffer.castedPointer<LightData>() };
	}

	std::tuple<GeometryData, GeometryData*> createGeometryData(std::shared_ptr<graph::Mesh> meshNode)
	{
		VTX_INFO("Computing BLAS");

		CUDA_SYNC_CHECK();

		OptixDeviceContext& optixContext = optix::getState()->optixContext;
		CUstream&           stream       = optix::getState()->stream;

		/// Uploading Vertex and Index Buffer ///

		CUDABuffer& vertexBuffer = GET_BUFFER(Buffers::GeometryBuffers, meshNode->getID(), vertexBuffer);
		CUDABuffer& indexBuffer  = GET_BUFFER(Buffers::GeometryBuffers, meshNode->getID(), indexBuffer);
		CUDABuffer& faceBuffer   = GET_BUFFER(Buffers::GeometryBuffers, meshNode->getID(), faceBuffer);

		vertexBuffer.upload(meshNode->vertices);
		indexBuffer.upload(meshNode->indices);
		faceBuffer.upload(meshNode->faceAttributes);


		const CUdeviceptr vertexData = vertexBuffer.dPointer();
		const CUdeviceptr indexData  = indexBuffer.dPointer();

		const OptixTraversableHandle traversable = optix::createGeometryAcceleration(vertexData,
			static_cast<uint32_t>(meshNode->vertices.size()),
			sizeof(graph::VertexAttributes),
			indexData,
			static_cast<uint32_t>(meshNode->indices.size()),
			sizeof(vtxID) * 3);

		GeometryData data{};

		data.type                = PT_TRIANGLES;
		data.traversable         = traversable;
		data.vertexAttributeData = vertexBuffer.castedPointer<graph::VertexAttributes>();
		data.indicesData         = indexBuffer.castedPointer<vtxID>();
		data.faceAttributeData   = faceBuffer.castedPointer<graph::FaceAttributes>();
		data.numVertices         = meshNode->vertices.size();
		data.numIndices          = meshNode->indices.size();
		data.numFaces            = meshNode->faceAttributes.size();

		CUDABuffer& geometryDataBuffer = GET_BUFFER(Buffers::GeometryBuffers, meshNode->getID(), geometryDataBuffer);
		geometryDataBuffer.upload(data);

		std::tuple<GeometryData, GeometryData*> tuple = std::make_tuple(data, geometryDataBuffer.castedPointer<GeometryData>());
		return tuple;
	}

	DeviceShaderConfiguration* createDeviceShaderConfiguration(std::shared_ptr<graph::Material> material)
	{
		const graph::DevicePrograms& dp = material->getPrograms();
		const graph::Configuration& config = material->getConfiguration();
		optix::PipelineOptix* rp = optix::getRenderingPipeline();
		//const CudaMap<vtxID, optix::sbtPosition>& sbtMap = rp->getSbtMap();
		DeviceShaderConfiguration                 dvConfig;

		// The constant expression values:
		//bool thin_walled; // Stored inside flags.
		// Simplify the conditions by translating all constants unconditionally.

		dvConfig.isThinWalled = material->isThinWalled();
		dvConfig.hasOpacity = material->useOpacity();
		dvConfig.isEmissive = material->useEmission();
		dvConfig.directCallable = (getOptions()->mdlCallType == MDL_DIRECT_CALL);

		if (getOptions()->mdlCallType == MDL_DIRECT_CALL)
		{
			dvConfig.surfaceIntensity = math::vec3f(config.surfaceIntensity[0], config.surfaceIntensity[1], config.surfaceIntensity[2]);
			dvConfig.surfaceIntensityMode = config.surfaceIntensityMode;
			dvConfig.backfaceIntensity = math::vec3f(config.backfaceIntensity[0], config.backfaceIntensity[1], config.backfaceIntensity[2]);
			dvConfig.backfaceIntensityMode = config.backfaceIntensityMode;
			dvConfig.ior = math::vec3f(config.ior[0], config.ior[1], config.ior[2]);
			dvConfig.absorptionCoefficient = math::vec3f(config.absorptionCoefficient[0], config.absorptionCoefficient[1], config.absorptionCoefficient[2]);
			dvConfig.scatteringCoefficient = math::vec3f(config.scatteringCoefficient[0], config.scatteringCoefficient[1], config.scatteringCoefficient[2]);
			dvConfig.cutoutOpacity = config.cutoutOpacity;

			if (dp.pgInit) {
				dvConfig.idxCallInit = rp->getProgramSbt(dp.pgInit->name);
			}

			if (dp.pgThinWalled) {
				dvConfig.idxCallThinWalled = rp->getProgramSbt(dp.pgThinWalled->name);
			}

			if (dp.pgSurfaceScatteringSample) {
				dvConfig.idxCallSurfaceScatteringSample = rp->getProgramSbt(dp.pgSurfaceScatteringSample->name);
			}

			if (dp.pgSurfaceScatteringEval) {
				dvConfig.idxCallSurfaceScatteringEval = rp->getProgramSbt(dp.pgSurfaceScatteringEval->name);
			}

			if (dp.pgSurfaceScatteringAuxiliary) {
				dvConfig.idxCallSurfaceScatteringAuxiliary = rp->getProgramSbt(dp.pgSurfaceScatteringAuxiliary->name);
			}

			if (dp.pgBackfaceScatteringSample) {
				dvConfig.idxCallBackfaceScatteringSample = rp->getProgramSbt(dp.pgBackfaceScatteringSample->name);
			}

			if (dp.pgBackfaceScatteringEval) {
				dvConfig.idxCallBackfaceScatteringEval = rp->getProgramSbt(dp.pgBackfaceScatteringEval->name);
			}

			if (dp.pgBackfaceScatteringAuxiliary) {
				dvConfig.idxCallBackfaceScatteringAuxiliary = rp->getProgramSbt(dp.pgBackfaceScatteringAuxiliary->name);
			}

			if (dp.pgSurfaceEmissionEval) {
				dvConfig.idxCallSurfaceEmissionEval = rp->getProgramSbt(dp.pgSurfaceEmissionEval->name);
			}

			if (dp.pgSurfaceEmissionIntensity) {
				dvConfig.idxCallSurfaceEmissionIntensity = rp->getProgramSbt(dp.pgSurfaceEmissionIntensity->name);
			}

			if (dp.pgSurfaceEmissionIntensityMode) {
				dvConfig.idxCallSurfaceEmissionIntensityMode = rp->getProgramSbt(dp.pgSurfaceEmissionIntensityMode->name);
			}

			if (dp.pgBackfaceEmissionEval) {
				dvConfig.idxCallBackfaceEmissionEval = rp->getProgramSbt(dp.pgBackfaceEmissionEval->name);
			}

			if (dp.pgBackfaceEmissionIntensity) {
				dvConfig.idxCallBackfaceEmissionIntensity = rp->getProgramSbt(dp.pgBackfaceEmissionIntensity->name);
			}

			if (dp.pgBackfaceEmissionIntensityMode) {
				dvConfig.idxCallBackfaceEmissionIntensityMode = rp->getProgramSbt(dp.pgBackfaceEmissionIntensityMode->name);
			}

			if (dp.pgIor) {
				dvConfig.idxCallIor = rp->getProgramSbt(dp.pgIor->name);
			}

			if (dp.pgVolumeAbsorptionCoefficient) {
				dvConfig.idxCallVolumeAbsorptionCoefficient = rp->getProgramSbt(dp.pgVolumeAbsorptionCoefficient->name);
			}

			if (dp.pgVolumeScatteringCoefficient) {
				dvConfig.idxCallVolumeScatteringCoefficient = rp->getProgramSbt(dp.pgVolumeScatteringCoefficient->name);
			}

			if (dp.pgVolumeDirectionalBias) {
				dvConfig.idxCallVolumeDirectionalBias = rp->getProgramSbt(dp.pgVolumeDirectionalBias->name);
			}

			if (dp.pgGeometryCutoutOpacity) {
				dvConfig.idxCallGeometryCutoutOpacity = rp->getProgramSbt(dp.pgGeometryCutoutOpacity->name);
			}

			if (dp.pgHairSample) {
				dvConfig.idxCallHairSample = rp->getProgramSbt(dp.pgHairSample->name);
			}

			if (dp.pgHairEval) {
				dvConfig.idxCallHairEval = rp->getProgramSbt(dp.pgHairEval->name);
			}
		}
		else if( getOptions()->mdlCallType == MDL_INLINE)
		{
			if (dp.pgEvaluateMaterial) {
				dvConfig.idxCallEvaluateMaterialStandard = rp->getProgramSbt(dp.pgEvaluateMaterial->name);
				dvConfig.idxCallEvaluateMaterialWavefront = rp->getProgramSbt(dp.pgEvaluateMaterial->name, "wfShade");
				VTX_WARN("Fetching Shader {} EvaluateMaterial STANDARD program {} with SBT {}", material->name, dp.pgEvaluateMaterial->name, dvConfig.idxCallEvaluateMaterialStandard);
				VTX_WARN("Fetching Shader {} EvaluateMaterial WAVEFRONT program {} with SBT {}", material->name, dp.pgEvaluateMaterial->name, dvConfig.idxCallEvaluateMaterialWavefront);
			}
		}
		else if (getOptions()->mdlCallType == MDL_CUDA)
		{
			dvConfig.idxCallEvaluateMaterialWavefront = mdl::getMdlCudaLinker().getMdlFunctionIndices(material->name);
		}


		CUDABuffer& materialConfigBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), materialConfigBuffer);
		materialConfigBuffer.upload(dvConfig);
		return materialConfigBuffer.castedPointer<DeviceShaderConfiguration>();
	}

	TextureHandler* createTextureHandler(std::shared_ptr<graph::Material> material)
	{
		TextureHandler textureHandler;

		const CudaMap<vtxID, std::tuple<TextureData, TextureData*>>& textureDataMap = UPLOAD_DATA->textureDataMap;
		const CudaMap<vtxID, std::tuple<BsdfData, BsdfData*>>& bsdfDataMap = UPLOAD_DATA->bsdfDataMap;
		const CudaMap<vtxID, std::tuple<LightProfileData, LightProfileData*>>& lightProfileDataMap = UPLOAD_DATA->lightProfileDataMap;

		// The following code was based on different assumptions on how mdl can access textures, it was probably wrong
		// It seems like mdl will provide indices to shader resource based on the order by whihch they are declared in the mdl file (target code)
		// Thi following code will try to remap the order
		if (const size_t size = material->getTextures().size(); size > 0)
		{
			std::vector<TextureData*> textures(size);
			for (const std::shared_ptr<graph::Texture> texture : material->getTextures())
			{
				vtxID          textureId = texture->getID();
				const uint32_t mdlIndex = texture->mdlIndex;
				if (textureDataMap.contains(textureId))
				{
					textures[mdlIndex - 1] = std::get<1>(textureDataMap[textureId]);
					//textures.push_back(textureId);
				}
				else
				{
					VTX_ERROR("Requesting Texture device Data of texture Node {} but not found!", textureId);
				}
			}
			CUDABuffer& textureIdBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), textureIdBuffer);
			textureIdBuffer.upload(textures);
			textureHandler.numTextures = textures.size();
			textureHandler.textures = textureIdBuffer.castedPointer<TextureData*>();
		}
		else
		{
			textureHandler.numTextures = 0;
			textureHandler.textures = nullptr;
		}


		if (const size_t size = material->getBsdfs().size(); size > 0)
		{
			std::vector<BsdfData*> bsdfs(size);
			for (const std::shared_ptr<graph::BsdfMeasurement> bsdf : material->getBsdfs())
			{
				vtxID          bsdfId = bsdf->getID();
				const uint32_t mdlIndex = bsdf->mdlIndex;
				if (bsdfDataMap.contains(bsdfId))
				{
					bsdfs[mdlIndex] = std::get<1>(bsdfDataMap[bsdfId]);
				}
				else
				{
					VTX_ERROR("Requesting Bsdf device Data of Bsdf Node {} but not found!", bsdfId);
				}
			}
			CUDABuffer& bsdfIdBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), bsdfIdBuffer);
			bsdfIdBuffer.upload(bsdfs);
			textureHandler.bsdfs = bsdfIdBuffer.castedPointer<BsdfData*>();
			textureHandler.numBsdfs = bsdfs.size();
		}
		else
		{
			textureHandler.numBsdfs = 0;
			textureHandler.bsdfs = nullptr;
		}

		if (const size_t size = material->getLightProfiles().size(); size > 0)
		{
			std::vector<LightProfileData*> lightProfiles(size);
			for (const std::shared_ptr<graph::LightProfile> lightProfile : material->getLightProfiles())
			{
				vtxID          lightProfileId = lightProfile->getID();
				const uint32_t mdlIndex = lightProfile->mdlIndex;
				if (lightProfileDataMap.contains(lightProfileId))
				{
					lightProfiles[mdlIndex] = std::get<1>(lightProfileDataMap[lightProfileId]);
					//lightProfiles.push_back(lightProfileId);
				}
				else
				{
					VTX_ERROR("Requesting Light profile device Data of light profile Node {} but not found!", lightProfileId);
				}
			}
			CUDABuffer& lightProfileBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), lightProfileBuffer);
			lightProfileBuffer.upload(lightProfiles);
			textureHandler.lightProfiles = lightProfileBuffer.castedPointer<LightProfileData*>();
			textureHandler.numLightProfiles = lightProfiles.size();
		}
		else
		{
			textureHandler.numLightProfiles = 0;
			textureHandler.lightProfiles = nullptr;
		}



		CUDABuffer& textureHandlerBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), TextureHandlerBuffer);
		textureHandlerBuffer.upload(textureHandler);
		return textureHandlerBuffer.castedPointer<TextureHandler>();
		
	}

	std::tuple<MaterialData, MaterialData*> createMaterialData(std::shared_ptr<graph::Material> material)
	{

		//CUDA_SYNC_CHECK();

		MaterialData materialData   = {}; // Set everything to zero.
		CUDABuffer&  argBlockBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), argBlockBuffer);
		// If the material has an argument block, allocate and upload it.
		if (const size_t sizeArgumentBlock = material->getArgumentBlockSize(); sizeArgumentBlock > 0)
		{
			argBlockBuffer.upload(material->getArgumentBlockData(), sizeArgumentBlock);
		}

		materialData.materialConfiguration = createDeviceShaderConfiguration(material);
		materialData.textureHandler = createTextureHandler(material);

		materialData.argBlock = argBlockBuffer.castedPointer<char>();

		CUDABuffer& materialDataBuffer = GET_BUFFER(Buffers::MaterialBuffers, material->getID(), materialDataBuffer);
		materialDataBuffer.upload(materialData);

		return std::tuple(materialData, materialDataBuffer.castedPointer<MaterialData>());

	}

	std::tuple<TextureData, TextureData *> createTextureData(std::shared_ptr<vtx::graph::Texture>& textureNode)
	{
		VTX_INFO("Creating Texture Data for Texture Node ID: {} Name: {}", textureNode->getID(), textureNode->databaseName);
		CUDA_ARRAY3D_DESCRIPTOR descArray3D = {};
		descArray3D.Format = textureNode->format;
		descArray3D.Width = textureNode->dimension.x;
		descArray3D.Height = textureNode->dimension.y;
		descArray3D.Depth = textureNode->dimension.z;
		descArray3D.NumChannels = textureNode->dimension.w;

		const CUaddress_mode addrMode = (textureNode->shape == mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_cube) ? CU_TR_ADDRESS_MODE_CLAMP : CU_TR_ADDRESS_MODE_WRAP;
		// Create filtered textureNode object
		CUDA_TEXTURE_DESC textureDesc = {};  // This contains all textureNode parameters which can be set individually or as a whole.

		// If the flag CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP.
		textureDesc.addressMode[0] = addrMode;
		textureDesc.addressMode[1] = addrMode;
		textureDesc.addressMode[2] = addrMode;

		textureDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.

		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
		if (textureNode->effectiveGamma != 1.0f && textureNode->pixelBytesSize == 1)
		{
			textureDesc.flags |= CU_TRSF_SRGB;
		}

		textureDesc.maxAnisotropy = 1;

		// LOD 0 only by default.
		// This means when using mipmaps it's the developer's responsibility to set at least 
		// maxMipmapLevelClamp > 0.0f before calling Texture::create() to make sure mipmaps can be sampled!
		textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDesc.mipmapLevelBias = 0.0f;
		textureDesc.minMipmapLevelClamp = 0.0f;
		textureDesc.maxMipmapLevelClamp = 0.0f; // This should be set to Picture::getNumberOfLevels() when using mipmaps.

		textureDesc.borderColor[0] = 0.0f;
		textureDesc.borderColor[1] = 0.0f;
		textureDesc.borderColor[2] = 0.0f;
		textureDesc.borderColor[3] = 0.0f;

		// Create unfiltered textureNode object if necessary (cube textures have no texel functions)
		CUDA_TEXTURE_DESC textureDescUnfiltered = textureDesc;

		if (textureNode->shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
		{
			// Use a black border for access outside of the textureNode
			textureDescUnfiltered.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.filterMode = CU_TR_FILTER_MODE_POINT;
		}

		TextureData textureData;

		CUresult result;
		CUarray& textureArray = GET_BUFFER(Buffers::TextureBuffers, textureNode->getID(), textureArray);
		cudaTextureObject_t& texObj = GET_BUFFER(Buffers::TextureBuffers, textureNode->getID(), texObj);
		cudaTextureObject_t& texObjUnfiltered = GET_BUFFER(Buffers::TextureBuffers, textureNode->getID(), texObjUnfiltered);

		CUDA_RESOURCE_DESC resourceDescription = uploadTexture(textureNode->imageLayersPointers, descArray3D, textureNode->pixelBytesSize, textureArray);


		result = cuTexObjectCreate(&texObj, &resourceDescription, &textureDesc, nullptr);
		CU_CHECK(result);
		result = cuTexObjectCreate(&texObjUnfiltered, &resourceDescription, &textureDescUnfiltered, nullptr);
		CU_CHECK(result);

		textureData.texObj = texObj;
		textureData.texObjUnfiltered = texObjUnfiltered;
		textureData.dimension = textureNode->dimension;
		textureData.invSize = math::vec3f{ 1.0f / textureData.dimension.x, 1.0f / textureData.dimension.y, 1.0f / textureData.dimension.z };

		CUDABuffer& textureDataBuffer = GET_BUFFER(Buffers::TextureBuffers, textureNode->getID(), textureDataBuffer);
		textureDataBuffer.upload(textureData);

		return { textureData, textureDataBuffer.castedPointer<TextureData>() };
	}

	BsdfSamplingPartData createBsdfPartData(graph::BsdfMeasurement::BsdfPartData& bsdfData, Buffers::BsdfPartBuffer& buffers)
	{
		BsdfSamplingPartData bsdfSamplingData{};
		CUDABuffer& sampleData = buffers.sampleData;
		CUDABuffer& albedoData = buffers.albedoData;
		CUtexObject& evalData = buffers.evalData;

		sampleData.upload(bsdfData.sampleData);
		albedoData.upload(bsdfData.albedoData);

		bsdfSamplingData.sampleData = sampleData.castedPointer<float>();
		bsdfSamplingData.albedoData = albedoData.castedPointer<float>();
		bsdfSamplingData.maxAlbedo = bsdfData.maxAlbedo;
		bsdfSamplingData.angularResolution = bsdfData.angularResolution;
		bsdfSamplingData.numChannels = bsdfData.numChannels;
		bsdfSamplingData.invAngularResolution = math::vec2f(1.0f / static_cast<float>(bsdfData.angularResolution.x), 1.0f / static_cast<float>(bsdfData.angularResolution.y));

		// Prepare 
		// uation data:
		// - Simply store the measured data in a volume texture.
		// - In case of color data, we store each sample in a vector4 to get texture support.

		// Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
		CUDA_ARRAY3D_DESCRIPTOR descArray3D = {};

		descArray3D.Width = bsdfData.angularResolution.y;
		descArray3D.Height = bsdfData.angularResolution.x;
		descArray3D.Depth = bsdfData.angularResolution.x;
		descArray3D.Format = CU_AD_FORMAT_FLOAT;
		descArray3D.NumChannels = (bsdfData.numChannels == 3) ? 4 : 1;
		descArray3D.Flags = 0;

		std::vector<const void*> pointers;
		pointers.push_back(bsdfData.lookupData.data());

		CUDA_RESOURCE_DESC resourceDescription = uploadTexture(pointers, descArray3D, sizeof(float), buffers.lookUpArray);

		bsdfSamplingData.deviceMbsdfData = buffers.lookUpArray;

		CUDA_TEXTURE_DESC textureDescription = {};  // This contains all texture parameters which can be set individually or as a whole.
		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES;
		textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
		textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.maxAnisotropy = 1;
		// DAR The default initialization handled all these. Just for code clarity.
		textureDescription.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDescription.mipmapLevelBias = 0.0f;
		textureDescription.minMipmapLevelClamp = 0.0f;
		textureDescription.maxMipmapLevelClamp = 0.0f;
		textureDescription.borderColor[0] = 0.0f;
		textureDescription.borderColor[1] = 0.0f;
		textureDescription.borderColor[2] = 0.0f;
		textureDescription.borderColor[3] = 0.0f;

		const CUresult result = cuTexObjectCreate(&evalData, &resourceDescription, &textureDescription, nullptr);
		CU_CHECK(result);

		bsdfSamplingData.evalData = evalData;
		return bsdfSamplingData;
	}

	std::tuple < BsdfData, BsdfData *> createBsdfData(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement)
	{
		BsdfData bsdfData;
		if (bsdfMeasurement->reflectionBsdf.isValid)
		{
			Buffers::BsdfPartBuffer& reflectionPartBuffer = GET_BUFFER(Buffers::BsdfBuffers, bsdfMeasurement->getID(), reflectionPartBuffer);
			const BsdfSamplingPartData reflectionBsdfDeviceData = createBsdfPartData(bsdfMeasurement->reflectionBsdf, reflectionPartBuffer);

			CUDABuffer& partBuffer = reflectionPartBuffer.partBuffer;
			partBuffer.upload(reflectionBsdfDeviceData);
			bsdfData.reflectionBsdf = partBuffer.castedPointer<BsdfSamplingPartData>();
			bsdfData.hasReflectionBsdf = true;

		}
		if (bsdfMeasurement->transmissionBsdf.isValid)
		{
			Buffers::BsdfPartBuffer& transmissionPartBuffer = GET_BUFFER(Buffers::BsdfBuffers, bsdfMeasurement->getID(), transmissionPartBuffer);
			const BsdfSamplingPartData transmissionBsdfDeviceData = createBsdfPartData(bsdfMeasurement->transmissionBsdf, transmissionPartBuffer);

			CUDABuffer& partBuffer = transmissionPartBuffer.partBuffer;
			partBuffer.upload(transmissionBsdfDeviceData);
			bsdfData.transmissionBsdf = partBuffer.castedPointer<BsdfSamplingPartData>();
			bsdfData.hasTransmissionBsdf = true;
		}

		CUDABuffer& bsdfDataBuffer = GET_BUFFER(Buffers::BsdfBuffers, bsdfMeasurement->getID(), bsdfDataBuffer);
		bsdfDataBuffer.upload(bsdfData);

		return { bsdfData, bsdfDataBuffer.castedPointer<BsdfData>() };
	}

	std::tuple < LightProfileData, LightProfileData *> createLightProfileData(std::shared_ptr<graph::LightProfile> lightProfile)
	{
		// Copy entire CDF data buffer to GPU
		CUDABuffer& cdfBuffer = GET_BUFFER(Buffers::LightProfileBuffers, lightProfile->getID(), cdfBuffer);
		cdfBuffer.upload(lightProfile->lightProfileData.cdfData);

		// --------------------------------------------------------------------------------------------
		// Prepare evaluation data.
		//  - Use a 2d texture that allows bilinear interpolation.
		// Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
		CUDA_ARRAY3D_DESCRIPTOR descArray3D = {};

		descArray3D.Width       = lightProfile->lightProfileData.resolution.x;
		descArray3D.Height      = lightProfile->lightProfileData.resolution.y;
		descArray3D.Depth       = 0; // A 2D array is allocated if only Depth extent is zero.
		descArray3D.Format      = CU_AD_FORMAT_FLOAT;
		descArray3D.NumChannels = 1;
		descArray3D.Flags       = 0;
		std::vector<const void*> pointers;
		pointers.push_back(lightProfile->lightProfileData.sourceData);

		CUarray& lightProfileSourceArray = GET_BUFFER(Buffers::LightProfileBuffers, lightProfile->getID(), lightProfileSourceArray);
		const CUDA_RESOURCE_DESC resourceDescription = uploadTexture(pointers, descArray3D, sizeof(float), lightProfileSourceArray);


		CUDA_TEXTURE_DESC textureDescription = {}; // This contains all texture parameters which can be set individually or as a whole.
		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDescription.flags          = CU_TRSF_NORMALIZED_COORDINATES;
		textureDescription.filterMode     = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
		textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; // FIXME Shouldn't phi use wrap?
		textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.maxAnisotropy  = 1;
		// DAR The default initialization handled all these. Just for code clarity.
		textureDescription.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDescription.mipmapLevelBias = 0.0f;
		textureDescription.minMipmapLevelClamp = 0.0f;
		textureDescription.maxMipmapLevelClamp = 0.0f;
		textureDescription.borderColor[0] = 1.0f; // DAR DEBUG Why 1.0f? Shouldn't matter with clamp.
		textureDescription.borderColor[1] = 1.0f;
		textureDescription.borderColor[2] = 1.0f;
		textureDescription.borderColor[3] = 1.0f;
		CUtexObject& evalData = GET_BUFFER(Buffers::LightProfileBuffers, lightProfile->getID(), evalData);
		CU_CHECK(cuTexObjectCreate(&evalData, &resourceDescription, &textureDescription, nullptr));

		LightProfileData lightProfileData{};
		lightProfileData.lightProfileArray = lightProfileSourceArray;
		lightProfileData.evalData = evalData;
		lightProfileData.cdfData = cdfBuffer.castedPointer<float>();
		lightProfileData.angularResolution = lightProfile->lightProfileData.resolution;
		lightProfileData.invAngularResolution = math::vec2f(1.0f / static_cast<float>(lightProfileData.angularResolution.x), 1.0f / static_cast<float>(lightProfileData.angularResolution.y));
		lightProfileData.thetaPhiStart = lightProfile->lightProfileData.start;        // start of the grid
		lightProfileData.thetaPhiDelta = lightProfile->lightProfileData.delta;        // angular step size
		lightProfileData.thetaPhiInvDelta.x = (lightProfileData.thetaPhiDelta.x != 0.0f) ? 1.0f / lightProfileData.thetaPhiDelta.x : 0.0f;
		lightProfileData.thetaPhiInvDelta.y = (lightProfileData.thetaPhiDelta.y != 0.0f) ? 1.0f / lightProfileData.thetaPhiDelta.y : 0.0f;
		lightProfileData.candelaMultiplier = static_cast<float>(lightProfile->lightProfileData.candelaMultiplier);
		lightProfileData.totalPower = static_cast<float>(lightProfile->lightProfileData.totalPower * lightProfile->lightProfileData.candelaMultiplier);


		CUDABuffer& bsdfDataBuffer = GET_BUFFER(Buffers::LightProfileBuffers, lightProfile->getID(), lightProfileDataBuffer);
		bsdfDataBuffer.upload(lightProfileData);

		return { lightProfileData, bsdfDataBuffer.castedPointer<LightProfileData>() };
	}


#define PREPARE_BUFFER(cudaBuffer, bufferPoitner, dataType)\
	CUDABuffer& cudaBuffer = GET_BUFFER(Buffers::FrameBufferBuffers, rendererNode->getID(), cudaBuffer); \
	cudaBuffer.resize((size_t)rendererNode->width* (size_t)rendererNode->height * sizeof(dataType)); \
	UPLOAD_DATA->frameBufferData.bufferPoitner = cudaBuffer.castedPointer<dataType>()\
	
	void setRendererData(std::shared_ptr<graph::Renderer> rendererNode)
	{
		//auto* tW = new float4[3];
		//tW[0] = float4{ 1.0f, 0.0f, 0.0f, 0.0f };
		//tW[1] = float4{ 0.0f, 1.0f, 0.0f, 0.0f };
		//tW[2] = float4{ 1.0f, 0.0f, 2.0f, 0.0f };
		//
		//const float4* tt = tW;
		//
		//const math::affine3f objectToWorld(tt);

		if (rendererNode->resized) {

			rendererNode->threadData.bufferUpdateReady = false;

			CUDABuffer& cudaOutputBuffer = GET_BUFFER(Buffers::FrameBufferBuffers, rendererNode->getID(), cudaOutputBuffer);
			cudaOutputBuffer.resize((size_t)rendererNode->width * (size_t)rendererNode->height * sizeof(math::vec4f));
			UPLOAD_DATA->frameBufferData.outputBuffer = cudaOutputBuffer.dPointer();


			PREPARE_BUFFER(rawRadiance, rawRadiance, math::vec3f);
			PREPARE_BUFFER(directLight, directLight, math::vec3f);
			PREPARE_BUFFER(diffuseIndirect, diffuseIndirect, math::vec3f);
			PREPARE_BUFFER(glossyIndirect, glossyIndirect, math::vec3f);
			PREPARE_BUFFER(transmissionIndirect, transmissionIndirect, math::vec3f);
			PREPARE_BUFFER(tmRadiance, tmRadiance, math::vec3f);
			PREPARE_BUFFER(tmDirectLight, tmDirectLight, math::vec3f);
			PREPARE_BUFFER(tmDiffuseIndirect, tmDiffuseIndirect, math::vec3f);
			PREPARE_BUFFER(tmGlossyIndirect, tmGlossyIndirect, math::vec3f);
			PREPARE_BUFFER(tmTransmissionIndirect, tmTransmissionIndirect, math::vec3f);
			PREPARE_BUFFER(albedo, albedo, math::vec3f);
			PREPARE_BUFFER(normal, normal, math::vec3f);
			PREPARE_BUFFER(trueNormal, trueNormal, math::vec3f);
			PREPARE_BUFFER(tangent, tangent, math::vec3f);
			PREPARE_BUFFER(orientation, orientation, math::vec3f);
			PREPARE_BUFFER(uv, uv, math::vec3f);
			PREPARE_BUFFER(debugColor1, debugColor1, math::vec3f);
			PREPARE_BUFFER(fireflyRemoval, fireflyPass, math::vec3f);

			optix::getState()->denoiser.resize(rendererNode->width, rendererNode->height);

			//TODO Smarter way to set these value, I don't want to waste memory if we are not using adaptive samplig
			{
				//PREPARE_BUFFER(toneMappedRadianceBuffer, toneMappedRadiance, math::vec3f);
				PREPARE_BUFFER(noiseDataBuffer, noiseBuffer, NoiseData);

				const int size = rendererNode->width * rendererNode->height;
				constexpr int threadsPerBlock = 256;
				const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

				CUDABuffer& radianceRangeBuffer = GET_BUFFER(Buffers::FrameBufferBuffers, rendererNode->getID(), radianceRangeBuffer);
				radianceRangeBuffer.resize(numBlocks * sizeof(math::vec2f));

				CUDABuffer& albedoRangeBuffer = GET_BUFFER(Buffers::FrameBufferBuffers, rendererNode->getID(), albedoRangeBuffer);
				albedoRangeBuffer.resize(numBlocks * sizeof(math::vec2f));

				CUDABuffer& normalRangeBuffer = GET_BUFFER(Buffers::FrameBufferBuffers, rendererNode->getID(), normalRangeBuffer);
				normalRangeBuffer.resize(numBlocks * sizeof(math::vec2f));

			}

			UPLOAD_DATA->frameBufferData.frameSize.x = rendererNode->width;
			UPLOAD_DATA->frameBufferData.frameSize.y = rendererNode->height;

			rendererNode->resized             = false;
			UPLOAD_DATA->isFrameBufferUpdated = true;
			rendererNode->resizeGlBuffer      = true;

			{
				int maxQueueSize = rendererNode->width * rendererNode->height;

				CUDABufferManager::deallocateAll();
				const WorkQueueSOA<PixelWorkItem> pixelQueue(maxQueueSize, "pixelQueue");
				CUDABuffer pixelQueueBuffer;
				pixelQueueBuffer.upload(pixelQueue);
				UPLOAD_DATA->launchParams.pixelQueue = pixelQueueBuffer.castedPointer<WorkQueueSOA<PixelWorkItem>>();

				const WorkQueueSOA<RayWorkItem> radianceTraceQueue(maxQueueSize, "radianceTraceQueue");
				CUDABuffer radianceTraceQueueBuffer;
				radianceTraceQueueBuffer.upload(radianceTraceQueue);
				UPLOAD_DATA->launchParams.radianceTraceQueue = radianceTraceQueueBuffer.castedPointer<WorkQueueSOA<RayWorkItem>>();

				const WorkQueueSOA<RayWorkItem> shadeQueue(maxQueueSize, "shadeQueue");
				CUDABuffer shadeQueueBuffer;
				shadeQueueBuffer.upload(shadeQueue);
				UPLOAD_DATA->launchParams.shadeQueue = shadeQueueBuffer.castedPointer<WorkQueueSOA<RayWorkItem>>();

				const WorkQueueSOA<RayWorkItem> escapedQueue(maxQueueSize, "escapedQueue");
				CUDABuffer escapedQueueBuffer;
				escapedQueueBuffer.upload(escapedQueue);
				UPLOAD_DATA->launchParams.escapedQueue = escapedQueueBuffer.castedPointer<WorkQueueSOA<RayWorkItem>>();

				const WorkQueueSOA<RayWorkItem> accumulationQueue(maxQueueSize, "accumulationQueue");
				CUDABuffer accumulationQueueBuffer;
				accumulationQueueBuffer.upload(accumulationQueue);
				UPLOAD_DATA->launchParams.accumulationQueue = accumulationQueueBuffer.castedPointer<WorkQueueSOA<RayWorkItem>>();


				//CUDABuffer rayDataBuffer;
				//rayDataBuffer.alloc(sizeof(RayData) * maxQueueSize);
				//UPLOAD_DATA->launchParams.rays = rayDataBuffer.castedPointer<RayData>();

				//CUDABuffer raydataMissBuffer;
				//raydataMissBuffer.alloc(sizeof(RayData*) * maxQueueSize);
				//UPLOAD_DATA->launchParams.escapedQueue = raydataMissBuffer.castedPointer<RayData*>();
				////CUDABuffer missQueueBuffer;
				////WorkQueue missQueue;
				////missQueue.maxSize = maxQueueSize;
				////missQueueBuffer.upload(missQueue);
				////UPLOAD_DATA->launchParams.escapedQueue = missQueueBuffer.castedPointer<WorkQueue>();

				//CUDABuffer raydataHitBuffer;
				//raydataHitBuffer.alloc(sizeof(RayData*) * maxQueueSize);
				//UPLOAD_DATA->launchParams.shadeQueue = raydataHitBuffer.castedPointer<RayData*>();
				////CUDABuffer hitQueueBuffer;
				////WorkQueue hitQueue;
				////hitQueue.maxSize = maxQueueSize;
				////hitQueueBuffer.upload(hitQueue);
				////UPLOAD_DATA->launchParams.shadeQueue = hitQueueBuffer.castedPointer<WorkQueue>();

				//CUDABuffer traceHitBuffer;
				//traceHitBuffer.alloc(sizeof(RayData*) * maxQueueSize);
				//UPLOAD_DATA->launchParams.radianceTraceQueue = traceHitBuffer.castedPointer<RayData*>();
				////CUDABuffer traceHitQueueBuffer;
				////WorkQueue traceHitQueue;
				////traceHitQueue.maxSize = maxQueueSize;
				////traceHitQueueBuffer.upload(traceHitQueue);
				////UPLOAD_DATA->launchParams.radianceTraceQueue = traceHitQueueBuffer.castedPointer<WorkQueue>();

				//CUDABuffer traceShadowBuffer;
				//traceShadowBuffer.alloc(sizeof(RayData*) * maxQueueSize);
				//UPLOAD_DATA->launchParams.shadowTraceQueue = traceShadowBuffer.castedPointer<RayData*>();
				////CUDABuffer traceShadowQueueBuffer;
				////WorkQueue traceShadowQueue;
				////traceShadowQueue.maxSize = maxQueueSize;
				////traceShadowQueueBuffer.upload(traceShadowQueue);
				////UPLOAD_DATA->launchParams.shadowTraceQueue = traceShadowQueueBuffer.castedPointer<WorkQueue>();

				//CUDABuffer accumulationDataBuffer;
				//accumulationDataBuffer.alloc(sizeof(RayData*) * maxQueueSize);
				//UPLOAD_DATA->launchParams.accumulationQueue = accumulationDataBuffer.castedPointer<RayData*>();
				////CUDABuffer accumulationQueueBuffer;
				////WorkQueue accumulationQueue;
				////accumulationQueue.maxSize = maxQueueSize;
				////accumulationQueueBuffer.upload(accumulationQueue);
				////UPLOAD_DATA->launchParams.accumulationQueue = accumulationQueueBuffer.castedPointer<WorkQueue>();

				//constexpr int startSize = 0;
				//CUDABuffer radianceTraceQueueSizeBuffer;
				//radianceTraceQueueSizeBuffer.upload(startSize);
				//UPLOAD_DATA->launchParams.radianceTraceQueueSize = radianceTraceQueueSizeBuffer.castedPointer<int>();

				//CUDABuffer shadowTraceQueueSizeBuffer;
				//shadowTraceQueueSizeBuffer.upload(startSize);
				//UPLOAD_DATA->launchParams.shadowTraceQueueSize = shadowTraceQueueSizeBuffer.castedPointer<int>();

				//CUDABuffer shadeQueueSizeBuffer;
				//shadeQueueSizeBuffer.upload(startSize);
				//UPLOAD_DATA->launchParams.shadeQueueSize = shadeQueueSizeBuffer.castedPointer<int>();

				//CUDABuffer escapedQueueSizeBuffer;
				//escapedQueueSizeBuffer.upload(startSize);
				//UPLOAD_DATA->launchParams.escapedQueueSize = escapedQueueSizeBuffer.castedPointer<int>();

				//CUDABuffer accumulationQueueSizeBuffer;
				//accumulationQueueSizeBuffer.upload(startSize);
				//UPLOAD_DATA->launchParams.accumulationQueueSize = accumulationQueueSizeBuffer.castedPointer<int>();

				//CUDABuffer maxQueueSizeBuffer;
				//maxQueueSizeBuffer.upload(maxQueueSize);
				//UPLOAD_DATA->launchParams.maxQueueSize = maxQueueSizeBuffer.castedPointer<int>();
			}
		}

		if(rendererNode->settings.isUpdated)
		{
			UPLOAD_DATA->settings.iteration         = rendererNode->settings.iteration;
			UPLOAD_DATA->settings.accumulate        = rendererNode->settings.accumulate;
			UPLOAD_DATA->settings.maxBounces        = rendererNode->settings.maxBounces;
			UPLOAD_DATA->launchParams.remainingBounces = rendererNode->settings.maxBounces;
			UPLOAD_DATA->settings.displayBuffer     = rendererNode->settings.displayBuffer;
			UPLOAD_DATA->settings.samplingTechnique = rendererNode->settings.samplingTechnique;
			UPLOAD_DATA->settings.minClamp          = rendererNode->settings.minClamp;
			UPLOAD_DATA->settings.maxClamp          = rendererNode->settings.maxClamp;

			UPLOAD_DATA->settings.adaptiveSampling = rendererNode->settings.adaptiveSampling;
			UPLOAD_DATA->settings.minAdaptiveSamples = rendererNode->settings.minAdaptiveSamples;
			UPLOAD_DATA->settings.minPixelSamples = rendererNode->settings.minPixelSamples;
			UPLOAD_DATA->settings.maxPixelSamples = rendererNode->settings.maxPixelSamples;
			UPLOAD_DATA->settings.noiseCutOff = rendererNode->settings.noiseCutOff;
			UPLOAD_DATA->settings.removeFirefly = rendererNode->settings.removeFireflies;
			UPLOAD_DATA->settings.enableDenoiser = rendererNode->settings.enableDenoiser;
			UPLOAD_DATA->settings.useRussianRoulette = rendererNode->settings.useRussianRoulette;

			UPLOAD_DATA->isSettingsUpdated   = true;
			rendererNode->settings.isUpdated = false;
		}

		if (rendererNode->toneMapperSettings.isUpdated)
		{
			UPLOAD_DATA->toneMapperSettings.invWhitePoint = 1.0f/ rendererNode->toneMapperSettings.whitePoint;
			UPLOAD_DATA->toneMapperSettings.colorBalance = rendererNode->toneMapperSettings.colorBalance;
			UPLOAD_DATA->toneMapperSettings.burnHighlights = rendererNode->toneMapperSettings.burnHighlights;
			UPLOAD_DATA->toneMapperSettings.crushBlacks = rendererNode->toneMapperSettings.crushBlacks;
			UPLOAD_DATA->toneMapperSettings.saturation = rendererNode->toneMapperSettings.saturation;
			UPLOAD_DATA->toneMapperSettings.invGamma = 1.0f/ rendererNode->toneMapperSettings.gamma;

			UPLOAD_DATA->isToneMapperSettingsUpdated = true;
			rendererNode->toneMapperSettings.isUpdated = false;
		}


	}

	void setCameraData(std::shared_ptr<graph::Camera> cameraNode)
	{
		if (cameraNode->updated) {
			UPLOAD_DATA->cameraData.position   = cameraNode->position;
			UPLOAD_DATA->cameraData.direction  = cameraNode->direction;
			UPLOAD_DATA->cameraData.vertical   = cos(cameraNode->fovY) * cameraNode->vertical;
			UPLOAD_DATA->cameraData.horizontal = cos(cameraNode->fovY) * cameraNode->aspect * cameraNode->horizontal;
			cameraNode->updated                = false;
			UPLOAD_DATA->isCameraUpdated       = true;
		}
	}


	SbtProgramIdx setProgramsSbt()
	{
		// This is hard Coded, I didn't find a better way which didn't have some hardcoded design or some hash map conversion from string on the device
		SbtProgramIdx                             pg;
		optix::PipelineOptix*                     rp     = optix::getRenderingPipeline();

		pg.raygen          = rp->getProgramSbt("raygen");
		pg.exception       = rp->getProgramSbt("exception");
		pg.miss            = rp->getProgramSbt("miss");
		pg.hit             = rp->getProgramSbt("hit");
		pg.pinhole         = rp->getProgramSbt("pinhole");
		pg.meshLightSample = rp->getProgramSbt("meshLightSample");
		pg.envLightSample  = rp->getProgramSbt("envLightSample");

		return pg;
	}

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR&  descArray3D,
		const size_t&                   sizeBytesPerElement,
		CUarray&                        array)
	{
		CUresult result;


		result = cuArray3DCreate(&array, &descArray3D);
		CU_CHECK(result);

		for (size_t i = 0; i < imageLayers.size(); ++i)
		{
			const void* layerSource = imageLayers[i];

			CUDA_MEMCPY3D params = {};

			params.srcMemoryType = CU_MEMORYTYPE_HOST;
			params.srcHost       = layerSource;
			params.srcPitch      = descArray3D.Width * sizeBytesPerElement * descArray3D.NumChannels;
			params.srcHeight     = descArray3D.Height;

			params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			params.dstArray      = array;

			if (descArray3D.Depth != 0)
			{
				params.dstXInBytes = 0;
				params.dstY        = 0;
				params.dstZ        = i;
			}

			params.WidthInBytes = params.srcPitch;
			params.Height       = descArray3D.Height;
			params.Depth        = 1;

			result = cuMemcpy3D(&params);
			CU_CHECK(result);
		}

		CUDA_RESOURCE_DESC resourceDescription = {};
		resourceDescription.resType            = CU_RESOURCE_TYPE_ARRAY;
		resourceDescription.res.array.hArray   = array;

		return resourceDescription;
	}
}

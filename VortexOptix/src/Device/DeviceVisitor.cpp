#include "DeviceVisitor.h"
#include "CUDAChecks.h"
#include "MDL/mdlWrapper.h"
#include "Device/OptixWrapper.h"
#include "Scene/Graph.h"
#include "DevicePrograms/LaunchParams.h"
#include "UploadCode/UploadBuffers.h"
#include "UploadCode/UploadData.h"
#include "UploadCode/UploadFunctions.h"

namespace vtx::device
{
	void DeviceVisitor::visit(const std::shared_ptr<graph::Instance> instance)
	{
		// If the child node is a mesh, then it's leaf therefore we can safely create the instance.
		// This supposes that child and transform are traversed before the instance visitor is accepted.
		if (const std::shared_ptr<graph::Mesh> meshNode = std::dynamic_pointer_cast<graph::Mesh>(instance->getChild())) {
			// TODO Check if transforms meshes or material have been changed
			if (const vtxID instanceId = instance->getID(); !UPLOAD_DATA->instanceDataMap.contains(instanceId)) {

				const vtxID meshId = meshNode->getID();

				const OptixTraversableHandle& traversable = UPLOAD_DATA->geometryDataMap[meshId].traversable;

				if (isTransformStackUpdated() || !(instance->finalTransformStack==transformIndexStack))
				{
					instance->finalTransform = getFinalTransform();
					instance->finalTransformStack = transformIndexStack;
				}

				const OptixInstance optixInstance = optix::createInstance(instanceId, instance->finalTransform, traversable);
				const InstanceData instanceData = createInstanceData(instance, instance->finalTransform);

				UPLOAD_DATA->instanceDataMap.insert(instanceId, instanceData);
				UPLOAD_DATA->optixInstances.push_back(optixInstance);

			}
		}
		popTransform();
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Transform> transform)
	{
		pushTransform(transform->getID(), transform->isUpdated);
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Group> group)
	{
		popTransform();
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Mesh> mesh)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID meshId = mesh->getID(); !UPLOAD_DATA->geometryDataMap.contains(meshId)) {
			const GeometryData geometryData = createGeometryData(mesh);
			UPLOAD_DATA->geometryDataMap.insert(meshId, geometryData);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Material> material)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID materialId = material->getID(); !UPLOAD_DATA->materialDataMap.contains(materialId)) {
			const MaterialData materialData = createMaterialData(material);
			UPLOAD_DATA->materialDataMap.insert(materialId, materialData);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Camera> camera)
	{
		setCameraData(camera);
	}

	void DeviceVisitor::visit(const std::shared_ptr < graph::Renderer > renderer)
	{
		setRendererData(renderer);
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Shader> shader)
	{
		if (const vtxID shaderId = shader->getID(); !UPLOAD_DATA->shaderDataMap.contains(shaderId)) {
			const ShaderData shaderData = createShaderData(shader);
			UPLOAD_DATA->shaderDataMap.insert(shaderId, shaderData);
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Texture> textureNode)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID textureId = textureNode->getID(); !UPLOAD_DATA->textureDataMap.contains(textureId)) {
			const TextureData textureData = createTextureData(textureNode);
			UPLOAD_DATA->textureDataMap.insert(textureNode->getID(), textureData);
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurementNode)
	{
		///TODO : Check if the texture has been updated
		if (const vtxID bsdfId = bsdfMeasurementNode->getID(); !UPLOAD_DATA->bsdfDataMap.contains(bsdfId)) {
			const BsdfData bsdfData = createBsdfData(bsdfMeasurementNode);
			UPLOAD_DATA->bsdfDataMap.insert(bsdfId, bsdfData);
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::LightProfile> lightProfile)
	{
		///TODO : Check if the light Profile has been updated
		if (const vtxID lightProfileId = lightProfile->getID(); !UPLOAD_DATA->lightProfileDataMap.contains(lightProfileId)) {
			const LightProfileData lightProfileData = createLightProfileData(lightProfile);
			UPLOAD_DATA->lightProfileDataMap.insert(lightProfileId, lightProfileData);
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Light> lightNode)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID lightId = lightNode->getID(); !UPLOAD_DATA->lightDataMap.contains(lightId)) {
			LightData lightData = createLightData(lightNode);
			UPLOAD_DATA->lightDataMap.insert(lightId, lightData);
			if(lightData.type == LightType::L_ENV)
			{
				// TODO : How do we handle the presence of another env Light?
				UPLOAD_DATA->launchParams.envLightId = lightId;
			}
		}
	}

	void finalizeUpload()
	{
		UploadData* uploadData = UPLOAD_DATA;
		Buffers* uploadBuffers = UPLOAD_BUFFERS;

		bool isLaunchParamsUpdated = false;
		if (uploadData->instanceDataMap.isUpdated)
		{
			uploadData->launchParams.topObject = optix::createInstanceAcceleration(uploadData->optixInstances);
			uploadData->launchParams.instanceMap = uploadData->instanceDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->geometryDataMap.isUpdated)
		{
			uploadData->launchParams.geometryMap = uploadData->geometryDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->materialDataMap.isUpdated)
		{
			uploadData->launchParams.materialMap = uploadData->materialDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->lightDataMap.isUpdated)
		{
			uploadData->launchParams.lightMap = uploadData->lightDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->shaderDataMap.isUpdated)
		{
			uploadData->launchParams.shaderMap = uploadData->shaderDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->textureDataMap.isUpdated)
		{
			uploadData->launchParams.textureMap = uploadData->textureDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->bsdfDataMap.isUpdated)
		{
			uploadData->launchParams.bsdfMap = uploadData->bsdfDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if (uploadData->lightProfileDataMap.isUpdated)
		{
			uploadData->launchParams.lightProfileMap = uploadData->lightProfileDataMap.upload();
			isLaunchParamsUpdated = true;
		}
		if(uploadData->isCameraUpdated)
		{
			uploadData->launchParams.cameraData = uploadData->cameraData;
			uploadData->isCameraUpdated = false;
			isLaunchParamsUpdated = true;
		}
		if(uploadData->isFrameBufferUpdated)
		{
			uploadData->launchParams.frameBuffer = uploadData->frameBufferData;
			uploadData->isCameraUpdated = false;
			isLaunchParamsUpdated = true;
		}
		if(uploadData->isFrameIdUpdated)
		{
			// TEST I don't really want to upload the whole launchparams just because I updated the frame count
			// Here I ma trying to check if it is the first time I upload the frame count, in that case, launch params needs to store the proper device ptr
			// However, if it is not the first time, since frameID will always have the same sizeof(int) the device ptr will be the same
			// and I shouldn't need to update the launch params
			uploadBuffers->frameIdBuffer.upload(uploadData->frameId);
			if(!uploadData->launchParams.frameID || uploadData->launchParams.frameID != uploadBuffers->frameIdBuffer.castedPointer<int>())
			{
				uploadData->launchParams.frameID = uploadBuffers->frameIdBuffer.castedPointer<int>();
				isLaunchParamsUpdated = true;
			}
		}
		if(uploadData->isSettingsUpdated)
		{
			uploadBuffers->rendererSettingsBuffer.upload(uploadData->settings);
			if (!uploadData->launchParams.settings || uploadData->launchParams.settings != uploadBuffers->frameIdBuffer.castedPointer<RendererDeviceSettings>())
			{
				uploadData->launchParams.settings = uploadBuffers->rendererSettingsBuffer.castedPointer<RendererDeviceSettings>();
				isLaunchParamsUpdated = true;
			}
		}

		//ATTENTION We do this only once! (hence the no update) However, this might be a way to swap rendering pipeline by changing the set of programSbt
		if (!uploadData->launchParams.programs)
		{
			uploadData->programs = setProgramsSbt();
			uploadBuffers->sbtProgramIdxBuffer.upload(uploadData->programs);
			uploadData->launchParams.programs = uploadBuffers->sbtProgramIdxBuffer.castedPointer<SbtProgramIdx>();
			isLaunchParamsUpdated = true;
		}

		if(isLaunchParamsUpdated)
		{
			uploadBuffers->launchParamsBuffer.upload(uploadData->launchParams);
		}
	}

	void incrementFrame() {
		UPLOAD_DATA->frameId++;
		UPLOAD_DATA->isFrameIdUpdated = true;
	}

}

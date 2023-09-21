#include "DeviceVisitor.h"
#include "CUDAChecks.h"
#include "MDL/mdlWrapper.h"
#include "Device/OptixWrapper.h"
#include "Scene/Graph.h"
#include "DevicePrograms/LaunchParams.h"
#include "Scene/Nodes/EnvironmentLight.h"
#include "Scene/Nodes/MeshLight.h"
#include "UploadCode/UploadBuffers.h"
#include "UploadCode/UploadData.h"
#include "UploadCode/UploadFunctions.h"
#include "UploadCode/CUDAMap.h"


namespace vtx::device
{
	void DeviceVisitor::visit(const std::shared_ptr<graph::Instance>& instance)
	{
		// If the child node is a mesh, then it's leaf therefore we can safely create the instance.
		// This supposes that child and transform are traversed before the instance visitor is accepted.
		if (const std::shared_ptr<graph::Mesh> meshNode = std::dynamic_pointer_cast<graph::Mesh>(instance->getChild())) {
			// TODO Check if transforms meshes or material have been changed
			if (const vtxID instanceId = instance->getID(); !UPLOAD_DATA->instanceDataMap.contains(instanceId)) {

				const vtxID meshId = meshNode->getID();

				const OptixTraversableHandle& traversable = std::get<0>(UPLOAD_DATA->geometryDataMap[meshId]).traversable;

				//if (isTransformStackUpdated() || instance->finalTransformStack!=transformIndexStack)
				//{
				//	instance->finalTransform = getFinalTransform();
				//	instance->finalTransformStack = transformIndexStack;
				//}

				const math::affine3f& globalTransform = instance->transform->globalTransform;
				const OptixInstance                           optixInstance = optix::createInstance(instanceId, globalTransform, traversable);
				const std::tuple<InstanceData, InstanceData*> instanceData = createInstanceData(instance, globalTransform);

				UPLOAD_DATA->instanceDataMap.insert(instanceId, instanceData);
				UPLOAD_DATA->optixInstances.push_back(optixInstance);

			}
		}
		//popTransform();
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Transform>& transform)
	{
		//pushTransform(transform->getID(), transform->isUpdated);
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Group>& group)
	{
		//popTransform();
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Mesh>& mesh)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID meshId = mesh->getID(); !UPLOAD_DATA->geometryDataMap.contains(meshId)) {
			const std::tuple<GeometryData, GeometryData*> geo = createGeometryData(mesh);
			UPLOAD_DATA->geometryDataMap.insert(meshId, geo);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Material>& material)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID materialId = material->getID(); (!UPLOAD_DATA->materialDataMap.contains(materialId) || material->isUpdated)) {
			int matQueueId = UPLOAD_DATA->materialDataMap.size;
			const std::tuple<MaterialData, MaterialData*> mat = createMaterialData(material, matQueueId);
			UPLOAD_DATA->materialDataMap.insert(materialId, mat);
			material->isUpdated = false;
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Camera>& camera)
	{
		setCameraData(camera);
	}

	void DeviceVisitor::visit(const std::shared_ptr < graph::Renderer >& renderer)
	{
		setRendererData(renderer);
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::Texture>& textureNode)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID textureId = textureNode->getID(); !UPLOAD_DATA->textureDataMap.contains(textureId)) {
			const std::tuple<TextureData, TextureData*> textureData = createTextureData(textureNode);
			UPLOAD_DATA->textureDataMap.insert(textureNode->getID(), textureData);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurementNode)
	{
		///TODO : Check if the texture has been updated
		if (const vtxID bsdfId = bsdfMeasurementNode->getID(); !UPLOAD_DATA->bsdfDataMap.contains(bsdfId)) {
			const std::tuple<BsdfData, BsdfData*> bsdfData = createBsdfData(bsdfMeasurementNode);
			UPLOAD_DATA->bsdfDataMap.insert(bsdfId, bsdfData);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::LightProfile>& lightProfile)
	{
		///TODO : Check if the light Profile has been updated
		if (const vtxID lightProfileId = lightProfile->getID(); !UPLOAD_DATA->lightProfileDataMap.contains(lightProfileId)) {
			const std::tuple<LightProfileData, LightProfileData*> lightProfileData = createLightProfileData(lightProfile);
			UPLOAD_DATA->lightProfileDataMap.insert(lightProfileId, lightProfileData);
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::MeshLight>& meshLight)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID lightId = meshLight->getID(); !UPLOAD_DATA->lightDataMap.contains(lightId)) {
			const std::tuple<LightData, LightData*> lightData = createMeshLightData(meshLight);
			// We keep a list on the device of all potential mesh lights but only the mesh lights associated with a material
			// markes as light will be sampled in the mis.
			if (meshLight->material->useAsLight)
			{
				UPLOAD_DATA->lightDataMap.insert(lightId, std::tuple(std::get<0>(lightData), std::get<1>(lightData), true));
			}
			else
			{
				UPLOAD_DATA->lightDataMap.insert(lightId, std::tuple(std::get<0>(lightData), std::get<1>(lightData), false));
			}
		}
	}

	void DeviceVisitor::visit(const std::shared_ptr<graph::EnvironmentLight>& envLight)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID lightId = envLight->getID(); !UPLOAD_DATA->lightDataMap.contains(lightId)) {
			const std::tuple<LightData, LightData*> lightData = createEnvLightData(envLight);
			// We keep a list on the device of all potential mesh lights but only the mesh lights associated with a material
			// markes as light will be sampled in the mis.
			// TODO : How do we handle the presence of another env Light?
			UPLOAD_DATA->lightDataMap.insert(lightId, std::tuple(std::get<0>(lightData), std::get<1>(lightData), true));
			UPLOAD_DATA->launchParams.envLight = std::get<1>(lightData);
		}
	}

	void finalizeUpload()
	{
		UploadData* uploadData    = UPLOAD_DATA;
		Buffers*    uploadBuffers = UPLOAD_BUFFERS;

		bool isLaunchParamsUpdated = false;
		if (uploadData->instanceDataMap.isUpdated && uploadData->instanceDataMap.size != 0)
		{
			std::vector<InstanceData*> instances;
			uploadData->instanceDataMap.finalize();
			uploadData->instanceDataMap.isUpdated = false;
			for (std::pair<vtxID, std::tuple<InstanceData, InstanceData*>> it : uploadData->instanceDataMap)
			{
				InstanceData* instanceData = std::get<1>(it.second);
				vtxID& instanceId = it.first;
				if (instanceId >= instances.size())
				{
					instances.resize(instanceId + 1);
				}
				instances[instanceId] = instanceData;
			}
			uploadBuffers->instancesBuffer.upload(instances);

			uploadData->launchParams.topObject = optix::createInstanceAcceleration(uploadData->optixInstances);
			uploadData->launchParams.instances = uploadBuffers->instancesBuffer.castedPointer<InstanceData*>();
			isLaunchParamsUpdated = true;
			uploadData->instanceDataMap.isUpdated = false;
		}
		if (uploadData->lightDataMap.isUpdated && uploadData->lightDataMap.size != 0)
		{
			std::vector<LightData*> lightData;
			for (auto lightEntry : uploadData->lightDataMap)
			{
				if(std::get<2>(lightEntry.second)) // we only add the lights that are marked as lights
				{
					LightData* light = std::get<1>(lightEntry.second);
					lightData.push_back(light);
				}
			}
			uploadBuffers->lightsDataBuffer.upload(lightData);
			uploadData->launchParams.lights = uploadBuffers->lightsDataBuffer.castedPointer<LightData*>();
			uploadData->launchParams.numberOfLights = lightData.size();
			isLaunchParamsUpdated = true;
			uploadData->lightDataMap.isUpdated = false;
		}
		if(uploadData->isCameraUpdated)
		{
			uploadData->launchParams.cameraData = uploadData->cameraData;
			uploadData->isCameraUpdated = false;
			isLaunchParamsUpdated = true;
		}
		if (uploadData->materialDataMap.isUpdated)
		{
			uploadData->materialDataMap.isUpdated = false;
		}

		if(uploadData->isFrameBufferUpdated)
		{
			uploadData->launchParams.frameBuffer = uploadData->frameBufferData;
			uploadData->isCameraUpdated = false;
			uploadData->isFrameBufferUpdated = false;
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
			uploadData->isFrameIdUpdated = false;
		}
		if(uploadData->isSettingsUpdated)
		{
			OnDeviceSettings* onDeviceSettings = uploadBuffers->rendererSettingsBuffer.upload(uploadData->settings);
			if (!uploadData->launchParams.settings || uploadData->launchParams.settings != onDeviceSettings)
			{
				uploadData->launchParams.settings = onDeviceSettings;
				isLaunchParamsUpdated = true;
			}
			uploadData->isSettingsUpdated = false;
		}

		//ATTENTION We do this only once! (hence the no update) However, this might be a way to swap rendering pipeline by changing the set of programSbt
		//if (!uploadData->launchParams.programs)
		//{
		//	uploadData->programs = setProgramsSbt();
		//	uploadBuffers->sbtProgramIdxBuffer.upload(uploadData->programs);
		//	uploadData->launchParams.programs = uploadBuffers->sbtProgramIdxBuffer.castedPointer<SbtProgramIdx>();
		//	isLaunchParamsUpdated = true;
		//}
		if(uploadData->isNetworkInterfaceUpdated)
		{
			isLaunchParamsUpdated = true;
			uploadData->isNetworkInterfaceUpdated = false;
		}
		if(uploadData->isWorkQueueUpdated)
		{
			isLaunchParamsUpdated = true;
			uploadData->isWorkQueueUpdated = false;
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

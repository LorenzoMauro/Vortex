#pragma once
#include "UploadBuffers.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Scene/Nodes/Shader/BsdfMeasurement.h"
#include "Scene/Node.h"

namespace vtx::device
{
	std::tuple<InstanceData, InstanceData*> createInstanceData(const std::shared_ptr<graph::Instance>& instanceNode, const math::affine3f& transform);

	std::tuple<LightData, LightData*> createMeshLightData(const std::shared_ptr<graph::MeshLight>& meshLight);

	std::tuple<LightData, LightData*> createEnvLightData(std::shared_ptr<graph::EnvironmentLight> envLight);

	/*Create BLAS and GeometryDataStruct given vertices attributes and indices*/
	std::tuple<GeometryData, GeometryData*> createGeometryData(const std::shared_ptr<graph::Mesh>& meshNode);

	void uploadMaps();

	std::tuple<MaterialData, MaterialData*> createMaterialData(const std::shared_ptr<graph::Material>& material, int matQueueId);

	DeviceShaderConfiguration* createDeviceShaderConfiguration(const std::shared_ptr<graph::Material>& material);

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR& descArray3D,
		const size_t& sizeBytesPerElement,
		CUarray& array);

	std::tuple< TextureData, TextureData*> createTextureData(const std::shared_ptr<vtx::graph::Texture>& textureNode);

	BsdfSamplingPartData createBsdfPartData(const graph::BsdfMeasurement::BsdfPartData& bsdfData, Buffers::BsdfPartBuffer& buffers);

	std::tuple < BsdfData, BsdfData*> createBsdfData(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement);

	std::tuple < LightProfileData, LightProfileData*> createLightProfileData(std::shared_ptr<graph::LightProfile> lightProfile);

	void setRendererData(const std::shared_ptr<graph::Renderer>& rendererNode);

	void setCameraData(const std::shared_ptr<graph::Camera>& cameraNode);

	SbtProgramIdx setProgramsSbt();
}

#pragma once
#include "Device/DevicePrograms/LaunchParams.h"
#include "Scene/Nodes/Shader/BsdfMeasurement.h"
#include "Scene/Node.h"

namespace vtx::device
{
	struct BsdfPartBuffer;
	InstanceData createInstanceData(const std::shared_ptr<graph::Instance>& instanceNode);

	LightData createMeshLightData(const std::shared_ptr<graph::MeshLight>& meshLight);

	LightData createEnvLightData(std::shared_ptr<graph::EnvironmentLight> envLight);

	/*Create BLAS and GeometryDataStruct given vertices attributes and indices*/
	GeometryData createGeometryData(const std::shared_ptr<graph::Mesh>& meshNode);

	void uploadMaps();

	MaterialData createMaterialData(const std::shared_ptr<graph::Material>& material);

	DeviceShaderConfiguration* createDeviceShaderConfiguration(const std::shared_ptr<graph::Material>& material);

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR& descArray3D,
		const size_t& sizeBytesPerElement,
		CUarray& array);

	TextureData createTextureData(const std::shared_ptr<vtx::graph::Texture>& textureNode);

	BsdfSamplingPartData createBsdfPartData(const graph::BsdfMeasurement::BsdfPartData& bsdfData, BsdfPartBuffer& buffers);

	BsdfData createBsdfData(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement);

	LightProfileData createLightProfileData(std::shared_ptr<graph::LightProfile> lightProfile);

	void setRendererData(const std::shared_ptr<graph::Renderer>& rendererNode);

	CameraData createCameraData(const std::shared_ptr<graph::Camera>& cameraNode);

}

#pragma once
#include "UploadBuffers.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Scene/Nodes/Shader/BsdfMeasurement.h"

namespace vtx
{
	namespace graph
	{
		class Camera;
		class Renderer;
		class LightProfile;
		class Texture;
		class Shader;
		class Material;
		class Mesh;
		class Light;
		class Instance;
	}
}

namespace vtx::device
{
	std::tuple<InstanceData, InstanceData*> createInstanceData(std::shared_ptr<graph::Instance> instanceNode, const math::affine3f& transform);

	std::tuple<LightData, LightData*> createLightData(std::shared_ptr<graph::Light> lightNode);

	/*Create BLAS and GeometryDataStruct given vertices attributes and indices*/
	std::tuple<GeometryData, GeometryData*> createGeometryData(std::shared_ptr<graph::Mesh> meshNode);

	void uploadMaps();

	std::tuple<MaterialData, MaterialData*>createMaterialData(std::shared_ptr<graph::Material> material);

	DeviceShaderConfiguration createDeviceShaderConfiguration(std::shared_ptr<graph::Shader> shader);

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR& descArray3D,
		const size_t& sizeBytesPerElement,
		CUarray& array);

	std::tuple< TextureData, TextureData*> createTextureData(std::shared_ptr<vtx::graph::Texture>& textureNode);

	BsdfSamplingPartData createBsdfPartData(graph::BsdfMeasurement::BsdfPartData& bsdfData, Buffers::BsdfPartBuffer& buffers);

	std::tuple < BsdfData, BsdfData*> createBsdfData(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement);

	std::tuple < LightProfileData, LightProfileData*> createLightProfileData(std::shared_ptr<graph::LightProfile> lightProfile);

	void computeNoiseInfo(std::shared_ptr<graph::Renderer> rendererNode);

	void setRendererData(std::shared_ptr<graph::Renderer> rendererNode);

	void setCameraData(std::shared_ptr<graph::Camera> cameraNode);

	SbtProgramIdx setProgramsSbt();
}

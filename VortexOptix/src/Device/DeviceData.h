#pragma once
#include "DevicePrograms/LaunchParams.h"
#include "Scene/Traversal.h"
#include "CUDAMap.h"
#include <map>

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "NodesBuffers.h"
#include "Core/Input/keycodes.h"
#include "Scene/Nodes/Shader/BsdfMeasurement.h"

namespace vtx::graph {
	class Node;
	class Transform;
	class Instance;
	class Group;
	class Mesh;
	class Material;
	class Camera;
	class Renderer;
	struct VertexAttributes;
}

namespace vtx::device
{
	struct Buffers
	{
		struct GeometryBuffers
		{
			CUDABuffer vertexBuffer;
			CUDABuffer indexBuffer;

			GeometryBuffers() = default;
			~GeometryBuffers()
			{
				vertexBuffer.free();
				indexBuffer.free();
			}
		};

		struct InstanceBuffers
		{
			CUDABuffer materialsIdBuffer;

			InstanceBuffers() = default;
			~InstanceBuffers()
			{
				materialsIdBuffer.free();
			}
		};

		struct MaterialBuffers
		{
			CUDABuffer argBlockBuffer;

			MaterialBuffers() = default;
			~MaterialBuffers()
			{
				if(argBlockBuffer.dPointer())
				{
					argBlockBuffer.free();
				}
			}
		};

		struct ShaderBuffers
		{
			CUDABuffer shaderConfigBuffer;
			CUDABuffer textureIdBuffer;
			CUDABuffer bsdfIdBuffer;
			CUDABuffer lightProfileBuffer;
			CUDABuffer TextureHandlerBuffer;

			ShaderBuffers() = default;
			~ShaderBuffers()
			{
				if (shaderConfigBuffer.dPointer()) { shaderConfigBuffer.free(); }
				if (textureIdBuffer.dPointer()) { textureIdBuffer.free(); }
				if (bsdfIdBuffer.dPointer()) { bsdfIdBuffer.free(); }
				if (lightProfileBuffer.dPointer()) { lightProfileBuffer.free(); }
				if (TextureHandlerBuffer.dPointer()) {TextureHandlerBuffer.free();}
			}
		};

		struct BsdfPartBuffer
		{
			CUDABuffer sampleData;
			CUDABuffer albedoData;
			CUDABuffer partBuffer;
			CUarray	   lookUpArray;

			BsdfPartBuffer() = default;
			~BsdfPartBuffer()
			{
				if (sampleData.dPointer()) { sampleData.free(); }
				if (albedoData.dPointer()) { albedoData.free(); }
				if (partBuffer.dPointer()){partBuffer.free();}
				const cudaError result = cudaFree((void*)lookUpArray);
				CUDA_CHECK(result);
			}

		};

		struct BsdfBuffers
		{
			BsdfPartBuffer	reflectionPartBuffer;
			BsdfPartBuffer	transmissionPartBuffer;
		};

		struct LightProfileBuffers
		{
			CUDABuffer cdfBuffer;
			CUarray	   lightProfileSourceArray;

			LightProfileBuffers() = default;
			~LightProfileBuffers()
			{
				cdfBuffer.free();
				const cudaError result = cudaFree((void*)lightProfileSourceArray);
				CUDA_CHECK(result);
			}
		};

		struct FrameBufferBuffers
		{
			CUDABuffer cudaColorBuffer;

			FrameBufferBuffers() = default;
			~FrameBufferBuffers()
			{
				cudaColorBuffer.free();
			}
		};

		struct TextureBuffers
		{
			CUarray textureArray;

			TextureBuffers() = default;
			~TextureBuffers()
			{
				const cudaError result = cudaFree((void*)textureArray);
				CUDA_CHECK(result);
			}
		};


		/*InstanceBuffers& getBufferCollectionElement(std::map<vtxID, InstanceBuffers>& bufferCollectionMap, const vtxID nodeId)
		{
			InstanceBuffers* bufferCollection;
			if (const auto& it = bufferCollectionMap.find(nodeId); it != bufferCollectionMap.end())
				bufferCollection = &(it->second);
			else
			{
				bufferCollectionMap.insert({nodeId, InstanceBuffers{}});
				bufferCollection = &(bufferCollectionMap[nodeId]);
			}

			return *bufferCollection;
		}*/

		template<typename T>
		T& getBufferCollectionElement(std::map<vtxID, T>& bufferCollectionMap, const vtxID nodeId)
		{
			if (const auto it = bufferCollectionMap.find(nodeId); it != bufferCollectionMap.end())
				return it->second;
			bufferCollectionMap.try_emplace(nodeId, T());
			return bufferCollectionMap[nodeId];
		}

		template<typename T>
		T& getBuffer(const vtxID nodeId);

		template<>
		InstanceBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(instance, nodeId);
		}

		template<>
			GeometryBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(geometry, nodeId);
		}

		template<>
			MaterialBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(material, nodeId);
		}

		template<>
			ShaderBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(shader, nodeId);
		}

		template<>
			TextureBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(texture, nodeId);
		}

		template<>
			BsdfBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(bsdf, nodeId);
		}

		template<>
			LightProfileBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(lightProfile, nodeId);
		}

		template<>
			FrameBufferBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(frameBuffer, nodeId);
		}

		std::map<vtxID, InstanceBuffers>			instance;
		std::map<vtxID, GeometryBuffers>			geometry;
		std::map<vtxID, MaterialBuffers>			material;
		std::map<vtxID, ShaderBuffers>				shader;
		std::map<vtxID, TextureBuffers>				texture;
		std::map<vtxID, BsdfBuffers>				bsdf;
		std::map<vtxID, LightProfileBuffers>		lightProfile;
		std::map<vtxID, FrameBufferBuffers>			frameBuffer;
		CUDABuffer									frameIdBuffer;
	};

	struct UploadData {
		//This first Set of Data is used to manage the cuda memory buffer memory, not create a new memory everytime something changes.
		// Each node will use its own buffer, and the buffer will be updated when the node is visited.
		Buffers									buffers;

#define GET_BUFFER(type, index, element) \
		uploadData.buffers.getBuffer<type>(index).element
		// The following maps will be uploaded to the device
		// the launch params will contain the pointers to the maps
		// data is reference by ids;
		CudaMap<vtxID, InstanceData>			instanceDataMap;
		CudaMap<vtxID, GeometryData>			geometryDataMap;
		CudaMap<vtxID, MaterialData>			materialDataMap;
		CudaMap<vtxID, ShaderData>				shaderDataMap;
		CudaMap<vtxID, TextureData>				textureDataMap;
		CudaMap<vtxID, BsdfData>				bsdfDataMap;
		CudaMap<vtxID, LightProfileData>		lightProfileDataMap;
		CameraData								cameraData;
		bool									isCameraUpdated;
		FrameBufferData							frameBufferData;
		bool									isFrameBufferUpdated;
		int										frameId = 0;
		bool									isFrameIdUpdated;

		std::vector<OptixInstance>				optixInstances;
		LaunchParams							launchParams;
		CUDABuffer								launchParamsBuffer;
	};

	class DeviceVisitor : public NodeVisitor {
	public:
		DeviceVisitor():
			currentTransform(math::Identity),
			previousTransform(math::Identity)
		{};
		void visit(std::shared_ptr<graph::Instance> instance) override;
		void visit(std::shared_ptr<graph::Transform> transform) override;
		void visit(std::shared_ptr<graph::Group> group) override;
		void visit(std::shared_ptr<graph::Mesh> mesh) override;
		void visit(std::shared_ptr<graph::Material> material) override;
		void visit(std::shared_ptr<graph::Camera> camera) override;
		void visit(std::shared_ptr<graph::Renderer> renderer) override;
		void visit(std::shared_ptr<graph::Shader> shader) override;
		void visit(std::shared_ptr<graph::Texture> textureNode) override;
		void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurementNode) override;
		void visit(std::shared_ptr<graph::LightProfile> lightProfile) override;

		//InstanceData instanceData;

		math::affine3f currentTransform;
		math::affine3f previousTransform;
	};

	LaunchParams&	getLaunchParams();

	CUDABuffer&		getLaunchParamsBuffer();

	void finalizeUpload();

	void incrementFrame();

	InstanceData createInstanceData(std::shared_ptr<graph::Instance> instanceNode);

	/*Create BLAS and GeometryDataStruct given vertices attributes and indices*/
	GeometryData createGeometryData(std::shared_ptr<graph::Mesh> meshNode);

	void uploadMaps();

	MaterialData createMaterialData(std::shared_ptr<graph::Material> material);

	DeviceShaderConfiguration createDeviceShaderConfiguration(std::shared_ptr<graph::Shader> shader);

	ShaderData createShaderData(std::shared_ptr<graph::Shader> shaderNode);

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR& descArray3D,
		const size_t& sizeBytesPerElement,
		CUarray array);

	TextureData createTextureDeviceData(std::shared_ptr<vtx::graph::Texture>& textureNode);

	BsdfSamplingPartData createBsdfPartDeviceData(graph::BsdfMeasurement::BsdfPartData& bsdfData, Buffers::BsdfPartBuffer& buffers);

	BsdfData createBsdfDeviceData(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement);

	LightProfileData createLightProfileDeviceData(std::shared_ptr<graph::LightProfile> lightProfile);

}

/*

// launchParameters
	TLAS handle
	GeometryinstanceData device pointer

// InstanceData = struct that contain the transform and the id of the instance
		id Geometry??

// MeshData = struct that contain the id of the mesh and the id of the material
		Primitive Type (custom enumeration)
		Owner = it's the first device that created the Mesh Data, it's the one that is responsible to freeing it??
		Device Pointer of the VertexAttribute struct
		Device Pointer of the IndexAttribute struct
		number of vertices
		number of indices
		GAS handle
		Device Pointer of the GAS -- it first gets used as the outputBuffer d_pointer and then the compacted size buffer pointer

// Geometry instance Data
	Material, Light, Object, ID (+ pad) // HOWEVER in our case we don't have a material data per instance but per triangle so we have to think about this
	device pointer to vertex attributes
	device pointer to indices
*/

/*
* 
* create vector of :
*	MeshData
*	OptixInstances
*	InstanceData

// Create empty InstanceData and pass it to the traverse
// Create MeshData vector of the size of the meshes (leaf nodes?)

// Traverse Scene
// As a result we have filled the 3 vectors defined above and created the GAS
// The size of the mesh vector is equal to the number of mesh nodes while the isntance vector can be greater

// Create TLAS 
// As a result we have filled the top level handle to pass as launch parameter and the member variable pointer to the TLAS buffer

// Create Geometry Instance Data
// As a result we have filled the member vector of the DeviceGeometryInstanceData and upload it to a cuda Buffer + we have set the device pointer in the launchParams


*/

#pragma once
#include "LaunchParams.h"
#include "Scene/Traversal.h"
#include "CUDAMap.h"
#include <map>

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>


namespace vtx::graph {
	class Node;
	class Transform;
	class Instance;
	class Group;
	class Mesh;
	class Material;
	class Camera;
	class Renderer;
}

namespace vtx::device
{


	enum PrimitiveType {
		PT_TRIANGLES,

		NUM_PT
	};

	/// This Struct Are used during the elaboration of the scene but shouldn't be uploaded
	/*struct InstanceData {
			vtxID						InstanceId;
			vtxID						sequentialInstance;
			CUdeviceptr					GeometryData;
			CUdeviceptr					d_MaterialArray=NULL;
		};*/

	struct GeometryData {
		PrimitiveType				type;
		OptixTraversableHandle		traversable;
		CUdeviceptr					d_Gas;
		CUdeviceptr					d_VertexAttribute;
		CUdeviceptr					d_IndexAttribut;
		size_t						numVertices;
		size_t						numIndices;
	};

	struct MaterialData {
		vtxID						index;
	};

	struct UploadData {
		CudaMap<vtxID, GeometryData>			meshIdToGeometryData;
		CudaMap<vtxID, MaterialData>			materialIdToGeometryData;
		CudaMap<vtxID, GeometryInstanceData>	geometryInstanceDataMap;
		std::vector<OptixInstance>				optixInstances;
		LaunchParams							launchParams;
		CUDABuffer								launchParamsBuffer;
		CUDABuffer								cudaColorBuffer;
	};

	class DeviceVisitor : public NodeVisitor {
	public:
		DeviceVisitor():
			previousIntanceId(-1),
			currentInstanceID(-1),
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

		//InstanceData instanceData;

		vtxID previousIntanceId;
		vtxID currentInstanceID;
		math::affine3f currentTransform;
		math::affine3f previousTransform;
	};

	LaunchParams&	getLaunchParams();

	CUDABuffer&		getLaunchParamsBuffer();

	void finalizeUpload();

	void incrementFrame();

	void createInstanceData(GeometryData geometryData, vtxID instanceID, math::affine3f transform);

	GeometryData createBLAS(std::shared_ptr<graph::Mesh> mesh);

	OptixTraversableHandle createTLAS();

	CudaMap<vtxID, GeometryInstanceData>* uploadGeometryInstanceData();

	//CameraData* uploadCamera();
	//void setCameraData(std::shared_ptr<scene::Camera> camera);

	MaterialData createMaterialData(std::shared_ptr<graph::Material> material);

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
// As a result we have filled the member vector of the GeometryInstanceData and upload it to a cuda Buffer + we have set the device pointer in the launchParams


*/

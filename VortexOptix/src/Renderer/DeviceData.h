#pragma once
#include "Scene/SceneGraph.h"
#include "Scene/Camera.h"
#include "LaunchParams.h"
#include <map>

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace vtx {

	enum PrimitiveType {
		PT_TRIANGLES,

		NUM_PT
	};

	/// This Struct Are used during the elaboration of the scene but shouldn't be uploaded
	struct InstanceData {
		vtxID						InstanceId;
		vtxID						sequentialInstance;
		CUdeviceptr					GeometryData;
		CUdeviceptr					d_MaterialArray=NULL;
	};

	struct GeometryData {
		PrimitiveType				type;
		OptixTraversableHandle		traversable;
		CUdeviceptr					d_Gas;
		CUdeviceptr					d_VertexAttribute;
		CUdeviceptr					d_IndexAttribut;
		size_t						numVertices;
		size_t						numIndices;
	};


	class DeviceScene {
	public:
		//copy operator default
		void Traverse(std::shared_ptr<scene::Node> node, InstanceData instanceData, math::affine3f transform);

		void createInstanceData(GeometryData geometryData, InstanceData instanceData, math::affine3f transform);

		void createCameraData(std::shared_ptr<scene::Camera> camera);

		GeometryData createBLAS(std::shared_ptr<scene::Mesh> mesh);

		OptixTraversableHandle createTLAS();

		GeometryInstanceData* uploadGeometryInstanceData();

		CameraData* uploadCamera();

	public:
		OptixDeviceContext						optixContext;
		CUstream								stream;
		std::map<vtxID, CUdeviceptr>			HDM;
		std::map<vtxID, GeometryData>			mehsToGeometryData;
		vtxID									m_SequentialInstanceID = 0;
		std::vector<GeometryInstanceData>		m_GeometryInstanceData;
		std::vector<OptixInstance>				m_OptixInstanceData;
		CameraData								m_CameraData;
	};
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

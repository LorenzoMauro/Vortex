#include "DeviceData.h"
#include "CUDABuffer.h"
#include "CUDAChecks.h"
#include "Scene/Material/mdlTools.h"
#include "Device/OptixUtils.h"
#include "Scene/Graph.h"
#include <cudaGL.h>

namespace vtx::device
{

	static UploadData uploadData;

	void DeviceVisitor::visit(std::shared_ptr<graph::Instance> instance)
	{
		currentInstanceID = instance->getID();
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Transform> transform)
	{
		currentTransform = currentTransform * transform->transformationAttribute.affineTransform;
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Group> group)
	{
		previousTransform = currentTransform;
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Mesh> mesh)
	{
		vtxID meshID = mesh->getID();
		GeometryData geometryData;
		if (uploadData.meshIdToGeometryData.contains(meshID)) {
			geometryData = uploadData.meshIdToGeometryData[meshID];
		}
		else {
			geometryData = createBLAS(mesh);
			uploadData.meshIdToGeometryData.insert(meshID, geometryData);
		}
		createInstanceData(geometryData, currentInstanceID, currentTransform);
		currentTransform = previousTransform;
		currentInstanceID = previousIntanceId;
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Material> material)
	{
		vtxID materialID = material->getID();
		MaterialData materialData;
		if (uploadData.meshIdToGeometryData.contains(materialID)) {
			materialData = uploadData.materialIdToGeometryData[materialID];
		}
		else {
			materialData = createMaterialData(material);
			uploadData.materialIdToGeometryData.insert(materialID, materialData);
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Camera> camera)
	{
		if (camera->updated) {
			uploadData.launchParams.CameraData.position = camera->position;
			uploadData.launchParams.CameraData.direction = camera->direction;
			uploadData.launchParams.CameraData.vertical = cos(camera->fovY) * camera->vertical;
			uploadData.launchParams.CameraData.horizontal = cos(camera->fovY) * camera->aspect * camera->horizontal;
			camera->updated = false;
		}
	}

	void DeviceVisitor::visit(std::shared_ptr < graph::Renderer > renderer)
	{
		if (renderer->resized) {
			if (renderer->cudaGraphicsResource != nullptr) {
				CUresult result = cuGraphicsUnregisterResource(renderer->cudaGraphicsResource);
				CU_CHECK(result);
			}

			uploadData.cudaColorBuffer.resize(renderer->width * renderer->height * sizeof(uint32_t));
			uploadData.launchParams.fbSize.x = renderer->width;
			uploadData.launchParams.fbSize.y = renderer->height;
			uploadData.launchParams.colorBuffer = uploadData.cudaColorBuffer.d_pointer();
			renderer->glFrameBuffer.SetSize(renderer->width, renderer->height);

			CUresult result = cuGraphicsGLRegisterImage(&(renderer->cudaGraphicsResource),
			                                            renderer->glFrameBuffer.m_ColorAttachment,
			                                            GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
			CU_CHECK(result);
			renderer->resized = false;
		}
	}

	void DeviceVisitor::visit(std::shared_ptr<graph::Shader> shader)
	{
	}

	LaunchParams& getLaunchParams()
	{
		return uploadData.launchParams;
	}

	CUDABuffer& getLaunchParamsBuffer()
	{
		return uploadData.launchParamsBuffer;
	}

	void finalizeUpload()
	{
		uploadData.launchParams.topObject = createTLAS();
		uploadData.launchParams.geometryInstanceData = uploadGeometryInstanceData();
		if (!uploadData.launchParamsBuffer.d_pointer()) {
			uploadData.launchParamsBuffer.alloc(sizeof(LaunchParams));
		}
		uploadData.launchParamsBuffer.upload(&uploadData.launchParams, 1);
	}

	void incrementFrame() {
		uploadData.launchParams.frameID++;
	}

	void createInstanceData(GeometryData geometryData, vtxID instanceID, math::affine3f transform)
	{
		CUDA_SYNC_CHECK();

		// First check if there is a valid material assigned to this instance.
		OptixInstance OptixInstance = {};

		float* matrix = transform;
		memcpy(OptixInstance.transform, transform, sizeof(float) * 12);

		//++m_SequentialInstanceID;
		//OptixInstance.instanceId = m_SequentialInstanceID; // User defined instance index, queried with optixGetInstanceId().
		OptixInstance.instanceId = instanceID; // User defined instance index, queried with optixGetInstanceId().
		OptixInstance.visibilityMask = 255;
		OptixInstance.sbtOffset = 0;
		OptixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		OptixInstance.traversableHandle = geometryData.traversable;

		GeometryInstanceData gid;
		gid.InstanceId = instanceID;
		gid.Vertexattributes = geometryData.d_VertexAttribute;
		gid.MaterialArray = geometryData.d_IndexAttribut;

		uploadData.geometryInstanceDataMap.insert(instanceID, gid);
		uploadData.optixInstances.push_back(OptixInstance);
	}

	GeometryData createBLAS(std::shared_ptr<graph::Mesh> mesh)
	{
		VTX_INFO("Computing BLAS of Mesh: {}", mesh->getID());

		CUDA_SYNC_CHECK();

		auto& optixContext = optix::getState()->optixContext;
		auto& stream = optix::getState()->stream;

		/// Uploading Vertex and Index Buffer ///

		CUDABuffer vertexBuffer;
		CUDABuffer indexBuffer;

		vertexBuffer.alloc_and_upload(mesh->vertices);
		indexBuffer.alloc_and_upload(mesh->indices);

		CUdeviceptr d_vertexBuffer = vertexBuffer.d_pointer();
		CUdeviceptr d_indexBuffer = indexBuffer.d_pointer();

		/// BLAS Inputs ///
		OptixBuildInput buildInput = {};

		buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		buildInput.triangleArray.vertexStrideInBytes = sizeof(graph::VertexAttributes);
		buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh->vertices.size());
		buildInput.triangleArray.vertexBuffers = &d_vertexBuffer;

		buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		buildInput.triangleArray.indexStrideInBytes = sizeof(vtxID) * 3;
		buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh->indices.size()) / 3;
		buildInput.triangleArray.indexBuffer = d_indexBuffer;

		unsigned int inputFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

		buildInput.triangleArray.flags = inputFlags;
		buildInput.triangleArray.numSbtRecords = 1;

		/// BLAS Options ///
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		/// Prepare Compaction ///
		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
			&accelOptions,
			&buildInput,
			1,
			&blasBufferSizes));

		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc = {};
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();

		/// First build ///

		CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OptixTraversableHandle traversable;

		OPTIX_CHECK(optixAccelBuild(optixContext,
			stream,
			&accelOptions,
			&buildInput,
			1,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,
			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,
			&traversable,
			&emitDesc, 1));
		CUDA_SYNC_CHECK();

		/// Compaction ///
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		CUdeviceptr d_gas = outputBuffer.d_pointer();
		if (compactedSize < outputBuffer.sizeInBytes) {
			CUDABuffer outputBuffer_compacted;
			outputBuffer_compacted.alloc(compactedSize);
			OPTIX_CHECK(optixAccelCompact(optixContext,
				/*stream:*/0,
				traversable,
				outputBuffer_compacted.d_pointer(),
				outputBuffer_compacted.sizeInBytes,
				&traversable));
			d_gas = outputBuffer_compacted.d_pointer();

			auto savedBytes = outputBuffer.sizeInBytes - compactedSize;
			VTX_INFO("Compacted GAS, saved {} bytes", savedBytes);
			CUDA_SYNC_CHECK();
			outputBuffer.free(); // << the UNcompacted, temporary output buffer
		}

		/// Clean Up ///
		tempBuffer.free();
		compactedSizeBuffer.free();


		GeometryData data;

		data.type = PT_TRIANGLES;
		data.traversable = traversable;
		data.d_Gas = d_gas;
		data.d_VertexAttribute = d_vertexBuffer;
		data.d_IndexAttribut = d_indexBuffer;
		data.numVertices = mesh->vertices.size();
		data.numIndices = mesh->indices.size();

		return data;
	}

	OptixTraversableHandle createTLAS() {
		CUDA_SYNC_CHECK();
		VTX_INFO("Computing TLAS");

		auto& optixContext = optix::getState()->optixContext;
		auto& stream = optix::getState()->stream;

		// Construct the TLAS by attaching all flattened instances.
		const size_t instancesSizeInBytes = sizeof(OptixInstance) * uploadData.optixInstances.size();

		CUDABuffer instancesBuffer;
		instancesBuffer.alloc_and_upload(uploadData.optixInstances);

		OptixBuildInput instanceInput = {};

		instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instanceInput.instanceArray.instances = instancesBuffer.d_pointer();
		instanceInput.instanceArray.numInstances = static_cast<unsigned int>(uploadData.optixInstances.size());


		OptixAccelBuildOptions accelBuildOptions = {};

		accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accelBuildOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
		accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;


		OptixAccelBufferSizes accelBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
			&accelBuildOptions,
			&instanceInput,
			1,
			&accelBufferSizes));


		CUDABuffer tempBuffer;
		tempBuffer.alloc(accelBufferSizes.tempSizeInBytes);

		CUDABuffer IAS;
		IAS.alloc(accelBufferSizes.outputSizeInBytes);

		OptixTraversableHandle TopTraversable;

		OPTIX_CHECK(optixAccelBuild(optixContext,
			stream,
			&accelBuildOptions,
			&instanceInput,
			1,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,
			IAS.d_pointer(),
			IAS.sizeInBytes,
			&TopTraversable,
			nullptr, 0));

		CUDA_SYNC_CHECK();

		/// Clean Up ///
		tempBuffer.free();
		instancesBuffer.free();

		return TopTraversable;
	}

	CudaMap<vtxID, GeometryInstanceData>* uploadGeometryInstanceData()
	{
		return uploadData.geometryInstanceDataMap.allocAndUpload();
		//CUDABuffer geometryInstanceBuffer;
		//geometryInstanceBuffer.alloc_and_upload(m_GeometryInstanceData);
		//return reinterpret_cast<GeometryInstanceData*>(geometryInstanceBuffer.d_pointer());
	}
	MaterialData createMaterialData(std::shared_ptr<graph::Material> material)
	{
		return MaterialData();
	}
}

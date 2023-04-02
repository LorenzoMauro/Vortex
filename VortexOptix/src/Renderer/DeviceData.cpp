#include "DeviceData.h"
#include "CUDABuffer.h"
#include "CUDAChecks.h"

namespace vtx {
	void DeviceScene::ElaborateScene(std::shared_ptr<scene::Node> root)
	{
		Traverse(root, InstanceData(), math::affine3f());
		createTLAS();
	}
	void vtx::DeviceScene::Traverse(std::shared_ptr<scene::Node> node, InstanceData instanceData, math::affine3f transform) {
		switch (node->getType()) {

			case scene::NodeType::NT_GROUP: {
				std::shared_ptr<scene::Group> group = std::static_pointer_cast<scene::Group>(node);
				for (std::shared_ptr<scene::Node> child : group->getChildren()) {
					Traverse(child, instanceData, transform);
				}
			}
			break;

			case scene::NodeType::NT_INSTANCE: {
				std::shared_ptr<scene::Instance> instance = std::static_pointer_cast<scene::Instance>(node);

				std::vector<std::shared_ptr<scene::Material>>& Materials = instance->getMaterials();

				
				if (Materials.size() != 0) {
					std::vector<CUdeviceptr> d_MaterialArray;
					for (auto& material : Materials) {
						Traverse(material, instanceData, transform);
						vtxID matID = material->getID();
						CUdeviceptr d_Material = HDM[matID];
						d_MaterialArray.push_back(d_Material);
					}
					CUDABuffer MaterialBuffer;
					MaterialBuffer.alloc_and_upload(d_MaterialArray);
					instanceData.d_MaterialArray = MaterialBuffer.d_pointer();
				}
				else {
					instanceData.d_MaterialArray = NULL;
				}

				instanceData.InstanceId = instance->getID();

				transform = transform * instance->getTransform()->transformationAttribute.AffineTransform;
				Traverse(instance->getChild(), instanceData, transform);
			}
			break;

			case scene::NodeType::NT_MESH:{

				std::shared_ptr<scene::Mesh> mesh = std::static_pointer_cast<scene::Mesh>(node);
				vtxID meshID = mesh->getID();
				GeometryData geometryData;
				if (mehsToGeometryData.find(meshID) == mehsToGeometryData.end()) {
					geometryData = createBLAS(mesh);
					mehsToGeometryData.insert({ meshID, geometryData });
				}
				else {
					geometryData = mehsToGeometryData[meshID];
				}

				createInstanceData(geometryData, instanceData, transform);
			}
			break;

			case scene::NodeType::NT_MATERIAL: {

			}

		}
	}

	void DeviceScene::createInstanceData(GeometryData geometryData, InstanceData instanceData, math::affine3f transform)
	{
		CUDA_SYNC_CHECK();

		// First check if there is a valid material assigned to this instance.
		OptixInstance OptixInstance = {};

		float* matrix = transform;
		memcpy(OptixInstance.transform, transform, sizeof(float) * 12);

		++m_SequentialInstanceID;
		OptixInstance.instanceId = m_SequentialInstanceID; // User defined instance index, queried with optixGetInstanceId().
		OptixInstance.visibilityMask = 255;
		OptixInstance.sbtOffset = 0;
		OptixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
		OptixInstance.traversableHandle = geometryData.traversable;

		GeometryInstanceData gid;
		gid.InstanceId = instanceData.InstanceId;
		gid.Vertexattributes = geometryData.d_VertexAttribute;
		gid.MaterialArray = geometryData.d_IndexAttribut;

		m_GeometryInstanceData.push_back(gid);
		m_OptixInstanceData.push_back(OptixInstance);
	}

	GeometryData DeviceScene::createBLAS(std::shared_ptr<scene::Mesh> mesh)
	{
		VTX_INFO("Computing BLAS of Mesh: {}", mesh->getID());
		
		CUDA_SYNC_CHECK();

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
		buildInput.triangleArray.vertexStrideInBytes = sizeof(scene::VertexAttributes);
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

	OptixTraversableHandle DeviceScene::createTLAS() {
		CUDA_SYNC_CHECK();

		// Construct the TLAS by attaching all flattened instances.
		const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_OptixInstanceData.size();

		CUDABuffer instancesBuffer;
		instancesBuffer.alloc_and_upload(m_OptixInstanceData);

		OptixBuildInput instanceInput = {};

		instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instanceInput.instanceArray.instances = instancesBuffer.d_pointer();
		instanceInput.instanceArray.numInstances = static_cast<unsigned int>(m_OptixInstanceData.size());


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

	GeometryInstanceData* DeviceScene::uploadGeometryInstanceData()
	{
		CUDABuffer geometryInstanceBuffer;
		geometryInstanceBuffer.alloc_and_upload(m_GeometryInstanceData);
		return reinterpret_cast<GeometryInstanceData*>(geometryInstanceBuffer.d_pointer());
	}

}

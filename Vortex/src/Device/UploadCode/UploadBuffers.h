#pragma once
#include <map>
#include "CUDABuffer.h"
#include "Core/VortexID.h"
#include <memory>

#define GET_BUFFER(type, index, element) \
		vtx::device::Buffers::getInstance()->getBuffer<type>(index).element

#define UPLOAD_BUFFERS \
		vtx::device::Buffers::getInstance()

namespace vtx::device
{
	struct Buffers
	{
		static Buffers* getInstance()
		{
			static Buffers buffersInstance;
			return &buffersInstance;
		}

		Buffers(const Buffers&)            = delete; // Disable copy constructor
		Buffers& operator=(const Buffers&) = delete; // Disable assignment operator
		Buffers(Buffers&&) = delete;                  // Disable move constructor
		Buffers& operator=(Buffers&&) = delete;       // Disable move assignment operator

		void shutDown()
		{
			VTX_INFO("Shutting Down Buffers");
			frameIdBuffer.free();
			launchParamsBuffer.free();
			rendererSettingsBuffer.free();
			sbtProgramIdxBuffer.free();
			instancesBuffer.free();
			toneMapperSettingsBuffer.free();
		}


		struct GeometryBuffers
		{
			CUDABuffer vertexBuffer;
			CUDABuffer indexBuffer;
			CUDABuffer faceBuffer;
			CUDABuffer geometryDataBuffer;

			GeometryBuffers() = default;
			~GeometryBuffers()
			{
				VTX_INFO("ShutDown: Destroying Geometry Buffers");
				vertexBuffer.free();
				indexBuffer.free();
				faceBuffer.free();
			}
		};

		struct InstanceBuffers
		{
			CUDABuffer materialSlotsBuffer;
			CUDABuffer instanceDataBuffer;

			InstanceBuffers() = default;
			~InstanceBuffers()
			{
				VTX_INFO("ShutDown: Destroying Instance Buffers");
				materialSlotsBuffer.free();
				instanceDataBuffer.free();
			}
		};

		struct MaterialBuffers
		{
			CUDABuffer argBlockBuffer;
			CUDABuffer materialDataBuffer;
			CUDABuffer materialConfigBuffer;
			CUDABuffer textureIdBuffer;
			CUDABuffer bsdfIdBuffer;
			CUDABuffer lightProfileBuffer;
			CUDABuffer TextureHandlerBuffer;
			CUDABuffer shaderDataBuffer;

			MaterialBuffers() = default;
			~MaterialBuffers()
			{
				VTX_INFO("ShutDown: Material Buffers");
				argBlockBuffer.free();
				materialDataBuffer.free();
				materialConfigBuffer.free();
				textureIdBuffer.free();
				bsdfIdBuffer.free();
				lightProfileBuffer.free();
				TextureHandlerBuffer.free();
				shaderDataBuffer.free();
			}
		};

		struct TextureBuffers
		{
			CUarray					textureArray;
			cudaTextureObject_t		texObj;
			cudaTextureObject_t		texObjUnfiltered;
			CUDABuffer				textureDataBuffer;

			TextureBuffers() = default;
			~TextureBuffers()
			{
				VTX_INFO("ShutDown: Texture Buffers");
				CU_CHECK_CONTINUE(cuArrayDestroy(textureArray));
				CU_CHECK_CONTINUE(cuTexObjectDestroy(texObj));
				CU_CHECK_CONTINUE(cuTexObjectDestroy(texObjUnfiltered));
				textureDataBuffer.free();

			}
		};

		struct BsdfPartBuffer
		{
			CUDABuffer	sampleData;
			CUDABuffer	albedoData;
			CUDABuffer	partBuffer;
			CUarray		lookUpArray;
			CUtexObject	evalData;

			BsdfPartBuffer() = default;
			~BsdfPartBuffer()
			{
				VTX_INFO("ShutDown: Destroying BSDF Part Buffers");
				sampleData.free();
				albedoData.free();
				partBuffer.free();
				CU_CHECK_CONTINUE(cuArrayDestroy(lookUpArray));
				CU_CHECK_CONTINUE(cuTexObjectDestroy(evalData));
			}

		};

		struct BsdfBuffers
		{
			BsdfPartBuffer	reflectionPartBuffer;
			BsdfPartBuffer	transmissionPartBuffer;
			CUDABuffer		bsdfDataBuffer;

			~BsdfBuffers()
			{
				VTX_INFO("ShutDown: Destroying BSDF Buffers");
				bsdfDataBuffer.free();
			}
		};

		struct LightProfileBuffers
		{
			CUDABuffer cdfBuffer;
			CUarray	   lightProfileSourceArray;
			CUtexObject	evalData;
			CUDABuffer	lightProfileDataBuffer;

			LightProfileBuffers() = default;
			~LightProfileBuffers()
			{
				VTX_INFO("ShutDown: Light Buffers");
				cdfBuffer.free();
				lightProfileDataBuffer.free();
				CU_CHECK_CONTINUE(cuArrayDestroy(lightProfileSourceArray));
				CU_CHECK_CONTINUE(cuTexObjectDestroy(evalData));
			}
		};

		struct FrameBufferBuffers
		{
			CUDABuffer cudaOutputBuffer;
			CUDABuffer radianceBuffer;
			CUDABuffer toneMappedRadianceBuffer;
			CUDABuffer albedoBuffer;
			CUDABuffer normalBuffer;
			CUDABuffer noiseDataBuffer;
			CUDABuffer radianceRangeBuffer;
			CUDABuffer normalRangeBuffer;
			CUDABuffer albedoRangeBuffer;
			FrameBufferBuffers() = default;
			~FrameBufferBuffers()
			{
				VTX_INFO("ShutDown: Frame Buffers");
				cudaOutputBuffer.free();
				radianceBuffer.free();
				toneMappedRadianceBuffer.free();
				albedoBuffer.free();
				normalBuffer.free();
				noiseDataBuffer.free();
				noiseDataBuffer.free();
				radianceRangeBuffer.free();
				normalRangeBuffer.free();
				albedoRangeBuffer.free();
			}
		};


		struct LightBuffers
		{
			////////////////////////////////////////
			//////////// Mesh Light ////////////////
			////////////////////////////////////////
			CUDABuffer areaCdfBuffer;
			CUDABuffer actualTriangleIndices;

			////////////////////////////////////////
			//////////// Env Light /////////////////
			////////////////////////////////////////
			CUDABuffer cdfUBuffer;
			CUDABuffer cdfVBuffer;
			CUDABuffer aliasBuffer;

			////////////////////////////////////////
			/// General Attributes for all Lights //
			////////////////////////////////////////
			CUDABuffer attributeBuffer;
			CUDABuffer lightDataBuffer;

			LightBuffers() = default;
			~LightBuffers()
			{
				VTX_INFO("ShutDown: Light Buffers");
				areaCdfBuffer.free();
				actualTriangleIndices.free();
				attributeBuffer.free();
				cdfUBuffer.free();
				cdfVBuffer.free();
				aliasBuffer.free();
			}
		};

		template<typename T>
		T& getBufferCollectionElement(std::map<vtxID, T>& bufferCollectionMap, const vtxID nodeId)
		{
			if (const auto it = bufferCollectionMap.find(nodeId); it != bufferCollectionMap.end())
				return it->second;
			// Use emplace to construct the object directly in the map
			bufferCollectionMap.emplace(std::piecewise_construct,
																std::forward_as_tuple(nodeId),
																std::tuple<>());

			//bufferCollectionMap.try_emplace(nodeId, T());
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

		template<>
		LightBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(light, nodeId);
		}

		std::map<vtxID, InstanceBuffers>     instance;
		std::map<vtxID, GeometryBuffers>     geometry;
		std::map<vtxID, MaterialBuffers>     material;
		std::map<vtxID, TextureBuffers>      texture;
		std::map<vtxID, BsdfBuffers>         bsdf;
		std::map<vtxID, LightProfileBuffers> lightProfile;
		std::map<vtxID, FrameBufferBuffers>  frameBuffer;
		std::map<vtxID, LightBuffers>        light;
		CUDABuffer                           frameIdBuffer;
		CUDABuffer                           launchParamsBuffer;
		CUDABuffer                           rendererSettingsBuffer;
		CUDABuffer                           sbtProgramIdxBuffer;
		CUDABuffer                           lightsDataBuffer;
		CUDABuffer                           instancesBuffer;
		CUDABuffer                           toneMapperSettingsBuffer;

	private:
		~Buffers() = default;
		Buffers() = default;
	};
}

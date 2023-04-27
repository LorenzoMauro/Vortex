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

		Buffers(const Buffers&) = delete;             // Disable copy constructor
		Buffers& operator=(const Buffers&) = delete;  // Disable assignment operator
		Buffers(Buffers&&) = delete;                  // Disable move constructor
		Buffers& operator=(Buffers&&) = delete;       // Disable move assignment operator
		

		struct GeometryBuffers
		{
			CUDABuffer vertexBuffer;
			CUDABuffer indexBuffer;
			CUDABuffer faceBuffer;

			GeometryBuffers() = default;
			~GeometryBuffers()
			{
				vertexBuffer.free();
				indexBuffer.free();
				faceBuffer.free();
			}
		};

		struct InstanceBuffers
		{
			CUDABuffer materialsIdBuffer;
			CUDABuffer meshLightIdBuffer;

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
				if (argBlockBuffer.dPointer())
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
				if (TextureHandlerBuffer.dPointer()) { TextureHandlerBuffer.free(); }
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
				if (partBuffer.dPointer()) { partBuffer.free(); }
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
			CUDABuffer cudaOutputBuffer;
			CUDABuffer radianceBuffer;

			FrameBufferBuffers() = default;
			~FrameBufferBuffers()
			{
				cudaOutputBuffer.free();
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

		struct LightBuffers
		{
			CUDABuffer areaCdfBuffer;
			CUDABuffer actualTriangleIndices;
			CUDABuffer attributeBuffer;


			LightBuffers() = default;
			~LightBuffers()
			{
				areaCdfBuffer.free();
			}
		};

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

		template<>
		LightBuffers& getBuffer(const vtxID nodeId) {
			return getBufferCollectionElement(light, nodeId);
		}

		std::map<vtxID, InstanceBuffers>			instance;
		std::map<vtxID, GeometryBuffers>			geometry;
		std::map<vtxID, MaterialBuffers>			material;
		std::map<vtxID, ShaderBuffers>				shader;
		std::map<vtxID, TextureBuffers>				texture;
		std::map<vtxID, BsdfBuffers>				bsdf;
		std::map<vtxID, LightProfileBuffers>		lightProfile;
		std::map<vtxID, FrameBufferBuffers>			frameBuffer;
		std::map<vtxID, LightBuffers>				light;
		CUDABuffer									frameIdBuffer;
		CUDABuffer									launchParamsBuffer;
		CUDABuffer									rendererSettingsBuffer;

	private:
		~Buffers() = default;
		Buffers() = default;
	};
}

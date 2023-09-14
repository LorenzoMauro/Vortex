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
		Buffers(Buffers&&)                 = delete; // Disable move constructor
		Buffers& operator=(Buffers&&)      = delete; // Disable move assignment operator

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
			CUarray             textureArray;
			cudaTextureObject_t texObj;
			cudaTextureObject_t texObjUnfiltered;
			CUDABuffer          textureDataBuffer;

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
			CUDABuffer  sampleData;
			CUDABuffer  albedoData;
			CUDABuffer  partBuffer;
			CUarray     lookUpArray;
			CUtexObject evalData;

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
			BsdfPartBuffer reflectionPartBuffer;
			BsdfPartBuffer transmissionPartBuffer;
			CUDABuffer     bsdfDataBuffer;

			~BsdfBuffers()
			{
				VTX_INFO("ShutDown: Destroying BSDF Buffers");
				bsdfDataBuffer.free();
			}
		};

		struct LightProfileBuffers
		{
			CUDABuffer  cdfBuffer;
			CUarray     lightProfileSourceArray;
			CUtexObject evalData;
			CUDABuffer  lightProfileDataBuffer;

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

		struct PathsBuffer
		{
			CUDABuffer pathStructBuffer;
			CUDABuffer bouncesBuffer;
			CUDABuffer pathsArrayBuffer;
			CUDABuffer validPixelsBuffer;
			CUDABuffer pathsAccumulatorBuffer;
			CUDABuffer resetBouncesBuffer;
			CUDABuffer validLightSampleBuffer;
			CUDABuffer pixelsWithContributionBuffer;

			~PathsBuffer()
			{
				bouncesBuffer.free();
				pathStructBuffer.free();
				pathsArrayBuffer.free();
				validPixelsBuffer.free();
				pathsAccumulatorBuffer.free();
				resetBouncesBuffer.free();
				validLightSampleBuffer.free();
				pixelsWithContributionBuffer.free();
			}
		};

		struct NetworkInputBuffers
		{
			CUDABuffer networkStateStructBuffer;
			CUDABuffer positionBuffer;
			CUDABuffer normalBuffer;
			CUDABuffer woBuffer;
			CUDABuffer encodedPositionBuffer;
			CUDABuffer encodedNormalBuffer;
			CUDABuffer encodedWoBuffer;

			~NetworkInputBuffers()
			{
				networkStateStructBuffer.free();
				positionBuffer.free();
				normalBuffer.free();
				woBuffer.free();
				encodedPositionBuffer.free();
				encodedNormalBuffer.free();
				encodedWoBuffer.free();
			}
		};

		struct ReplayBufferBuffers
		{
			CUDABuffer          replayBufferStructBuffer;
			CUDABuffer          actionBuffer;
			CUDABuffer          rewardBuffer;
			CUDABuffer          nextActionBuffer;
			CUDABuffer          doneBuffer;
			CUDABuffer          maxSize;
			NetworkInputBuffers stateBuffer;
			NetworkInputBuffers nextStatesBuffer;

			~ReplayBufferBuffers()
			{
				replayBufferStructBuffer.free();
				actionBuffer.free();
				rewardBuffer.free();
				nextActionBuffer.free();
				doneBuffer.free();
				maxSize.free();
			}
		};

		struct NpgTrainingDataBuffers
		{
			CUDABuffer          npgTrainingDataStructBuffers;
			NetworkInputBuffers inputBuffer;
			CUDABuffer          outgoingRadianceBuffer;
			CUDABuffer          incomingDirectionBuffer;
			CUDABuffer          bsdfProbabilitiesBuffer;

			~NpgTrainingDataBuffers()
			{
				npgTrainingDataStructBuffers.free();
				outgoingRadianceBuffer.free();
				incomingDirectionBuffer.free();
				bsdfProbabilitiesBuffer.free();
			}
		};

		struct InferenceBuffers
		{
			CUDABuffer          inferenceStructBuffer;
			NetworkInputBuffers stateBuffer;
			CUDABuffer          distributionParameters;
			CUDABuffer          inferenceSize;
			CUDABuffer          decodedStateBuffer;
			CUDABuffer          samplingFractionArrayBuffer;
			CUDABuffer          distributionTypeBuffer;
			CUDABuffer          samplesBuffer;
			CUDABuffer          probabilitiesBuffer;
			CUDABuffer          mixtureWeightBuffer;

			~InferenceBuffers()
			{
				inferenceStructBuffer.free();
				distributionParameters.free();
				inferenceSize.free();
				decodedStateBuffer.free();
				samplingFractionArrayBuffer.free();
				distributionTypeBuffer.free();
				samplesBuffer.free();
				probabilitiesBuffer.free();
				mixtureWeightBuffer.free();
			}
		};

		struct NetworkInterfaceBuffer
		{
			PathsBuffer            pathsBuffers;
			ReplayBufferBuffers    replayBufferBuffers;
			NpgTrainingDataBuffers npgTrainingDataBuffers;
			InferenceBuffers       inferenceBuffers;

			CUDABuffer networkInterfaceBuffer;
			CUDABuffer seedsBuffer;

			CUDABuffer debugBuffer1Buffer;
			CUDABuffer debugBuffer2Buffer;
			CUDABuffer debugBuffer3Buffer;

			~NetworkInterfaceBuffer()
			{
				pathsBuffers.~PathsBuffer();
				replayBufferBuffers.~ReplayBufferBuffers();
				inferenceBuffers.~InferenceBuffers();
				networkInterfaceBuffer.free();
				seedsBuffer.free();
				debugBuffer1Buffer.free();
				debugBuffer2Buffer.free();
				debugBuffer3Buffer.free();
			}
		};

		struct NoiseComputationBuffers
		{
			CUDABuffer radianceRangeBuffer;
			CUDABuffer normalRangeBuffer;
			CUDABuffer albedoRangeBuffer;
			CUDABuffer globalRadianceRangeBuffer;
			CUDABuffer globalAlbedoRangeBuffer;
			CUDABuffer globalNormalRangeBuffer;
			CUDABuffer noiseSumBuffer;
			CUDABuffer remainingSamplesBuffer;

			NoiseComputationBuffers() = default;

			~NoiseComputationBuffers()
			{
				VTX_INFO("ShutDown: Noise Computation Buffers");
				radianceRangeBuffer.free();
				normalRangeBuffer.free();
				albedoRangeBuffer.free();
				globalRadianceRangeBuffer.free();
				globalAlbedoRangeBuffer.free();
				globalNormalRangeBuffer.free();
				noiseSumBuffer.free();
				remainingSamplesBuffer.free();
			}
		};

		struct FrameBufferBuffers
		{
			CUDABuffer rawRadiance;
			CUDABuffer directLight;
			CUDABuffer diffuseIndirect;
			CUDABuffer glossyIndirect;
			CUDABuffer transmissionIndirect;

			CUDABuffer tmRadiance;
			CUDABuffer hdriRadiance;
			CUDABuffer normalNormalized;
			CUDABuffer albedoNormalized;
			CUDABuffer tmTransmissionIndirect;

			CUDABuffer albedo;
			CUDABuffer normal;
			CUDABuffer trueNormal;
			CUDABuffer tangent;
			CUDABuffer orientation;
			CUDABuffer uv;
			CUDABuffer debugColor1;

			CUDABuffer fireflyRemoval;

			CUDABuffer samples;
			CUDABuffer cudaOutputBuffer;
			CUDABuffer noiseDataBuffer;


			FrameBufferBuffers() = default;

			~FrameBufferBuffers()
			{
				VTX_INFO("ShutDown: Frame Buffers");
				cudaOutputBuffer.free();
				rawRadiance.free();
				directLight.free();
				diffuseIndirect.free();
				glossyIndirect.free();
				transmissionIndirect.free();
				tmRadiance.free();
				hdriRadiance.free();
				normalNormalized.free();
				albedoNormalized.free();
				tmTransmissionIndirect.free();
				albedo.free();
				normal.free();
				trueNormal.free();
				tangent.free();
				orientation.free();
				uv.free();
				debugColor1.free();
				fireflyRemoval.free();
				noiseDataBuffer.free();
				noiseDataBuffer.free();
				directLight.free();
				diffuseIndirect.free();
				glossyIndirect.free();
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

		struct WorkQueueBuffers
		{
			CUDABuffer radianceTraceQueueBuffer;
			CUDABuffer shadeQueueBuffer;
			CUDABuffer escapedQueueBuffer;
			CUDABuffer accumulationQueueBuffer;
			CUDABuffer shadowQueueBuffer;
			CUDABuffer countersBuffer;

			WorkQueueBuffers() = default;
			~WorkQueueBuffers()
			{
				VTX_INFO("ShutDown: Work Queue Buffers");
				radianceTraceQueueBuffer.free();
				shadeQueueBuffer.free();
				escapedQueueBuffer.free();
				accumulationQueueBuffer.free();
				shadowQueueBuffer.free();
				countersBuffer.free();
			}
		};

		template <typename T>
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

		template <typename T>
		T& getBuffer(const vtxID nodeId);

		template <>
		InstanceBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(instance, nodeId);
		}

		template <>
		GeometryBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(geometry, nodeId);
		}

		template <>
		MaterialBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(material, nodeId);
		}

		template <>
		TextureBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(texture, nodeId);
		}

		template <>
		BsdfBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(bsdf, nodeId);
		}

		template <>
		LightProfileBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(lightProfile, nodeId);
		}

		template <>
		FrameBufferBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(frameBuffer, nodeId);
		}

		template <>
		LightBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(light, nodeId);
		}

		template <>
		NoiseComputationBuffers& getBuffer(const vtxID nodeId)
		{
			return getBufferCollectionElement(noiseComputationBuffer, nodeId);
		}

		std::map<vtxID, InstanceBuffers>         instance;
		std::map<vtxID, GeometryBuffers>         geometry;
		std::map<vtxID, MaterialBuffers>         material;
		std::map<vtxID, TextureBuffers>          texture;
		std::map<vtxID, BsdfBuffers>             bsdf;
		std::map<vtxID, LightProfileBuffers>     lightProfile;
		std::map<vtxID, FrameBufferBuffers>      frameBuffer;
		std::map<vtxID, LightBuffers>            light;
		std::map<vtxID, NoiseComputationBuffers> noiseComputationBuffer;
		NetworkInterfaceBuffer                   networkInterfaceBuffer;
		WorkQueueBuffers                         workQueueBuffers;
		CUDABuffer                               frameIdBuffer;
		CUDABuffer                               launchParamsBuffer;
		CUDABuffer                               rendererSettingsBuffer;
		CUDABuffer                               sbtProgramIdxBuffer;
		CUDABuffer                               lightsDataBuffer;
		CUDABuffer                               instancesBuffer;
		CUDABuffer                               toneMapperSettingsBuffer;

	private:
		~Buffers() = default;
		Buffers()  = default;
	};
}

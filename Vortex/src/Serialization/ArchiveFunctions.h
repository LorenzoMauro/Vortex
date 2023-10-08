#pragma once
#include "NodeSaveData.h"
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/polymorphic.hpp>

//#include <cereal/types/memory.hpp>

#define nvp(data, field) make_nvp(#field, (data).field)
//#define nvp(data, field) (data).field

namespace cereal
{
#ifdef ARCHIVE_ENUMS
    // Serialization for NodeType
    template<class Archive>
    void save(Archive& archive, vtx::graph::NodeType const& data)
    {
        archive(cereal::make_nvp("NodeType", vtx::graph::nodeNames[data]));
    }

    template<class Archive>
    void load(Archive& archive, vtx::graph::NodeType& data)
    {
        std::string s;
        archive(s);
        data = vtx::graph::nameToNodeType[s];
    }
    //Serialization for DisplayBuffer
    template<class Archive>
    void save(Archive& archive, vtx::DisplayBuffer const& data)
    {
        std::string displayBufferName = vtx::displayBufferNames[(int)data];
        archive(cereal::make_nvp("displayBufferName", displayBufferName));
    }

    //Serialization for DisplayBuffer
    template<class Archive>
    void load(Archive& archive, vtx::DisplayBuffer& data)
    {
        std::string s;
        archive(s);
        data = vtx::displayBufferNameToEnum[s];
    }

    //Serialization for SamplingTechnique
    template<class Archive>
    void save(Archive& archive, vtx::SamplingTechnique const& data)
    {
        std::string samplingTechniqueName = vtx::samplingTechniqueNames[(int)data];
        archive(cereal::make_nvp("samplingTechniqueName", samplingTechniqueName));
    }

    template<class Archive>
    void load(Archive& archive, vtx::SamplingTechnique& data)
    {
        std::string s;
        archive(s);
        data = vtx::samplingTechniqueNameToEnum[s];
    }

    //Serialization for NetworkType
    template<class Archive>
    void save(Archive& archive, vtx::network::NetworkType const& data)
    {
        std::string networkType = vtx::network::networkNames[(int)data];
        archive(cereal::make_nvp("networkType", networkType));
    }
    template<class Archive>
    void load(Archive& archive, vtx::network::NetworkType& data)
    {
        std::string s;
        archive(s);
        data = vtx::network::networkNameToEnum[s];
    }

    //Serialization for SamplingStrategy
    template<class Archive>
    void save(Archive& archive, vtx::network::SamplingStrategy const& data)
    {
        std::string samplingStrategy = vtx::network::samplingStrategyNames[(int)data];
        archive(cereal::make_nvp("samplingStrategy", samplingStrategy));
    }
    template<class Archive>
    void load(Archive& archive, vtx::network::SamplingStrategy& data)
    {
        std::string s;
        archive(s);
        data = vtx::network::samplingStrategyNameToEnum[s];
    }
    //Serialization for EncodingType
    template<class Archive>
    void save(Archive& archive, vtx::network::EncodingType const& data)
    {
        std::string encodingType = vtx::network::encodingNames[(int)data];
        archive(cereal::make_nvp("encodingType", encodingType));
    }
    template<class Archive>
    void load(Archive& archive, vtx::network::EncodingType& data)
    {
        std::string s;
        archive(s);
        data = vtx::network::encodingNameToEnum[s];
    }

    //Serialization for DistributionType
    template<class Archive>
    void save(Archive& archive, vtx::network::DistributionType const& data)
    {
        std::string distributionType = vtx::network::distributionNames[(int)data];
        archive(cereal::make_nvp("distributionType", distributionType));
    }
    template<class Archive>
    void load(Archive& archive, vtx::network::DistributionType& data)
    {
        std::string s;
        archive(s);
        data = vtx::network::distributionNameToEnum[s];
    }

    //Serialization for LossType
    template<class Archive>
    void save(Archive& archive, vtx::network::LossType const& data)
    {
        std::string lossType = vtx::network::lossNames[(int)data];
        archive(cereal::make_nvp("lossType", lossType));
    }
    template<class Archive>
    void load(Archive& archive, vtx::network::LossType& data)
    {
        std::string s;
        archive(s);
        data = vtx::network::lossNameToEnum[s];
    }
    //Serialization for ParameterInfo::ParamKind
    template<class Archive>
    void save(Archive& archive, vtx::graph::shader::ParamKind const& data)
    {
        archive(cereal::make_nvp("paramKind", vtx::graph::shader::paramKindNames[data]));
    }

    template<class Archive>
    void load(Archive& archive, vtx::graph::shader::ParamKind& data)
    {
        std::string s;
        archive(s);
        data = vtx::graph::shader::paramKindNamesToEnum[s];
    }
#endif
}

namespace cereal
{
    
    // Serialization for BaseNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::BaseNodeSaveData& data)
    {
        archive(nvp(data,UID), nvp(data,TID), nvp(data,name), nvp(data,type));
    }

    // Serialization for math::vec3f
    template<class Archive>
    void serialize(Archive& archive, vtx::math::vec3f& data)
    {
	    archive(nvp(data,x), nvp(data,y), nvp(data,z));
	}

    // Serialization for math::affine3f
    template<class Archive>
    void serialize(Archive& archive, vtx::math::affine3f& data)
    {
	    archive(nvp(data,l.vx), nvp(data,l.vy), nvp(data,l.vz), nvp(data, p));
    }

    // Serialization for VertexAttributes
    template<class Archive>
    void serialize(Archive& archive, vtx::graph::VertexAttributes& data)
    {
        archive(nvp(data,position), nvp(data,texCoord), nvp(data,normal), nvp(data,tangent), nvp(data,bitangent));
    }

    //Serialization for FaceAttributes
    template<class Archive>
    void serialize(Archive& archive, vtx::graph::FaceAttributes& data)
    {
	    archive(nvp(data,materialSlotId));
	}

    //serialization for MeshStatus
    template<class Archive>
    void serialize(Archive& archive, vtx::graph::MeshStatus& data)
    {
	    archive(nvp(data,hasTangents), nvp(data,hasNormals), nvp(data,hasFaceAttributes));
    }

    // Serialization for MeshNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::MeshNodeSaveData& data)
    {
        archive(nvp(data,base), nvp(data,vertices), nvp(data,indices), nvp(data,faceAttributes), nvp(data,status));
    }

    // Serialization for TransformNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::TransformNodeSaveData& data)
    {
		archive(nvp(data,base), nvp(data,affineTransform));
	}

    // Serialization for MaterialNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::MaterialNodeSaveData& data)
    {
        archive(nvp(data, base), nvp(data, materialGraphUID), nvp(data, materialDbName), nvp(data, path), nvp(data, materialCallName));// , nvp(data, useAsLight));
    }

    // Serialization for MaterialSlotSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::MaterialSlotSaveData& data)
    {
	    archive(nvp(data, materialUID), nvp(data, slotIndex));
	}

    // Serialization for InstanceNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::InstanceNodeSaveData& data)
    {
		archive(nvp(data, base), nvp(data, transformUID), nvp(data, childUID), nvp(data, materialSlots));
    }

    //Serialization for groupNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::GroupNodeSaveData& data)
    {
	    archive(nvp(data, base), nvp(data,transformUID), nvp(data, childUIDs));
    }

    //Serialization for EnvironmentLightNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::EnvironmentLightNodeSaveData& data)
    {
	    archive(nvp(data, base), nvp(data, transformUID), nvp(data, texturePath));
	}

    //Serialization for CameraNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::CameraNodeSaveData& data)
    {
	    archive(nvp(data, base), nvp(data, transformUID), nvp(data, fov));
    }

    //Serialization for DenoiserSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::ToneMapperSettings& data)
    {
        archive(
            nvp(data, whitePoint),
            nvp(data, invWhitePoint),
            nvp(data, colorBalance),
            nvp(data, burnHighlights),
            nvp(data, crushBlacks),
            nvp(data, saturation),
            nvp(data, gamma),
            //nvp(data, isUpdated = true),
            nvp(data, invGamma)
        );
    }

    //Serialization for DenoiserSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::DenoiserSettings& data)
    {
        archive(
            nvp(data, active),
            nvp(data, denoiserStart),
            //nvp(data, isUpdated),
            nvp(data, denoiserBlend)
        );
    }

    //Serialization for FireflySettings
    template<class Archive>
    void serialize(Archive& archive, vtx::FireflySettings& data)
    {
        archive(
            nvp(data, active),
            nvp(data, kernelSize),
            nvp(data, start),
            nvp(data, threshold)
        );
    }

    //Serialization for AdaptiveSamplingSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::AdaptiveSamplingSettings& data)
    {
        archive(
            nvp(data, active),
            nvp(data, noiseKernelSize),
            nvp(data, minAdaptiveSamples),
            nvp(data, minPixelSamples),
            nvp(data, maxPixelSamples),
            nvp(data, albedoNormalNoiseInfluence),
            //nvp(data, isUpdated = true),
            nvp(data, noiseCutOff)
        );
    }

    //Serialization for RendererSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::RendererSettings& data)
    {
        archive(
            nvp(data, iteration),
            nvp(data, maxBounces),
            nvp(data, maxSamples),
            nvp(data, accumulate),
            nvp(data, samplingTechnique),
            nvp(data, displayBuffer),
            nvp(data, minClamp),
            nvp(data, maxClamp),
            nvp(data, useRussianRoulette),
            nvp(data, runOnSeparateThread),
            //nvp(data, isUpdated),
            //nvp(data, isMaxBounceChanged),
            nvp(data, adaptiveSamplingSettings),
            nvp(data, fireflySettings),
            nvp(data, denoiserSettings),
            nvp(data, toneMapperSettings)
        );
    }


    //Serialization for WavefrontSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::WavefrontSettings& data)
    {
        archive(
            nvp(data, active),
            nvp(data, fitWavefront),
            nvp(data, optixShade),
            nvp(data, parallelShade),
            nvp(data, longPathPercentage),
            //nvp(data, isUpdated),
            nvp(data, useLongPathKernel)
        );
    }

    

    //Serialization for TrainingBatchGenerationSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::TrainingBatchGenerationSettings& data)
    {
        archive(
            nvp(data, strategy),
            nvp(data, weightByMis),
            //nvp(data, isUpdated),
            nvp(data, lightSamplingProb)
        );
    }
    
    //Serialization for EncodingSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::EncodingSettings& data)
    {
        archive(
            nvp(data, type),
            //nvp(data, isUpdated ),
            nvp(data, features)
        );
    }

    //Serialization for InputSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::InputSettings& data)
    {
        archive(
            nvp(data, positionEncoding),
            nvp(data, woEncoding),
            //nvp(data, isUpdated = true),
            nvp(data, normalEncoding)
        );
    }
    
    //Serialization for PathGuidingSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::PathGuidingNetworkSettings& data)
    {
        archive(
            nvp(data, hiddenDim),
            nvp(data, numHiddenLayers),
            nvp(data, distributionType),
            nvp(data, mixtureSize),
            //nvp(data, isUpdated ),
            nvp(data, produceSamplingFraction)
        );
    }


    //Serialization for SacSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::SacSettings& data)
    {
        archive(
            nvp(data, polyakFactor),
            nvp(data, logAlphaStart),
            nvp(data, gamma),
            nvp(data, neuralSampleFraction),
            nvp(data, policyLr),
            nvp(data, qLr),
            //nvp(data, isUpdated = true),
            nvp(data, alphaLr)
        );
    }
    

    //Serialization for NpgSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::NpgSettings& data)
    {
        archive(
            nvp(data, learningRate),
            nvp(data, e),
            nvp(data, constantBlendFactor),
            nvp(data, samplingFractionBlend),
            nvp(data, lossType),
            nvp(data, isUpdated),
            nvp(data, absoluteLoss),
            nvp(data, meanLoss)
        );
    }

    //Serialization for NetworkSettings
    template<class Archive>
    void serialize(Archive& archive, vtx::network::NetworkSettings& data)
    {
        archive(
            nvp(data, active),
            nvp(data, batchSize),
            nvp(data, maxTrainingStepPerFrame),
            nvp(data, doTraining),
            nvp(data, maxTrainingSteps),
            nvp(data, doInference),
            nvp(data, inferenceIterationStart),
            nvp(data, clearOnInferenceStart),
            nvp(data, type),
            nvp(data, trainingBatchGenerationSettings),
            nvp(data, inputSettings),
            nvp(data, pathGuidingSettings),
            nvp(data, sac),
            //nvp(data, isUpdated = true),
            //nvp(data, isDatasetSizeUpdated = true),
            //nvp(data, depthToDebug = 0),
            nvp(data, npg)
        );
    }

    //Serialization for ExperimentSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::ExperimentSaveData& data)
    {
        archive(
            nvp(data, name),
            nvp(data, mape),
            nvp(data, rendererSettings),
            nvp(data, networkSettings),
            nvp(data, wavefrontSettings)
        );
    }

    //Serialization for ExperimentManagerSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::ExperimentManagerSaveData& data)
    {
        archive(
            nvp(data, currentExperiment),
            nvp(data, currentExperimentStep),
            nvp(data, width),
            nvp(data, height),
            nvp(data, isGroundTruthReady),
            nvp(data, maxSamples),
            nvp(data, experiments),
            nvp(data, groundTruthImage)
        );
    }

    //Serialization for RendererNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::RendererNodeSaveData& data)
    {
        archive(
            nvp(data, base),
            nvp(data, rendererSettings),
            nvp(data, wavefrontSettings),
            nvp(data, networkSettings),
            nvp(data, cameraUID),
            nvp(data, sceneRootUID),
            nvp(data, environmentLightUID),
            nvp(data, experimentManagerSaveData)
        );
    }
    
    //Serialization for ShaderSocketSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::ShaderSocketSaveData& data)
    {
        archive(
            nvp(data,socketNodeInputUID),
	        nvp(data,socketName),
	        nvp(data,dataBuffer),
	        nvp(data,paramKind),
	        nvp(data,arrayParamKind),
	        nvp(data,arraySize),
	        nvp(data,arrayElemSize)
		);
	}

    //Serialization for FunctionInfo
    template<class Archive>
    void serialize(Archive& archive, vtx::mdl::MdlFunctionInfo& data)
    {
        archive(
            nvp(data, fullModulePath),
            nvp(data, module),
            nvp(data, name),
            nvp(data, signature)
		);
	}

    //Serialization for BaseShaderNodeSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::BaseShaderNodeSaveData& data)
    {
        archive(
            nvp(data, base),
			nvp(data, shaderSocketSaveData),
			nvp(data, functionInfo),
            nvp(data, texturePath),
            nvp(data, channel)
		);
	}

    //Serialization for GraphSaveData
    template<class Archive>
    void serialize(Archive& archive, vtx::serializer::GraphSaveData& data)
    {
        archive(
            nvp(data, meshes),
	        nvp(data, transforms),
	        nvp(data, materials),
	        nvp(data, instances),
	        nvp(data, groups),
	        nvp(data, environmentLights),
	        nvp(data, cameras),
	        nvp(data, renderers),
	        nvp(data, shaderNodes),
	        nvp(data, activeRendererUID),
	        nvp(data, activeCameraUID),
	        nvp(data, sceneRootUID)
		);
	}
}

#pragma once
#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H
#include "NetworkInputs.h"
#include "NpgTrainingData.h"
#include "Paths.h"
#include "ReplayBuffer.h"
#include "Scene/Nodes/RendererSettings.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "NeuralNetworks/Config/NetworkSettings.h"

namespace vtx
{
	namespace device
	{
		struct NetworkInterfaceBuffer;
	}

	struct InferenceQueries;
	struct NpgTrainingData;
	struct ReplayBuffer;
	struct Paths;

	struct NetworkInterface
	{
		struct WhatChanged
		{
			bool maxDepth = false;
			bool numberOfPixels = false;
			bool maxDatasetSize = false;
			bool distributionType = false;
		};

		static NetworkInterface* upload(
			const int&                      _numberOfPixels,
			const int&                      _maxDatasetSize,
			const int&                      _maxDepth,
			const int&                      frameId,
			const network::config::DistributionType _distributionType,
			const int                       _mixtureSize,
			const ToneMapperSettings&       _toneMapperSettings,
			const WhatChanged&              changed,
			device::NetworkInterfaceBuffer& networkInterfaceBuffers);

	private:
		NetworkInterface(
			const int& numberOfPixels,
			const int& maxDatasetSize,
			const int& maxDepth,
			const int& frameId,
			const network::config::DistributionType distributionType,
			int mixtureSize,
			const ToneMapperSettings& _toneMapperSettings,
			WhatChanged changed,
			device::NetworkInterfaceBuffer& networkInterfaceBuffers);
	public:

		__forceinline__ __device__ void sppDebugFrame(const int sampledPixel, const int sampledDepth)
		{
			debugBuffer3[sampledPixel].x += 1.0f;
		}

		__forceinline__ __device__ void shuffleDataset(const int& trainingDatasetIdx, const network::config::NetworkSettings& settings)
		{
			if (trainingDatasetIdx >= npgTrainingData->nAlloc)
			{
				return;
			}

			unsigned& seed = lcgSeeds[trainingDatasetIdx];
			Paths::Hit hit;
			Paths::Hit* nextHit = nullptr;
			/*if (settings.type == network::NT_SAC)
			{
				*nextHit = Paths::Hit();
			}*/
			Paths::LightContribution lightContribution;
			bool isTerminal = false;
			int sampledPixel = 0;
			int sampledDepth = 0;
			if(settings.trainingBatchGenerationSettings.strategy == network::config::SS_ALL || settings.trainingBatchGenerationSettings.strategy == network::config::SS_PATHS_WITH_CONTRIBUTION)
			{
				paths->getRandomPathSample(
					seed, settings,
					sampledPixel, sampledDepth,
					&hit, &lightContribution, isTerminal, nextHit
				);
			}
			else if(settings.trainingBatchGenerationSettings.strategy == network::config::SS_LIGHT_SAMPLES)
			{
				paths->getRandomLightSamplePath(
					seed, settings.trainingBatchGenerationSettings,
					sampledPixel, sampledDepth,
					&hit, &lightContribution, isTerminal, nextHit
				);
			}
			else
			{
				printf("Unknown training batch generation strategy, file: %s, line %d\n", __FILE__, __LINE__);
			}


			npgTrainingData->inputs->addState(trainingDatasetIdx, hit.position, hit.wOutgoing, hit.normal, (float)hit.instanceId, (float)hit.triangleId, (float)hit.materialId);
			npgTrainingData->incomingDirection[trainingDatasetIdx] = lightContribution.wIncoming;
			npgTrainingData->bsdfProbabilities[trainingDatasetIdx] = lightContribution.bsdfProb;
			npgTrainingData->outRadiance[trainingDatasetIdx] = lightContribution.outRadiance;
			npgTrainingData->inRadiance[trainingDatasetIdx] = lightContribution.inRadiance;
			npgTrainingData->throughput[trainingDatasetIdx] = lightContribution.throughput;
			npgTrainingData->overallProb[trainingDatasetIdx] = hit.overallProb;
			
			const float signal = utl::luminance(lightContribution.outRadiance);
			debugBuffer1[sampledPixel].x += signal;
			//debugBuffer2[sampledPixel] += lightContribution.outRadiance;
			debugBuffer1[sampledPixel].x = lightContribution.bsdfProb;
			debugBuffer1[sampledPixel].z += 1.0f;
			sppDebugFrame(sampledPixel, sampledDepth);
		}

		Paths* paths;
		ReplayBuffer* replayBuffer;
		NpgTrainingData* npgTrainingData;
		InferenceQueries* inferenceQueries;
		unsigned* lcgSeeds;
		ToneMapperSettings toneMapperSettings;

		math::vec3f* debugBuffer1;
		math::vec3f* debugBuffer2;
		math::vec3f* debugBuffer3;

		int numberOfPixels;
		int maxDatasetSize;
		int maxDepth;
		network::config::DistributionType distributionType;
		int mixtureSize;
	};
}

#endif
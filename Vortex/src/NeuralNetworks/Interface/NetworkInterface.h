#pragma once
#ifndef NETWORK_INTERFACE_H
#define NETWORK_INTERFACE_H
#include "NetworkInputs.h"
#include "NpgTrainingData.h"
#include "Paths.h"
#include "ReplayBuffer.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "NeuralNetworks/NetworkSettings.h"
#include "Scene/Nodes/RendererSettings.h"
#include "Device/DevicePrograms/ToneMapper.h"

namespace vtx
{
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
			const int& numberOfPixels,
			const int& maxDatasetSize,
			const int& maxDepth,
			const int& frameId,
			const network::DistributionType distributionType,
			const int mixtureSize,
			const ToneMapperSettings& _toneMapperSettings,
			const WhatChanged& changed);

	private:
		NetworkInterface(const int& numberOfPixels, const int& maxDatasetSize, const int& maxDepth, const int& frameId, const network::DistributionType distributionType, int mixtureSize, const ToneMapperSettings& _toneMapperSettings, WhatChanged changed);
	public:

		

		__forceinline__ __device__ void sppDebugFrame(const int sampledPixel, const int sampledDepth)
		{
			debugBuffer3[sampledPixel].x += 1.0f;
		}

		__forceinline__ __device__ void shuffleDataset(const int& trainingDatasetIdx, const network::NetworkSettings& settings)
		{
			if (trainingDatasetIdx >= npgTrainingData->nAlloc)
			{
				return;
			}

			unsigned& seed = lcgSeeds[trainingDatasetIdx];

			Paths::Hit hit;
			Paths::Hit* nextHit = nullptr;
			if (settings.type == network::NT_SAC)
			{
				*nextHit = Paths::Hit();
			}
			Paths::LightContribution lightContribution;
			bool isTerminal = false;
			int sampledPixel = 0;
			int sampledDepth = 0;
			if(settings.trainingBatchGenerationSettings.strategy == network::SS_ALL || settings.trainingBatchGenerationSettings.strategy == network::SS_PATHS_WITH_CONTRIBUTION)
			{
				paths->getRandomPathSample(
					seed, settings.trainingBatchGenerationSettings,
					sampledPixel, sampledDepth,
					&hit, &lightContribution, isTerminal, nextHit
				);
			}
			else if(settings.trainingBatchGenerationSettings.strategy == network::SS_LIGHT_SAMPLES)
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

			const float signal = utl::luminance(lightContribution.outLight);
			if (settings.type == network::NT_SAC)
			{
				replayBuffer->state->addState(trainingDatasetIdx, hit.position, hit.wOutgoing, hit.normal);
				replayBuffer->nextState->addState(trainingDatasetIdx, nextHit->position, nextHit->wOutgoing, nextHit->normal);
				replayBuffer->action[trainingDatasetIdx] = lightContribution.wIncoming; //hit incoming direction
				replayBuffer->reward[trainingDatasetIdx] = signal; //hit incoming direction
				replayBuffer->doneSignal[trainingDatasetIdx] = isTerminal; //hit incoming direction
			}
			else if (settings.type == network::NT_NGP)
			{
				npgTrainingData->inputs->addState(trainingDatasetIdx, hit.position, hit.wOutgoing, hit.normal);
				npgTrainingData->incomingDirection[trainingDatasetIdx] = lightContribution.wIncoming;
				npgTrainingData->bsdfProbabilities[trainingDatasetIdx] = lightContribution.bsdfProb;
				npgTrainingData->luminance[trainingDatasetIdx] = signal;
			}
			else
			{
				printf("Unknown network type, file: %s, line %d\n", __FILE__, __LINE__);
			}
			
			debugBuffer1[sampledPixel].x += (signal + 1.0f) / 2.0f;
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
	};
}

#endif
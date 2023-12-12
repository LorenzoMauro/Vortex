#include "NetworkInterface.h"

#include <random>

#include "InferenceQueries.h"
#include "NpgTrainingData.h"
#include "Paths.h"
#include "ReplayBuffer.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"

namespace vtx
{
	NetworkInterface* NetworkInterface::upload(const int&                              numberOfPixels, const int&  maxDatasetSize,
											   const int&                              maxDepth, const int&        frameId,
											   const network::config::DistributionType distributionType, const int mixtureSize,
											   const ToneMapperSettings&               _toneMapperSettings,
											   const WhatChanged&                      changed, device::NetworkInterfaceBuffer& networkInterfaceBuffers)
	{
		const NetworkInterface networkInterface(numberOfPixels, maxDatasetSize, maxDepth, frameId, distributionType,
												mixtureSize, _toneMapperSettings, changed, networkInterfaceBuffers);
		return networkInterfaceBuffers.networkInterfaceBuffer.upload(networkInterface);
	}

	NetworkInterface::NetworkInterface(const int&  _numberOfPixels, const int&                      _maxDatasetSize, const int& _maxDepth,
									   const int&  frameId, const network::config::DistributionType _distributionType,
									   int         _mixtureSize, const ToneMapperSettings&          _toneMapperSettings,
									   WhatChanged changed, device::NetworkInterfaceBuffer&         networkInterfaceBuffers)
	{
		numberOfPixels = _numberOfPixels;
		maxDatasetSize = _maxDatasetSize;
		maxDepth = _maxDepth;
		distributionType = _distributionType;
		mixtureSize = _mixtureSize;

		if (changed.maxDatasetSize)
		{
			const unsigned        chronoSeed = std::chrono::system_clock::now().time_since_epoch().count();
			std::vector<unsigned> seeds(maxDatasetSize);
			unsigned              seed = tea<4>(frameId, chronoSeed);
			std::mt19937 mt(chronoSeed); // Initialize with chronoSeed

			for (int i = 0; i < maxDatasetSize; i++)
			{
				seeds[i] = mt();
				//lcg(seed);
				//seeds[i] = seed;
			}
			lcgSeeds = networkInterfaceBuffers.seedsBuffer.upload(seeds);
			VTX_INFO("Allocating ReplayBuffer and NpgTrainingData");
			replayBuffer    = ReplayBuffer::upload(networkInterfaceBuffers.replayBufferBuffers, maxDatasetSize);
			npgTrainingData = NpgTrainingData::upload(networkInterfaceBuffers.npgTrainingDataBuffers, maxDatasetSize);
		}
		else
		{
			lcgSeeds        = networkInterfaceBuffers.seedsBuffer.castedPointer<unsigned>();
			replayBuffer    = ReplayBuffer::getPreviouslyUploaded(networkInterfaceBuffers.replayBufferBuffers);
			npgTrainingData = NpgTrainingData::getPreviouslyUploaded(networkInterfaceBuffers.npgTrainingDataBuffers);
		}


		if (changed.distributionType || changed.numberOfPixels)
		{
			VTX_INFO("Allocating InferenceQueries");
			inferenceQueries = InferenceQueries::upload(networkInterfaceBuffers.inferenceBuffers, numberOfPixels, distributionType, mixtureSize);
		}
		else
		{
			inferenceQueries = InferenceQueries::getPreviouslyUploaded(networkInterfaceBuffers.inferenceBuffers);
		}

		if (changed.numberOfPixels || changed.maxDepth)
		{
			VTX_INFO("Allocating Paths");
			paths = Paths::upload(networkInterfaceBuffers.pathsBuffers, maxDepth, numberOfPixels);
		}
		else
		{
			//Paths::resetBounces(maxDepth, numberOfPixels);
			paths = Paths::getPreviouslyUploaded(networkInterfaceBuffers.pathsBuffers);
		}

		if (changed.numberOfPixels)
		{
			debugBuffer1 = networkInterfaceBuffers.debugBuffer1Buffer.alloc<math::vec3f>(numberOfPixels);
			debugBuffer2 = networkInterfaceBuffers.debugBuffer2Buffer.alloc<math::vec3f>(numberOfPixels);
			debugBuffer3 = networkInterfaceBuffers.debugBuffer3Buffer.alloc<math::vec3f>(numberOfPixels);
		}
		else
		{
			debugBuffer1 = networkInterfaceBuffers.debugBuffer1Buffer.castedPointer<math::vec3f>();
			debugBuffer2 = networkInterfaceBuffers.debugBuffer2Buffer.castedPointer<math::vec3f>();
			debugBuffer3 = networkInterfaceBuffers.debugBuffer3Buffer.castedPointer<math::vec3f>();
		}

		toneMapperSettings = _toneMapperSettings;
	}
}

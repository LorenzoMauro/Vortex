#include "NetworkInterface.h"

#include "InferenceQueries.h"
#include "NpgTrainingData.h"
#include "Paths.h"
#include "ReplayBuffer.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	NetworkInterface* NetworkInterface::upload(const int& numberOfPixels, const int& maxDatasetSize,
											   const int& maxDepth, const int& frameId,
											   const network::DistributionType distributionType, const int mixtureSize,
											   const ToneMapperSettings& _toneMapperSettings,
											   const WhatChanged& changed)
	{
		const NetworkInterface networkInterface(numberOfPixels, maxDatasetSize, maxDepth, frameId, distributionType,
												mixtureSize, _toneMapperSettings, changed);
		return UPLOAD_BUFFERS->networkInterfaceBuffer.networkInterfaceBuffer.upload(networkInterface);
	}

	NetworkInterface::NetworkInterface(const int&  numberOfPixels, const int& maxDatasetSize, const int& maxDepth,
									   const int&  frameId, const network::DistributionType distributionType,
									   int         mixtureSize, const ToneMapperSettings& _toneMapperSettings,
									   WhatChanged changed)
	{
		//NetworkInterface networkInterface;
		device::Buffers::NetworkInterfaceBuffer& networkInterfaceBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer;

		if (changed.maxDatasetSize)
		{
			const unsigned        chronoSeed = std::chrono::system_clock::now().time_since_epoch().count();
			std::vector<unsigned> seeds(maxDatasetSize);
			unsigned              seed = tea<4>(frameId, chronoSeed);
			for (int i = 0; i < maxDatasetSize; i++)
			{
				lcg(seed);
				seeds[i] = seed;
			}
			lcgSeeds = networkInterfaceBuffers.seedsBuffer.upload(seeds);

			replayBuffer    = ReplayBuffer::upload(maxDatasetSize);
			npgTrainingData = NpgTrainingData::upload(maxDatasetSize);
		}
		else
		{
			lcgSeeds        = networkInterfaceBuffers.seedsBuffer.castedPointer<unsigned>();
			replayBuffer    = ReplayBuffer::getPreviouslyUploaded();
			npgTrainingData = NpgTrainingData::getPreviouslyUploaded();
		}


		if (changed.distributionType || changed.numberOfPixels)
		{
			inferenceQueries = InferenceQueries::upload(numberOfPixels, distributionType, mixtureSize);
		}
		else
		{
			inferenceQueries = InferenceQueries::getPreviouslyUploaded();
		}

		if (changed.numberOfPixels || changed.maxDepth)
		{
			paths = Paths::upload(maxDepth, numberOfPixels);
		}
		else
		{
			//Paths::resetBounces(maxDepth, numberOfPixels);
			paths = Paths::getPreviouslyUploaded();
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

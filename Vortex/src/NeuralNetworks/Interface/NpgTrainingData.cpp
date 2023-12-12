#include "NpgTrainingData.h"
#include "NetworkInputs.h"
#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	NpgTrainingData* NpgTrainingData::upload(device::NpgTrainingDataBuffers& buffers, const int& maxDatasetSize)
	{
		const NpgTrainingData npgTrainingData(buffers, maxDatasetSize);
		return buffers.npgTrainingDataStructBuffers.upload(npgTrainingData);
	}

	NpgTrainingData* NpgTrainingData::getPreviouslyUploaded(const device::NpgTrainingDataBuffers& buffers)
	{
		return buffers.npgTrainingDataStructBuffers.castedPointer<NpgTrainingData>();
	}

	NpgTrainingData::NpgTrainingData(device::NpgTrainingDataBuffers& buffers, const int& maxDatasetSize)
	{
		outRadiance = buffers.outgoingRadianceBuffer.alloc<math::vec3f>(maxDatasetSize);
		inRadiance = buffers.incomingRadianceBuffer.alloc<math::vec3f>(maxDatasetSize);
		throughput = buffers.trhoughputBuffer.alloc<math::vec3f>(maxDatasetSize);
		incomingDirection = buffers.incomingDirectionBuffer.alloc<math::vec3f>(maxDatasetSize);
		bsdfProbabilities = buffers.bsdfProbabilitiesBuffer.alloc<float>(maxDatasetSize);
		inputs = NetworkInput::upload(maxDatasetSize, buffers.inputBuffer);
		nAlloc = maxDatasetSize;
	}
}


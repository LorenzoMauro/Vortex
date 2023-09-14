#include "NpgTrainingData.h"
#include "NetworkInputs.h"
#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	NpgTrainingData* NpgTrainingData::upload(const int& maxDatasetSize)
	{
		const NpgTrainingData npgTrainingData(maxDatasetSize);
		return UPLOAD_BUFFERS->networkInterfaceBuffer.npgTrainingDataBuffers.npgTrainingDataStructBuffers.upload(npgTrainingData);
	}

	NpgTrainingData* NpgTrainingData::getPreviouslyUploaded()
	{
		return UPLOAD_BUFFERS->networkInterfaceBuffer.npgTrainingDataBuffers.npgTrainingDataStructBuffers.castedPointer<NpgTrainingData>();
	}

	NpgTrainingData::NpgTrainingData(const int& maxDatasetSize)
	{
		device::Buffers::NpgTrainingDataBuffers& ngpTrainingDataBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.npgTrainingDataBuffers;
		luminance = ngpTrainingDataBuffers.outgoingRadianceBuffer.alloc<float>(maxDatasetSize);
		incomingDirection = ngpTrainingDataBuffers.incomingDirectionBuffer.alloc<math::vec3f>(maxDatasetSize);
		bsdfProbabilities = ngpTrainingDataBuffers.bsdfProbabilitiesBuffer.alloc<float>(maxDatasetSize);
		inputs = NetworkInput::upload(maxDatasetSize, ngpTrainingDataBuffers.inputBuffer);
		nAlloc = maxDatasetSize;
	}
}


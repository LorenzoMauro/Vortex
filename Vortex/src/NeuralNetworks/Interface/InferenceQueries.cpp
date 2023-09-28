#include "InferenceQueries.h"

#include "Device/UploadCode/UploadBuffers.h"

namespace vtx
{
	InferenceQueries* InferenceQueries::upload(device::InferenceBuffers& buffers, const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize)
	{
		const InferenceQueries inferenceQueries(buffers, numberOfPixels, type, _mixtureSize);
		return buffers.inferenceStructBuffer.upload(inferenceQueries);
	}

	InferenceQueries* InferenceQueries::getPreviouslyUploaded(const device::InferenceBuffers& buffers)
	{
		return buffers.inferenceStructBuffer.castedPointer<InferenceQueries>();
	}

	InferenceQueries::InferenceQueries(device::InferenceBuffers& buffers, const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize)
	{
		device::InferenceBuffers& inferenceBuffers = buffers;
		distributionParameters = inferenceBuffers.distributionParameters.alloc<float>(numberOfPixels * _mixtureSize * distribution::Mixture::getDistributionParametersCount(type));
		mixtureWeights = inferenceBuffers.mixtureWeightBuffer.alloc<float>(numberOfPixels * _mixtureSize);
		samplingFractionArray = inferenceBuffers.samplingFractionArrayBuffer.alloc<float>(numberOfPixels);
		samples = inferenceBuffers.samplesBuffer.alloc<math::vec3f>(numberOfPixels);
		prob = inferenceBuffers.probabilitiesBuffer.alloc<float>(numberOfPixels);

		distributionType = type;// inferenceBuffers.distributionTypeBuffer.upload(type);
		state = NetworkInput::upload(numberOfPixels, inferenceBuffers.stateBuffer);
		size = inferenceBuffers.inferenceSize.upload(0);
		maxSize = numberOfPixels;
		mixtureSize = _mixtureSize;
	}

}

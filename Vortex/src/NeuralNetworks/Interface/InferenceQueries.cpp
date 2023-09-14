#include "InferenceQueries.h"

namespace vtx
{
	InferenceQueries* InferenceQueries::upload(const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize)
	{
		const InferenceQueries inferenceQueries(numberOfPixels, type, _mixtureSize);
		return UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers.inferenceStructBuffer.upload(inferenceQueries);
	}

	InferenceQueries* InferenceQueries::getPreviouslyUploaded()
	{
		return UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers.inferenceStructBuffer.castedPointer<InferenceQueries>();
	}

	InferenceQueries::InferenceQueries(const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize)
	{
		device::Buffers::InferenceBuffers& inferenceBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers;
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

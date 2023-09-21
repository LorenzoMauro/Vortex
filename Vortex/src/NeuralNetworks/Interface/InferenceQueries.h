#pragma once
#ifndef INFERENCE_QUERIES_H
#define INFERENCE_QUERIES_H
#include "NetworkInputs.h"
#include "Core/Math.h"
#include "NeuralNetworks/NetworkSettings.h"
#include "Device/Wrappers/WorkItems.h"
#include "Device/DevicePrograms/nvccUtils.h"

#define CUDA_INTERFACE
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx
{
	struct RayWorkItem;
	struct NetworkInput;

	struct InferenceQueries
	{
		static InferenceQueries* upload(const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize);

		static InferenceQueries* getPreviouslyUploaded();

	private:
		InferenceQueries(const int& numberOfPixels, const network::DistributionType& type, const int& _mixtureSize);
	public:

		__forceinline__ __device__ void reset()
		{
			*size = 0;
		}

		__forceinline__ __device__ int addInferenceQuery(const RayWorkItem& prd, const int index)
		{
			const int newSize = cuAtomicAdd(size, 1);
			state->addState(index, prd.hitProperties.position, prd.direction, prd.hitProperties.shadingNormal);
			return index;
		}

		__forceinline__ __device__ float evaluate(const int& idx, const math::vec3f& action)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			const float  sampleProb = distribution::Mixture::evaluate(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, action);
			return sampleProb;
		}

		__forceinline__ __device__ math::vec4f sample(const int& idx, unsigned& seed) const
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			math::vec3f  sample = distribution::Mixture::sample(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, seed);
			const float  sampleProb = distribution::Mixture::evaluate(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, sample);

			/*if(idx<10)
			{
				float sampleNorm = math::length(sample);

				printf(
					"Sample : %f %f %f\n"
					"Sample norm : %f\n"
					"Sample prob : %f\n\n",
					sample.x, sample.y, sample.z,
					sampleNorm,
					sampleProb
				);
			}*/
			

			return { sample, sampleProb };
		}

		__forceinline__ __device__ void getSampleMixtureParameters(const int& idx, const float*& sampleMixtureParams, const float*& sampleMixtureWeights) const
		{
			const int mixtureParamsOffset = idx * mixtureSize * distribution::Mixture::getDistributionParametersCount(distributionType);
			const int mixtureWeightsOffset = idx * mixtureSize;
			sampleMixtureParams = distributionParameters + mixtureParamsOffset;
			sampleMixtureWeights = mixtureWeights + mixtureWeightsOffset;
		}

		__forceinline__ __device__ math::vec3f& getStatePosition(const int& idx)
		{
			return state->position[idx];
		}

		__forceinline__ __device__ math::vec3f& getStateDirection(const int& idx)
		{
			return state->wo[idx];
		}

		__forceinline__ __device__ math::vec3f& getStateNormal(const int& idx)
		{
			return state->normal[idx];
		}

		__forceinline__ __device__ math::vec3f getMean(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAvarageAxis(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}

		__forceinline__ __device__ float getConcentration(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAvarageConcentration(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}

		__forceinline__ __device__ float& getSamplingFraction(const int& idx)
		{
			return samplingFractionArray[idx];
		}

		__forceinline__ __device__ float getAnisotropy(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAverageAnisotropy(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}

		NetworkInput* state;

		float* distributionParameters;
		float* mixtureWeights;
		float* samplingFractionArray;
		math::vec3f* samples;
		float* prob;

		int* size;
		int                        maxSize;
		network::DistributionType  distributionType;
		int                        mixtureSize;
	};
}
#endif
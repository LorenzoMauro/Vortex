#pragma once

#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/Wrappers/WorkItems.h"

#define EPS 1e-6f
#define N_STATE_VECTOR 3
#define LIGHT_PATH_SAMPLING_PROB 0.5f

#define UPLOAD(bufferName, variableName, size, type) \
	(bufferName).resize((size_t)(size) * sizeof(type)); \
	(variableName) = (bufferName).castedPointer<type>()

namespace vtx {
	__forceinline__ __device__ bool isNan(const math::vec3f& vec)
	{
		return (isnan(vec.x) || isnan(vec.y) || isnan(vec.z));
	}

	__forceinline__ __device__ void constructState(
		float* state,const int& index,
		const math::vec3f& position,
		const math::vec3f& incomingDirection,
		const math::vec3f& shadingNormal
	)
	{
		state[index * 3 * N_STATE_VECTOR + 0] = position.x;
		state[index * 3 * N_STATE_VECTOR + 1] = position.y;
		state[index * 3 * N_STATE_VECTOR + 2] = position.z;
		state[index * 3 * N_STATE_VECTOR + 3] = incomingDirection.x;
		state[index * 3 * N_STATE_VECTOR + 4] = incomingDirection.y;
		state[index * 3 * N_STATE_VECTOR + 5] = incomingDirection.z;
		state[index * 3 * N_STATE_VECTOR + 6] = shadingNormal.x;
		state[index * 3 * N_STATE_VECTOR + 7] = shadingNormal.y;
		state[index * 3 * N_STATE_VECTOR + 8] = shadingNormal.z;
	}

	__forceinline__ __device__ float luminanceLocal(const math::vec3f& rgb)
	{
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };
		return dot(rgb, ntscLuminance);
	}

	struct Bounce
	{
		math::vec3f* position;
		math::vec3f* normal;
		math::vec3f* wo;

		math::vec3f* bsWi;
		math::vec3f* bsBsdf;
		math::vec3f* bsRadiance;
		bool*		 isBsValid;

		math::vec3f* lsPosition;
		math::vec3f* lsNormal;
		math::vec3f* lsBsdf;
		math::vec3f* lsRadiance;
		bool* isLsValid;

		bool* isBounceValid;
		int depth;
		int originPixel;

		__forceinline__ __device__ void print()
		{
			printf(
				"\nPosition %f %f %f\n"
				"Normal %f %f %f\n"
				"Wo %f %f %f\n"
				"BsWi %f %f %f\n"
				"BsBsdf %f %f %f\n"
				"BsRadiance %f %f %f\n"
				"LsPosition %f %f %f\n"
				"LsBsdf %f %f %f\n"
				"LsRadiance %f %f %f\n"
				"IsBounceValid %d\n"
				"IsLsValid %d\n"
				"IsBsValid %d\n"
				"Depth %d\n"
				"OriginPixel %d\n",
				position->x, position->y, position->z,
				normal->x, normal->y, normal->z,
				wo->x, wo->y, wo->z,
				bsWi->x, bsWi->y, bsWi->z,
				bsBsdf->x, bsBsdf->y, bsBsdf->z,
				bsRadiance->x, bsRadiance->y, bsRadiance->z,
				lsPosition->x, lsPosition->y, lsPosition->z,
				lsBsdf->x, lsBsdf->y, lsBsdf->z,
				lsRadiance->x, lsRadiance->y, lsRadiance->z,
				*isBounceValid,
				*isLsValid,
				*isBsValid,
				depth,
				originPixel
			);
		}
	};

	struct Paths
	{
		Paths(const int& _maxDepth, const int& _numberOfPixels)
		{
			numberOfPixels = _numberOfPixels;
			maxDepth = _maxDepth;

			device::Buffers::PathsBuffer& pathBuffer = UPLOAD_BUFFERS->networkInterfaceBuffer.pathsBuffers;

			CUDABuffer& positionBuffer = pathBuffer.positionBuffer;
			CUDABuffer& normalBuffer = pathBuffer.normalBuffer;
			CUDABuffer& woBuffer = pathBuffer.woBuffer;
			CUDABuffer& bsWiBuffer = pathBuffer.bsWiBuffer;
			CUDABuffer& bsBsdfBuffer = pathBuffer.bsBsdfBuffer;
			CUDABuffer& bsRadianceBuffer = pathBuffer.bsRadianceBuffer;
			CUDABuffer& lsPositionBuffer = pathBuffer.lsPositionBuffer;
			CUDABuffer& lsNormalBuffer = pathBuffer.lsNormalBuffer;
			CUDABuffer& lsBsdfBuffer = pathBuffer.lsBsdfBuffer;
			CUDABuffer& lsRadianceBuffer = pathBuffer.lsRadianceBuffer;

			CUDABuffer& maxActualDepthBuffer = pathBuffer.maxActualDepthBuffer;
			CUDABuffer& validPixelsBuffer = pathBuffer.validPixelsBuffer;

			CUDABuffer& isBounceValidBuffer = pathBuffer.isBounceValidBuffer;
			CUDABuffer& isLsValidBuffer = pathBuffer.isLsValidBuffer;
			CUDABuffer& isBsValidBuffer = pathBuffer.isBsValidBuffer;

			UPLOAD(positionBuffer, position, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(normalBuffer, normal, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(woBuffer, wo, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(bsWiBuffer, bsWi, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(bsBsdfBuffer, bsBsdf, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(bsRadianceBuffer, bsRadiance, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(lsPositionBuffer, lsPosition, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(lsNormalBuffer, lsNormal, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(lsBsdfBuffer, lsBsdf, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(lsRadianceBuffer, lsRadiance, (numberOfPixels * maxDepth), math::vec3f);
			UPLOAD(validPixelsBuffer, validPixels, (numberOfPixels), int);

			std::vector<bool> vecBool((numberOfPixels * maxDepth), false);
			std::vector<char> falseVector(vecBool.begin(), vecBool.end());

			const std::vector<int>      maxActualDepthVector((size_t)(numberOfPixels), 0);

			isBsValidBuffer.upload(falseVector);
			isBsValid = isBsValidBuffer.castedPointer<bool>();

			isLsValidBuffer.upload(falseVector);
			isLsValid = isLsValidBuffer.castedPointer<bool>();

			isBounceValidBuffer.upload(falseVector);
			isBounceValid = isBounceValidBuffer.castedPointer<bool>();

			maxActualDepthBuffer.upload(maxActualDepthVector);
			maxActualDepth = maxActualDepthBuffer.castedPointer<int>();

			validPixelsSize = 0;
		}

		__forceinline__ __device__ int bounceAddress(const int& originPixel, const int& depth)
		{
			return depth * numberOfPixels + originPixel;
		}

		__forceinline__ __device__ Bounce getBounce(const int& originPixel, const int depth)
		{
			Bounce bounce;
			const int idx = bounceAddress(originPixel, depth);
			bounce.position = &position[idx];
			bounce.normal = &normal[idx];
			bounce.wo = &wo[idx];

			bounce.bsWi = &bsWi[idx];
			bounce.bsBsdf = &bsBsdf[idx];
			bounce.bsRadiance = &bsRadiance[idx];
			bounce.isBsValid = &isBsValid[idx];

			bounce.lsPosition = &lsPosition[idx];
			bounce.lsNormal = &lsNormal[idx];
			bounce.lsBsdf = &lsBsdf[idx];
			bounce.lsRadiance = &lsRadiance[idx];
			bounce.isLsValid = &isLsValid[idx];

			bounce.isBounceValid = &isBounceValid[idx];

			bounce.depth = depth;
			bounce.originPixel = originPixel;
			return bounce;
		}

		__forceinline__ __device__ void recordHit(const RayWorkItem& prd)
		{
			if (prd.depth == 0)
			{
				const int newValidPixelIndex = cuAtomicAdd(&validPixelsSize, 1);
				validPixels[newValidPixelIndex] = prd.originPixel;
			}
			const Bounce bounce = getBounce(prd.originPixel, prd.depth);
			*bounce.position = prd.hitProperties.position;
			*bounce.normal = prd.hitProperties.shadingNormal;
			*bounce.wo = prd.direction;
			*bounce.isBounceValid = true;
			maxActualDepth[prd.originPixel] = prd.depth;
			*bounce.isBsValid = false;
			*bounce.isLsValid = false;
		}

		__forceinline__ __device__ void recordLightSample(const int& originPixel, const int& depth, const math::vec3f& bxdf, const math::vec3f& radiance, const math::vec3f& lightPosition, const math::vec3f& lightSampleNormal)
		{
			const Bounce bounce = getBounce(originPixel, depth - 1);
			if(bxdf == math::vec3f(0.0f))
			{
				// bad sampled Direction
				*bounce.isLsValid = true; // negative reward
			}
			*bounce.lsBsdf = bxdf;
			*bounce.lsRadiance = radiance;
			*bounce.lsPosition = lightPosition;
			*bounce.lsNormal = lightSampleNormal;
		}

		__forceinline__ __device__ void recordBsRadiance(const math::vec3f& radiance, const RayWorkItem& prd)
		{
			const Bounce bounce = getBounce(prd.originPixel, prd.depth - 1);
			*bounce.bsRadiance = radiance;
		}

		__forceinline__ __device__ void recordBsdfSample(const math::vec3f& bsdf, const RayWorkItem& prd, bool isAbsorb)
		{
			const Bounce bounce = getBounce(prd.originPixel, prd.depth);
			if(isAbsorb)
			{
				*bounce.isBsValid = false;
				return;
			}
			*bounce.bsBsdf = bsdf;
			*bounce.bsWi = prd.direction;
			*bounce.isBsValid = true;
		}

		__forceinline__ __device__ void validateLightSample(const ShadowWorkItem& swi)
		{
			const Bounce bounce = getBounce(swi.originPixel, swi.depth - 1);
			*bounce.isLsValid = true;
		}

		__forceinline__ __device__ void recordRadianceMiss(const math::vec3f& radiance, const EscapedWorkItem& ewi, const float& maxClamp)
		{
			const Bounce missBounce = getBounce(ewi.originPixel, ewi.depth);
			const Bounce previousBounce = getBounce(ewi.originPixel, ewi.depth - 1);
			maxActualDepth[ewi.originPixel] = ewi.depth;
			*missBounce.position = *previousBounce.position + maxClamp * ewi.direction;
			*missBounce.normal = -ewi.direction;
			*missBounce.isBounceValid = true;
			*missBounce.isBsValid = false;
			*missBounce.isLsValid = false;
			*previousBounce.bsRadiance = radiance;
		}

		__forceinline__ __device__ void resetPath(const int& originPixel)
		{
			maxActualDepth[originPixel] = -1;
		}

		__forceinline__ __device__ void resetValidPixels()
		{
			validPixelsSize = 0;
		}

		math::vec3f* position;
		math::vec3f* normal;
		math::vec3f* wo;

		math::vec3f* bsWi;
		math::vec3f* bsBsdf;
		math::vec3f* bsRadiance;
		bool* isBsValid;

		math::vec3f* lsPosition;
		math::vec3f* lsNormal;
		math::vec3f* lsBsdf;
		math::vec3f* lsRadiance;
		bool* isLsValid;
		bool* isBounceValid;

		int* maxActualDepth;

		int* validPixels;
		int validPixelsSize;

		int numberOfPixels;
		int maxDepth;
	};

	struct NetworkState
	{
		NetworkState() = default;

		NetworkState(const int& maxSize, device::Buffers::NetworkStateBuffers& buffers)
		{
			CUDABuffer& positionBuffer = buffers.positionBuffer;
			CUDABuffer& woBuffer = buffers.woBuffer;
			CUDABuffer& normalBuffer = buffers.normalBuffer;

			UPLOAD(positionBuffer, position, maxSize, math::vec3f);
			UPLOAD(woBuffer, wo, maxSize, math::vec3f);
			UPLOAD(normalBuffer, normal, maxSize, math::vec3f);
		}

		__forceinline__ __device__ void addState(const int& index, const math::vec3f& position, const math::vec3f& wo, const math::vec3f& normal)
		{
			this->position[index] = position;
			this->wo[index] = wo;
			this->normal[index] = normal;
		}
		math::vec3f* position;
		math::vec3f* wo;
		math::vec3f* normal;
	};

	struct ReplayBuffer
	{
		ReplayBuffer(const int& maxReplayBufferSize)
		{
			device::Buffers::ReplayBufferBuffers& replayBufferBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.replayBufferBuffers;

			CUDABuffer& actionBuffer = replayBufferBuffers.actionBuffer;
			CUDABuffer& rewardBuffer = replayBufferBuffers.rewardBuffer;
			CUDABuffer& doneBuffer = replayBufferBuffers.doneBuffer;
			CUDABuffer& networkStateStructBuffer = replayBufferBuffers.stateBuffer.networkStateStructBuffer;
			CUDABuffer& nextStateStructBuffer = replayBufferBuffers.nextStatesBuffer.networkStateStructBuffer;

			UPLOAD(actionBuffer, action, maxReplayBufferSize, math::vec3f);
			UPLOAD(rewardBuffer, reward, maxReplayBufferSize, float);
			UPLOAD(doneBuffer, doneSignal, maxReplayBufferSize, int);


			const NetworkState hostState(maxReplayBufferSize, replayBufferBuffers.stateBuffer);
			networkStateStructBuffer.upload(hostState);
			state = networkStateStructBuffer.castedPointer<NetworkState>();


			const NetworkState hostNextState(maxReplayBufferSize, replayBufferBuffers.nextStatesBuffer);
			nextStateStructBuffer.upload(hostNextState);
			nextState = nextStateStructBuffer.castedPointer<NetworkState>();

			nAlloc = maxReplayBufferSize;
		}

		NetworkState* state;
		NetworkState* nextState;
		math::vec3f* action;
		float*		 reward;
		int*		 doneSignal;
		int nAlloc;
	};

	struct InferenceQueries
	{
		InferenceQueries(const int& numberOfPixels)
		{
			device::Buffers::InferenceBuffers& inferenceBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer.inferenceBuffers;
			CUDABuffer& networkStateStructBuffer = inferenceBuffers.stateBuffer.networkStateStructBuffer;

			UPLOAD(inferenceBuffers.meanBuffer, meanArray, numberOfPixels, math::vec3f);
			UPLOAD(inferenceBuffers.concentrationBuffer, concentrationArray, numberOfPixels, float);

			const NetworkState hostState(numberOfPixels, inferenceBuffers.stateBuffer);
			networkStateStructBuffer.upload(hostState);
			state = networkStateStructBuffer.castedPointer<NetworkState>();

			inferenceBuffers.inferenceSize.upload(0);
			size = inferenceBuffers.inferenceSize.castedPointer<int>();
			maxSize = numberOfPixels;
		}

		__forceinline__ __device__ void reset()
		{
			*size = 0;
		}
		__forceinline__ __device__ int addInferenceQuery(const RayWorkItem& prd, int index)
		{
			const int newSize = cuAtomicAdd(size, 1);
			state->addState(index, prd.hitProperties.position, prd.direction, prd.hitProperties.shadingNormal);
			return index;
		}

		__forceinline__ __device__ float evaluateVmf(const math::vec3f& mean, const float& k, const math::vec3f& action)
		{
			const float pdf = k / (2.0f * M_PI * (1 - expf(-2.0f * k))) * expf(k * dot(mean,action) - 1.0f);
			if (isnan(pdf))
			{
				return 0.0f;
			}
			return pdf;
		}

		__forceinline__ __device__ math::vec4f sampleVmf(const math::vec3f& mean, const float& k, unsigned& seed)
		{
			const float uniform = rng(seed);
			const float w = 1.0f + logf(uniform + (1.0f-uniform)*expf(-2.0f * k) + EPS) / (k + EPS);

			const float angleUniform = rng(seed) * 2.0f * M_PI;
			const math::vec2f v = math::vec2f(cosf(angleUniform), sinf(angleUniform));

			float      w_ = sqrtf(math::max(0.0f, 1.0f - w * w));
			const auto x = math::vec3f(w, w_ * v.x, w_ * v.y);

			const auto  e1 = math::vec3f(1.0f, 0.0f, 0.0f);
			math::vec3f u = e1 - mean;
			u = math::normalize(u);
			math::vec3f sample = x - 2.0f * math::dot(x, u) * u;
			const float pdf = evaluateVmf(mean, k, sample);

			return { sample, pdf };
		}

		__forceinline__ __device__ float evaluate(const int& idx, const math::vec3f& action)
		{
			const math::vec3f mean = meanArray[idx];
			const float k = concentrationArray[idx];
			return evaluateVmf(mean, k, action);
		}

		__forceinline__ __device__ math::vec4f sample(const int& idx, unsigned& seed)
		{
			const math::vec3f mean = meanArray[idx];
			const float k = concentrationArray[idx];
			return sampleVmf(mean, k, seed);
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

		__forceinline__ __device__ math::vec3f& getMean(const int& idx)
		{
			return meanArray[idx];
		}

		__forceinline__ __device__ float& getConcentration(const int& idx)
		{
			return concentrationArray[idx];
		}

		NetworkState* state;
		math::vec3f* meanArray;
		float* concentrationArray;
		int* size;
		int maxSize;
	};

	struct NetworkInterface
	{
		NetworkInterface(const int& numberOfPixels, const int& maxReplayBufferSize, const int& maxDepth, const int& frameId)
		{
			//NetworkInterface networkInterface;
			device::Buffers::NetworkInterfaceBuffer& networkInterfaceBuffers = UPLOAD_BUFFERS->networkInterfaceBuffer;

			CUDABuffer& inferenceStructBuffer = networkInterfaceBuffers.inferenceBuffers.inferenceStructBuffer;
			CUDABuffer& replayStructBuffer = networkInterfaceBuffers.replayBufferBuffers.replayBufferStructBuffer;
			CUDABuffer& pathsStructBuffer = networkInterfaceBuffers.pathsBuffers.pathStructBuffer;
			CUDABuffer& seedsBuffer = networkInterfaceBuffers.seedsBuffer;
			CUDABuffer& debugBuffer1Buffer = networkInterfaceBuffers.debugBuffer1Buffer;
			CUDABuffer& debugBuffer2Buffer = networkInterfaceBuffers.debugBuffer2Buffer;
			CUDABuffer& debugBuffer3Buffer = networkInterfaceBuffers.debugBuffer3Buffer;

			const Paths hostPaths(maxDepth, numberOfPixels);
			pathsStructBuffer.upload(hostPaths);
			paths = pathsStructBuffer.castedPointer<Paths>();

			const ReplayBuffer hostReplayBuffer(maxReplayBufferSize);
			replayStructBuffer.upload(hostReplayBuffer);
			replayBuffer = replayStructBuffer.castedPointer<ReplayBuffer>();

			const InferenceQueries hostInferenceQueries(numberOfPixels);
			inferenceStructBuffer.upload(hostInferenceQueries);
			inferenceQueries = inferenceStructBuffer.castedPointer<InferenceQueries>();

			debugBuffer1Buffer.resize(numberOfPixels * sizeof(math::vec3f));
			debugBuffer1 = debugBuffer1Buffer.castedPointer<math::vec3f>();

			debugBuffer2Buffer.resize(numberOfPixels * sizeof(math::vec3f));
			debugBuffer2 = debugBuffer2Buffer.castedPointer<math::vec3f>();

			debugBuffer3Buffer.resize(numberOfPixels * sizeof(math::vec3f));
			debugBuffer3 = debugBuffer3Buffer.castedPointer<math::vec3f>();

			const unsigned chronoSeed = std::chrono::system_clock::now().time_since_epoch().count();
			std::vector<unsigned> seeds(maxReplayBufferSize);
			unsigned seed = tea<4>(frameId, chronoSeed);
			for (int i = 0; i < maxReplayBufferSize; i++)
			{
				lcg(seed);
				seeds[i] = seed;
			}
			seedsBuffer.upload(seeds);
			lcgSeeds = seedsBuffer.castedPointer<unsigned>();
		}

		__forceinline__ __device__ void randomFillReplayBuffer(const int& replayBufferIdx) const
		{
			if (replayBufferIdx >= replayBuffer->nAlloc)
			{
				return;
			}

			unsigned& seed = lcgSeeds[replayBufferIdx];

			float rndDepth = rng(seed);
			float rndLight = rng(seed);

			bool isValid = false;
			int randomOriginPixel;
			int maxDepth;
			while(!isValid)
			{
				float rndPixel = rng(seed);
				randomOriginPixel = paths->validPixels[(int)(rndPixel * (float)paths->validPixelsSize)];
				maxDepth = paths->maxActualDepth[randomOriginPixel];	
				isValid = maxDepth > 0; //HACK, Something is wrong, all paths in validPixels should have at least depth 1
			}

			const int randomDepth = (int)round(fmaxf(0.0f, rndDepth * (float)(maxDepth-1)));
			Bounce bounce = paths->getBounce(randomOriginPixel, randomDepth);

			math::vec3f nextPosition;
			math::vec3f nextDirection;
			math::vec3f nextNormal;
			math::vec3f radiance;
			float reward;
			int doneSignal = 0;
			if(*bounce.isLsValid && rndLight <= LIGHT_PATH_SAMPLING_PROB)
			{
				nextPosition = *bounce.lsPosition;
				nextDirection = nextPosition - *bounce.position;
				nextNormal = *bounce.lsNormal;
				doneSignal = 1;
				if (*bounce.lsBsdf < math::vec3f(0.00001f))
				{
					radiance = -0.10f;
				}
				else
				{
					radiance = *bounce.lsRadiance * *bounce.lsBsdf;
				}
			}
			else
			{
				if(!*bounce.isBsValid)
				{
					nextPosition = *bounce.position;
					nextNormal = *bounce.normal;
					radiance = -0.10f;
					doneSignal = 1;
				}
				else
				{
					nextPosition = paths->position[paths->bounceAddress(randomOriginPixel, randomDepth + 1)];
					nextNormal = paths->normal[paths->bounceAddress(randomOriginPixel, randomDepth + 1)];
					radiance = *bounce.bsRadiance * *bounce.bsBsdf;
					if (randomDepth == maxDepth - 1)
					{
						doneSignal = 1;
						if(radiance == math::vec3f(0.0f))
						{
							radiance = -0.10f;
						}
					}
				}
				nextDirection = *bounce.bsWi;
			}

			//reward = fmaxf(-1.0f, fminf(1.0f, 0.30f * radiance.x + 0.59f * radiance.y + 0.11f * radiance.z));
			reward = radiance.x + radiance.y + radiance.z;

			replayBuffer->state->addState(replayBufferIdx, *bounce.position, *bounce.wo, *bounce.normal);
			replayBuffer->nextState->addState(replayBufferIdx, nextPosition, nextDirection, nextNormal);
			replayBuffer->action[replayBufferIdx] = nextDirection; //hit incoming direction
			replayBuffer->reward[replayBufferIdx] = reward; //hit incoming direction
			replayBuffer->doneSignal[replayBufferIdx] = doneSignal; //hit incoming direction

			debugBuffer1[randomOriginPixel].x += (reward+1.0f)/2.0f;
			debugBuffer1[randomOriginPixel].z += 1.0f;
			debugBuffer3[randomOriginPixel] += floatToScientificRGB(debugBuffer1[randomOriginPixel].z / 1000.0f);

			if(isNan(*bounce.wo) || maxDepth == 0)
			{
				bounce.print();
				printf(
					"\nShuffled Bounce %d origin Pixel %d to ReplayBuffer position %d\n"
					"State: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n"
					"NextState: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n"
					"Action: %.3f, %.3f, %.3f\n"
					"Reward: %.3f\n"
					"DoneSignal: %d\n"
					"Max Depth of Path: %d\n"
					"Radiance: %.3f, %.3f, %.3f\n"
					,
					randomDepth, randomOriginPixel, replayBufferIdx,
					replayBuffer->state->position[replayBufferIdx].x, replayBuffer->state->position[replayBufferIdx].y, replayBuffer->state->position[replayBufferIdx].z,
					replayBuffer->state->wo[replayBufferIdx].x, replayBuffer->state->wo[replayBufferIdx].y, replayBuffer->state->wo[replayBufferIdx].z,
					replayBuffer->nextState->position[replayBufferIdx].x, replayBuffer->nextState->position[replayBufferIdx].y, replayBuffer->nextState->position[replayBufferIdx].z,
					replayBuffer->nextState->wo[replayBufferIdx].x, replayBuffer->nextState->wo[replayBufferIdx].y, replayBuffer->nextState->wo[replayBufferIdx].z,
					replayBuffer->action[replayBufferIdx].x, replayBuffer->action[replayBufferIdx].y, replayBuffer->action[replayBufferIdx].z,
					replayBuffer->reward[replayBufferIdx],
					replayBuffer->doneSignal[replayBufferIdx],
					maxDepth,
					radiance.x, radiance.y, radiance.z
				);
			}
			
		}

		Paths* paths;
		ReplayBuffer* replayBuffer;
		InferenceQueries* inferenceQueries;
		unsigned* lcgSeeds;

		math::vec3f* debugBuffer1;
		math::vec3f* debugBuffer2;
		math::vec3f* debugBuffer3;
	};

}
#endif
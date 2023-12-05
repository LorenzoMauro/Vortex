#pragma once

#include "Core/Math.h"
#include "Device/DevicePrograms/nvccUtils.h"
#include "Device/DevicePrograms/Utils.h"
#include "NeuralNetworks/NetworkSettings.h"

namespace vtx {
	namespace device
	{
		struct PathsBuffer;
	}

	struct Hit
	{
		math::vec3f position;
		math::vec3f normal;
		math::vec3f outgoingRadiance;
		float       misWeight;
		bool         isEmissive;
	};

	struct RayExtension
	{
		math::vec3f throughputMultiplier;
		math::vec3f direction;
		float       bsdfPdf;
		float       sampleProb;
		bool        isValid = false;
	};

	struct Bounce
	{
		Hit				hit;
		math::vec3f		outgoingDirection;
		RayExtension	bsdfSample;
		RayExtension	lightSample;
		Hit				lightHit;
		bool 			isValid = false;
		bool			isTerminal = false;
	};

	enum EmissionSampleType
	{
		ES_SURFACE_EMISSION,
		ES_DIRECT_LIGHT,
		ES_MISS_EMISSION,
		ES_COUNT
	};

	struct Path
	{
		Path(Bounce* deviceBouncePtr, int _maxAllocDepth);

		__forceinline__ __device__ Bounce& operator[](const int& index)
		{
			assert(index >= 0);
			assert(index < maxAllocDepth);
			return bounces[index];
		}
		

		Bounce* bounces = nullptr;
		int maxDepth = 0;
		int maxAllocDepth = 0;
		bool isDepthAtTerminalZero = false;
		bool hasEmission = false;
	};

	struct Paths
	{
		static Paths* upload(device::PathsBuffer& buffers, const int& _maxDepth, const int& _numberOfPixels);

		static Paths* getPreviouslyUploaded(const device::PathsBuffer& buffers);


	private:
		Paths(device::PathsBuffer& buffers, const int& _maxDepth, const int& _numberOfPixels);

	public:

		__forceinline__ __device__ Path& operator[](const int& index) const
		{
			assert(index >= 0 && index < numberOfPaths);
			return paths[index];
		}


		__forceinline__ __device__ void resetValidPixels()
		{
			validPixelsCount = 0;
			pixelsWithContributionCount = 0;
			validLightSampleCount = 0;
		}

		__forceinline__ __device__ void resetPixel(const int& id)
		{
			paths[id].maxDepth = -1;
			paths[id].isDepthAtTerminalZero = false;
			paths[id].hasEmission = false;
		}


		__forceinline__ __device__ void assertBounceValidity(const int& originPixel, const int& depth)
		{
			assert(depth >= 0 && depth < maxAllocDepth);
			assert(originPixel >= 0 && originPixel < numberOfPaths);
			assert(paths[originPixel].bounces != nullptr);
		}

		__forceinline__ __device__ void recordBounceHit(
			const int& originPixel, const int& depth,
			const math::vec3f& position, const math::vec3f& normal, const math::vec3f& outgoingDirection)
		{
			assertBounceValidity(originPixel, depth);
			assert(!utl::isNan(position));
			assert(!utl::isNan(normal));

			Bounce& bounce = paths[originPixel][depth];
			bounce.hit.position = position;
			bounce.hit.normal = normal;
			bounce.hit.outgoingRadiance = math::vec3f(0.f);
			bounce.hit.misWeight = 1.0f;
			bounce.hit.isEmissive = false;
			bounce.isTerminal = false;
			bounce.outgoingDirection = outgoingDirection;
			bounce.isValid = true;
			bounce.bsdfSample.isValid = false;
			bounce.lightSample.isValid = false;

			if (depth < maxAllocDepth - 1)
			{
				paths[originPixel][depth + 1].isValid = false;
			}
			paths[originPixel].maxDepth += 1;
		}

		__forceinline__ __device__ void setBounceAsTerminal(const int& originPixel, const int& depth)
		{
			assertBounceValidity(originPixel, depth);
			Bounce& bounce = paths[originPixel][depth];
			bounce.isTerminal = true;

			// ATTENTION
			// If the first bounce gets absorbed then it's going to be terminated
			// now I won't add it to the valid pixels list, however these bounces might have direct lighting information
			// that could be used. Maybe a better approach is needed here.
			if (depth != 0)// && paths[originPixel].hasEmission)
			{
				validPixels[cuAtomicAdd(&validPixelsCount, 1)] = originPixel;
				if (paths[originPixel].hasEmission)
				{
					pixelsWithContribution[cuAtomicAdd(&pixelsWithContributionCount, 1)] = originPixel;
				}
			}
			else
			{
				paths[originPixel].isDepthAtTerminalZero = true;
			}
		}

		__forceinline__ __device__ void recordBsdfRayExtension(
			const int& originPixel, const int& depth,
			const math::vec3f& direction, const math::vec3f& throughputMultiplier, const float& bsdfPdf, const float& selectionProb)
		{
			assertBounceValidity(originPixel, depth);
			assert(!utl::isNan(direction));
			assert(!isnan(bsdfPdf));
			assert(!utl::isNan(throughputMultiplier));

			Bounce& bounce = paths[originPixel][depth];
			bounce.bsdfSample.isValid = true;
			bounce.bsdfSample.direction = direction;
			bounce.bsdfSample.sampleProb = selectionProb;
			bounce.bsdfSample.bsdfPdf = bsdfPdf;
			bounce.bsdfSample.throughputMultiplier = throughputMultiplier;
		}

		__forceinline__ __device__ void recordLightRayExtension(
			const int& originPixel, const int& depth,
			const math::vec3f& position, const math::vec3f& direction, const math::vec3f& normal,
			const math::vec3f& throughputMultiplier, const math::vec3f& outgoingRadiance, const float& bsdfPdf, const float& sampleProb, const float& misWeight)
		{
			assertBounceValidity(originPixel, depth);
			assert(!utl::isNan(direction));
			assert(!utl::isNan(position));
			assert(!utl::isNan(normal));
			assert(!utl::isNan(throughputMultiplier));
			assert(!utl::isNan(outgoingRadiance));
			assert(!isnan(bsdfPdf));

			Bounce& bounce = paths[originPixel][depth];
			bounce.lightHit.position = position;
			bounce.lightHit.normal = normal;
			bounce.lightHit.outgoingRadiance = outgoingRadiance;
			bounce.lightHit.misWeight = misWeight;
			bounce.lightSample.direction = direction;
			bounce.lightSample.throughputMultiplier = throughputMultiplier;
			bounce.lightSample.bsdfPdf = bsdfPdf;
			bounce.lightSample.sampleProb = sampleProb;
			bounce.lightSample.isValid = false; // To be validated after shadow tracing
		}

		__forceinline__ __device__ void recordBounceEmission(
			const int& originPixel, const int& depth,
			const math::vec3f& emission, const float& misWeight)
		{
			assertBounceValidity(originPixel, depth);
			assert(!utl::isNan(emission));
			assert(!isnan(misWeight));
			if (!math::isZero(emission))
			{
				Bounce& bounce = paths[originPixel][depth];
				bounce.hit.outgoingRadiance = emission;
				bounce.hit.misWeight = misWeight;
				bounce.hit.isEmissive = true;
				paths[originPixel].hasEmission = true;
				registerValidLightSample(originPixel, depth, ES_SURFACE_EMISSION); //This gets called by the miss too, but for now it's not a problem to set the value to ES_SURFACE_EMISSION
			}
		}

		__forceinline__ __device__ void validateLightSample(const int& originPixel, const int& depth)
		{
			assertBounceValidity(originPixel, depth);
			Bounce& bounce = paths[originPixel][depth];
			if(!math::isZero(bounce.lightHit.outgoingRadiance))
			{
				registerValidLightSample(originPixel, depth, ES_DIRECT_LIGHT);
				bounce.lightSample.isValid = true;
				paths[originPixel].hasEmission = true;
			}
		}

		__forceinline__ __device__ void recordMissBounce(
			const int& originPixel, const int& depth,
			const float& maxClamp, const math::vec3f& emission, const float& misWeight)
		{
			assertBounceValidity(originPixel, depth);
			assert(!utl::isNan(emission));
			assert(!isnan(misWeight));
			const Bounce& previousBounce = paths[originPixel][depth - 1];
			const math::vec3f virtualPosition = previousBounce.hit.position + previousBounce.bsdfSample.direction * maxClamp;
			const math::vec3f virtualNormal = -previousBounce.bsdfSample.direction;

			recordBounceHit(originPixel, depth, virtualPosition, virtualNormal, previousBounce.bsdfSample.direction);
			recordBounceEmission(originPixel, depth, emission, misWeight);
			Bounce& bounce = paths[originPixel][depth];
			bounce.hit.isEmissive = true;
			bounce.hit.outgoingRadiance = emission;
			bounce.hit.misWeight = misWeight;
			paths[originPixel].hasEmission = !math::isZero(emission) ? true : paths[originPixel].hasEmission;

			setBounceAsTerminal(originPixel, depth);
		}

		__forceinline__ __device__ math::vec3f outgoingDirectLight(const int& originPixel, const int& depth, const bool useMIS)
		{
			assertBounceValidity(originPixel, depth);
			const Bounce& bounce = paths[originPixel][depth];
			if (bounce.lightSample.isValid)
			{
				const float misWeight = useMIS ? bounce.lightHit.misWeight : 1.0f;
				const math::vec3f directLight = bounce.lightSample.throughputMultiplier * bounce.lightHit.outgoingRadiance * misWeight;
				return directLight;
			}
			return { 0.0f };
		}

		__forceinline__ __device__ math::vec3f outgoingSurfaceEmission(const int& originPixel, const int& depth, const bool useMIS)
		{
			assertBounceValidity(originPixel, depth);
			const Bounce& bounce = paths[originPixel][depth];
			if (bounce.hit.isEmissive)
			{
				const float misWeight = useMIS ? bounce.hit.misWeight : 1.0f;
				const math::vec3f surfaceEmission = bounce.hit.outgoingRadiance * misWeight;
				return surfaceEmission;
			}
			return { 0.0f };
		}

		__forceinline__ __device__ math::vec3f outgoingIndirectLight(const int& originPixel, const int& depth, const bool useMis)
		{
			assertBounceValidity(originPixel, depth);
			math::vec3f indirectLight(0.0f);

			const int& pathMaxDepth = paths[originPixel].maxDepth;

			const Bounce& startingBounce = paths[originPixel][depth];
			if (startingBounce.isTerminal)
			{
				return { 0.0f };
			}

			math::vec3f throughput = startingBounce.bsdfSample.throughputMultiplier;

			for (int i = depth + 1; i <= pathMaxDepth; ++i)
			{
				const Bounce& currentBounce = paths[originPixel][i];
				math::vec3f surfaceEmission = outgoingSurfaceEmission(originPixel, i, useMis);
				math::vec3f directLight = outgoingDirectLight(originPixel, i, useMis);
				math::vec3f outgoingRadianceAtBounce = surfaceEmission + directLight;
				indirectLight += throughput * outgoingRadianceAtBounce;

				assert(!utl::isNan(indirectLight));

				if (currentBounce.isTerminal)
				{
					return indirectLight;
				}
				throughput *= currentBounce.bsdfSample.throughputMultiplier;
			}

			return indirectLight;
		}

		__forceinline__ __device__ math::vec3f getTotalOutgoingRadiance(const int& originPixel, const int& depth)
		{
			assertBounceValidity(originPixel, depth);
			math::vec3f outgoingRadiance = outgoingIndirectLight(originPixel, depth,true) + outgoingSurfaceEmission(originPixel, depth,true) + outgoingDirectLight(originPixel, depth,true);
			return outgoingRadiance;
		}

		__forceinline__ __device__ math::vec3f oneBounceIndirectLight(const int& originPixel, const int& depth, const bool& useMis)
		{
			assertBounceValidity(originPixel, depth);
			math::vec3f indirectLight(0.0f);

			const int& pathMaxDepth = paths[originPixel].maxDepth;

			const Bounce& startingBounce = paths[originPixel][depth];
			if (startingBounce.isTerminal || depth == pathMaxDepth)
			{
				return { 0.0f };
			}

			const math::vec3f throughput = startingBounce.bsdfSample.throughputMultiplier;
			const math::vec3f outgoingRadianceAtBounce = outgoingSurfaceEmission(originPixel, depth + 1, useMis) + outgoingDirectLight(originPixel, depth + 1, useMis);
			indirectLight += throughput * outgoingRadianceAtBounce;
			assert(!utl::isNan(indirectLight));

			return indirectLight;
		}

		__forceinline__ __device__ void accumulatePath(const int& originPixel)
		{
			assertBounceValidity(originPixel, 0);
			if (paths[originPixel].maxDepth == -1)
			{
				return;
			}
			pathsAccumulator[originPixel] += getTotalOutgoingRadiance(originPixel, 0);
		}

		__forceinline__ __device__ void registerValidLightSample(const int& originPixel, const int& depth, const EmissionSampleType type)
		{
			if (depth == 0)
			{
				return;
			}
			int idx = cuAtomicAdd(&validLightSampleCount, 1);
			validLightSample[idx].x = originPixel;
			validLightSample[idx].y = depth;
			validLightSample[idx].z = type;
		}
		
		__forceinline__ __device__ void selectRandomBounce(unsigned& seed, int* selectedPixel, int* selectedDepth, const network::SamplingStrategy strategy)
		{

			int* validPixelsArray;
			int* validPixelsCounter;
			if (strategy == network::SS_PATHS_WITH_CONTRIBUTION)
			{
				validPixelsArray = pixelsWithContribution;
				validPixelsCounter = &pixelsWithContributionCount;
			}
			else
			{
				validPixelsArray = validPixels;
				validPixelsCounter = &validPixelsCount;
			}

			const float rnd = rng(seed);
			const int   randomOriginPixel = validPixelsArray[(int)(rnd * (float)(*validPixelsCounter))];
			const int   maxDepth = paths[randomOriginPixel].maxDepth;
			*selectedPixel = randomOriginPixel;

			// we don't want to select the last bounce which is either termination or miss shader
			// but some paths might have just one bounce, so we need to handle that case
			if(false){
				if (maxDepth == 0)
				{
					// This is not going to happen as it is because of how the setBounceAsTerminal is implemented
					*selectedDepth = 0;
				}
				else
				{
					const float rnd2 = rng(seed);
					*selectedDepth = (int)round(rnd2 * (float)(maxDepth - 1));
					assert(*selectedDepth < maxDepth);
					assert(*selectedDepth >= 0);
				}
			}
			else
			{
				*selectedDepth = 0;
			}


		}

		struct Hit
		{
			math::vec3f position;
			math::vec3f normal;
			math::vec3f wOutgoing;
		};

		struct LightContribution
		{
			math::vec3f outLight;
			math::vec3f wIncoming;
			float bsdfProb;
		};

		__forceinline__ __device__ void getRandomPathSample(
			unsigned&                                      seed,
			const network::TrainingBatchGenerationSettings settings,
			int&                                           sampledPixel, int&      sampledDepth,
			Hit*                                           hit, LightContribution* lightContribution, bool& isTerminal, Hit* nextHit = nullptr)
		{
			int randomOriginPixel;
			int randomDepth;
			selectRandomBounce(seed, &randomOriginPixel, &randomDepth, settings.strategy);
			const Bounce& bounce = paths[randomOriginPixel][randomDepth];

			sampledPixel = randomOriginPixel;
			sampledDepth = randomDepth;

			if (bounce.lightSample.isValid && rng(seed) <= settings.lightSamplingProb)
			{
				lightContribution->outLight = outgoingDirectLight(randomOriginPixel, randomDepth, settings.weightByMis);
				lightContribution->bsdfProb = bounce.lightSample.bsdfPdf;
				lightContribution->wIncoming = bounce.lightSample.direction;
				if (nextHit != nullptr)
				{
					nextHit->position = bounce.lightHit.position;
					nextHit->normal = bounce.lightHit.normal;
					nextHit->wOutgoing = lightContribution->wIncoming;
				}
				isTerminal = true;
			}
			else
			{
				lightContribution->outLight = outgoingIndirectLight(randomOriginPixel, randomDepth, settings.weightByMis);
				lightContribution->bsdfProb = bounce.bsdfSample.bsdfPdf;
				lightContribution->wIncoming = bounce.bsdfSample.direction;
				if (nextHit != nullptr)
				{
					if(randomDepth < paths[randomOriginPixel].maxDepth)
					{
						const Bounce& nextBounce = paths[randomOriginPixel][randomDepth + 1];
						nextHit->position = nextBounce.hit.position;
						nextHit->normal = nextBounce.hit.normal;
						nextHit->wOutgoing = nextBounce.outgoingDirection;
						if (randomDepth + 1 == paths[randomOriginPixel].maxDepth)
						{
							isTerminal = true;
						}
						else
						{
							isTerminal = false;
						}
					}
					else
					{
						isTerminal = true;
						printf("We shouldn't be here\n");
						nextHit->position = bounce.hit.position;
						nextHit->normal = bounce.hit.normal;
						nextHit->wOutgoing = bounce.outgoingDirection;
					}
				}
			}

			hit->position = bounce.hit.position;
			hit->normal = bounce.hit.normal;
			hit->wOutgoing = bounce.outgoingDirection;
			bool print = false;
			//printf("Seed: %u SampledPixel: %d, SampledDepth: %d\n", seed, sampledPixel, sampledDepth);
			if(print)
			{
				printf(
					"Seed: %u\n"
					" sampledPixel: %d, sampledDepth: %d, isTerminal: %d\n"
					"Position (%f, %f, %f)\n"
					"Normal (%f, %f, %f)\n"
					"wOutgoing (%f, %f, %f)\n"
					"Bsdf Prob %f\n"
					"Light Contribution (%f, %f, %f)\n"
					"wIncoming (%f, %f, %f)\n",
					seed,
					sampledPixel, sampledDepth, isTerminal,
					hit->position.x, hit->position.y, hit->position.z,
					hit->normal.x, hit->normal.y, hit->normal.z,
					hit->wOutgoing.x, hit->wOutgoing.y, hit->wOutgoing.z,
					lightContribution->bsdfProb,
					lightContribution->outLight.x, lightContribution->outLight.y, lightContribution->outLight.z,
					lightContribution->wIncoming.x, lightContribution->wIncoming.y, lightContribution->wIncoming.z
				);
			}
			
		}

		__forceinline__ __device__ void getRandomLightSamplePath(
			unsigned&                                      seed,
			const network::TrainingBatchGenerationSettings settings,
			int&                                           sampledPixel, int&      sampledDepth,
			Hit*                                           hit, LightContribution* lightContribution, bool& isTerminal, Hit* nextHit = nullptr)
		{
			math::vec3i& sample = validLightSample[(int)(rng(seed) * (float)validLightSampleCount)];
			const int& originPixel = sample.x;
			const int& sampleDepth = sample.y;
			const auto   type = (EmissionSampleType)sample.z;
			int startDepth = 0;
		
			switch (type)
			{
			case ES_MISS_EMISSION:
			case ES_SURFACE_EMISSION:
				startDepth = (int)(rng(seed) * (float)(sampleDepth - 1));
				lightContribution->outLight = outgoingSurfaceEmission(originPixel, startDepth, settings.weightByMis);
				break;

			case ES_DIRECT_LIGHT:
				startDepth = (int)(rng(seed) * (float)sampleDepth);
				lightContribution->outLight = outgoingDirectLight(originPixel, startDepth, settings.weightByMis);
				break;
			}

			sampledPixel = originPixel;
			sampledDepth = startDepth;
		
			if (type == ES_DIRECT_LIGHT && sampleDepth == startDepth)
			{
				const Bounce& prevBounce = paths[originPixel][startDepth];
				hit->position = prevBounce.hit.position;
				hit->normal = prevBounce.hit.normal;
				hit->wOutgoing = prevBounce.outgoingDirection;
				lightContribution->wIncoming = prevBounce.lightSample.direction;
				lightContribution->bsdfProb = prevBounce.lightSample.bsdfPdf;
				if (nextHit != nullptr)
				{
					nextHit->position = paths[originPixel][sampleDepth].lightHit.position;
					nextHit->normal = paths[originPixel][sampleDepth].lightHit.normal;
					nextHit->wOutgoing = lightContribution->wIncoming;
				}
				isTerminal = true;
			}
			else
			{
				if (startDepth >= paths[originPixel].maxDepth - 1)
				{
					isTerminal = true;
				}
				else
				{
					isTerminal = false;
				}
				for (int i = sampleDepth - 1; i >= startDepth; --i)
				{
					const Bounce& prevBounce = paths[originPixel][i];
					const math::vec3f throughput = prevBounce.bsdfSample.throughputMultiplier;
					lightContribution->outLight *= throughput;

					if(nextHit != nullptr && i == startDepth +1)
					{
						nextHit->position = prevBounce.hit.position;
						nextHit->normal = prevBounce.hit.normal;
						nextHit->wOutgoing = prevBounce.outgoingDirection;
					}
					if (i == startDepth)
					{
						hit->position = prevBounce.hit.position;
						hit->normal = prevBounce.hit.normal;
						hit->wOutgoing = prevBounce.outgoingDirection;
						lightContribution->wIncoming = prevBounce.bsdfSample.direction;
						lightContribution->bsdfProb = prevBounce.bsdfSample.bsdfPdf;
					}
				}
			}
		}

		Path* paths;

		int* validPixels;
		int validPixelsCount;

		int* pixelsWithContribution;
		int pixelsWithContributionCount;

		math::vec3i* validLightSample;
		int validLightSampleCount;

		math::vec3f* pathsAccumulator;
		int numberOfPaths;
		int maxAllocDepth = 0;
	};
}

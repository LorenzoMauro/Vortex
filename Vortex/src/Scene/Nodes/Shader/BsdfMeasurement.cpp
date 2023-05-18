#include "BsdfMeasurement.h"
#include "Scene/Traversal.h"
#include "MDL/mdlWrapper.h"
namespace vtx::graph {


	void BsdfMeasurement::init()
	{
		reflectionBsdf = mdl::fetchBsdfData(databaseName, MBSDF_DATA_REFLECTION);
		if(reflectionBsdf.srcData != nullptr)
		{
			prepareSampling(reflectionBsdf);
		}
		transmissionBsdf = mdl::fetchBsdfData(databaseName, MBSDF_DATA_TRANSMISSION);
		if (transmissionBsdf.srcData != nullptr)
		{
			prepareSampling(transmissionBsdf);
		}
	}

	void BsdfMeasurement::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if(!isInitialized)
		{
			init();
		}
		ACCEPT(BsdfMeasurement, orderedVisitors)
	}

	/*void BsdfMeasurement::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<BsdfMeasurement>());
	}*/
	void BsdfMeasurement::prepareSampling(BsdfPartData bsdfData)
	{
		// CDF of the probability to select a certain theta_out for a given theta_in.

		const math::vec2ui& res = bsdfData.angularResolution;
		const unsigned int& numChannels = bsdfData.numChannels;
		const float* srcData = bsdfData.srcData;

		const unsigned int cdfThetaSize = res.x * res.x;

		// For each of theta_in x theta_out combination, a CDF of the probabilities to select a certain theta_out is stored.
		const unsigned sampleDataSize = cdfThetaSize + cdfThetaSize * res.y;

		bsdfData.sampleData.resize(sampleDataSize);
		bsdfData.albedoData.resize(res.x);

		float* sampleDataTheta = bsdfData.sampleData.data();                  // begin of the first (theta) CDF
		float* sampleDataPhi = bsdfData.sampleData.data() + cdfThetaSize; // begin of the second (phi) CDFs

		const float sTheta = static_cast<float>((M_PI * 0.5)) / static_cast<float>(res.x); // step size
		const float sPhi = (float)(M_PI) / static_cast<float>(res.y); // step size

		float maxAlbedo = 0.0f;

		for (unsigned int tIn = 0; tIn < res.x; ++tIn)
		{
			float sumTheta = 0.0f;
			float sintheta0Sqd = 0.0f;

			for (unsigned int tOut = 0; tOut < res.x; ++tOut)
			{
				const float sintheta1 = sinf(static_cast<float>(tOut + 1) * sTheta);
				const float sintheta1_sqd = sintheta1 * sintheta1;

				// BSDFs are symmetric: f(w_in, w_out) = f(w_out, w_in)
				// Take the average of both measurements.

				// Area of two the surface elements (the ones we are averaging).
				const float mu = (sintheta1_sqd - sintheta0Sqd) * sPhi * 0.5f;

				sintheta0Sqd = sintheta1_sqd;

				// Offset for both the thetas into the measurement data (select row in the volume).
				const unsigned int offsetPhi = (tIn * res.x + tOut) * res.y;
				const unsigned int offsetPhi2 = (tOut * res.x + tIn) * res.y;

				// Build CDF for phi
				float sumPhi = 0.0f;

				for (unsigned int pOut = 0; pOut < res.y; ++pOut)
				{
					const unsigned int idx = offsetPhi + pOut;
					const unsigned int idx2 = offsetPhi2 + pOut;

					float value = 0.0f;

					if (numChannels == 3)
					{
						value = fmax(fmaxf(srcData[3 * idx + 0], srcData[3 * idx + 1]), fmaxf(srcData[3 * idx + 2], 0.0f))
							+ fmax(fmaxf(srcData[3 * idx2 + 0], srcData[3 * idx2 + 1]), fmaxf(srcData[3 * idx2 + 2], 0.0f));
					}
					else /* num_channels == 1 */
					{
						value = fmaxf(srcData[idx], 0.0f) + fmaxf(srcData[idx2], 0.0f);
					}

					sumPhi += value * mu;

					sampleDataPhi[idx] = sumPhi;
				}

				// Normalize CDF for phi.
				for (unsigned int pOut = 0; pOut < res.y; ++pOut)
				{
					const unsigned int idx = offsetPhi + pOut;

					sampleDataPhi[idx] = sampleDataPhi[idx] / sumPhi;
				}

				// Build CDF for theta.
				sumTheta += sumPhi;
				sampleDataTheta[tIn * res.x + tOut] = sumTheta;
			}

			if (sumTheta > maxAlbedo)
			{
				maxAlbedo = sumTheta;
			}

			bsdfData.albedoData[tIn] = sumTheta;

			// normalize CDF for theta
			for (unsigned int tOut = 0; tOut < res.x; ++tOut)
			{
				const unsigned int idx = tIn * res.x + tOut;

				sampleDataTheta[idx] = sampleDataTheta[idx] / sumTheta;
			}
		}

		bsdfData.maxAlbedo = maxAlbedo;
		const unsigned int lookupChannels = (numChannels == 3) ? 4 : 1;

		// Make lookup data symmetric
		bsdfData.albedoData.resize(res.y * res.x * res.x * lookupChannels);

		for (unsigned int tIn = 0; tIn < res.x; ++tIn)
		{
			for (unsigned int tOut = 0; tOut < res.x; ++tOut)
			{
				const unsigned int offsetPhi = (tIn * res.x + tOut) * res.y;
				const unsigned int offsetPhi2 = (tOut * res.x + tIn) * res.y;

				for (unsigned int pOut = 0; pOut < res.y; ++pOut)
				{
					const unsigned int idx = offsetPhi + pOut;
					const unsigned int idx2 = offsetPhi2 + pOut;

					if (numChannels == 3)
					{
						bsdfData.lookupData[4 * idx + 0] = (srcData[3 * idx + 0] + srcData[3 * idx2 + 0]) * 0.5f;
						bsdfData.lookupData[4 * idx + 1] = (srcData[3 * idx + 1] + srcData[3 * idx2 + 1]) * 0.5f;
						bsdfData.lookupData[4 * idx + 2] = (srcData[3 * idx + 2] + srcData[3 * idx2 + 2]) * 0.5f;
						bsdfData.lookupData[4 * idx + 3] = 1.0f;
					}
					else
					{
						bsdfData.lookupData[idx] = (srcData[idx] + srcData[idx2]) * 0.5f;
					}
				}
			}
		}

		bsdfData.isValid = true;
	}
}

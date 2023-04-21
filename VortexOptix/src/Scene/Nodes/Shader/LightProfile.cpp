#include "LightProfile.h"
#include "Scene/Traversal.h"
#include "MDL/MdlWrapper.h"

namespace vtx::graph {
	void LightProfile::init()
	{
		lightProfileData = mdl::fetchLightProfileData(databaseName);
		prepareSampling();
	}
	void LightProfile::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if (!isInitialized)
		{
			init();
		}
		ACCEPT(orderedVisitors);
	}

	void LightProfile::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<LightProfile>());
	}
	void LightProfile::prepareSampling()
	{
		const math::vec2ui& res		= lightProfileData.resolution;
		const math::vec2f& start	= lightProfileData.start;
		const math::vec2f& delta	= lightProfileData.delta;
		const float* data			= lightProfileData.sourceData;

		// Compute total power.
		// Compute inverse CDF data for sampling.
		// Sampling will work on cells rather than grid nodes (used for evaluation).

		// First (res.x-1) for the cdf for sampling theta.
		// Rest (rex.x-1) * (res.y-1) for the individual cdfs for sampling phi (after theta).
		lightProfileData.cdfDataSize = (res.x - 1) + (res.x - 1) * (res.y - 1);

		lightProfileData.cdfData.resize(lightProfileData.cdfDataSize);

		float debugTotalArea = 0.0f;
		float sumTheta = 0.0f;
		float totalPower = 0.0f;

		float cosTheta0 = cosf(start.x);

		for (unsigned int t = 0; t < res.x - 1; ++t)
		{
			const float cosTheta1 = cosf(start.x + static_cast<float>(t + 1) * delta.x);

			// Area of the patch (grid cell)
			// \mu = int_{theta0}^{theta1} sin{theta} \delta theta
			const float mu = cosTheta0 - cosTheta1;
			cosTheta0 = cosTheta1;

			// Build CDF for phi.
			float* cdfDataPhi = lightProfileData.cdfData.data() + (res.x - 1) + t * (res.y - 1);

			float sumPhi = 0.0f;
			for (unsigned int p = 0; p < res.y - 1; ++p)
			{
				// The probability to select a patch corresponds to the value times area.
				// The value of a cell is the average of the corners.
				// Omit the *1/4 as we normalize in the end.
				const float value = data[p * res.x + t]
									+ data[p * res.x + t + 1]
									+ data[(p + 1) * res.x + t]
									+ data[(p + 1) * res.x + t + 1];

				sumPhi += value * mu;
				cdfDataPhi[p] = sumPhi;

				debugTotalArea += mu;
			}

			// Normalize CDF for phi.
			for (unsigned int p = 0; p < res.y - 2; ++p)
			{
				cdfDataPhi[p] = (0.0f < sumPhi) ? (cdfDataPhi[p] / sumPhi) : 0.0f;
			}

			cdfDataPhi[res.y - 2] = 1.0f;

			// Build CDF for theta
			sumTheta += sumPhi;
			lightProfileData.cdfData[t] = sumTheta;
		}

		lightProfileData.totalPower = sumTheta * 0.25f * delta.y;

		// Normalize CDF for theta.
		for (unsigned int t = 0; t < res.x - 2; ++t)
		{
			lightProfileData.cdfData[t] = (0.0f < sumTheta) ? (lightProfileData.cdfData[t] / sumTheta) : lightProfileData.cdfData[t];
		}

		lightProfileData.cdfData[res.x - 2] = 1.0f;
	}
}
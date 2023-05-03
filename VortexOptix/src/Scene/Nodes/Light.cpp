#include "Light.h"
#include "Scene/Traversal.h"


namespace vtx::graph
{
	

	void Light::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		if(!attributes->isInitialized)
		{
			attributes->init();
		}
		if(attributes->lightType == L_ENV)
		{
			std::dynamic_pointer_cast<EvnLightAttributes>(attributes)->envTexture->traverse(orderedVisitors);
		}
		ACCEPT(orderedVisitors);
	}

	void Light::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Light>());
	}

	EvnLightAttributes::EvnLightAttributes()
	{
		setType(L_ENV);
		transform = ops::createNode<Transform>();
	}

	EvnLightAttributes::EvnLightAttributes(const std::string& filePath)
	{
		setType(L_ENV);
		setImage(filePath);
		transform = ops::createNode<Transform>();
	}

	void EvnLightAttributes::init()
	{
		//computeSphericalCdf();
		computeCdfAliasMaps();
		isInitialized = true;
	}

	void EvnLightAttributes::setImage(const std::string& filePath)
	{
		envTexture = mdl::createTextureFromFile(filePath);
		SIM::record(envTexture);
	}

	void EvnLightAttributes::computeSphericalCdf()
	{
		VTX_INFO("Computing Env Area Light for Texture {}", envTexture->databaseName);

		const unsigned int texWidth = envTexture->dimension[0];
		const unsigned int texHeight = envTexture->dimension[1];
		const auto* rgba = static_cast<const float*>(envTexture->imageLayersPointers[0]);

		// The original data needs to be retained. The Gaussian filter does not work in place.
		//float* funcU = new float[texWidth * texHeight];
		//float* funcV = new float[texHeight + 1];

		std::vector<float> funcU(texWidth * texHeight);
		std::vector<float> funcV(texHeight + 1);

		float sum = 0.0f;
		// First generate the function data.
		for (unsigned int y = 0; y < texHeight; ++y)
		{
			// Scale distribution by the sine to get the sampling uniform. (Avoid sampling more values near the poles.)
			// See Physically Based Rendering v2, chapter 14.6.5 on Infinite Area Lights, page 728.
			float sinTheta = float(sin(M_PI * (double(y) + 0.5) / double(texHeight))); // Make this as accurate as possible.

			for (unsigned int x = 0; x < texWidth; ++x)
			{
				// Filter to keep the piecewise linear function intact for samples with zero value next to non-zero values.
				const float value = vtx::ops::gaussianFilter(rgba, texWidth, texHeight, x, y, true);
				funcU[y * texWidth + x] = value * sinTheta;

				// Compute integral over the actual function.
				const float* p = rgba + (y * texWidth + x) * 4;
				const float intensity = (p[0] + p[1] + p[2]) / 3.0f;
				sum += intensity * sinTheta;
			}
		}

		// This integral is used inside the light sampling function (see sysData.envIntegral).
		invIntegral = 1.0f / (sum * 2.0f * M_PI * M_PI / float(texWidth * texHeight));

		// Now generate the CDF data.
		// Normalized 1D distributions in the rows of the 2D buffer, and the marginal CDF in the 1D buffer.
		// Include the starting 0.0f and the ending 1.0f to avoid special cases during the continuous sampling.

		cdfU.resize((texWidth+1) * texHeight);
		cdfV.resize(texHeight + 1);

		for (unsigned int y = 0; y < texHeight; ++y)
		{
			unsigned int row = y * (texWidth + 1); // Watch the stride!
			cdfU[row + 0] = 0.0f; // CDF starts at 0.0f.

			for (unsigned int x = 1; x <= texWidth; ++x)
			{
				unsigned int i = row + x;
				cdfU[i] = cdfU[i - 1] + funcU[y * texWidth + x - 1]; // Attention, funcU is only texWidth wide! 
			}

			const float integral = cdfU[row + texWidth]; // The integral over this row is in the last element.
			funcV[y] = integral;                        // Store this as function values of the marginal CDF.

			if (integral != 0.0f)
			{
				for (unsigned int x = 1; x <= texWidth; ++x)
				{
					cdfU[row + x] /= integral;
				}
			}
			else // All texels were black in this row. Generate an equal distribution.
			{
				for (unsigned int x = 1; x <= texWidth; ++x)
				{
					cdfU[row + x] = float(x) / float(texWidth);
				}
			}
		}

		// Now do the same thing with the marginal CDF.
		cdfV[0] = 0.0f; // CDF starts at 0.0f.
		for (unsigned int y = 1; y <= texHeight; ++y)
		{
			cdfV[y] = cdfV[y - 1] + funcV[y - 1];
		}

		const float cdfIntegral = cdfV[texHeight]; // The integral over this marginal CDF is in the last element.
		funcV[texHeight] = cdfIntegral;            // For completeness, actually unused.

		if (cdfIntegral != 0.0f)
		{
			for (unsigned int y = 1; y <= texHeight; ++y)
			{
				cdfV[y] /= cdfIntegral;
			}
		}
		else // All texels were black in the whole image. Seriously? :-) Generate an equal distribution.
		{
			for (unsigned int y = 1; y <= texHeight; ++y)
			{
				cdfV[y] = float(y) / float(texHeight);
			}
		}

		isValid = true;
		VTX_INFO("Finished Computing Env Area Light for Texture {}", envTexture->databaseName);
	}

	float buildAliasMap(
		const std::vector<float>& data,
		const unsigned int size,
		std::vector<AliasData>& aliasMap)
	{
		// create qs (normalized)
		float sum = 0.0f;
		for (unsigned int i = 0; i < size; ++i)
			sum += data[i];

		for (unsigned int i = 0; i < size; ++i)
			aliasMap[i].q = (static_cast<float>(size) * data[i] / sum);

		// create partition table
		std::vector<unsigned> partitionTable(size);
		unsigned int s = 0u, large = size;
		for (unsigned int i = 0; i < size; ++i)
			partitionTable[(aliasMap[i].q < 1.0f) ? (s++) : (--large)] = aliasMap[i].alias = i;

		// create alias map
		for (s = 0; s < large && large < size; ++s)
		{
			const unsigned int j = partitionTable[s], k = partitionTable[large];
			aliasMap[j].alias = k;
			aliasMap[k].q += aliasMap[j].q - 1.0f;
			large = (aliasMap[k].q < 1.0f) ? (large + 1u) : large;
		}
		return sum;
	}

	void EvnLightAttributes::computeCdfAliasMaps()
	{
		const unsigned int width     = envTexture->dimension[0];
		const unsigned int height     = envTexture->dimension[1];
		const math::vec3f ntscLuminance{ 0.30f, 0.59f, 0.11f };

		const auto         pixels = static_cast<const float*>(envTexture->imageLayersPointers[0]);
		importanceData.resize(width * height);
		aliasMap.resize(width * height);
		// Create importance sampling data
		float cosTheta0 = 1.0f;
		const float stepPhi = float(2.0 * M_PI) / float(width);
		const float stepTheta = float(M_PI) / float(height);
		for (unsigned int y = 0; y < height; ++y)
		{
			const float theta1 = float(y + 1) * stepTheta;
			const float cosTheta1 = std::cos(theta1);
			const float area = (cosTheta0 - cosTheta1) * stepPhi;
			cosTheta0 = cosTheta1;

			for (unsigned int x = 0; x < width; ++x) {
				const unsigned int idx = y * width + x;
				const unsigned int idx4 = idx * 4;
				//importanceData[idx] = area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
				//importanceData[idx] = area * (pixels[idx4] + pixels[idx4 + 1] + pixels[idx4 + 2]) * 0.3333333333f;
				const float       luminance =dot(math::vec3f(pixels[idx4],pixels[idx4 + 1],pixels[idx4 + 2]), ntscLuminance);
				const float value = vtx::ops::gaussianFilter(pixels, width, height, x, y, true);
				importanceData[idx] = area * (value);
			}
		}
		
		const float invEnvIntegral = 1.0f / buildAliasMap(importanceData, width * height, aliasMap);
		for (unsigned int y = 0; y < height; ++y) {
			for (unsigned int x = 0; x < width; ++x) {
				const unsigned int idx = y * width + x;
				const unsigned int idx4 = idx * 4;
				//aliasMap[i].pdf = std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2])) * invEnvIntegral;
				const float       luminance = dot(math::vec3f(pixels[idx4], pixels[idx4 + 1], pixels[idx4 + 2]), ntscLuminance);

				const float value = vtx::ops::gaussianFilter(pixels, width, height, x, y, true);

				aliasMap[idx].pdf = value * invEnvIntegral;
			}
		}
	}

	SpotLightAttributes::SpotLightAttributes()
	{
		setType(L_SPOT);
	}

	PointLightAttributes::PointLightAttributes()
	{
		setType(L_POINT);
	}

}


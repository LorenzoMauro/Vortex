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
		computeSphericalCdf();
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

	SpotLightAttributes::SpotLightAttributes()
	{
		setType(L_SPOT);
	}

	PointLightAttributes::PointLightAttributes()
	{
		setType(L_POINT);
	}

}


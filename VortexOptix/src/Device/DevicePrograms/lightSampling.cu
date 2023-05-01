#include "DataFetcher.h"
#include "mdlDeviceWrapper.h"
#include "randomNumberGenerator.h"
#include "RayData.h"
#include "Utils.h"

namespace vtx
{
    extern "C" __constant__ LaunchParams optixLaunchParams;

    extern "C" __device__ LightSample __direct_callable__meshLightSample(const LightData& light, PerRayData* prd)
	{
        MeshLightAttributesData meshLightAttributes = *reinterpret_cast<MeshLightAttributesData*>(light.attributes);
        LightSample lightSample;

        lightSample.pdf = 0.0f;

        const float* cdfArea = meshLightAttributes.cdfArea;
        const float3 sample3D = rng3(prd->seed);

        // Uniformly sample the triangles over their surface area.
        // Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
        // The cdfU is one bigger than res.y.

        unsigned int idxTriangle = utl::binarySearchCdf(cdfArea, meshLightAttributes.size, sample3D.z);
        idxTriangle = meshLightAttributes.actualTriangleIndices[idxTriangle];

        // Unit square to triangle via barycentric coordinates.
        const float sqrtSampleX = sqrtf(sample3D.x);
        // Barycentric coordinates.
        const float alpha = 1.0f - sqrtSampleX;
        const float beta = sample3D.y * sqrtSampleX;
        const float gamma = 1.0f - alpha - beta;

        HitProperties hitP;
        hitP.baricenter = math::vec3f(alpha, beta, gamma);
        utl::getInstanceAndGeometry(&hitP, meshLightAttributes.instanceId);
        utl::getVertices(&hitP, idxTriangle);
        utl::fetchTransformsFromInstance(&hitP);
        utl::computeHit(&hitP, prd->position);

        lightSample.position = hitP.position;
        lightSample.direction = -hitP.direction;
        lightSample.distance = hitP.distance;
        if (lightSample.distance < DENOMINATOR_EPSILON)
        {
            return lightSample;
        }

        utl::computeGeometricHitProperties(&hitP, true);
        utl::determineMaterialHitProperties(&hitP, idxTriangle);


        mdl::MdlData mdlData;
        mdl::InitConfig mdlConfig;
        mdlConfig.evaluateOpacity = true;
        mdlConfig.evaluateEmission = true;

        mdl::initMdl(hitP, &mdlData, mdlConfig);

        if(mdlData.opacity <= 0.0f)
        {
            return lightSample;
		}
      
        //Evauluate Sampled Point Emission
        if (mdlData.emissionFunctions.hasEmission)
        {
            mdl::EmissionEvaluateData evalData = mdl::evaluateEmission(mdlData, -lightSample.direction);
            if (evalData.isValid)
            {
                const float totArea = meshLightAttributes.totalArea;

                // Modulate the emission with the cutout opacity value to get the correct value.
                // The opacity value must not be greater than one here, which could happen for HDR textures.
                float opacity = math::min(mdlData.opacity, 1.0f);

                // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
                const float factor = (mdlData.emissionFunctions.mode == 0) ? mdlData.opacity : mdlData.opacity / totArea;

                lightSample.pdf = lightSample.distance * lightSample.distance / (totArea * evalData.data.cos); // Solid angle measure.
                lightSample.radianceOverPdf = mdlData.emissionFunctions.intensity * evalData.data.edf * (factor / lightSample.pdf);
                lightSample.isValid = true;
            }
        }

        return lightSample;
	}

    extern "C" __device__ LightSample __direct_callable__envLightSample(const LightData & light, PerRayData * prd)
    {
        EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
		auto texture = getData<TextureData>(attrib.textureId);

        unsigned width = texture->dimension.x;
        unsigned height = texture->dimension.y;

		LightSample            lightSample;

        lightSample.pdf = 0.0f;

        // Importance-sample the spherical environment light direction in object space.
        // TODO The binary searches are generating a lot of memory traffic. Replace this with an alias-map lookup.
        const float2 sample = rng2(prd->seed);

        // Note that the marginal CDF is one bigger than the texture height. As index this is the 1.0f at the end of the CDF.
        const float*       cdfV = attrib.cdfV;
        const unsigned int idxV = utl::binarySearchCdf(cdfV, height, sample.y);

        const float* cdfU = attrib.cdfU;
        cdfU += (width + 1) * idxV; // Horizontal CDF is one bigger than the texture width!
        const unsigned int idxU = utl::binarySearchCdf(cdfU, width, sample.x);

        // Continuous sampling of the CDF.
        const float cdfLowerU = cdfU[idxU];
        const float cdfUpperU = cdfU[idxU + 1];
        const float du = (sample.x - cdfLowerU) / (cdfUpperU - cdfLowerU);
        float u = (float(idxU) + du) / float(width);

        const float cdfLowerV = cdfV[idxV];
        const float cdfUpperV = cdfV[idxV + 1];
        const float dv = (sample.y - cdfLowerV) / (cdfUpperV - cdfLowerV);
        float v = (float(idxV) + dv) / float(height);

        // Light sample direction vector in object space polar coordinates.
        //float theta = 2.0f * M_PI * u - M_PI / 2.0f; // azimuth angle (theta)
        //float phi = M_PI * (1.0f - v); // inclination angle (phi)
        //
        //float x = cosf(theta) * sinf(phi);
        //float y = sinf(theta) * sinf(phi);
        //float z = cosf(phi);
        float theta = (1.0f - u) * (2.0f * M_PI) - M_PI / 2.0f;
        float phi = (1.0f - v) * M_PI;

        float x = sinf(phi) * cosf(theta);
        float y = sinf(phi) * sinf(theta);
        float z = cosf(phi);

        math::vec3f dir{ x,y,z };
        
        // Now rotate that normalized object space direction into world space. 
        lightSample.direction = math::transformNormal3F(attrib.transformation, dir);


        lightSample.distance = optixLaunchParams.settings->maxClamp; // Environment light.

        // Get the emission from the spherical environment texture.
        //float4 emissionLookUp = tex2D<float4>(texture->texObj, u, v);
        //const math::vec3f emission{ emissionLookUp.x, emissionLookUp.y, emissionLookUp.z };
        math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
        // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
        // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
        // (Note that this does not contain the light.emission which just modulates the texture.)
        lightSample.pdf = utl::intensity(emission) * attrib.invIntegral;

        if (DENOMINATOR_EPSILON < lightSample.pdf)
        {
            //lightSample.radianceOverPdf = light.emission * emission / lightSample.pdf;
            lightSample.radianceOverPdf = emission / lightSample.pdf;
        }

        lightSample.isValid = true;
        return lightSample;

    }
}

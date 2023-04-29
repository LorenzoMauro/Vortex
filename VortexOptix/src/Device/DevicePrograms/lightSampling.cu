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
        // The cdfU is one bigger than light.width.

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
}

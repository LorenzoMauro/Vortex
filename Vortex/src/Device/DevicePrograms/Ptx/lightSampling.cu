#include "../DataFetcher.h"
#include "../randomNumberGenerator.h"
#include "../RayData.h"
#include "../Utils.h"
#include "Device/DevicePrograms/Mdl/directMdlWrapper.h"
#include "Scene/Nodes/Light.h"

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


        mdl::MdlRequest request;
        request.edf = true;
        request.opacity = true;
        request.lastRayDirection = -lightSample.direction;

        mdl::MaterialEvaluation matEval;
        if (hitP.shaderConfiguration->directCallable)
        {
            const int sbtIndex = hitP.shaderConfiguration->idxCallEvaluateMaterial;
            matEval = optixDirectCall<mdl::MaterialEvaluation, mdl::MdlRequest*>(sbtIndex, &request);
        }
        else
        {
            matEval = mdl::evaluateMdlMaterial(&request);
        }

        if(matEval.opacity <= 0.0f)
        {
            return lightSample;
		}
      
        if (matEval.edf.isValid)
        {
            const float totArea = meshLightAttributes.totalArea;

            // Modulate the emission with the cutout opacity value to get the correct value.
            // The opacity value must not be greater than one here, which could happen for HDR textures.
            float opacity = math::min(matEval.opacity, 1.0f);

            // Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
            const float factor = (matEval.edf.mode == 0) ? matEval.opacity : matEval.opacity / totArea;

            lightSample.pdf = lightSample.distance * lightSample.distance / (totArea * matEval.edf.cos); // Solid angle measure.
            lightSample.radianceOverPdf = matEval.edf.intensity * matEval.edf.edf * (factor / lightSample.pdf);
            lightSample.isValid = true;
        }

        return lightSample;
	}

    extern "C" __device__ LightSample __direct_callable__envLightSample(const LightData & light, PerRayData * prd)
    {
        EnvLightAttributesData attrib = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
		auto texture = attrib.texture;

        unsigned width = texture->dimension.x;
        unsigned height = texture->dimension.y;

		LightSample            lightSample;

        // importance sample an envmap pixel using an alias map
        const float3       sample = rng3(prd->seed);
        const unsigned int size   = width * height;
        const auto         idx    = math::min<unsigned>((unsigned int)(sample.x * (float)size), size - 1);
        unsigned int       envIdx;
        float              sampleY = sample.y;
        if (sampleY < attrib.aliasMap[idx].q) {
            envIdx = idx;
            sampleY /= attrib.aliasMap[idx].q;
        }
        else {
            envIdx = attrib.aliasMap[idx].alias;
            sampleY = (sampleY - attrib.aliasMap[idx].q) / (1.0f - attrib.aliasMap[idx].q);
        }

        const unsigned int py   = envIdx / width;
        const unsigned int px   = envIdx % width;
        lightSample.pdf         = attrib.aliasMap[envIdx].pdf;

        const float u = (float)(px + sampleY) / (float)width;
        //const float phi = (M_PI_2)*(1.0f-u);

        //const float phi = (float)M_PI  -u * (float)(2.0 * M_PI);
        const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
        float sinPhi, cosPhi;
        sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sinPhi, &cosPhi);
        const float stepTheta = (float)M_PI / (float)height;
        const float theta0 = (float)(py)*stepTheta;
        const float cosTheta = cosf(theta0) * (1.0f - sample.z) + cosf(theta0 + stepTheta) * sample.z;
        const float theta = acosf(cosTheta);
        const float sinTheta = sinf(theta);
        const float v = theta * (float)(1.0 / M_PI);

        float x = cosPhi * sinTheta;
        float y = sinPhi * sinTheta;
        float z = -cosTheta;

        math::vec3f dir{ x,y,z };
        // Now rotate that normalized object space direction into world space. 
        lightSample.direction = math::transformNormal3F(attrib.transformation, dir);

        lightSample.distance = optixLaunchParams.settings->maxClamp; // Environment light.

        // Get the emission from the spherical environment texture.
        math::vec3f emission = tex2D<float4>(texture->texObj, u, v);
        // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
        // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
        // (Note that this does not contain the light.emission which just modulates the texture.)
        if (DENOMINATOR_EPSILON < lightSample.pdf)
        {
            //lightSample.radianceOverPdf = light.emission * emission / lightSample.pdf;
            lightSample.radianceOverPdf = emission / lightSample.pdf;
        }

        lightSample.isValid = true;
        return lightSample;

    }
}

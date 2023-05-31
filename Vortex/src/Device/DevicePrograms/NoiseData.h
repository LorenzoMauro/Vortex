#pragma once

#ifndef NOISEDATA_H
#define NOISEDATA_H

namespace vtx
{

	enum NoiseType
	{
		LUMINANCE,
		COLOR
	};

	enum NoiseDataType
	{
		RADIANCE,
		ALBEDO,
		NORMAL
	};

    struct NoiseData
    {
        float noise;
        float radianceNoise;
        float albedoNoise;
        float normalNoise;
        int samples;
    };
}

#endif
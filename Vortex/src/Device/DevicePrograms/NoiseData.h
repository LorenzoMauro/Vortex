#pragma once

#ifndef NOISEDATA_H
#define NOISEDATA_H

namespace vtx
{

	enum NoiseType
	{
		LUMINANCE,
		COLOR,
		HUE
	};

	enum NoiseDataType
	{
		RADIANCE,
		ALBEDO,
		NORMAL
	};

    struct NoiseData
    {
		float normalizedNoise;
		float prevNoise = 0.0f;
        float noiseAbsolute;
        int	  adaptiveSamples = 1;
    };
}

#endif
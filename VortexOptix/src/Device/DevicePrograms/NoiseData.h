#pragma once

#ifndef NOISEDATA_H
#define NOISEDATA_H

namespace vtx
{
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
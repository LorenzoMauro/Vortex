#pragma once
#include "NoiseData.h"
#include "Core/Math.h"


namespace vtx
{
	void noiseComputation(NoiseData* noiseBuffer,
						  const math::vec3f* radianceBuffer, const math::vec3f* albedoBuffer, const math::vec3f* normalBuffer,
						  math::vec2f* radianceRange, math::vec2f* albedoRange, math::vec2f* normalRange,
						  const int          width, const int                   height, const int                noiseKernelSize, const float albedoNormalInfluence);
}

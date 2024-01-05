#pragma once
#ifndef LIGHT_TYPES_H
#define LIGHT_TYPES_H

#include "Core/Math.h"

namespace vtx
{
	enum LightType
	{
		L_POINT,
		L_SPOT,
		L_MESH,
		L_ENV,

		L_NUM_LIGHT_TYPES
	};

	struct LightSample
	{
		math::vec3f position;
		math::vec3f direction;
		math::vec3f radianceOverPdf;
		math::vec3f normal;
		float       distance;
		float       pdf;
		bool        isValid = false;
		LightType   typeLight;
		math::vec3f         radiance;
	};
}

#endif



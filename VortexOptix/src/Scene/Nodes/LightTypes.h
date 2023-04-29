#pragma once
#ifndef LIGHT_TYPES_H
#define LIGHT_TYPES_H

#include "Core/math.h"

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
		float		distance;
		float		pdf;
		bool		isValid = false;
	};
}

#endif



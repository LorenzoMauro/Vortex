#pragma once
#ifndef VERTEXATTRIBUTE_H
#define VERTEXATTRIBUTE_H

#include "Core/Math.h"
namespace vtx::graph
{
	struct VertexAttributes {
		math::vec3f position;
		math::vec3f texCoord;
		math::vec3f normal;
		math::vec3f tangent;
		math::vec3f bitangent;
	};

	struct FaceAttributes
	{
		unsigned int materialSlotId = 0;
		//math::vec3f normal;
		//math::vec3f tangent;
		//math::vec3f bitangent;
	};
}
#endif
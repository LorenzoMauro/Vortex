#pragma once
#include "SceneGraph.h"


namespace vtx {
	namespace scene {

		class Camera : public Node
		{
			public:
			Camera() :
				Node(NT_CAMERA),
				right(		math::vec3f{ 1.0f, 0.0f, 0.0f }),
				up(			math::vec3f{ 0.0f, 1.0f, 0.0f }),
				direction(	math::vec3f{ 0.0f, 0.0f, -1.0f }),
				position(	math::vec3f{ 0.0f, 0.0f, 0.0f })
			{
				
			}

			void UpdateDirections() {
				right = transform->TransformVector(right);
				up = transform->TransformVector(up);
				direction = transform->TransformVector(direction);
				position = transform->TransformPoint(position);
			}

		public:
			std::shared_ptr< Transform> transform;
			math::vec3f position;
			math::vec3f direction;
			math::vec3f up;
			math::vec3f right;
		};
	}

}
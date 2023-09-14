#pragma once
#include "Scene/Node.h"
#include "Transform.h"

#include "Core/Options.h"
#include "Core/Input/Input.h"
#include "Core/Log.h"

namespace vtx::graph
{

	enum NavigationType {
		NAV_NONE,
		NAV_PAN,
		NAV_ORBIT,
		NAV_ZOOM,
		NAV_DOLLY,
		NAV_FOV
	};

	class Camera : public Node
	{
	public:
		Camera();

		void updateDirections();

		void resize(const uint32_t width, const uint32_t height);

		void onUpdate(const float ts);

		void orbitNavigation(const float ts);

		std::vector<std::shared_ptr<Node>> getChildren() const override;

	protected:
		void accept(NodeVisitor& visitor) override;
	public:
		std::shared_ptr<Transform>			transform;
		math::vec3f							position;
		math::vec3f							direction;
		math::vec3f							vertical;
		math::vec3f							horizontal;
		float								fovY;
		float								aspect;
		math::vec2i							resolution;
		math::vec2f							mousePosition;
		math::vec2f							mouseDelta;
		bool								navigationActive = false;
		NavigationType						navigationMode = NAV_ORBIT;
		float								movementSensibility;
		float								rotationSensibility;
		float								zoomSensibility;
		bool								lockCamera = false;
	};

}

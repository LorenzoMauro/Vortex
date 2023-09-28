#pragma once 
#include <imgui.h>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "Core/Math.h"
#include "Core/VortexID.h"

namespace vtx
{
	namespace graph
	{
		class Instance;
		class Camera;
	}
}

namespace vtx::gui
{
	enum TransformType
	{
		TT_TRANSLATE,
		TT_ROTATE,
		TT_SCALE,
		TT_NONE
	};

	enum TransformAxis
	{
		TA_X,
		TA_Y,
		TA_Z,
		TA_NONE
	};

	class TransformUI
	{
	public:
		void monitorTransformUI(std::shared_ptr<graph::Camera> camera);
		bool  isTransforming();

	private:
		math::vec2f toVec2f(ImVec2 vec);

		bool checkTransformTypeKey(ImGuiKey key, TransformType type);

		bool checkAxisTypeKey(ImGuiKey key, TransformAxis axis);

		void applyTransform(TransformType tt, TransformAxis ta, const math::vec2f& mouseReferencePosition, const math::vec2f& deltaPos);

		math::affine3f computeDeltaTransform(TransformType tt, TransformAxis ta,const math::vec2f& mouseReferencePosition, const math::vec2f& deltaPos);

		float getRotationDelta(const math::vec2f& pivotScreenPos, const math::vec2f& mousePreviousPosition, const math::vec2f& deltaPos) const;

		float getScaleDelta(const math::vec2f& mouseDelta);

		math::vec3f axisFromType(const TransformAxis& axis) const;
		math::vec3f getTranslationDelta(const math::vec2f& mouseDelta, const TransformAxis& axisType);

		math::vec3f gui::TransformUI::getPivotWorldPosition() const;

		math::vec2f getPivotScreenPosition() const;

		void resetTransform();

		math::vec3f gui::TransformUI::projectPixelOnCameraPlaneThroughPivot(const math::vec2f pixel);

		math::vec2f frameMousePosition();

		math::vec2f mouseStartPos;
		math::vec2f mouseDeltaPos;
		math::vec2f mousePreviousDelta;
		bool wasPrevDeltaZero = true;

		TransformType prevTransformType = TT_NONE;
		TransformType transformType = TT_NONE;
		TransformAxis prevTransformAxis = TA_NONE;
		TransformAxis transformAxis = TA_NONE;

		std::shared_ptr<graph::Camera>             camera;
		std::set<std::shared_ptr<graph::Instance>> selectedInstances;
		math::vec3f                                pivotWorld;
		math::vec2f                                pivotScreen;
		std::map<vtxID, math::affine3f>            initialTransforms;
	};
}

#include "transformInterface.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Camera.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Instance.h"

namespace vtx
{

	math::vec2f gui::TransformUI::toVec2f(ImVec2 vec)
	{
		return math::vec2f(vec.x, vec.y);
	}

	bool gui::TransformUI::checkTransformTypeKey(ImGuiKey key, TransformType type)
	{
		bool transformTypeChanged = false;
		if (ImGui::IsKeyPressed(key)) {
			if (transformType == TT_NONE) {
				mouseStartPos     = frameMousePosition();
				selectedInstances = graph::Scene::get()->getSelectedInstances();
				pivotWorld        = getPivotWorldPosition();
				pivotScreen       = camera.lock()->project(pivotWorld);
				for(const auto& instance : selectedInstances)
				{
					initialTransforms[instance->getUID()] = instance->transform->affineTransform;
				}
			}
			else if (transformType != type)
			{
				transformTypeChanged = true;
			}
			prevTransformType = transformType;
			transformType = type;
		}
		return transformTypeChanged;
	}

	bool gui::TransformUI::checkAxisTypeKey(ImGuiKey key, TransformAxis axis)
	{
		bool axisChanged = false;
		if (ImGui::IsKeyPressed(key))
		{
			if (transformAxis == axis)
			{
				axisChanged = true;
				prevTransformAxis = transformAxis;
				prevTransformType = transformType;
				transformAxis = TA_NONE;
			}
			else
			{
				axisChanged = true;
				prevTransformAxis = transformAxis;
				prevTransformType = transformType;
				transformAxis = axis;
			}
		}
		return axisChanged;
	}

	void gui::TransformUI::resetTransform()
	{
		for (const auto& instance : selectedInstances)
		{
			instance->transform->setAffine(initialTransforms[instance->getUID()]);
		}
		wasPrevDeltaZero = true;
		mousePreviousDelta = { 0.0f, 0.0f };
	}

	math::vec2f gui::TransformUI::frameMousePosition()
	{
		const ImVec2 mousePos = ImGui::GetMousePos();
		const ImVec2 windowPos = ImGui::GetWindowPos();

		const float mouseXRelativeToWindow =(mousePos.x - windowPos.x);
		const float mouseYRelativeToWindow = (float)(camera.lock()->resolution.y) - (mousePos.y - windowPos.y);

		return { mouseXRelativeToWindow , mouseYRelativeToWindow };
	}

	void gui::TransformUI::monitorTransformUI(std::shared_ptr<graph::Camera> _camera)
	{
		camera = _camera;
		bool transformTypeChanged = false;

		transformTypeChanged |= checkTransformTypeKey(ImGuiKey_R, TT_ROTATE);
		transformTypeChanged |= checkTransformTypeKey(ImGuiKey_G, TT_TRANSLATE);
		transformTypeChanged |= checkTransformTypeKey(ImGuiKey_S, TT_SCALE);

		if (transformType != TT_NONE) {
			bool axisChanged = false;
			axisChanged |= checkAxisTypeKey(ImGuiKey_X, TA_X);
			axisChanged |= checkAxisTypeKey(ImGuiKey_Y, TA_Y);
			axisChanged |= checkAxisTypeKey(ImGuiKey_Z, TA_Z);
			if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				transformType = TT_NONE;
				transformAxis = TA_NONE;
				selectedInstances.clear();
				wasPrevDeltaZero = true;
				mousePreviousDelta = { 0.0f, 0.0f };
				return;
			}
			if (ImGui::IsKeyPressed(ImGuiKey_Escape))
			{
				resetTransform();
				transformType = TT_NONE;
				transformAxis = TA_NONE;
				selectedInstances.clear();
				return;
			}

			if (transformTypeChanged || axisChanged)
			{
				resetTransform();
			}
			else
			{
				mousePreviousDelta = mouseDeltaPos;
			}

			mouseDeltaPos = frameMousePosition() - mouseStartPos;
			if(mouseDeltaPos.x != 0.0f && mouseDeltaPos.y != 0.0f)
			{
				if(mouseDeltaPos != mousePreviousDelta)
				{
					applyTransform(transformType, transformAxis, mouseStartPos, mouseDeltaPos);
				}
			}
			else if(!wasPrevDeltaZero)
			{
				resetTransform();
			}
		}
	}

	bool gui::TransformUI::isTransforming()
	{
		return transformType != TT_NONE;
	}

	void gui::TransformUI::applyTransform(TransformType tt, TransformAxis ta, const math::vec2f& mouseReferencePosition, const math::vec2f& deltaPos)
	{
		const math::affine3f deltaTransform = computeDeltaTransform(tt, ta, mouseReferencePosition, deltaPos);
		if(deltaTransform == math::affine3f(math::Identity))
		{
			if(!wasPrevDeltaZero)
			{
				resetTransform();
			}
			return;
		}
		for (const std::shared_ptr<graph::Instance>& instance : selectedInstances)
		{
			// rcp for reciprocal, GT for global transform
			const math::affine3f& rcpParentGT = instance->transform->reciprocalParentGlobalTransform;
			const math::affine3f& parentGT = instance->transform->parentGlobalTransform;
			const math::affine3f& localTransform = initialTransforms[instance->getUID()];
			const math::affine3f newTransform = rcpParentGT * deltaTransform * parentGT * localTransform;
			instance->transform->setAffine(newTransform);
			wasPrevDeltaZero = false;
		}
	}

	math::affine3f gui::TransformUI::computeDeltaTransform(TransformType tt, TransformAxis ta, const math::vec2f& mouseReferencePosition, const math::vec2f& deltaPos)
	{
		switch (tt)
		{
		case TT_TRANSLATE:
		{
			const math::vec3f translationDelta = getTranslationDelta(deltaPos, ta);
			if (translationDelta == math::vec3f(0.0f, 0.0f, 0.0f))
			{
				return math::Identity;
			}
			math::affine3f translationMatrix = math::affine3f::translate(translationDelta);
			return translationMatrix;
		}
		case TT_ROTATE:
		{
			const float rotationDelta = getRotationDelta(pivotScreen, mouseReferencePosition, deltaPos);
			if (gdt::isEqual(rotationDelta, 0.0f))
			{
				return math::Identity;
			}
			math::vec3f rotationAxis = axisFromType(ta);
			if(dot(rotationAxis, -camera.lock()->direction) < 0.0f)
			{
				rotationAxis = -rotationAxis;
			}
			math::affine3f rotationMatrix = math::affine3f::rotate(pivotWorld, rotationAxis, rotationDelta);
			return rotationMatrix;
		}
		case TT_SCALE:
		{
			const float scaleDelta = getScaleDelta(deltaPos);
			if (gdt::isEqual(scaleDelta, 0.0f))
			{
				return math::Identity;
			}
			math::vec3f axis;
			if (ta== TA_NONE)
			{
				axis = { 1.0f, 1.0f, 1.0f };
			}
			else
			{
				axis = axisFromType(ta);
			}
			math::affine3f scaleMatrix = math::affine3f::scale(pivotWorld, axis*scaleDelta);
			return scaleMatrix;
		}
		case TT_NONE:
			return math::Identity;
		}
		return math::Identity;
	}

	float gui::TransformUI::getRotationDelta(const math::vec2f& pivotScreenPos, const math::vec2f& mousePreviousPosition, const math::vec2f& deltaPos) const
	{
		if(deltaPos.x == 0.0f && deltaPos.y == 0.0f)
		{
			return 0.0f;
		}
		const math::vec2f radius = mousePreviousPosition - pivotScreenPos;
		const math::vec2f newRadius = radius + deltaPos;

		// Normalize vectors to avoid domain issues in acos
		const math::vec2f normalizedRadius = normalize(radius);
		const math::vec2f normalizedNewRadius = normalize(newRadius);

		// Calculate the angle
		float dotProduct = dot(normalizedRadius, normalizedNewRadius);
		dotProduct = std::clamp(dotProduct, -1.0f, 1.0f); // To avoid domain errors

		float deltaAngle = acosf(dotProduct);

		// Determine the orientation of the rotation
		if (cross(normalizedRadius, normalizedNewRadius) < 0)
		{
			deltaAngle = -deltaAngle;
		}

		return deltaAngle;
	}

	math::vec3f gui::TransformUI::projectPixelOnCameraPlaneThroughPivot(const math::vec2f pixel)
	{
		const math::vec2f screen {(float)camera.lock()->resolution.x, (float)camera.lock()->resolution.y};
		const math::vec2f fragment = pixel;                    // Jitter the sub-pixel location
		const math::vec2f ndc = (fragment / screen) * 2.0f - 1.0f;      // Normalized device coordinates in range [-1, 1].
		const math::vec3f rayOrigin = camera.lock()->position;
		const math::vec3f rayDirection = math::normalize(camera.lock()->horizontal * ndc.x + camera.lock()->vertical * ndc.y + camera.lock()->direction);
		const math::vec3f& planeNormal = -camera.lock()->direction;

		const math::vec3f pixelWorld = rayOrigin + dot(planeNormal, (pivotWorld-rayOrigin))/ dot(planeNormal, rayDirection) * rayDirection;


		math::vec2f onScreenPivot = camera.lock()->project(pixelWorld, true);
		// add circle
		vtxImGui::drawOrigin(onScreenPivot);

		return pixelWorld;
	}
	float gui::TransformUI::getScaleDelta(const math::vec2f& mouseDelta)
	{
		const math::vec2f startPixel = mouseStartPos;
		const math::vec2f endPixel = mouseStartPos + mouseDelta;

		const math::vec3f startWorldProjection = camera.lock()->projectPixelAtPointDepth(startPixel, pivotWorld);
		const math::vec3f endWorldProjection = camera.lock()->projectPixelAtPointDepth(endPixel, pivotWorld);
		const math::vec3f pivotToStart = startWorldProjection - pivotWorld;
		const math::vec3f pivotToEnd = endWorldProjection - pivotWorld;

		const float lengthPivotToStart = length(pivotToStart);
		const float lengthPivotToEnd = length(pivotToEnd);

		//const math::vec2f pivotToStart = startPixel - pivotScreen;
		//const math::vec2f pivotToEnd = endPixel - pivotScreen;
		//const float lengthPivotToStart = length(pivotToStart);
		//const float lengthPivotToEnd = length(pivotToEnd);
		if (gdt::isEqual(lengthPivotToStart, 0.0f))
		{
			return lengthPivotToEnd;
		}
		const float scaleFactor = lengthPivotToEnd / lengthPivotToStart;
		
		return scaleFactor;

	}

	math::vec3f gui::TransformUI::axisFromType(const TransformAxis& axis) const
	{
		switch (axis)
		{
		case TA_X:
			return math::xAxis;
		case TA_Y:
			return math::yAxis;
		case TA_Z:
			return math::zAxis;
		case TA_NONE:
			return camera.lock()->direction;
		}
		return 0.0f;
	}

	math::vec3f gui::TransformUI::getTranslationDelta(const math::vec2f& mouseDelta, const TransformAxis& axisType)
	{
		const math::vec2f endPosition = pivotScreen + mouseDelta;

		const math::vec3f projectedEndPosition = camera.lock()->projectPixelAtPointDepth(endPosition, pivotWorld);

		math::vec3f deltaTranslation = projectedEndPosition - pivotWorld;

		if (axisType != TA_NONE)
		{
			const math::vec3f axis = axisFromType(axisType);
			deltaTranslation = axis * deltaTranslation;
		}

		return deltaTranslation;


	}

	math::vec3f gui::TransformUI::getPivotWorldPosition() const
	{
		math::vec3f selectedInstancesCenter = math::vec3f(0.0f);
		for (const std::shared_ptr<graph::Instance>& instance : selectedInstances)
		{
			const math::affine3f& transform = instance->transform->globalTransform;
			selectedInstancesCenter += transform.p;
		}
		selectedInstancesCenter /= selectedInstances.size();
		return selectedInstancesCenter;
	}

	math::vec2f gui::TransformUI::getPivotScreenPosition() const
	{
		const math::vec3f centerToCamera = camera.lock()->position - pivotWorld;
		return  math::vec2f(dot(camera.lock()->horizontal, centerToCamera), dot(camera.lock()->vertical, centerToCamera));
	}

	

}
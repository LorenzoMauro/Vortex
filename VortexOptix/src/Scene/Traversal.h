#pragma once
#include <memory>
#include <stack>

#include "SIM.h"
#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Nodes/Transform.h"

namespace vtx {

	namespace graph
	{
		class Light;
		class Node;
		class Transform;
		class Instance;
		class Group;
		class Mesh;
		class Material;
		class Camera;
		class Renderer;
		class Shader;
		class Texture;
		class BsdfMeasurement;
		class LightProfile;
	}

	class NodeVisitor {
	public:
		virtual ~NodeVisitor() = default;
		virtual void visit(std::shared_ptr<graph::Transform> transform)= 0;
		virtual void visit(std::shared_ptr<graph::Instance> instance)= 0;
		virtual void visit(std::shared_ptr<graph::Group> group)= 0;
		virtual void visit(std::shared_ptr<graph::Mesh> mesh)= 0;
		virtual void visit(std::shared_ptr<graph::Material> material)= 0;
		virtual void visit(std::shared_ptr<graph::Camera> camera)= 0;
		virtual void visit(std::shared_ptr<graph::Renderer> renderer) = 0;
		virtual void visit(std::shared_ptr<graph::Shader> shader) = 0;
		virtual void visit(std::shared_ptr<graph::Texture> texture)= 0;
		virtual void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement) = 0;
		virtual void visit(std::shared_ptr<graph::LightProfile> lightProfile) = 0;
		virtual void visit(std::shared_ptr<graph::Light> lightProfile) = 0;

		virtual void popTransform() {
			transformIndexStack.pop_back();
			transformUpdateStack.pop_back();
		}

		virtual void pushTransform(const vtxID transformId, const bool transformUpdated) {
			transformIndexStack.push_back(transformId);
			transformUpdateStack.push_back(transformUpdated || isTransformStackUpdated());
		}

		virtual bool isTransformStackUpdated()
		{
			if(!transformUpdateStack.empty())
			{
				return transformUpdateStack.back();
			}
			return false;
		}

		math::affine3f getFinalTransform() const
		{
			math::affine3f finalTransform{ math::Identity };
			for (const auto& index : transformIndexStack) {
				const auto transformNode = graph::SIM::getNode<graph::Transform>(index);
				finalTransform     = finalTransform * transformNode->transformationAttribute.affineTransform;
			}
			return finalTransform;
		}


		std::vector<vtxID> transformIndexStack;
		std::vector<bool>  transformUpdateStack;
	};

}

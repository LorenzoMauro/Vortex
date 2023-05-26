#pragma once
#include <memory>
#include <stack>

#include "SIM.h"
#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Nodes/Transform.h"

namespace vtx {

	class NodeVisitor {
	public:
		virtual ~NodeVisitor() = default;
		virtual void visit(std::shared_ptr<graph::Transform> transform){}
		virtual void visit(std::shared_ptr<graph::Instance> instance){}
		virtual void visit(std::shared_ptr<graph::Group> group){}
		virtual void visit(std::shared_ptr<graph::Mesh> mesh){}
		virtual void visit(std::shared_ptr<graph::Material> material){}
		virtual void visit(std::shared_ptr<graph::Camera> camera){}
		virtual void visit(std::shared_ptr<graph::Renderer> renderer) {}
		virtual void visit(std::shared_ptr<graph::Texture> texture){}
		virtual void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement) {}
		virtual void visit(std::shared_ptr<graph::LightProfile> lightProfile) {}
		virtual void visit(std::shared_ptr<graph::Light> lightProfile) {}

		//////////////////////////////////////////////////////////////////////////////////
		//////////////////////// Shaders Nodes ///////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////

		//virtual void visit(std::shared_ptr<graph::shader::TextureFile> textureFileNode) {};
		//virtual void visit(std::shared_ptr<graph::shader::TextureReturn> textureReturnNode) {};
		virtual void visit(std::shared_ptr<graph::shader::ShaderNode> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::DiffuseReflection> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::MaterialSurface> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::Material> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::ImportedNode> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::PrincipledMaterial> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::ColorTexture> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::MonoTexture> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::NormalTexture> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::BumpTexture> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::TextureTransform> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::NormalMix> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::GetChannel> shaderNode) {}

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

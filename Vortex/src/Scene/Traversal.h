#pragma once
#include <memory>
#include <stack>

#include "SIM.h"
#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Nodes/Transform.h"

namespace vtx {

	/*namespace graph
	{
		namespace shader
		{
			class TextureFile;
			class TextureReturn;
			class ShaderNode;
			class Material;
			class DiffuseReflection;
			class MaterialSurface;
		}

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
	}*/

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
		virtual void visit(std::shared_ptr<graph::Shader> shader) {}
		virtual void visit(std::shared_ptr<graph::Texture> texture){}
		virtual void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement) {}
		virtual void visit(std::shared_ptr<graph::LightProfile> lightProfile) {}
		virtual void visit(std::shared_ptr<graph::Light> lightProfile) {}

		//////////////////////////////////////////////////////////////////////////////////
		//////////////////////// Shaders Nodes ///////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////

		virtual void visit(std::shared_ptr<graph::shader::TextureFile> textureFileNode) {};
		virtual void visit(std::shared_ptr<graph::shader::TextureReturn> textureReturnNode) {};
		virtual void visit(std::shared_ptr<graph::shader::ShaderNode> shaderNode) {}
		virtual void visit(std::shared_ptr<graph::shader::DiffuseReflection> shaderDiffuseReflection) {}
		virtual void visit(std::shared_ptr<graph::shader::MaterialSurface> materialSurfaceNode) {}
		virtual void visit(std::shared_ptr<graph::shader::Material> materialNode) {}
		virtual void visit(std::shared_ptr<graph::shader::ImportedNode> importedNode) {};


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

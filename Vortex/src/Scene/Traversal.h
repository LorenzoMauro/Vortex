#pragma once
#include <memory>
#include <stack>

#include "Core/Math.h"
#include "Core/VortexID.h"
#include "Scene/Node.h"

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
		virtual void visit(std::shared_ptr<graph::EnvironmentLight> node) {}
		virtual void visit(std::shared_ptr<graph::MeshLight> node) {}

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

		void visitBegin(const std::shared_ptr<graph::Node>& node);

		void visitEnd(std::shared_ptr<graph::Node> node);

		bool collectWidthsAndDepths = false;
		std::map<vtxID, std::tuple<int, int, int>> nodesDepthsAndWidths;
		std::map<vtxID, std::vector<vtxID>> nodesParents;
		std::stack<int> depthStack;
		std::stack<int> widthStack;
		std::vector<vtxID> parentPath;

		bool collectTransforms = false;
		std::stack<bool> resetTransforms;
		math::affine3f currentTransform = math::Identity;
		std::stack<math::affine3f> tmpTransforms;
	};

}

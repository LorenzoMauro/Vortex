#pragma once
#include "Traversal.h"

namespace vtx
{
	class HostVisitor : public NodeVisitor {
	public:
		HostVisitor(){
			collectTransforms = true;
			collectWidthsAndDepths = true;
		};
		void visit(const std::shared_ptr<graph::Instance>& instance) override;
		void visit(const std::shared_ptr<graph::Transform>& transform) override;
		void visit(const std::shared_ptr<graph::Group>& group) override;
		void visit(const std::shared_ptr<graph::Mesh>& mesh) override;
		void visit(const std::shared_ptr<graph::Material>& material) override;
		void visit(const std::shared_ptr<graph::Camera>& camera) override;
		void visit(const std::shared_ptr<graph::Renderer>& renderer) override;
		void visit(const std::shared_ptr<graph::Texture>& texture) override;
		void visit(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement) override;
		void visit(const std::shared_ptr<graph::LightProfile>& lightProfile) override;
		void visit(const std::shared_ptr<graph::EnvironmentLight>& lightNode) override;
		void visit(const std::shared_ptr<graph::MeshLight>& lightNode) override;
	};
}

#pragma once
#include "Traversal.h"

namespace vtx
{
	class HostVisitor : public NodeVisitor {
	public:
		HostVisitor() = default;
		void visit(std::shared_ptr<graph::Instance> instance) override;
		void visit(std::shared_ptr<graph::Transform> transform) override;
		void visit(std::shared_ptr<graph::Group> group) override;
		void visit(std::shared_ptr<graph::Mesh> mesh) override;
		void visit(std::shared_ptr<graph::Material> material) override;
		void visit(std::shared_ptr<graph::Camera> camera) override;
		void visit(std::shared_ptr<graph::Renderer> renderer) override;
		void visit(std::shared_ptr<graph::Shader> shader) override;
		void visit(std::shared_ptr<graph::Texture> texture) override;
		void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement) override;
		void visit(std::shared_ptr<graph::LightProfile> lightProfile) override;
	};
}

#include "HostVisitor.h"
#include "MDL//mdlWrapper.h"
namespace vtx
{
	void HostVisitor::visit(std::shared_ptr<graph::Instance> instance) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Transform> transform) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Group> group) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Mesh> mesh) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Material> material) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Camera> camera) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Renderer> renderer) {
	};

	void HostVisitor::visit(std::shared_ptr<graph::Shader> shader) {
		
	};

	void HostVisitor::visit(std::shared_ptr<graph::Texture> texture) {

	};

	void HostVisitor::visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurement) {

	};

	void HostVisitor::visit(std::shared_ptr<graph::LightProfile> lightProfile) {

	};
}

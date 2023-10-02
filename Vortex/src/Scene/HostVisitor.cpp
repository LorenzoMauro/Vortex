#include "HostVisitor.h"
#include "SceneIndexManager.h"
#include "MDL//mdlWrapper.h"
#include "Nodes/Instance.h"
#include "Nodes/Mesh.h"

namespace vtx
{
	void HostVisitor::visit(const std::shared_ptr<graph::Instance>& instance) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Transform>& transform) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Group>& group) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Mesh>& mesh) {
		if(!mesh->status.hasFaceAttributes)
		{
			ops::computeFaceAttributes(mesh);
		}
		if(!mesh->status.hasNormals)
		{
			ops::computeVertexNormals(mesh);
		}
		if(!mesh->status.hasTangents)
		{
			ops::computeVertexTangentSpace(mesh);
		}
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Material>& material) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Camera>& camera) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Renderer>& renderer) {
	};

	void HostVisitor::visit(const std::shared_ptr<graph::Texture>& texture) {

	};

	void HostVisitor::visit(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement) {

	};

	void HostVisitor::visit(const std::shared_ptr<graph::LightProfile>& lightProfile) {

	};

	void HostVisitor::visit(const std::shared_ptr<graph::EnvironmentLight>& lightNode) {

	};


	void HostVisitor::visit(const std::shared_ptr<graph::MeshLight>& lightNode) {

	};
	
}

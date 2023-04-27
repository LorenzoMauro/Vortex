#pragma once
#include "DevicePrograms/LaunchParams.h"
#include "Scene/Traversal.h"
#include <map>
#include "Scene/Nodes/Shader/BsdfMeasurement.h"


namespace vtx::device
{

	class DeviceVisitor : public NodeVisitor {
	public:
		DeviceVisitor():
			currentTransform(math::Identity),
			previousTransform(math::Identity)
		{};
		void visit(std::shared_ptr<graph::Instance> instance) override;
		void visit(std::shared_ptr<graph::Transform> transform) override;
		void visit(std::shared_ptr<graph::Group> group) override;
		void visit(std::shared_ptr<graph::Mesh> mesh) override;
		void visit(std::shared_ptr<graph::Material> material) override;
		void visit(std::shared_ptr<graph::Camera> camera) override;
		void visit(std::shared_ptr<graph::Renderer> renderer) override;
		void visit(std::shared_ptr<graph::Shader> shader) override;
		void visit(std::shared_ptr<graph::Texture> textureNode) override;
		void visit(std::shared_ptr<graph::BsdfMeasurement> bsdfMeasurementNode) override;
		void visit(std::shared_ptr<graph::LightProfile> lightProfile) override;
		void visit(std::shared_ptr<graph::Light> lightNode) override;

		//InstanceData instanceData;

		math::affine3f currentTransform;
		math::affine3f previousTransform;
	};

	void finalizeUpload();

	void incrementFrame();
}

#pragma once
#include <memory>

namespace vtx {

	namespace graph
	{
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
	};

}
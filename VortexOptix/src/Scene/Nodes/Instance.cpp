#include "Instance.h"
#include "Scene/Traversal.h"

namespace vtx::graph
{
	Instance::Instance() : Node(NT_INSTANCE) {}

	std::shared_ptr<Node> Instance::getChild() {
		return child;
	}

	void Instance::setChild(std::shared_ptr<Node> _child) {
		child = _child;
	}

	std::shared_ptr<Transform> Instance::getTransform() {
		return transform;
	}

	void Instance::setTransform(std::shared_ptr<Transform>& _transform) {
		transform = _transform;
	}

	std::vector<std::shared_ptr<Material>>& Instance::getMaterials() {
		return materials;
	}

	void Instance::addMaterial(std::shared_ptr<Material> _material) {
		materials.push_back(_material);
	}

	void Instance::removeMaterial(vtxID matID) {
		for (auto it = materials.begin(); it != materials.end(); ++it)
		{
			if ((*it)->getID() == matID)
			{
				materials.erase(it);
				break;
			}
		}
	}

	void Instance::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& material : materials) {
			material->traverse(orderedVisitors);
		}
		transform->traverse(orderedVisitors);
		child->traverse(orderedVisitors);
		ACCEPT(orderedVisitors);
	}

	void Instance::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Instance>());
	}
}


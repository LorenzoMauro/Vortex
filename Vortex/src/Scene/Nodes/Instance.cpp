#include "Instance.h"
#include "Scene/Traversal.h"
#include "Mesh.h"
#include "Scene/SIM.h"

namespace vtx::graph
{
	Instance::Instance() : Node(NT_INSTANCE)
	{
		transform = ops::createNode<Transform>();
	}

	void Instance::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{

		transform->traverse(orderedVisitors);
		child->traverse(orderedVisitors);
		for (auto& materialSlot : materialSlots) {
			materialSlot.material->traverse(orderedVisitors);

			if(!materialSlot.isMeshLightEvaluated)
			{
				materialSlot.meshLight->attributes->init();
				materialSlot.isMeshLightEvaluated = true;
			}
			if(materialSlot.meshLight->attributes->isValid )
			{
				materialSlot.meshLight->traverse(orderedVisitors);
			}
		}
		ACCEPT(Instance,orderedVisitors);
	}

	/*void Instance::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Instance>());
	}*/

	std::shared_ptr<Node> Instance::getChild() {
		return child;
	}

	void Instance::setChild(const std::shared_ptr<Node>& _child) {
		if (child && childIsMesh) {
			if (_child->getID() != child->getID()) {
				clearMeshLights();
			}
		}
		child = _child;
		if (child->getType() == NT_MESH)
		{
			childIsMesh = true;
			for (auto& [slotIndex, material, meshLight, isMeshLightEvaluated] : materialSlots)
			{
				const auto& attributes = std::dynamic_pointer_cast<MeshLightAttributes>(meshLight->attributes);
				attributes->mesh = std::dynamic_pointer_cast<Mesh>(child);
				isMeshLightEvaluated = false;
			}
		}
		else
		{
			childIsMesh = false;
		}
	}

	std::shared_ptr<Transform> Instance::getTransform() {
		return transform;
	}

	void Instance::setTransform(std::shared_ptr<Transform>& _transform) {
		transform = _transform;
	}

	void Instance::addMaterial(const std::shared_ptr<Material>& _material, const int slot) {

		MaterialSlot* matSlot;
		if(slot <= materialSlots.size()-1 && slot!= -1)
		{
			const vtxID matId = materialSlots[slot].material->getID();
			removeMaterial(matId);
			matSlot = &materialSlots[slot];
		}
		else
		{
			materialSlots.emplace_back();
			matSlot = &materialSlots.back();
		}

		matSlot->material = _material;
		matSlot->meshLight = ops::createNode<graph::Light>();
		matSlot->isMeshLightEvaluated = false;
		const auto attributes = std::make_shared<graph::MeshLightAttributes>();
		attributes->material = matSlot->material;
		attributes->materialRelativeIndex = materialSlots.size() - 1;
		attributes->parentInstanceId = getID();

		SIM::record(matSlot->meshLight);

		if (childIsMesh)
		{
			attributes->mesh = std::dynamic_pointer_cast<graph::Mesh>(getChild());
		}
		matSlot->meshLight->attributes = attributes;
	} 

	void Instance::removeMaterial(const vtxID matID) {

		//clearMeshLight(matID);

		for (auto& [slotIndex, material, meshLight, isMeshLightEvaluated] : materialSlots)
		{
			if (material->getID() == matID)
			{
				material = nullptr;
				SIM::releaseIndex(meshLight->getID());
				meshLight            = nullptr;
				isMeshLightEvaluated = true;
				break;
			}
		}
	}

	std::vector<std::shared_ptr<Material>>& Instance::getMaterials() {
		std::vector<std::shared_ptr<Material>> materials;
		for(auto it: materialSlots)
		{
			materials.push_back(it.material);
		}
		return materials;
	}

	void Instance::clearMeshLights()
	{
		for (auto it : materialSlots)
		{
			SIM::releaseIndex(it.meshLight->getID());
			it.meshLight = nullptr;
			it.isMeshLightEvaluated = true;
		}
	}

	void Instance::clearMeshLight(const vtxID matID)
	{
		for (auto it : materialSlots)
		{
			if(it.material->getID()==matID)
			{
				SIM::releaseIndex(it.meshLight->getID());
				it.meshLight = nullptr;
				it.isMeshLightEvaluated = true;
			}
		}
	}

	std::shared_ptr<graph::Light> Instance::getMeshLight(vtxID materialID)
	{
		for (auto it : materialSlots)
		{
			if (it.material->getID() == materialID)
			{
				return it.meshLight;
			}
		}
	}

	std::vector<MaterialSlot>& Instance::getMaterialSlots()
	{
		return materialSlots;
	}

}


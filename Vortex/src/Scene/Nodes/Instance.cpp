#include "Instance.h"
#include "Scene/Traversal.h"
#include "Mesh.h"
#include "MeshLight.h"
#include "Scene/SIM.h"

namespace vtx::graph
{
	Instance::Instance() : Node(NT_INSTANCE)
	{
		transform = ops::createNode<Transform>();
		child = nullptr;
		typeID = SIM::get()->getTypeId<Instance>();
	}

	Instance::~Instance()
	{
		SIM::get()->releaseTypeId<Transform>(typeID);
	}

	void Instance::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Instance>());
	}

	std::shared_ptr<Node> Instance::getChild() {
		return child;
	}

	void Instance::setChild(const std::shared_ptr<Node>& _child) {
		if (child && childIsMesh) {
			if (_child->getUID() != child->getUID()) {
				clearMeshLights();
			}
		}
		child = _child ;
		if (child->getType() == NT_MESH)
		{
			childIsMesh = true;
			for (auto& [material, slotIndex, meshLight] : materialSlots)
			{
				meshLight->mesh = child->as<Mesh>();
				meshLight->state.isInitialized = false;
			}
		}
		else
		{
			childIsMesh = false;
		}
	}

	void Instance::addMaterial(const std::shared_ptr<Material>& _material, const int slot) {

		MaterialSlot* materialSlot;
		if(slot <= materialSlots.size() - 1 && slot != -1)
		{
			const vtxID matId = materialSlots[slot].material->getUID();
			removeMaterial(matId);
			materialSlot = &materialSlots[slot];
		}
		else
		{
			materialSlots.emplace_back();
			materialSlot = &materialSlots.back();
		}
		materialSlot->material = _material;
		materialSlot->meshLight = ops::createNode<graph::MeshLight>();
		materialSlot->meshLight->material = materialSlot->material;
		materialSlot->meshLight->materialRelativeIndex = materialSlots.size() - 1;
		materialSlot->meshLight->parentInstanceId = getTypeID();

		if (childIsMesh)
		{
			materialSlot->meshLight->mesh = std::dynamic_pointer_cast<graph::Mesh>(getChild());
		}
	} 

	void Instance::removeMaterial(const vtxID matID) {

		// Remove Related Material Slot
		for(int i = 0; i < materialSlots.size(); i++)
		{
			if (materialSlots[i].material->getUID() == matID)
			{
				materialSlots[i].material = nullptr;
				SIM::get()->releaseUID(materialSlots[i].meshLight->getUID());
				SIM::get()->releaseTypeId<MeshLight>(materialSlots[i].meshLight->getTypeID());
				materialSlots[i].meshLight = nullptr;

				materialSlots.erase(materialSlots.begin() + i);
				break;
			}
		}
	}

	std::vector<std::shared_ptr<Material>> Instance::getMaterials() {
		std::vector<std::shared_ptr<Material>> materials;
		for(const auto& it: materialSlots)
		{
			materials.push_back(it.material);
		}
		return materials;
	}

	void Instance::clearMeshLights() const
	{
		for (auto it : materialSlots)
		{
			SIM::get()->releaseUID(it.meshLight->getUID());
			SIM::get()->releaseTypeId<MeshLight>(it.meshLight->getTypeID());
			it.meshLight = nullptr;
		}
	}

	void Instance::clearMeshLight(const vtxID matID) const
	{
		for (auto it : materialSlots)
		{
			if(it.material->getUID() == matID)
			{
				SIM::get()->releaseUID(it.meshLight->getUID());
				SIM::get()->releaseTypeId<MeshLight>(it.meshLight->getTypeID());
				it.meshLight = nullptr;
			}
		}
	}

	std::shared_ptr<graph::MeshLight> Instance::getMeshLight(const vtxID materialId) const
	{
		for (auto& it : materialSlots)
		{
			if (it.material->getUID() == materialId)
			{
				return it.meshLight;
			}
		}
		return nullptr;
	}

	std::vector<std::shared_ptr<Node>> Instance::getChildren() const
	{
		std::vector<std::shared_ptr<Node>> children;
		children.push_back(transform);
		children.push_back(child);
		for (auto& materialSlot : materialSlots)
		{
			children.push_back(materialSlot.material);
			children.push_back(materialSlot.meshLight);
		}
		return children;
	}

	std::vector<MaterialSlot>& Instance::getMaterialSlots()
	{
		return materialSlots;
	}

}


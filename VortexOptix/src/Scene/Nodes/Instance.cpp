#include "Instance.h"
#include "Scene/Traversal.h"
#include "Mesh.h"
#include "Scene/SIM.h"

namespace vtx::graph
{
	Instance::Instance() : Node(NT_INSTANCE) {}

	void Instance::traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors)
	{
		for (auto& material : materials) {
			material->traverse(orderedVisitors);
		}
		transform->traverse(orderedVisitors);
		child->traverse(orderedVisitors);
		for(auto& meshLight: meshLights)
		{
			meshLight->traverse(orderedVisitors);
		}

		ACCEPT(orderedVisitors);
	}

	void Instance::accept(std::shared_ptr<NodeVisitor> visitor)
	{
		visitor->visit(sharedFromBase<Instance>());
	}

	std::shared_ptr<Node> Instance::getChild() {
		return child;
	}

	void Instance::setChild(const std::shared_ptr<Node>& _child) {
		if(!child || _child->getID() != child->getID())
		{
			clearMeshLights();
			child = _child;
			if(child->getType() == NT_MESH)
			{
				childIsMesh = true;
				createMeshLight();
			}
			else
			{
				childIsMesh = false;
			}
		}

	}

	std::shared_ptr<Transform> Instance::getTransform() {
		return transform;
	}

	void Instance::setTransform(std::shared_ptr<Transform>& _transform) {
		transform = _transform;
	}

	void Instance::addMaterial(const std::shared_ptr<Material>& _material) {
		materials.push_back(_material);
		if(childIsMesh)
		{
			createMeshLight(_material, materials.size()-1);
		}
	}

	void Instance::removeMaterial(vtxID matID) {

		//TODO slot change for materials

		clearMeshLight(matID);

		for (auto it = materials.begin(); it != materials.end(); ++it)
		{
			if ((*it)->getID() == matID)
			{
				materials.erase(it);
				break;
			}
		}
	}

	std::vector<std::shared_ptr<Material>>& Instance::getMaterials() {
		return materials;
	}

	void Instance::createMeshLight()
	{
		unsigned int relativeSlot = 0;
		for(const std::shared_ptr<Material>& material : materials)
		{
			createMeshLight(material, relativeSlot);
			relativeSlot++;
		}
	}

	void Instance::createMeshLight(const std::shared_ptr<graph::Material>& materialNode, unsigned int relativeSlot)
	{
		if (const std::shared_ptr<graph::Shader>& shader = materialNode->getShader(); shader->isEmissive())
		{
			const std::shared_ptr<Mesh> meshNode = std::dynamic_pointer_cast<graph::Mesh>(getChild());
			const auto light = std::make_shared<graph::Light>();
			auto attributes = std::make_shared<graph::MeshLightAttributes>();
			light->attributes = attributes;
			attributes->material = materialNode;
			attributes->mesh = meshNode;
			attributes->materialRelativeIndex = relativeSlot;
			attributes->init();
			if (attributes->isValid)
			{
				meshLights.push_back(light);
				std::pair<vtxID, vtxID> meshMatPair = std::make_pair(meshNode->getID(), materialNode->getID());
				meshLightMap.insert({ meshMatPair, light->getID() });
				SIM::record(light);
			}
			else
			{
				SIM::releaseIndex(light->getID());
			}
		}
	}

	void Instance::clearMeshLights()
	{
		for(const std::shared_ptr<graph::Light>& light : meshLights)
		{
			SIM::releaseIndex(light->getID());
		}
		meshLights.clear();
		meshLightMap.clear();
	}

	void Instance::clearMeshLight(const vtxID matID)
	{
		for (auto& [pair, lightId] : meshLightMap)
		{
			if (auto& [meshID, materialID] = pair; materialID == matID)
			{
				SIM::releaseIndex(lightId);

				for (size_t i = 0; i < meshLights.size(); i++)
				{
					if (meshLights[i]->getID() == lightId)
					{
						meshLights.erase(meshLights.begin() + i);
						break;
					}
				}
			}
		}
	}

	std::shared_ptr<graph::Light> Instance::getMeshLight(vtxID materialID)
	{
		//We supppose that we already checked if the child is a mesh by now!
		std::pair<vtxID, vtxID> meshMatPair = std::make_pair(getChild()->getID(), materialID);

		if(meshLightMap.find(meshMatPair) != meshLightMap.end())
		{
			return SIM::getNode<Light>(meshLightMap[meshMatPair]);
		}
		return nullptr;
	}

}


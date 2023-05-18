#pragma once
#include <set>

#include "Scene/Node.h"
#include "Light.h"
#include "Material.h"
#include "Transform.h"
#include <unordered_map>

namespace vtx::graph
{
	struct MaterialSlot
	{
		int slotIndex;
		std::shared_ptr<Material> material;
		std::shared_ptr <Light> meshLight;
		bool isMeshLightEvaluated;
	};
	struct PairHash {
		template <class T1, class T2>
		std::size_t operator () (const std::pair<T1, T2>& pair) const {
			auto h1 = std::hash<T1>{}(pair.first);
			auto h2 = std::hash<T2>{}(pair.second);

			// A simple hashing technique to combine h1 and h2
			return h1 ^ h2;
		}
	};

	class Instance : public Node {
	public:

		Instance();

		std::shared_ptr<Node> getChild();

		void setChild(const std::shared_ptr<Node>& _child);

		std::shared_ptr<Transform> getTransform();

		void setTransform(std::shared_ptr<Transform>& _transform);

		std::vector<std::shared_ptr<Material>>& getMaterials();

		//TODO Slot addition and removal, currently we can add or remove materials
		//Removing Material won't delete the slot
		void Instance::addMaterial(const std::shared_ptr<Material>& _material, int slot=-1);

		void removeMaterial(vtxID matID);

		void clearMeshLights();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

		void createMeshLight();

		void createMeshLight(const std::shared_ptr<graph::Material>& materialNode, unsigned int relativeSlot);

		void clearMeshLight(vtxID matID);

		//TODO : This is a temporary solution, since we are creating a hit program per material we will have to split instances according to the used materials
		unsigned getHitSbt(const size_t slot) const;

		std::shared_ptr<graph::Light> getMeshLight(vtxID materialID);

		std::vector<MaterialSlot>&     getMaterialSlots();

	public:
		std::shared_ptr<Transform>		transform;
		std::vector<vtxID>				finalTransformStack;
		math::affine3f                  finalTransform;

	private:
		std::shared_ptr<Node>							child;
		std::vector<MaterialSlot>                       materialSlots;
		bool											childIsMesh = false;
	};

}

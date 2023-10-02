#pragma once
#include "Scene/Node.h"
#include "Material.h"
#include "Transform.h"

namespace vtx::graph
{
	struct MaterialSlot
	{
		std::shared_ptr<Material> material = nullptr;
		int                       slotIndex;
		std::shared_ptr<MeshLight>    meshLight;
	};

	struct PairHash
	{
		template <class T1, class T2>
		std::size_t operator ()(const std::pair<T1, T2>& pair) const
		{
			auto h1 = std::hash<T1>{}(pair.first);
			auto h2 = std::hash<T2>{}(pair.second);

			// A simple hashing technique to combine h1 and h2
			return h1 ^ h2;
		}
	};

	class Instance : public Node
	{
	public:
		Instance();

		std::shared_ptr<Node> getChild();

		void setChild(const std::shared_ptr<Node>& _child);

		std::vector<std::shared_ptr<Material>> getMaterials();

		//TODO Slot addition and removal, currently we can add or remove materials
		//Removing Material won't delete the slot
		void addMaterial(const std::shared_ptr<Material>& _material, int slot = -1);

		void removeMaterial(vtxID matID);

		void clearMeshLights() const;

		void clearMeshLight(vtxID matID) const;

		std::shared_ptr<graph::MeshLight> getMeshLight(vtxID materialId) const;

		std::vector<MaterialSlot>& getMaterialSlots();

		std::vector<std::shared_ptr<Node>> getChildren() const override;

	protected:
		void accept(NodeVisitor& visitor) override;
	public:
		std::shared_ptr<Transform>		transform;

	private:
		std::vector<MaterialSlot>		materialSlots;
		std::shared_ptr<Node>			child;
		bool                      childIsMesh = false;

	};
}

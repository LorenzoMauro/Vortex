#pragma once
#include <set>

#include "Node.h"

namespace vtx::graph
{
	class SIM {
	public:

		static std::shared_ptr<SIM> Get();

		static vtxID getFreeIndex();

		static void releaseIndex(vtxID id);

		template<typename T>
		static void record(std::shared_ptr<T> node) {

			const auto sim = Get();
			const NodeType& type = node->getType();
			static_assert(std::is_base_of_v<Node, T>, "Template type is not a subclass of Node!");
			if (sim->map.size() == 0) {
				sim->map.resize(NT_NUM_NODE_TYPES + 1);
			}
			sim->map[static_cast<int>(type)].insert({ node->getID(), node });
			sim->idToType.insert({ node->getID(), type });

			if (sim->vectorsOfNodes.find(type) != sim->vectorsOfNodes.end())
			{
				sim->vectorsOfNodes[type].push_back(node);
			}
			else
			{
				std::vector<std::shared_ptr<Node>> vector;
				vector.push_back(node);
				sim->vectorsOfNodes.insert({ type,  vector});
			}
		}

		// Operator[] definition
		std::shared_ptr<Node>& operator[](const vtxID id) {
			const NodeType type = idToType[id];
			return map[static_cast<int>(type)][id];
		}

		// Template function to return the statically-casted shared_ptr based on NodeType
		template<typename T>
		static std::shared_ptr<T> getNode(vtxID id) {
			const auto sim = Get();

			static_assert(std::is_base_of_v<Node, T>, "Template type is not a subclass of Node!");
			const std::shared_ptr<Node>& nodePtr = (*sim)[id];
			return std::static_pointer_cast<T>(nodePtr);
		}

		static std::vector<std::shared_ptr<Node>> getAllNodeOfType(NodeType nodeType)
		{
			const auto sim = Get();
			return sim->vectorsOfNodes[nodeType];

		}

		vtxID														nextIndex = 1; // Index Zero is reserved for Invalid Index, maybe I can use invalid unsigned
		std::set<vtxID>												freeIndices;
		std::vector<std::map<vtxID, std::shared_ptr<Node>>>			map;
		std::map<NodeType, std::vector<std::shared_ptr<Node>>>		vectorsOfNodes;
		std::map<vtxID, NodeType>									idToType;

		static std::shared_ptr<SIM>									sInstance;
	};

}

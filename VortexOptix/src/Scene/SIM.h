#pragma once
#include <set>

#include "Node.h"

namespace vtx::graph
{

	class SIM {
	public:

		static std::shared_ptr<SIM> Get();

		vtxID getFreeIndex();

		void releaseIndex(vtxID id);

		template<typename T>
		void Record(std::shared_ptr<T> node) {
			static_assert(std::is_base_of_v<Node, T>, "Template type is not a subclass of Node!");
			if (Map.size() == 0) {
				Map.resize(int(NT_NUM_NODE_TYPES + 1));
			}
			Map[(int)node->getType()].insert({node->getID(), node});
			idToType.insert({ node->getID(), node->getType() });
		}

		// Operator[] definition
		std::shared_ptr<Node>& operator[](vtxID id) {
			NodeType type = idToType[id];
			return Map[static_cast<int>(type)][id];
		}

		// Template function to return the statically-casted shared_ptr based on NodeType
		template<typename T>
		std::shared_ptr<T> getNode(vtxID id) {
			static_assert(std::is_base_of_v<Node, T>, "Template type is not a subclass of Node!");
			std::shared_ptr<Node>& nodePtr = this[id];
			return std::static_pointer_cast<T>(nodePtr);
		}

		vtxID nextIndex = 0;
		std::set<vtxID> freeIndices;
		std::vector<std::map<vtxID, std::shared_ptr<Node>>> Map;
		std::map<vtxID, NodeType> idToType;
		static std::shared_ptr<SIM> s_Instance;
	};

}

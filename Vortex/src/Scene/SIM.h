#pragma once
#include <map>
#include <set>
#include <typeindex>

#include "Node.h"
#include "Core/Log.h"
#include "Core/Math.h"

namespace vtx::graph
{
	class SIM {
	public:

		SIM();
		static std::shared_ptr<SIM> get();

		vtxID getUID();


		template<typename T>
		vtxID getTypeId()
		{
			// No free indices for this type, create a new one
			const auto typeIndex = std::type_index(typeid(T));
			if (freeTID.find(typeIndex) == freeTID.end() || freeTID[std::type_index(typeid(T))].empty())
			{
				if (nextTID.find(typeIndex) == nextTID.end())
				{
					nextTID[typeIndex] = 1;
				}
				return nextTID[typeIndex]++;
			}

			const auto it = freeTID[typeIndex].begin();
			const vtxID idx = *it;
			freeTID[typeIndex].erase(it);
			return idx;
		}

		void releaseUID(vtxID id);

		template<typename T>
		void releaseTypeId(const vtxID id)
		{

			const auto typeIndex = std::type_index(typeid(T));
			if (id + 1 == nextTID[typeIndex])
			{
				--nextTID[typeIndex];
			}
			else
			{
				freeTID[typeIndex].insert(id);
			}
		}

		void record(const std::shared_ptr<Node>& node);

		std::shared_ptr<Node> operator[](const vtxID id) {
			const auto it = nodesByUID.find(id);
			if (it == nodesByUID.end()) {
				//VTX_WARN("The requested Node Id either doesn't exist or has not been registered!");
				return nullptr;
			}

			std::shared_ptr<Node> node = it->second.lock();
			if (!node) {
				//VTX_WARN("The requested Node Id has been deleted, removing references!");
				removeNodeReference(id);
			}
			return node;
		}

		void removeNodeReference(vtxID id);

		// Template function to return the statically-casted shared_ptr based on NodeType
		template<typename T>
		std::shared_ptr<T> getNode(const vtxID id) {
			static_assert(std::is_base_of_v<Node, T>, "Template type is not a subclass of Node!");
			const std::shared_ptr<T>& nodePtr = std::dynamic_pointer_cast<T>((*this)[id]);
			if(!nodePtr)
			{
				VTX_WARN("The requested Node Id doesn't match it's type!");

			}
			return nodePtr;
		}

		template<typename T>
		bool hasNode(const vtxID id)
		{
			const bool hasNode = getNode<T>(id) != nullptr;
			return hasNode;
		}

		template<typename T>
		std::vector<std::shared_ptr<T>> getAllNodeOfType(const NodeType nodeType)
		{
			std::vector<std::shared_ptr<T>> nodes;

			const std::vector<vtxID>& ids = nodesByType[nodeType];

			for (const vtxID id : ids)
			{
				if (const std::shared_ptr<T>& nodePtr = getNode<T>(id))
				{
					nodes.push_back(nodePtr);
				}
			}
			return nodes;

		}

		template<typename T>
		std::vector<vtxID> getAllNodeIdByType(const NodeType nodeType)
		{
			return nodesByType[nodeType];
		}

		std::vector<math::vec2ui> getDeletedNodesByType(const NodeType nodeType);

		void cleanDeletedNodesByType(const NodeType nodeType);

		vtxID UIDfromTID(const NodeType nodeType, const vtxID UID);

		vtxID TIDfromUID(const vtxID typeID);

		NodeType nodeTypeFromUID(const vtxID id);

	private:
		vtxID														nextUID = 1; // Index Zero is reserved for Invalid Index, maybe I can use invalid unsigned
		std::set<vtxID>												freeUID;

		// Currently the use of weak_ptr allows for the automatic removal of nodes which are not reference by any other node
		// However we can revert to shared_ptr if we want to keep the nodes alive even if they are not referenced by any other node have them be removed manually
		std::map<vtxID, std::weak_ptr<Node>>						nodesByUID; 
		std::map<NodeType, std::vector<vtxID>>						nodesByType;
		std::map<vtxID, NodeType>									UIDtoNodeType;
		std::map<std::type_index, std::set<vtxID>>					freeTID;
		std::map<std::type_index, vtxID>							nextTID;
		std::map<NodeType, std::vector<math::vec2ui>>				deletedNodes;
		std::map<vtxID,vtxID>										UIDtoTID;
		std::map<NodeType, std::map<vtxID, vtxID>>					TIDtoUID;
	};

}

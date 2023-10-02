#include "SceneIndexManager.h"


namespace vtx::graph
{
	vtxID SceneIndexManager::getUID() {

		if (freeUID.empty()) {
			// If no free indices available, create a new one

			vtxID id = nextUID++;
			if(nodesByUID.find(id) == nodesByUID.end())
			{
				return id;
			}

			while (nodesByUID.find(id) != nodesByUID.end())
			{
				id = nextUID++;
			}
			return id;
		}
		
		// If there are free indices, use the first one and remove it from the set
		const auto it = freeUID.begin();
		const vtxID idx = *it;
		freeUID.erase(it);
		return idx;
	}


	vtxID SceneIndexManager::getTypeId(NodeType type)
	{
		if (freeTID.find(type) == freeTID.end() || freeTID[type].empty())
		{
			if (nextTID.find(type) == nextTID.end())
			{
				nextTID[type] = 1;
			}

			vtxID id = nextTID[type]++;
			if (TIDtoUID.find(type) == TIDtoUID.end())
			{
				return id;
			}

			while (TIDtoUID[type].find(id) != TIDtoUID[type].end())
			{
				id = nextTID[type]++;
			}

			return id;
		}

		const auto it = freeTID[type].begin();
		const vtxID idx = *it;
		freeTID[type].erase(it);
		return idx;
	}

	void SceneIndexManager::releaseUID(const vtxID id, bool doRemoveNodeReference) {

		// Release the index
		if (id + 1 == nextUID) {
			// If the released index is the last one, just decrement nextUID
			--nextUID;
		}
		else {
			// Otherwise, add it to the set of free indices
			freeUID.insert(id);
		}
	}

	void SceneIndexManager::releaseTypeId(const vtxID id, NodeType type)
	{
		if (id + 1 == nextTID[type])
		{
			--nextTID[type];
		}
		else
		{
			freeTID[type].insert(id);
		}
	}

	void SceneIndexManager::record(const std::shared_ptr<Node>& node) {

		const NodeType& type = node->getType();
		// check if node is already in the map
		if (nodesByUID.find(node->getUID()) != nodesByUID.end())
		{
			const std::shared_ptr<Node>& previousNode = getNode<Node>(node->getUID());
			if (previousNode && previousNode == node)
			{
				return;
			}
		}

		nodesByUID.insert({ node->getUID(), node });
		if (nodesByType.find(type) == nodesByType.end())
		{
			nodesByType[type] = {};
		}
		nodesByType[type].push_back(node->getUID());
		UIDtoNodeType.insert({ node->getUID(), type });
		UIDtoTID[node->getUID()] = node->getTypeID();
		TIDtoUID[type][node->getTypeID()] = node->getUID();
	}

	void SceneIndexManager::removeNodeReference(const vtxID UID, const vtxID TID, NodeType type, bool addToDeleted)
	{
		UIDtoNodeType.erase(UID);
		nodesByUID.erase(UID);
		deletedNodes[type].push_back({ UID, TID });
		UIDtoTID.erase(UID);
		TIDtoUID[type].erase(TID);

		releaseTypeId(TID, type);
		releaseUID(UID);

		std::vector<vtxID>& nodesOfType = nodesByType[type];
		nodesOfType.erase(std::remove(nodesOfType.begin(), nodesOfType.end(), UID), nodesOfType.end());
	}
	std::vector<math::vec2ui> SceneIndexManager::getDeletedNodesByType(const NodeType nodeType)
	{
		if (deletedNodes.find(nodeType) == deletedNodes.end())
		{
			return std::vector<math::vec2ui>();
		}
		return deletedNodes[nodeType];
	}
	void SceneIndexManager::cleanDeletedNodesByType(const NodeType nodeType)
	{
		if (deletedNodes.find(nodeType) == deletedNodes.end())
		{
			return;
		}
		deletedNodes[nodeType].clear();
	}
	vtxID SceneIndexManager::UIDfromTID(const NodeType nodeType, const vtxID UID)
	{
		if (TIDtoUID.find(nodeType) == TIDtoUID.end())
		{
			return 0; //invalid
		}
		return TIDtoUID[nodeType][UID];
	}
	vtxID SceneIndexManager::TIDfromUID(const vtxID typeID)
	{
		if (UIDtoTID.find(typeID) == UIDtoTID.end())
		{
			return 0; //invalid
		}
		return UIDtoTID[typeID];
	}
	NodeType SceneIndexManager::nodeTypeFromUID(const vtxID id)
	{
		if (UIDtoNodeType.find(id) == UIDtoNodeType.end())
		{
			return NodeType::NT_NUM_NODE_TYPES;
		}
		return UIDtoNodeType[id];
	}
	std::vector<std::shared_ptr<Node>> SceneIndexManager::getAllNodes()
	{
		std::vector<std::shared_ptr<Node>> nodes;
		for (const auto& [nodeUID, _] : nodesByUID)
		{
			if (const std::shared_ptr<Node>& node = (*this)[nodeUID])
			{
				nodes.push_back(node);
			}
		}
		return nodes;
	}
}

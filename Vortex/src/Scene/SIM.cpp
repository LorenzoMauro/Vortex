#include "SIM.h"


namespace vtx::graph
{
	std::shared_ptr<SIM> sInstance = std::make_shared<SIM>();

	SIM::SIM()
	{
		sInstance = std::shared_ptr<SIM>(this);
	}

	std::shared_ptr<SIM> SIM::get() {
		return sInstance;
	}

	vtxID SIM::getUID() {

		if (freeUID.empty()) {
			// If no free indices available, create a new one
			return nextUID++;
		}
		
		// If there are free indices, use the first one and remove it from the set
		const auto it = freeUID.begin();
		const vtxID idx = *it;
		freeUID.erase(it);
		return idx;
	}


	void SIM::releaseUID(const vtxID id) {

		// Release the index
		if (id + 1 == nextUID) {
			// If the released index is the last one, just decrement nextUID
			--nextUID;
		}
		else {
			// Otherwise, add it to the set of free indices
			freeUID.insert(id);
		}
		removeNodeReference(id);
	}


	void SIM::record(const std::shared_ptr<Node>& node) {

		const NodeType& type = node->getType();
		// check if node is already in the map
		if (nodesByUID.find(node->getUID()) != nodesByUID.end())
		{
			const std::shared_ptr<Node>& previousNode = getNode<Node>(node->getUID());
			if (previousNode && previousNode == node)
			{
				return;
			}
			return;
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

	void SIM::removeNodeReference(const vtxID id)
	{
		VTX_WARN("Removing node {} reference from SIM", id);
		const NodeType type = UIDtoNodeType[id];
		UIDtoNodeType.erase(id);
		nodesByUID.erase(id);
		deletedNodes[type].push_back({ id, UIDtoTID[id]});
		UIDtoTID.erase(id);
		TIDtoUID[type].erase(UIDtoTID[id]);
		std::vector<vtxID>& nodesOfType = nodesByType[type];
		nodesOfType.erase(std::remove(nodesOfType.begin(), nodesOfType.end(), id), nodesOfType.end());
	}
	std::vector<math::vec2ui> SIM::getDeletedNodesByType(const NodeType nodeType)
	{
		if (deletedNodes.find(nodeType) == deletedNodes.end())
		{
			return std::vector<math::vec2ui>();
		}
		return deletedNodes[nodeType];
	}
	void SIM::cleanDeletedNodesByType(const NodeType nodeType)
	{
		if (deletedNodes.find(nodeType) == deletedNodes.end())
		{
			return;
		}
		deletedNodes[nodeType].clear();
	}
	vtxID SIM::UIDfromTID(const NodeType nodeType, const vtxID UID)
	{
		if (TIDtoUID.find(nodeType) == TIDtoUID.end())
		{
			return 0; //invalid
		}
		return TIDtoUID[nodeType][UID];
	}
	vtxID SIM::TIDfromUID(const vtxID typeID)
	{
		if (UIDtoTID.find(typeID) == UIDtoTID.end())
		{
			return 0; //invalid
		}
		return UIDtoTID[typeID];
	}
	NodeType SIM::nodeTypeFromUID(const vtxID id)
	{
		if (UIDtoNodeType.find(id) == UIDtoNodeType.end())
		{
			return NodeType::NT_NUM_NODE_TYPES;
		}
		return UIDtoNodeType[id];
	}
}

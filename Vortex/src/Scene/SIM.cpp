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

	vtxID SIM::getFreeIndex() {

		if (freeIndices.empty()) {
			// If no free indices available, create a new one
			return nextIndex++;
		}
		
		// If there are free indices, use the first one and remove it from the set
		const auto it = freeIndices.begin();
		const vtxID idx = *it;
		freeIndices.erase(it);
		return idx;
	}

	void SIM::releaseIndex(const vtxID id) {

		// Release the index
		if (id + 1 == nextIndex) {
			// If the released index is the last one, just decrement nextIndex
			--nextIndex;
		}
		else {
			// Otherwise, add it to the set of free indices
			freeIndices.insert(id);
		}
		removeNodeReference(id);
	}

	void SIM::record(const std::shared_ptr<Node>& node) {

		const NodeType& type = node->getType();
		// check if node is already in the map
		if (nodesById.find(node->getID()) != nodesById.end())
		{
			const std::shared_ptr<Node>& previousNode = getNode<Node>(node->getID());
			if (previousNode && previousNode == node)
			{
				return;
			}
			return;
		}

		nodesById.insert({ node->getID(), node });
		if (nodesByType.find(type) == nodesByType.end())
		{
			nodesByType[type] = {};
		}
		nodesByType[type].push_back(node->getID());
		idToType.insert({ node->getID(), type });
	}

	void SIM::removeNodeReference(const vtxID id)
	{
		const NodeType type = idToType[id];
		idToType.erase(id);
		nodesById.erase(id);

		std::vector<vtxID>& nodesOfType = nodesByType[type];
		nodesOfType.erase(std::remove(nodesOfType.begin(), nodesOfType.end(), id), nodesOfType.end());
	}
}

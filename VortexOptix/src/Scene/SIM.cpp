#include "SIM.h"


namespace vtx::graph
{
	std::shared_ptr<SIM> SIM::sInstance = nullptr;

	std::shared_ptr<SIM> SIM::Get() {
		return sInstance;
	}

	vtxID SIM::getFreeIndex() {
		const auto sim = Get();

		if (sim->freeIndices.empty()) {
			// If no free indices available, create a new one
			return sim->nextIndex++;
		}
		
		// If there are free indices, use the first one and remove it from the set
		const auto it = sim->freeIndices.begin();
		const vtxID idx = *it;
		sim->freeIndices.erase(it);
		return idx;
	}

	void SIM::releaseIndex(vtxID id) {

		const auto sim = Get();

		const std::shared_ptr<Node> nodePtr = (*sim)[id];
		const NodeType              type    = nodePtr->getType();
		auto&                       vector  = sim->vectorsOfNodes[type];

		int vectorReleaseIndex = -1;
		for(int i = 0; i<vector.size(); i++)
		{
			if(vector[i]->getID() == id)
			{
				vectorReleaseIndex = i;
				break;
			}
		}
		vector.erase(vector.begin() + vectorReleaseIndex);

		//verify id has been used to register a node:
		// This approach is caused by the fact that just generating a node will use a new index, but the node is not registered automatically
		if (sim->idToType.find(id) != sim->idToType.end())
		{
			const NodeType type = sim->idToType[id];
			sim->map[type].erase(id);
			sim->idToType.erase(id);
		}

		if (id + 1 == sim->nextIndex) {
			// If the released index is the last one, just decrement sim->nextIndex
			--sim->nextIndex;
		}
		else {
			// Otherwise, add it to the set of free indices
			sim->freeIndices.insert(id);
		}
	}
}

#include "SIM.h"


namespace vtx::graph
{
	std::shared_ptr<SIM> SIM::s_Instance = nullptr;

	std::shared_ptr<SIM> vtx::graph::SIM::Get() {
		return SIM::s_Instance;
	}

	vtxID SIM::getFreeIndex() {
		if (freeIndices.empty()) {
			// If no free indices available, create a new one
			return nextIndex++;
		}
		else {
			// If there are free indices, use the first one and remove it from the set
			auto it = freeIndices.begin();
			vtxID idx = *it;
			freeIndices.erase(it);
			return idx;
		}
	}

	void SIM::releaseIndex(vtxID id) {
		NodeType type = idToType[id];
		Map[type].erase(id);
		idToType.erase(id);
		if (id + 1 == nextIndex) {
			// If the released index is the last one, just decrement nextIndex
			--nextIndex;
		}
		else {
			// Otherwise, add it to the set of free indices
			freeIndices.insert(id);
		}
	}
}

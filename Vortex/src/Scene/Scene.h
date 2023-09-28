#pragma once
#include "Graph.h"

namespace vtx::graph
{
	class Scene {
	public:
		static std::shared_ptr<Scene> getScene();

		std::set<vtxID> getSelectedInstancesIds() const;
		std::set<std::shared_ptr<Instance>> getSelectedInstances() const;
		void               addNodesToSelection(const std::set<vtxID>& selected);
		void               removeNodesToSelection(const std::set<vtxID>& selected);
		void               removeInstancesFromSelection();
		void               setSelected(const std::set<vtxID>& selected);
		std::set<vtxID> getSelected() const;

		std::shared_ptr<Group>						sceneRoot;
		std::shared_ptr<Renderer>					renderer;

	private:
		std::set<vtxID>							selectedIds;
	};
}


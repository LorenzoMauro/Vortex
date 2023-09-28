#include "Scene.h"

namespace vtx::graph
{
	static std::shared_ptr<Scene> gScene;

	std::shared_ptr<Scene> Scene::getScene()
	{
		if (!gScene)
		{
			gScene = std::make_shared<Scene>();
			gScene->sceneRoot = std::make_shared<Group>();
			gScene->renderer = std::make_shared<Renderer>();
			gScene->sceneRoot->addChild(gScene->renderer);
		}
		return gScene;
	}

	std::set<vtxID> fetchInstances(vtxID id)
	{
		const std::shared_ptr<SIM> sim = SIM::get();
		if (sim->nodeTypeFromUID(id) == NT_INSTANCE)
		{
			return { id };
		}
		if (sim->nodeTypeFromUID(id) == NT_GROUP)
		{
			std::set<vtxID> instances;
			const std::shared_ptr<Group> group = sim->getNode<Group>(id);
			for (const std::shared_ptr<Node>& child : group->getChildren())
			{
				if (child->getType() == NT_INSTANCE)
				{
					instances.insert(child->getUID());
				}
				else if (child->getType() == NT_GROUP)
				{
					const std::set<vtxID> childInstances = fetchInstances(child->getUID());
					instances.insert(childInstances.begin(), childInstances.end());
				}
			}
			return instances;
		}
		return {};
	}
	std::set<vtxID> Scene::getSelectedInstancesIds() const
	{
		std::set<vtxID> selectedInstances;
		for (auto& id : selectedIds)
		{
			const std::set<vtxID> instances = fetchInstances(id);
			if (!instances.empty())
			{
				selectedInstances.insert(instances.begin(), instances.end());
			}
		}
		return selectedInstances;
	}

	std::set<std::shared_ptr<Instance>> Scene::getSelectedInstances() const
	{
		const std::set<vtxID> selectedInstanceIds = getSelectedInstancesIds();
		std::set<std::shared_ptr<graph::Instance>> selectedInstances;
		for (const vtxID& instanceID : selectedInstanceIds)
		{
			const std::shared_ptr<graph::Instance>& instance = graph::SIM::get()->getNode<graph::Instance>(instanceID);
			selectedInstances.insert(instance);
		}
		return selectedInstances;
	}

	void Scene::addNodesToSelection(const std::set<vtxID>& selected)
	{
		if (selected.empty())
		{
			return;
		}
		selectedIds.insert(selected.begin(), selected.end());
	}

	void Scene::removeNodesToSelection(const std::set<vtxID>& selected)
	{
		for (const vtxID id : selected)
		{
			selectedIds.erase(id);
		}
	}

	void Scene::removeInstancesFromSelection()
	{
		std::set<vtxID> newSelection;
		for (const vtxID id: selectedIds)
		{
			if (SIM::get()->nodeTypeFromUID(id) != NT_INSTANCE)
			{
				newSelection.insert(id);
			}
		}
		setSelected(newSelection);
	}
	void Scene::setSelected(const std::set<vtxID>& selected)
	{
		selectedIds = selected;
	}
	std::set<vtxID> Scene::getSelected() const { return selectedIds; }
}

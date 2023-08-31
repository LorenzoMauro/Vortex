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
}

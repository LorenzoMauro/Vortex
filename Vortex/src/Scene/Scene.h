#pragma once
#include "Graph.h"

namespace vtx::graph
{
	class Scene {
	public:
		static std::shared_ptr<Scene> getScene();
		std::shared_ptr<Group>						sceneRoot;
		std::shared_ptr<Renderer>					renderer;
	};
}


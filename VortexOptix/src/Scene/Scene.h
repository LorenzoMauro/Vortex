#pragma once
#include "Graph.h"
#include "SIM.h"

namespace vtx::graph
{
	class Scene {
	public:

		void start();

	public:
		std::shared_ptr<SIM>						graphIndexManager;
		std::shared_ptr<Group>						sceneRoot;
		std::shared_ptr<Renderer>					renderer;
	};
}


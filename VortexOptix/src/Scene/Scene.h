#pragma once
#include "SceneGraph.h"
#include "SIM.h"
#include "Utility/Operations.h"
#include "Core/Log.h"
namespace vtx {
	namespace scene {

		class Scene {
		public:

			void Start() {
				SceneIndexManager = std::make_shared<SIM>();
				SIM::s_Instance = SceneIndexManager;
				rootNode = ops::CreateNode<Node>();
				VTX_INFO("Starting Scene");
				std::shared_ptr<Mesh> Cube = ops::createBox();
				std::shared_ptr<Transform> Transformation = ops::CreateNode<Transform>();
				Cube->setTransform(Transformation);
				rootNode->addChild(Cube);
			}


		public:
			std::shared_ptr<SIM>						SceneIndexManager;
			std::shared_ptr<Node>						rootNode;
		};
	}
}


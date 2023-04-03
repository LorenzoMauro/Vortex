#pragma once
#include "SceneGraph.h"
#include "SIM.h"
#include "Utility/Operations.h"
#include "Core/Log.h"
#include "Camera.h"

namespace vtx {
	namespace scene {

		class Scene {
		public:

			void Start() {
				SceneIndexManager = std::make_shared<SIM>();
				SIM::s_Instance = SceneIndexManager;
				rootNode = ops::CreateNode<Group>();
				VTX_INFO("Starting Scene");
				std::shared_ptr<Mesh> Cube = ops::createBox();

				std::shared_ptr<Transform> Transformation_1 = ops::CreateNode<Transform>();
				Transformation_1->translate(math::xAxis, -2.0f);

				std::shared_ptr<Instance> instance_1 = ops::CreateNode<Instance>();
				instance_1->setChild(Cube);
				instance_1->setTransform(Transformation_1);
				rootNode->addChild(instance_1);

				std::shared_ptr<Transform> Transformation_2 = ops::CreateNode<Transform>();
				Transformation_1->translate(math::xAxis, 2.0f);

				std::shared_ptr<Instance> instance_2 = ops::CreateNode<Instance>();
				instance_2->setChild(Cube);
				instance_2->setTransform(Transformation_2);
				rootNode->addChild(instance_2);

				std::shared_ptr<Camera> camera = ops::CreateNode<Camera>();;
				rootNode->addChild(instance_2);

				camera->transform = std::make_shared<Transform>();
				camera->transform->translate(math::xAxis, 5.0f);
				camera->transform->rotateAroundPointDegree(math::origin, math::zAxis, 45.0f);
				camera->UpdateDirections();
				camera->transform->rotateAroundPointDegree(math::origin, camera->right, 45.0f);

			}


		public:
			std::shared_ptr<SIM>						SceneIndexManager;
			std::shared_ptr<Group>						rootNode;
		};
	}
}


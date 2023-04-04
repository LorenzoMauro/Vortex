#pragma once
#include "SceneGraph.h"
#include "SIM.h"
#include "Utility/Operations.h"
#include "Core/Log.h"
#include "Camera.h"
#include "Renderer/Renderer.h"

namespace vtx {
	namespace scene {

		class Graph {
		public:

			void Start() {
				graphIndexManager = std::make_shared<SIM>();
				SIM::s_Instance = graphIndexManager;

				//////////////////////////////////////////////////////////////////////////
				//////////////// Scene Graph /////////////////////////////////////////////
				//////////////////////////////////////////////////////////////////////////

				sceneRoot = ops::CreateNode<Group>();
				VTX_INFO("Starting Scene");
				std::shared_ptr<Mesh> Cube = ops::createBox();

				std::shared_ptr<Transform> Transformation_1 = ops::CreateNode<Transform>();
				Transformation_1->translate(math::xAxis, -2.0f);

				std::shared_ptr<Instance> instance_1 = ops::CreateNode<Instance>();
				instance_1->setChild(Cube);
				instance_1->setTransform(Transformation_1);
				sceneRoot->addChild(instance_1);

				std::shared_ptr<Transform> Transformation_2 = ops::CreateNode<Transform>();
				Transformation_2->translate(math::xAxis, 2.0f);

				std::shared_ptr<Instance> instance_2 = ops::CreateNode<Instance>();
				instance_2->setChild(Cube);
				instance_2->setTransform(Transformation_2);
				sceneRoot->addChild(instance_2);

				//////////////////////////////////////////////////////////////////////////
				//////////////// Graph Root /////////////////////////////////////////////
				//////////////////////////////////////////////////////////////////////////

				std::shared_ptr<Camera> camera = ops::CreateNode<Camera>();;
				camera->transform = std::make_shared<Transform>();
				camera->transform->rotateDegree(math::xAxis, 90.0f);
				camera->transform->translate(math::yAxis, -5.0f);
				//camera->transform->rotateAroundPointDegree(math::origin, math::yAxis, -45.0f);
				//camera->transform->rotateAroundPointDegree(math::origin, math::zAxis, 45.0f);
				camera->UpdateDirections();

				renderer = ops::CreateNode<Renderer>();
				renderer->setCamera(camera);
				renderer->setScene(sceneRoot);
			}


		public:
			std::shared_ptr<SIM>						graphIndexManager;
			std::shared_ptr<Group>						sceneRoot;
			std::shared_ptr<Renderer>					renderer;
		};
	}
}


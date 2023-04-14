#include "Scene.h"
#include "Utility/Operations.h"

namespace vtx::graph
{
	void Scene::start() {
		graphIndexManager = std::make_shared<SIM>();
		SIM::s_Instance = graphIndexManager;

		//////////////////////////////////////////////////////////////////////////
		//////////////// Scene Graph /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		sceneRoot = ops::createNode<Group>();
		VTX_INFO("Starting Scene");
		std::shared_ptr<Mesh> Cube = ops::createBox();

		std::shared_ptr<Transform> Transformation_1 = ops::createNode<Transform>();
		Transformation_1->translate(math::xAxis, -2.0f);

		std::shared_ptr<Instance> instance_1 = ops::createNode<Instance>();
		instance_1->setChild(Cube);
		instance_1->setTransform(Transformation_1);
		sceneRoot->addChild(instance_1);

		std::shared_ptr<Transform> Transformation_2 = ops::createNode<Transform>();
		Transformation_2->translate(math::xAxis, 2.0f);

		std::shared_ptr<Instance> instance_2 = ops::createNode<Instance>();
		instance_2->setChild(Cube);
		instance_2->setTransform(Transformation_2);
		sceneRoot->addChild(instance_2);

		std::shared_ptr<Shader> shader = ops::createNode<Shader>();
		shader->name = "Stone_Mediterranean";
		shader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
		std::shared_ptr<Material> material = ops::createNode<Material>();
		material->setShader(shader);

		//////////////////////////////////////////////////////////////////////////
		//////////////// Graph Root /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		std::shared_ptr<Camera> camera = ops::createNode<Camera>();;
		camera->transform = std::make_shared<Transform>();
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->translate(math::yAxis, -5.0f);
		//camera->transform->rotateAroundPointDegree(math::origin, math::yAxis, -45.0f);
		//camera->transform->rotateAroundPointDegree(math::origin, math::zAxis, 45.0f);
		camera->updateDirections();

		renderer = ops::createNode<Renderer>();
		renderer->setCamera(camera);
		renderer->setScene(sceneRoot);
	}

}

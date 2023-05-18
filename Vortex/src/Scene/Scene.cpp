#include "Scene.h"
#include "Utility/Operations.h"

namespace vtx::graph
{
	void Scene::start() {
		graphIndexManager = std::make_shared<SIM>();
		SIM::sInstance = graphIndexManager;

		//////////////////////////////////////////////////////////////////////////
		//////////////// Scene Graph /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		//sceneRoot = ops::simpleScene01();
		sceneRoot = ops::importedScene();

		//////////////////////////////////////////////////////////////////////////
		//////////////// Graph Root /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		float cameraDistance = 2.0f;
		std::shared_ptr<Camera> camera = ops::createNode<Camera>();;
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->rotateDegree(camera->horizontal, 30.0f);
		camera->transform->translate(math::yAxis, -cameraDistance);
		camera->transform->rotateDegree(math::zAxis, 90.0f);
		camera->transform->translate(math::zAxis, cameraDistance);
		camera->updateDirections();

		renderer = ops::createNode<Renderer>();
		renderer->setCamera(camera);
		renderer->setScene(sceneRoot);
		VTX_INFO("Finishing Scene Definition!");
	}

}

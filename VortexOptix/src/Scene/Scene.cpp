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

		std::shared_ptr<Camera> camera = ops::createNode<Camera>();;
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->rotateDegree(camera->horizontal, -45.0f);
		camera->transform->translate(math::yAxis, -7.0f);
		camera->transform->rotateDegree(math::zAxis, 135.0f);
		camera->transform->translate(math::zAxis, 7.0f);
		camera->updateDirections();

		renderer = ops::createNode<Renderer>();
		renderer->setCamera(camera);
		renderer->setScene(sceneRoot);
		VTX_INFO("Finishing Scene Definition!");
	}

}

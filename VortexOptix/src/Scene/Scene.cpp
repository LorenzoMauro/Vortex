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
		Transformation_1->translate(math::xAxis, 2.0f);

		std::shared_ptr<Instance> instance_1 = ops::createNode<Instance>();
		instance_1->setChild(Cube);
		instance_1->setTransform(Transformation_1);
		sceneRoot->addChild(instance_1);

		std::shared_ptr<Transform> Transformation_2 = ops::createNode<Transform>();
		Transformation_2->translate(math::yAxis, 2.0f);

		std::shared_ptr<Instance> instance_2 = ops::createNode<Instance>();
		instance_2->setChild(Cube);
		instance_2->setTransform(Transformation_2);
		sceneRoot->addChild(instance_2);


		std::shared_ptr<Transform> Transformation_3 = ops::createNode<Transform>();
		Transformation_3->translate(math::zAxis, 2.0f);

		std::shared_ptr<Instance> instance_3 = ops::createNode<Instance>();
		instance_3->setChild(Cube);
		instance_3->setTransform(Transformation_3);
		sceneRoot->addChild(instance_3);

		if(false)
		{
			std::shared_ptr<Shader> stoneMediteranneanShader = ops::createNode<Shader>();
			stoneMediteranneanShader->name = "Stone_Mediterranean";
			stoneMediteranneanShader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
			//stoneMediteranneanShader->name = "bsdf_diffuse_reflection";
			//stoneMediteranneanShader->path = "\\bsdf_diffuse_reflection.mdl";
			std::shared_ptr<Material> stoneMediteranneanMaterial = ops::createNode<Material>();
			stoneMediteranneanMaterial->setShader(stoneMediteranneanShader);


			std::shared_ptr<Shader> carbonFibershader = ops::createNode<Shader>();
			carbonFibershader->name = "Carbon_Fiber";
			carbonFibershader->path = "\\vMaterials_2\\Composite\\Carbon_Fiber.mdl";
			//carbonFibershader->name = "bsdf_specular_reflect_transmit";
			//carbonFibershader->path = "\\bsdf_specular_reflect_transmit.mdl";
			std::shared_ptr<Material> carbonFiberMaterial = ops::createNode<Material>();
			carbonFiberMaterial->setShader(carbonFibershader);

			instance_1->addMaterial(stoneMediteranneanMaterial);
			instance_2->addMaterial(carbonFiberMaterial);
		}

		//////////////////////////////////////////////////////////////////////////
		//////////////// Graph Root /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		std::shared_ptr<Camera> camera = ops::createNode<Camera>();;
		camera->transform = std::make_shared<Transform>();
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->rotateDegree(camera->horizontal, -45.0f);
		camera->transform->translate(math::yAxis, -5.0f);
		camera->transform->rotateDegree(math::zAxis, 135.0f);
		camera->transform->translate(math::zAxis, 5.0f);
		//camera->transform->rotateAroundPointDegree(math::origin, math::zAxis, 45.0f);
		camera->updateDirections();

		renderer = ops::createNode<Renderer>();
		renderer->setCamera(camera);
		renderer->setScene(sceneRoot);
	}

}

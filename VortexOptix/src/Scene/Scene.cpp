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

		sceneRoot = ops::createNode<Group>();
		VTX_INFO("Starting Scene");
		std::shared_ptr<Mesh> cube = ops::createBox();

		std::shared_ptr<Transform> transformation1 = ops::createNode<Transform>();
		transformation1->translate(math::xAxis, 2.0f);

		std::shared_ptr<Instance> instance1 = ops::createNode<Instance>();
		instance1->setChild(cube);
		instance1->setTransform(transformation1);

		std::shared_ptr<Transform> transformation2 = ops::createNode<Transform>();
		transformation2->translate(math::yAxis, 2.0f);

		std::shared_ptr<Instance> instance2 = ops::createNode<Instance>();
		instance2->setChild(cube);
		instance2->setTransform(transformation2);


		std::shared_ptr<Transform> transformation3 = ops::createNode<Transform>();
		transformation3->rotateDegree(math::xAxis, 45.0f);
		transformation3->translate(math::zAxis, 2.0f);

		std::shared_ptr<Instance> instance3 = ops::createNode<Instance>();
		instance3->setChild(cube);
		instance3->setTransform(transformation3);


		std::shared_ptr<Mesh> plane = ops::createPlane();

		std::shared_ptr<Transform> transformation4 = ops::createNode<Transform>();
		transformation4->scale(0.5f);
		transformation4->rotateDegree(math::xAxis, 180.0f);
		transformation4->translate(math::zAxis, 7.0f);

		std::shared_ptr<Instance> instance4 = ops::createNode<Instance>();
		instance4->setChild(plane);
		instance4->setTransform(transformation4);


		std::shared_ptr<Transform> transformation5 = ops::createNode<Transform>();
		transformation5->scale(100.0f);
		transformation5->translate(math::zAxis, -1.0f);

		std::shared_ptr<Instance> instance5 = ops::createNode<Instance>();
		instance5->setChild(plane);
		instance5->setTransform(transformation5);


		sceneRoot->addChild(instance1);
		sceneRoot->addChild(instance2);
		sceneRoot->addChild(instance3);
		sceneRoot->addChild(instance4);
		sceneRoot->addChild(instance5);

		if(true)
		{
			std::shared_ptr<Shader> shader1 = ops::createNode<Shader>();
			shader1->name = "Stone_Mediterranean";
			shader1->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
			//shader1->name = "Aluminum";
			//shader1->path = "\\vMaterials_2\\Metal\\Aluminum.mdl";
			//Shader1->name = "bsdf_diffuse_reflection";
			//Shader1->path = "\\bsdf_diffuse_reflection.mdl";
			std::shared_ptr<Material> material1 = ops::createNode<Material>();
			material1->setShader(shader1);


			//std::shared_ptr<Shader> Shader2 = ops::createNode<Shader>();
			//Shader2->name = "Carbon_Fiber";
			//Shader2->path = "\\vMaterials_2\\Composite\\Carbon_Fiber.mdl";
			////Shader2->name = "bsdf_specular_reflect_transmit";
			////Shader2->path = "\\bsdf_specular_reflect_transmit.mdl";
			//std::shared_ptr<Material> Material2 = ops::createNode<Material>();
			//Material2->setShader(Shader2);

			std::shared_ptr<Shader> shaderEmissive = ops::createNode<Shader>();
			shaderEmissive->name = "naturalwhite_4000k";
			shaderEmissive->path = "\\nvidia\\vMaterials\\AEC\\Lights\\Lights_Emitter.mdl";
			std::shared_ptr<Material> materialEmissive = ops::createNode<Material>();
			materialEmissive->setShader(shaderEmissive);

			instance1->addMaterial(material1);
			instance2->addMaterial(material1);
			instance3->addMaterial(material1);
			instance5->addMaterial(material1);
			//instance4->addMaterial(material1);
			//instance1->addMaterial(materialEmissive);
			//instance2->addMaterial(materialEmissive);
			//instance3->addMaterial(materialEmissive);
			instance4->addMaterial(materialEmissive);
		}

		std::string envMapPath = "data/sunset_in_the_chalk_quarry_1k.hdr";
		//std::string envMapPath = "data/studio_small_03_1k.hdr";
		//std::string envMapPath = "data/16x16-in-1024x1024.png";
		//std::string envMapPath = "data/sunset03_EXR.exr";
		//std::string envMapPath = "data/morning07_EXR.exr";
		std::shared_ptr<Light> envLight = ops::createNode<Light>();
		auto attrib = std::make_shared<EvnLightAttributes>(envMapPath);
		//attrib->transform->rotateDegree(math::xAxis, 90.0f);
		envLight->attributes = attrib;

		sceneRoot->addChild(envLight);
		//////////////////////////////////////////////////////////////////////////
		//////////////// Graph Root /////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////

		std::shared_ptr<Camera> camera = ops::createNode<Camera>();;
		camera->transform = std::make_shared<Transform>();
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->rotateDegree(camera->horizontal, -45.0f);
		camera->transform->translate(math::yAxis, -7.0f);
		camera->transform->rotateDegree(math::zAxis, 135.0f);
		camera->transform->translate(math::zAxis, 7.0f);
		//camera->transform->rotateAroundPointDegree(math::origin, math::zAxis, 45.0f);
		camera->updateDirections();

		renderer = ops::createNode<Renderer>();
		renderer->setCamera(camera);
		renderer->setScene(sceneRoot);
		VTX_INFO("Finishing Scene Definition!");
	}

}

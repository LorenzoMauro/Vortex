#include "Serializer.h"

#include <fstream>
#include <yaml-cpp/yaml.h>

#include "ExperimentSerializer.h"
#include "Core/Log.h"
#include "NeuralNetworks/Experiment.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/EnvironmentLight.h"
#include "Scene/Nodes/Light.h"
#include "Scene/Nodes/Renderer.h"
#include "Scene/Utility/ModelLoader.h"
#include "Scene/Utility/Operations.h"

namespace vtx::serializer
{

    enum class YamlKey
    {
        MODEL_PATH,
        HDRI_PATH
    };

    static std::map<YamlKey, std::string> keymap{
        {
            YamlKey::MODEL_PATH, "Model Path"
        },
		{
			YamlKey::HDRI_PATH, "HDRI Path"
		}
    };

    static std::string previousModelPath;
    static std::string previousHDRIPath;

    bool deserialize(const std::string& filePath, const std::shared_ptr<graph::Scene>& scene)
    {
        VTX_INFO("Deserializing scene from {0}", filePath);
        std::ifstream file(filePath);

        YAML::Node data = YAML::Load(file);

        std::shared_ptr<graph::Camera> camera;

        if (data[keymap[YamlKey::MODEL_PATH]])
        {
            const auto modelPath = data[keymap[YamlKey::MODEL_PATH]].as<std::string>();
            VTX_INFO("Model Path: {0}", modelPath);
            const auto [_sceneRoot, cameras] = importer::importSceneFile(modelPath);
            if (!cameras.empty())
            {
                camera = cameras[0];
            }
            else
            {
                camera = ops::standardCamera();
            }
            scene->sceneRoot = _sceneRoot;
            previousModelPath = modelPath;
        }

        if (data[keymap[YamlKey::HDRI_PATH]])
        {
            const auto                          hdriPath = data[keymap[YamlKey::HDRI_PATH]].as<std::string>();
            const std::shared_ptr<graph::EnvironmentLight> envLight = ops::createNode<graph::EnvironmentLight>();
            envLight->envTexture = ops::createNode<graph::Texture>(hdriPath);
            scene->sceneRoot->addChild(envLight);
            VTX_INFO("HDRI Path: {0}", hdriPath);
            previousHDRIPath = hdriPath;
        }

        if(!scene->renderer)
        {
	        scene->renderer = ops::createNode<graph::Renderer>();
        }
        scene->renderer->camera  = camera;
        if (scene->sceneRoot)
        {
            scene->renderer->sceneRoot  = scene->sceneRoot;
        }

        return true;
    }

    void serialize(const std::string& filePath)
    {
	    
    }

    bool serializeExperimentManger(const std::string& filePath, ExperimentsManager& em)
    {
		const YAML::Node node = vtx::serializer::ExperimentsManagerSerializer::encode(em, utl::getFolder(filePath));
        std::ofstream file(filePath);
        if (!file.is_open())
        {
	        VTX_ERROR("Could not open file: {0}", filePath);
			return false;
		}
        file << node;
        return true;
    }

    ExperimentsManager deserializeExperimentManager(const std::string& filePath)
    {
	    std::ifstream      file(filePath);
        if (!file.is_open())
        {
	        VTX_ERROR("Could not open file: {0}", filePath);
			return ExperimentsManager();
		}
		const YAML::Node   node = YAML::Load(file);
        ExperimentsManager em;
        vtx::serializer::ExperimentsManagerSerializer::decode(node, em);
        return em;
    }
    std::string getPreviousModelPath()
    {
        return previousModelPath;
    }
    std::string getPreviousHDRIPath()
    {
        return previousHDRIPath;
    }
}

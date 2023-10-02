#include "Serializer.h"
#include <fstream>
#include "Core/Log.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/EnvironmentLight.h"
#include "Scene/Nodes/Renderer.h"
#include "Scene/Utility/ModelLoader.h"
#include "Scene/Utility/Operations.h"

#include "ArchiveFunctions.h"
#include <cereal/cereal.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

namespace vtx::serializer
{
    bool deserialize(const std::string& filePath, bool importScene)
    {
        try
        {
            VTX_INFO("Deserializing scene from {0}", filePath);

			const std::string fileExtension = utl::getFileExtension(filePath);
            std::ifstream file;
            if (fileExtension == "vtx") {
                file.open(filePath, std::ios::binary);
            }
            else {
                file.open(filePath);
            }
            if (!file.is_open())
            {
                VTX_ERROR("Could not open file {0}", filePath);
                return false;
            }

            std::shared_ptr<graph::Camera> camera = nullptr;
            std::shared_ptr<graph::Renderer> rendererNode = nullptr;
            std::shared_ptr<graph::Group> sceneRootNode = nullptr;
            std::shared_ptr<graph::EnvironmentLight> envLight = nullptr;


            GraphSaveData graphSaveData;
            if (fileExtension == "json")
            {
                cereal::JSONInputArchive archiveIn(file);
                archiveIn(graphSaveData);
            }
            else if (fileExtension == "xml")
            {
                cereal::XMLInputArchive archiveIn(file);
                archiveIn(graphSaveData);
            }
            else if (fileExtension == "vtx")
            {
                cereal::BinaryInputArchive archiveIn(file);
                archiveIn(graphSaveData);
            }
            else
            {
                VTX_ERROR("Unsupported file extension: {0}", fileExtension);
                return false;
            }
            const auto [_renderer, _sceneRoot] = graphSaveData.restoreShaderGraph();
            rendererNode = _renderer;
            sceneRootNode = _sceneRoot;

            graph::Scene* scene = graph::Scene::get();
            if (sceneRootNode) {
                // if the scene is not imported we replace the old scene root with the new one
                if (importScene)
                {
                    scene->sceneRoot->addChild(sceneRootNode);
                }
                else
                {
                    scene->sceneRoot = sceneRootNode;
                }
            }
            else if (!scene->sceneRoot)
            {
                // if there is no scene root we create a new one
                scene->sceneRoot = ops::createNode<graph::Group>();
            }
            if (rendererNode)
            {
                // if there is a renderer node we replace the old one
                scene->renderer = rendererNode;
            }
            else if (!scene->renderer)
            {
                // if there is no renderer node we create a new one
                scene->renderer = ops::createNode<graph::Renderer>();
            }
            if (camera)
            {
                scene->renderer->camera = camera;
            }
            if (envLight)
            {
                scene->renderer->environmentLight = envLight;
            }
            // we make sure the renderer node is connected to the scene root
            scene->renderer->sceneRoot = scene->sceneRoot;
        }
        catch (const std::exception& e)
        {
	        VTX_ERROR("Could not deserialize scene from {0}: {1}", filePath, e.what());
			return false;
		}
        catch (...)
        {
            VTX_ERROR("Unknown exception caught during serialization");
        }
        

        return true;
    }

    void serialize(const std::string& filePath)
    {
        try
        {
            std::ofstream file;
            const std::string fileExtension = utl::getFileExtension(filePath);

            if (fileExtension == "vtx") {
                file.open(filePath, std::ios::binary); // Open in binary mode for binary files
            }
            else {
                file.open(filePath); // Open in text mode for text-based formats like XML or JSON
            }

            GraphSaveData graphSaveData;
            graphSaveData.prepareSaveData();

            if (fileExtension == "json")
            {
                cereal::JSONOutputArchive archiveOut(file);
                archiveOut(graphSaveData);
            }
            else if (fileExtension == "xml")
            {
                cereal::XMLOutputArchive archiveOut(file);
                archiveOut(graphSaveData);
            }
            else if (fileExtension == "vtx")
            {
                cereal::BinaryOutputArchive archiveOut(file);
                archiveOut(graphSaveData);
            }
            else
            {
                VTX_ERROR("Unsupported file extension: {0}", fileExtension);
            }
        }
        catch (const std::exception& e)
        {
            VTX_ERROR("Exception caught during serialization: {0}", e.what());
        }
        catch (...)
        {
            VTX_ERROR("Unknown exception caught during serialization");
        }
    }
}

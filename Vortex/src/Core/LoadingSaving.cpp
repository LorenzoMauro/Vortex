#include "LoadingSaving.h"
#include "Application.h"
#include "FileDialog.h"
#include "Gui/Windows/LoadingSavingWindow.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Renderer.h"
#include "Scene/Utility/ModelLoader.h"
#include "Scene/Utility/Operations.h"
#include "Serialization/Serializer.h"

namespace vtx
{
	LoadingSaving& LoadingSaving::get()
	{
		static LoadingSaving instance; // Guaranteed to be destroyed. Instantiated on first use.
		return instance;
	}

	void LoadingSaving::loadFileDialog(std::vector<std::string> fileExtensions)
	{
		if (fileExtensions.empty())
		{
			fileExtensions = getSupportedLoadingFileExtensions();
		}
		const std::string filePath = vtx::FileDialogs::openFileDialog(fileExtensions);
		if (!filePath.empty())
		{
			doLoadFile = true;
			filePathToLoad = filePath;
		}
	}

	void LoadingSaving::saveFileDialog(std::vector<std::string> fileExtensions)
	{
		if (fileExtensions.empty())
		{
			fileExtensions = getSupportedSavingFileExtensions();
		}
		const std::string filePath = vtx::FileDialogs::saveFileDialog(fileExtensions);
		if (!filePath.empty())
		{
			doSaveFile = true;
			filePathToSave = filePath;
		}
	}

	void LoadingSaving::loadFile(std::string filePath)
	{
		if(filePath.empty())
		{
			filePath = filePathToLoad;
		}
		else
		{
			filePathToLoad = filePath;
		}
		if (!filePath.empty())
		{
			const std::string   fileExtension = utl::getFileExtension(filePath);
			graph::Scene* scene         = graph::Scene::get();
			// switch on file extension
			if (fileExtension == "vtx" || fileExtension == "xml" || fileExtension == "json")
			{
				serializer::deserialize(filePath);
			}
			else if (fileExtension == "obj" || fileExtension == "fbx" || fileExtension == "gltf")
			{
				const auto [_sceneRoot, cameras] = importer::importSceneFile(filePath);

				if (!scene->renderer)
				{
					scene->renderer = ops::createNode<graph::Renderer>();
				}
				if (!cameras.empty())
				{
					scene->renderer->camera = cameras[0];
				}
				else
				{
					scene->renderer->camera = ops::standardCamera();
				}
				scene->sceneRoot = _sceneRoot;
				scene->renderer->sceneRoot = scene->sceneRoot;

				previousModelPath = filePath;
			}
			else
			{
				// Handle unsupported file extension
			}
		}
		doLoadFile = false;
	}

	void LoadingSaving::saveFile(std::string filePath)
	{
		if (filePath.empty())
		{
			filePath = filePathToSave;
		}
		else
		{
			filePathToSave = filePath;
		}
		if (!filePath.empty()) {
			serializer::serialize(filePath);
		}
		doSaveFile = false;
	}

	std::vector<std::string> LoadingSaving::getSupportedLoadingFileExtensions()
	{
		return {"*.vtx", "*.xml", "*.json", "*.obj", "*.fbx", "*.gltf"};
	}

	std::vector<std::string> LoadingSaving::getSupportedSavingFileExtensions()
	{
		return {"*.vtx", "*.xml", "*.json"};
	}

	bool LoadingSaving::isLoadFileRequested()
	{
		return doLoadFile;
	}

	bool LoadingSaving::isSaveFileRequested()
	{
		return doSaveFile;
	}
	void LoadingSaving::performLoadSave()
	{
		if (!doLoadFile && !doSaveFile && currentState == LoadSaveState::Idle)
		{
			return; // Early exit if there's no loading or saving to be done.
		}

		switch (currentState) {
		case LoadSaveState::Idle:
			if (doLoadFile) {
				Application::get()->windowManager->createWindow<LoadingWindow>();
				currentState = LoadSaveState::LoadingGuiShown;
			}
			else if (doSaveFile) {
				Application::get()->windowManager->createWindow<SavingWindow>();
				currentState = LoadSaveState::SavingGuiShown;
			}
			break;

		case LoadSaveState::LoadingGuiShown:
			loadFile();
			currentState = LoadSaveState::Loading;
			break;

		case LoadSaveState::SavingGuiShown:
			saveFile();
			currentState = LoadSaveState::Saving;
			break;

		case LoadSaveState::Loading:
			filePathToLoad = "";
			currentState = LoadSaveState::Idle; // Reset after operation
			break;
		case LoadSaveState::Saving:
			filePathToSave = "";
			currentState = LoadSaveState::Idle; // Reset after operation
			break;
		}
	}
	std::string LoadingSaving::getFilePathToLoad()
	{
		return filePathToLoad;
	}
	std::string LoadingSaving::getFilePathToSave()
	{
		return filePathToSave;
	}

	LoadingSaving::LoadSaveState LoadingSaving::getCurrentState()
	{
		return currentState;
	}
	void LoadingSaving::setManualLoad(const std::string& filePath)
	{
		doLoadFile = true;
		filePathToLoad = filePath;
	}
}

#pragma once
#include <string>
#include <vector>

namespace vtx
{
	class LoadingSaving
	{
	public:

		enum class LoadSaveState {
			Idle,
			LoadingGuiShown,
			SavingGuiShown,
			Loading,
			Saving
		};

		// Public method to get the single instance
		static LoadingSaving& get();
		static std::vector<std::string> getSupportedLoadingFileExtensions();
		static std::vector<std::string> getSupportedSavingFileExtensions();

		LoadingSaving(LoadingSaving const&) = delete; // Don't allow copy
		void operator=(LoadingSaving const&) = delete; // Don't allow assignment

		void                            loadFileDialog(std::vector<std::string> fileExtensions = {});
		void                            saveFileDialog(std::vector<std::string> fileExtensions = {});
		void                            loadFile(std::string filePath = "");
		void                            saveFile(std::string filePath = "");
		bool                            isLoadFileRequested();
		bool                            isSaveFileRequested();
		void							performLoadSave();
		std::string                     getFilePathToLoad();
		std::string                     getFilePathToSave();
		LoadSaveState                   getCurrentState();
		void							setManualLoad(const std::string& filePath);

	private:
		LoadSaveState currentState = LoadSaveState::Idle;
		std::string previousModelPath;
		std::string filePathToLoad;
		std::string filePathToSave;
		bool doLoadFile = false;
		bool isLoadingGuiOpen = false;
		bool doSaveFile = false;
		bool isSavingGuiOpen = false;

		// Private constructor
		LoadingSaving() = default;
	};

}

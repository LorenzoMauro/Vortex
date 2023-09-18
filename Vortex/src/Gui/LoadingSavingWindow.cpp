#include "LoadingSavingWindow.h"

#include "Core/LoadingSaving.h"

namespace vtx
{
	void savingLoadingPreRender()
	{
		const ImGuiIO& io         = ImGui::GetIO();
		const auto     windowSize = ImVec2(io.DisplaySize.x * 0.10f, io.DisplaySize.y * 0.10f);
		const auto     windowPos  = ImVec2((io.DisplaySize.x - windowSize.x) * 0.5f, (io.DisplaySize.y - windowSize.y) * 0.5f);


		ImGui::SetNextWindowPos(windowPos);
		ImGui::SetNextWindowSize(windowSize);
	}
	LoadingWindow::LoadingWindow()
	{
		windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;
		useToolbar = false;
	}

	void LoadingWindow::preRender()
	{
		savingLoadingPreRender();
	}

	void LoadingWindow::OnUpdate(float ts)
	{
		if(LoadingSaving::get().getCurrentState() == LoadingSaving::LoadSaveState::Idle)
		{
			isOpen = false;
		}
	}

	void LoadingWindow::renderMainContent()
	{
		const std::string filePathToLoad = LoadingSaving::get().getFilePathToLoad();
		ImGui::Text("Loading file: %s\n This might take a while, you can check the terminal for more info", filePathToLoad.c_str());
	}

	SavingWindow::SavingWindow()
	{
		windowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;
		useToolbar = false;
	}

	void SavingWindow::preRender()
	{
		savingLoadingPreRender();
	}

	void SavingWindow::OnUpdate(float ts)
	{
		if (LoadingSaving::get().getCurrentState() == LoadingSaving::LoadSaveState::Idle)
		{
			isOpen = false;
		}
	}

	void SavingWindow::renderMainContent()
	{
		const std::string filePathToSave = LoadingSaving::get().getFilePathToSave();
		ImGui::Text("Saving file: %s\n", filePathToSave.c_str());
	}
}

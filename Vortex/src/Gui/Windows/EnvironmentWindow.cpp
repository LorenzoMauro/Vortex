#include "EnvironmentWindow.h"

#include "Core/FileDialog.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/EnvironmentLight.h"
#include "Scene/Nodes/Shader/Texture.h"
#include "Scene/Utility/Operations.h"

namespace vtx
{
	EnvironmentWindow::EnvironmentWindow()
	{
		name = "Environment Dome Light";
		useToolbar = false;
		isBorderLess = true;
		renderCustomMenuBar = true;
	}

	void vtx::EnvironmentWindow::OnUpdate(float ts)
	{
		if(const auto& env = graph::Scene::get()->renderer->environmentLight)
		{
			if(texturePath.empty() || env->envTexture->filePath != texturePath)
			{
				texturePath = env->envTexture->filePath;
				glEnvironmentTexture = 0;
				if (!env->envTexture->isInitialized)
				{
					env->envTexture->init();
				}
				const void* const ptr = env->envTexture->imageLayersPointers[0];
				width = env->envTexture->dimension.x;
				height = env->envTexture->dimension.y;

				glGenTextures(1, &glEnvironmentTexture);
				glBindTexture(GL_TEXTURE_2D, glEnvironmentTexture);

				// Modified this line
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, ptr);

				// Adding texture parameters
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

				glBindTexture(GL_TEXTURE_2D, glEnvironmentTexture);
				isReady = true;
			}
		}
	}
	void EnvironmentWindow::mainContent()
	{
		if (isReady)
		{
			ImVec2 windowSize = ImGui::GetWindowSize(); // get the size of the current window
			ImVec2 imageSize = ImVec2((float)width, (float)height); // original image size

			// Calculate the scaling factor
			float scale = std::min(windowSize.x / imageSize.x, windowSize.y / imageSize.y);

			// Scaled image size
			ImVec2 scaledImageSize = ImVec2(imageSize.x * scale, imageSize.y * scale);

			// Calculate centering offsets
			float offsetX = (windowSize.x - scaledImageSize.x) * 0.5f;
			float offsetY = (windowSize.y - scaledImageSize.y) * 0.5f;

			// Use offsets to center the image
			ImGui::SetCursorPosX(offsetX);
			ImGui::SetCursorPosY(offsetY);

			ImGui::Image((ImTextureID)glEnvironmentTexture, scaledImageSize, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });
		}
	}
	void EnvironmentWindow::menuBarContent()
	{
		ImGui::BeginGroup();
		{
			const float windowWidth = ImGui::GetWindowWidth();
			ImGui::SetCursorPosY(ImGui::GetStyle().WindowPadding.y);

			float       xPos               = windowWidth;
			const float arrangeButtonWidth = ImGui::CalcTextSize("Load Hdri").x + 20.0f;
			ImGui::PushItemWidth(arrangeButtonWidth);
			ImGui::SameLine();
			xPos = xPos - arrangeButtonWidth - ImGui::GetStyle().WindowPadding.x;
			ImGui::SetCursorPosX(xPos);
			if (ImGui::Button("Load Hdri"))
			{
				const std::string filePath = vtx::FileDialogs::openFileDialog({"*.hdr"});
				const auto& renderer = graph::Scene::get()->renderer;
				renderer->environmentLight = ops::createNode<graph::EnvironmentLight>();
				renderer->environmentLight->envTexture = ops::createNode<graph::Texture>(filePath);
				renderer->environmentLight->envTexture->init();
				texturePath = "";
				glDeleteTextures(1, &glEnvironmentTexture); // delete the old texture
				ops::restartRender();
			}
			ImGui::PopItemWidth();
		}
		ImGui::EndGroup();
		menuBarHeight = ImGui::GetItemRectSize().y;
	}
}


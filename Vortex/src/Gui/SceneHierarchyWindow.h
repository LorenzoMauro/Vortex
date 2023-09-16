#pragma once
#include "GuiWindow.h"

namespace vtx
{
	namespace graph
	{
		class Renderer;
		class Node;
	}

	class SceneHierarchyWindow : public Window
	{
	public:
		SceneHierarchyWindow();

        void displayNode(const std::shared_ptr<graph::Node>& node);

		void renderMainContent() override;

	private:
		std::shared_ptr<graph::Renderer> renderer;
		int selection_mask = - 1;

	};

}
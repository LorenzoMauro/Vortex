#pragma once
#include "Gui/GuiWindow.h"

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

		void mainContent() override;
	};

}
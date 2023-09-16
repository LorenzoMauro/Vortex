#include "GraphWindow.h"
#include "Scene/Scene.h"


namespace vtx
{
	GraphWindow::GraphWindow()
	{
		name = "Node Graph";
		useToolbar = false;
		renderer = graph::Scene::getScene()->renderer;
		isBorderLess = true;
	}

	void GraphWindow::renderMainContent()
	{
		windowManager->selectedNodes["SceneGraphWindow"] = gui::SceneGraphGui::draw(renderer);
	}
}


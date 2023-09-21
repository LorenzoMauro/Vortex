#include "PropertiesWindow.h"
#include "Scene/Scene.h"

namespace vtx
{

	PropertiesWindow::PropertiesWindow()
	{
		name = "Properties";
		useToolbar = false;
		renderer = graph::Scene::getScene()->renderer;
	}

	void PropertiesWindow::mainContent()
	{
		bool areNodesSelected = false;
		if(!windowManager->selectedNodes.empty())
		{
			for(auto [key, nodes]: windowManager->selectedNodes)
			{
				for (const int idx : nodes)
				{
					if(idx != 0)
					{
						areNodesSelected = true;
						const vtxID nodeId = idx;
						const std::shared_ptr<graph::Node>& node = graph::SIM::get()->getNode<graph::Node>(nodeId);
						if(node)
						{
							ImGui::SetNextItemOpen(true, ImGuiCond_Once);
							guiVisitor.changed = false;
							node->accept(guiVisitor);
						}
					}
				}
			}
		}
		if (!areNodesSelected)
		{
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			renderer->accept(guiVisitor);
		}
		
	}
}

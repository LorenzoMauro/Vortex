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
		bool                     areNodesSelected = false;
		if(const std::set<vtxID> selected = graph::Scene::getScene()->getSelected(); !selected.empty())
		{
			areNodesSelected = true;
			for (const vtxID idx : selected)
			{
				if (idx != 0)
				{
					areNodesSelected = true;
					const vtxID nodeId = idx;
					if (const std::shared_ptr<graph::Node>& node = graph::SIM::get()->getNode<graph::Node>(nodeId))
					{
						ImGui::SetNextItemOpen(true, ImGuiCond_Once);
						guiVisitor.changed = false;
						node->accept(guiVisitor);
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

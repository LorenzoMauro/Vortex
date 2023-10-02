#include "PropertiesWindow.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Renderer.h"

namespace vtx
{

	PropertiesWindow::PropertiesWindow()
	{
		name = "Properties";
		useToolbar = false;
	}

	void PropertiesWindow::mainContent()
	{
		bool                     areNodesSelected = false;
		if(const std::set<vtxID> selected = graph::Scene::get()->getSelected(); !selected.empty())
		{
			areNodesSelected = true;
			for (const vtxID idx : selected)
			{
				if (idx != 0)
				{
					areNodesSelected = true;
					const vtxID nodeId = idx;
					if (const std::shared_ptr<graph::Node>& node = graph::Scene::getSim()->getNode<graph::Node>(nodeId))
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
			graph::Scene::get()->renderer->accept(guiVisitor);
		}
		
	}
}

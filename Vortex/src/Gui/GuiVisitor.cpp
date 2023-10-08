#include "GuiVisitor.h"

#include "GuiProvider.h"
#include "Scene/Graph.h"

namespace vtx::gui
{

	void GuiVisitor::callOnChildren(const std::shared_ptr<graph::Node>& node)
	{
		const std::vector<std::shared_ptr<graph::Node>> children = node->getChildren();
		if (!children.empty())
		{
			ImGui::PushID(node->getUID());
			if (ImGui::CollapsingHeader("Node Children"))
			{
				ImGui::Indent();
				for (const auto& child : children)
				{
					child->accept(*this);
				}
				ImGui::Unindent();
			}
			ImGui::PopID();
		}
		
		
	}
	void GuiVisitor::visit(const std::shared_ptr<graph::Instance>& node)
	{
		changed |= GuiProvider::drawEditGui(node);
		//callOnChildren(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Transform>& node)
	{
		changed |= GuiProvider::drawEditGui(node);
		//callOnChildren(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Group>& node)
	{
		changed |= GuiProvider::drawEditGui(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Mesh>& node)
	{

		changed |= GuiProvider::drawEditGui(node);
		//callOnChildren(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Material>& node)
	{
		changed |= GuiProvider::drawEditGui(node);

		const std::vector<std::shared_ptr<graph::Node>> children = node->getChildren();

		if (!children.empty())
		{
			ImGui::PushID(node->getUID());
			if (ImGui::CollapsingHeader("Node Children"))
			{
				ImGui::Indent();
				for (const auto& child : children)
				{
					if (child != node->materialGraph)
					{
						child->accept(*this);
					}
				}
				ImGui::Unindent();
			}
			ImGui::PopID();
		}
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Camera>& node)
	{

		changed |= GuiProvider::drawEditGui(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Renderer>& node)
	{
		changed |= GuiProvider::drawEditGui(node);
		callOnChildren(node);
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::Texture>& node)
	{
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::BsdfMeasurement>& node)
	{
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::LightProfile>& node)
	{
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::EnvironmentLight>& node)
	{
	}

	void GuiVisitor::visit(const std::shared_ptr<graph::MeshLight>& node)
	{
	}


	void GuiVisitor::visit(const std::shared_ptr<graph::shader::ShaderNode>& node)
	{
		changed |= GuiProvider::drawEditGui(node, isNodeEditor);
	}
}
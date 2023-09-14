#include "GuiVisitor.h"

#include "CameraNodeGui.h"
#include "TransformNodeGui.h"
#include "Layers/GuiElements/RendererNodeGui.h"
#include "Scene/Graph.h"

namespace vtx::gui
{

	void GuiVisitor::callOnChildren(std::shared_ptr<graph::Node> node)
	{
		const std::vector<std::shared_ptr<graph::Node>> children = node->getChildren();
		if (!children.empty())
		{
			if (ImGui::CollapsingHeader("Node Children"))
			{
				ImGui::Indent();
				for (const auto& child : children)
				{
					child->accept(*this);
				}
				ImGui::Unindent();
			}
		}
		
		
	}
	void GuiVisitor::visit(std::shared_ptr<graph::Instance> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Transform> node)
	{
		transformNodeGui(node);
		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Group> node)
	{


		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Mesh> node)
	{


		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Material> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Camera> node)
	{

		cameraNodeGui(node);
		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Renderer> node)
	{
		rendererNodeGui(node);
		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::Texture> node)
	{


		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::BsdfMeasurement> node)
	{


		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::LightProfile> node)
	{


		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::EnvironmentLight> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::MeshLight> node)
	{

		callOnChildren(node);
	}


	void GuiVisitor::visit(std::shared_ptr<graph::shader::DiffuseReflection> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::MaterialSurface> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::Material> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::ImportedNode> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::PrincipledMaterial> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::ColorTexture> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::MonoTexture> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::NormalTexture> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::BumpTexture> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::TextureTransform> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::NormalMix> node)
	{

		callOnChildren(node);
	}

	void GuiVisitor::visit(std::shared_ptr<graph::shader::GetChannel> node)
	{
		
		callOnChildren(node);
	}
}
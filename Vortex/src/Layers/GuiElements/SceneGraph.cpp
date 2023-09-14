#include "SceneGraph.h"
#include "imnodes.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Graph.h"
#include "Scene/Scene.h"

namespace vtx::gui
{
	static bool isSceneGraphOpened = false;

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Instance> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Transform> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Group> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Mesh> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Material> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Camera> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Renderer> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::Texture> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::BsdfMeasurement> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::LightProfile> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);

	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::EnvironmentLight> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::MeshLight> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}


	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::DiffuseReflection> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::MaterialSurface> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::Material> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::ImportedNode> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::PrincipledMaterial> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::ColorTexture> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::MonoTexture> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::NormalTexture> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::BumpTexture> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::TextureTransform> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::NormalMix> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	void SceneGraphVisitor::visit(std::shared_ptr<graph::shader::GetChannel> node)
	{
		auto [depth, width, overallWidth] = nodesDepthsAndWidths[node->getID()];
		nodeGraphUi.submitNode(node, depth, width, overallWidth);
	}

	static SceneGraphVisitor visitor = SceneGraphVisitor();

	void SceneGraphGui::draw()
	{
		std::shared_ptr<graph::Scene>      scene    = graph::Scene::getScene();
		std::shared_ptr<graph::Renderer>   renderer = scene->renderer;
		renderer->traverse(visitor);
		visitor.nodeGraphUi.draw();
	}
}

#pragma once
#include "NodeInfo.h"
#include "Scene/Traversal.h"
#include <ImNodes.h>
#include <set>

#include "Core/Options.h"
#include "Gui/GuiProvider.h"
#include "Gui/GuiVisitor.h"

namespace vtx::gui
{
	class NodeEditor;


	class NodeEditorVisitor : public NodeVisitor {
	public:

		NodeEditorVisitor();
		void visit(const std::shared_ptr<graph::Node>& node) override;

		NodeEditor* nodeEditor = nullptr;
	};

	class NodeEditor
	{
	public:


		NodeEditor();

		void updateNodeLinks(const std::shared_ptr<graph::Node>& node);

		void submitNode(const std::shared_ptr<graph::Node>& node, const int depth, const int width, const int overallWidth);

		void        traverseAndGetNodeInfos();
		static void pushNodeColorByNodeType(const graph::NodeType& nodeType);

		static void popNodeColorByNodeType();
		static void pushPinColorByChildrenNodeType(const graph::NodeType& nodeType);
		static void popPinColorByChildrenNodeType();
		static void pushLinkColorByChildNodeType(const graph::NodeType& nodeType);
		static void popLinkColorByChildNodeType();
		bool        verticalArrangementDrawNode(NodeInfo& nodeInfo) const;

		void nodeResizeButton(math::vec2f& size, const int id) const;

		bool horizontalArrangementDrawNode(NodeInfo& nodeInfo);

		bool drawNode(NodeInfo& node);

		void drawLinks(const NodeInfo& nodeInfo);

		enum class LayoutDirection
		{
			Vertical,
			Horizontal
		};

		void arrangeNodes(LayoutDirection direction);

		void removeUnvisitedNodes();

		bool draw();

		bool styleEditor();

		void toolBar();

		NodeEditorVisitor            visitor;
		GuiVisitor                   guiVisitor;
		std::map<vtxID, NodeInfo>    nodes;
		std::vector<int>             selectedNodes     = {};
		std::shared_ptr<graph::Node> rootNode          = nullptr;
		ImNodesEditorContext*        sceneGraphContext = ImNodes::EditorContextCreate();
		bool                         isFirstTime       = true;
		int                          spacing           = 40;
		ImNodesStyleFlags_           flags             = (ImNodesStyleFlags_)0;
		std::set<vtxID>              runVisitedNodes   = {};
		std::vector<vtxID>           depthFirstTraversal;
		float                        treeWidthPadding = 10.0f;
		float                        treeDepthPadding = 30.0f;
		bool                          nodesWithColor = false;
	};
}

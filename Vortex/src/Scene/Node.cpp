#include "SIM.h"
#include "Node.h"
#include "Traversal.h"

namespace vtx::graph
{

	Node::Node(const NodeType _type) : type(_type)
	{
		sim = SIM::get();
		id  = sim->getFreeIndex();
		name = nodeNames[type] + "_" + std::to_string(id);
	}

	Node::~Node()
	{
		sim->releaseIndex(id);
		VTX_WARN("Node ID: {} Name: {} destroyed", id, name);
	}
        
	NodeType Node::getType() const {
		return type;
	}
        
	vtxID Node::getID() const {
		return id;
	}

	void Node::traverse(NodeVisitor& visitor)
	{
		visitor.visitBegin(as<Node>());

		traverseChildren(visitor);

		if (!isInitialized)
		{
			init();
			isInitialized = true;
			isUpdated = true;
		}

		accept(visitor);

		visitor.visitEnd(as<Node>());

	}

	void Node::traverseChildren(NodeVisitor& visitor)
	{
		for(const std::shared_ptr<Node> node : getChildren())
		{
			node->traverse(visitor);
		}

	}

}

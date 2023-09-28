#include "SIM.h"
#include "Node.h"
#include "Traversal.h"

namespace vtx::graph
{

	Node::Node(const NodeType _type) : type(_type)
	{
		sim = SIM::get();
		UID  = sim->getUID();
		name = nodeNames[type] + "_" + std::to_string(UID);
	}

	Node::~Node()
	{
		sim->releaseUID(UID);
		VTX_WARN("Node ID: {} Name: {} destroyed", UID, name);
	}
        
	NodeType Node::getType() const {
		return type;
	}
        
	vtxID Node::getUID() const {
		return UID;
	}

	vtxID Node::getTypeID() const
	{
		return typeID;
	}

	void Node::traverse(NodeVisitor& visitor)
	{

		//TEMPORARY CHECK TO MAKE SURE TYPE ID DEFINITION IN CONSTRUCTOR AND DESTRUCTOR IN DERIVED CLASS IS DEFINED
		if (typeID == 0)
		{
			VTX_ERROR("Node ID: {} Name: {} typeID not defined", UID, name);
		}
		visitor.visitBegin(as<Node>());

		traverseChildren(visitor);

		if (!state.isInitialized)
		{
			init();
			state.isInitialized = true;
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

#include "Node.h"
#include "Scene.h"
#include "Traversal.h"

namespace vtx::graph
{

	Node::Node(const NodeType _type) : type(_type)
	{
		sim = Scene::getSim();
		UID  = sim->getUID();
		typeID = sim->getTypeId(type);
		name = nodeNames[type] + "." + std::to_string(typeID);
	}

	Node::~Node()
	{
		sim->removeNodeReference(UID, typeID, type);
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

	void Node::setUID(vtxID id)
	{
		UID = id;
	}

	void Node::setTID(vtxID id)
	{
		typeID = id;
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
			if(node)
			{
				node->traverse(visitor);
			}
		}

	}

}

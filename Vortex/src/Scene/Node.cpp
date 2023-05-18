#include "SIM.h"
#include "Node.h"
#include "Traversal.h"

namespace vtx::graph
{

	Node::Node(NodeType _type) : type(_type) {
		sim = SIM::Get();
		id = sim->getFreeIndex();
	}

	Node::~Node()
	{
		std::shared_ptr<SIM> sim = SIM::Get();
		sim->releaseIndex(id);
	}
        
	NodeType Node::getType() const {
		return type;
	}
        
	vtxID Node::getID() const {
		return id;
	}

}

#include "SIM.h"
#include "SceneGraph.h"

namespace vtx {
	namespace scene {


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
}
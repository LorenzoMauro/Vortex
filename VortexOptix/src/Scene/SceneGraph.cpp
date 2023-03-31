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
        
        void Node::addChild(std::shared_ptr<Node> child) {
            child->setParent(shared_from_this());
            children.push_back(child);
        }
        
        std::vector<std::shared_ptr<Node>>& Node::getChildren() {
            return children;
        }
        
        std::shared_ptr<Node> Node::getParent() {
            return parent.lock();
        }
        
        NodeType Node::getType() const {
            return type;
        }
        
        vtxID Node::getID() const {
            return id;
        }
        
        void Node::setParent(std::shared_ptr<Node> parentNode) {
            parent = parentNode;
        }

    }
}
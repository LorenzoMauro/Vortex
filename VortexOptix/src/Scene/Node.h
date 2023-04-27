#pragma once
#include "Core/VortexID.h"
#include "Core/math.h"
#include <vector>
#include <memory>
#include <map>



#define ACCEPT(visitors) \
		for (const std::shared_ptr<vtx::NodeVisitor> visitor : orderedVisitors)\
		{\
			accept(visitor);\
		};

namespace vtx
{
	class NodeVisitor;
}

namespace vtx::graph
{

	class SIM;

	enum NodeType {
		NT_GROUP,

		NT_INSTANCE,
		NT_MESH,
		NT_TRANSFORM,
		NT_LIGHT,

		NT_CAMERA,
		NT_RENDERER,

		NT_MATERIAL,
		NT_MDL_SHADER,
		NT_MDL_TEXTURE,
		NT_MDL_BSDF,
		NT_MDL_LIGHTPROFILE,

		NT_NUM_NODE_TYPES,
	};
        
	class Node : public std::enable_shared_from_this<Node> {
	public:

		Node(NodeType _type);

		virtual ~Node();

		NodeType getType() const;

		vtxID getID() const;

		virtual void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) = 0;

		virtual void accept(std::shared_ptr<NodeVisitor> visitor) = 0;
	public:
		bool isUpdated;
	protected:
		template <class Derived>
		std::shared_ptr<Derived> sharedFromBase()
		{
			return std::static_pointer_cast<Derived>(shared_from_this());
		}
	protected:
		std::shared_ptr<SIM> sim;
		NodeType type;
		vtxID id;
	};
}

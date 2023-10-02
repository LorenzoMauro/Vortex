#pragma once
#include "Core/VortexID.h"
#include <vector>
#include <memory>
#include "NodeTypes.h"

#define ACCEPT(derived, visitor) \
			visitor.visit(as<derived>()); \

namespace vtx
{
	class NodeVisitor;
}

namespace vtx::graph
{

	class SceneIndexManager;

	struct NodeState
	{
		bool isInitialized = false;
		bool isChangedByGui = false;
		bool updateOnDevice = false;
		bool isShaderCodeUpdated = false;
		bool isShaderArgBlockUpdated = false;
	};

	struct NodeTreePosition
	{
		int depth = 0;
		int width= 0;
		int overallWidth = 0;
	};

	class Node : public std::enable_shared_from_this<Node> {
	public:

		Node(NodeType _type);

		virtual ~Node();

		NodeType getType() const;

		vtxID getUID() const;
		vtxID getTypeID() const;

		void Node::setUID(vtxID id);
		void setTID(vtxID id);

		void traverse(NodeVisitor& visitor);

		template<class Derived>
		std::shared_ptr<Derived> as()
		{
			return std::dynamic_pointer_cast<Derived>(shared_from_this());
		}

		virtual void init() {};

		virtual std::vector<std::shared_ptr<Node>> getChildren() const
		{
			return {};
		}

		NodeState   state;
		std::string name;
		NodeTreePosition treePosition;

		virtual void accept(NodeVisitor& visitor) = 0;
	protected:
		virtual void traverseChildren(NodeVisitor& visitor);


		template <class Derived>
		std::shared_ptr<Derived> sharedFromBase()
		{
			return std::static_pointer_cast<Derived>(shared_from_this());
		}

		NodeType type;
		vtxID UID;
		vtxID typeID = 0;
		std::shared_ptr<SceneIndexManager> sim;
	};

	namespace shader {
		class TextureFile;
		class TextureReturn;
		class ShaderNode;
		class Material;
		class DiffuseReflection;
		class MaterialSurface;
		class ImportedNode;
		class PrincipledMaterial;
		class ColorTexture;
		class MonoTexture;
		class NormalTexture;
		class BumpTexture;
		class TextureTransform;
		class NormalMix;
		class GetChannel;

		struct EnumValue;
		struct EnumTypeInfo;
		struct Annotation;
		class ParameterInfo;
		struct ShaderNodeSocket;
	}

	class MeshLight;
	class EnvironmentLight;
	class Node;
	class Transform;
	class Instance;
	class Group;
	class Mesh;
	class Material;
	class Camera;
	class Renderer;
	class Texture;
	class BsdfMeasurement;
	class LightProfile;
	struct Configuration;
	struct FunctionNames;
	struct DevicePrograms;
}

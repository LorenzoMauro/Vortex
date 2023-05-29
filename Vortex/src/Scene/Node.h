#pragma once
#include "Core/VortexID.h"
#include <vector>
#include <memory>

#define ACCEPT(derived, visitors) \
		for (const std::shared_ptr<vtx::NodeVisitor> visitor : orderedVisitors)\
		{\
			visitor->visit(sharedFromBase<derived>()); \
		};\

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

		NT_SHADER_TEXTURE,
		NT_SHADER_DF,
		NT_SHADER_MATERIAL,
		NT_SHADER_SURFACE,
		NT_SHADER_IMPORTED,
		NT_SHADER_COLOR,
		NT_SHADER_FLOAT3,
		NT_SHADER_FLOAT,
		NT_SHADER_COORDINATE,

		NT_NUM_NODE_TYPES
	};
        
	class Node : public std::enable_shared_from_this<Node> {
	public:

		Node(NodeType _type);

		virtual ~Node();

		NodeType getType() const;

		vtxID getID() const;

		virtual void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) = 0;

		//virtual void accept(std::shared_ptr<NodeVisitor> visitor) = 0;

		void          updateMaterialSlots(int removedSlot);

	public:
		bool isUpdated = true;
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
		enum ParamKind;
		struct Annotation;
		class ParameterInfo;
		struct ShaderNodeSocket;
	}

	class Light;
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

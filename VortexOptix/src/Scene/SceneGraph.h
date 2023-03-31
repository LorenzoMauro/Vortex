#pragma once
#include "Core/VortexID.h"
#include "Core/math.h"
#include <vector>
#include <memory>
#include <map>


namespace vtx {
	namespace scene {

        class SIM;

        struct VertexAttributes {
            math::vec3f position;
            math::vec3f normal;
            math::vec3f tangent;
            math::vec3f texcoord;
        };

        struct TransformAttribute {
            math::vec3f scale;
            math::vec3f translation;
            math::vec3f eulerAngles;
            math::Affine3f AffineTransform;

            void updateFromVectors() {
				AffineTransform = math::Affine3f::translate(translation) * (math::Affine3f)math::AffineFromEuler<float>(eulerAngles) * math::Affine3f::scale(scale);
			}

            void updateFromAffine() {
                math::VectorFromAffine<float>(AffineTransform, scale, translation, eulerAngles);
            }
        };

		enum NodeType {
			NT_GROUP,
			NT_INSTANCE,
			NT_MESH,
            NT_MATERIAL,
            NT_TRANSFORM,

            NT_NUM_NODE_TYPES
		};
        
        class Node : public std::enable_shared_from_this<Node> {
        public:

            Node(NodeType _type = NT_GROUP);

            ~Node();

            void addChild(std::shared_ptr<Node> child);

            std::vector<std::shared_ptr<Node>>& getChildren();

            std::shared_ptr<Node> getParent();

            NodeType getType() const;

            vtxID getID() const;

        protected:
            void setParent(std::shared_ptr<Node> parentNode);
        public:
            std::shared_ptr<SIM> sim;
            std::vector<std::shared_ptr<Node>> children;
            std::weak_ptr<Node> parent;
            NodeType type;
            vtxID id;
        };

        class Material : public Node {
        public:
			Material() : Node(NT_MATERIAL) {};
        };

        class Transform : public Node {
        public:
            Transform() : Node(NT_TRANSFORM) {};

            TransformAttribute transformationAttribute;
        };

        class Mesh : public Node {
        public:
            Mesh() : Node(NT_MESH) {};

            std::shared_ptr<Transform> getTransform() {
                return transform;
            }

            void setTransform(std::shared_ptr<Transform>& _transform) {
                transform = _transform;
            }

        public: 
            std::shared_ptr<Transform>  transform;
            std::vector<VertexAttributes> vertices;
            std::vector<vtxID> indices; // indices for triangles (every 3 indices define a triangle)
            std::vector<std::shared_ptr<Material>> materials; // container for shared pointers to materials
            std::vector<std::shared_ptr<Material>> triangleMaterials; // mapping of triangle indices to materials
        };

        class Instance : public Node {
            Instance() : Node(NT_INSTANCE) {}

            std::shared_ptr<Transform> getTransform() {
                return transform;
            }

            void setTransform(std::shared_ptr<Transform>& _transform) {
                transform = _transform;
            }

        public:
            std::shared_ptr<Transform>  transform;
        };

	}
}
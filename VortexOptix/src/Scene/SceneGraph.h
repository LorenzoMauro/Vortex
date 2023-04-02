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
            int         instanceMaterialIndex;
        };

        struct TransformAttribute {
            math::vec3f scale{ 1.0f };
            math::vec3f translation{ 0.0f };
            math::vec3f eulerAngles{ 0.0f };
            math::affine3f AffineTransform = math::affine3f(math::Identity);

            void updateFromVectors() {
				AffineTransform = math::affine3f::translate(translation) * (math::affine3f)math::AffineFromEuler<math::LinearSpace3f>(eulerAngles) * math::affine3f::scale(scale);
			}

            void updateFromAffine() {
                math::VectorFromAffine<math::LinearSpace3f>(AffineTransform, scale, translation, eulerAngles);
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
        
        class Node {
        public:

            Node(NodeType _type);

            ~Node();

            NodeType getType() const;

            vtxID getID() const;

        protected:
            std::shared_ptr<SIM> sim;
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

        class Instance : public Node {
        public:

            Instance() : Node(NT_INSTANCE) {}

            std::shared_ptr<Node> getChild() {
                return child;
            }

            void setChild(std::shared_ptr<Node> _child) {
                child = _child;
            }

            std::shared_ptr<Transform> getTransform() {
                return transform;
            }

            void setTransform(std::shared_ptr<Transform>& _transform) {
                transform = _transform;
            }

            std::vector<std::shared_ptr<Material>>& getMaterials() {
                return materials;
            }

            void addmaterial(std::shared_ptr<Material> _material) {
                materials.push_back(_material);
            }

            void RemoveMaterial(vtxID matID) {

            }

        private:
            std::shared_ptr<Node>       child;
            std::shared_ptr<Transform>  transform;
            std::vector<std::shared_ptr<Material>> materials;

        };

        class Group : public Node {
        public:
            Group() : Node(NT_GROUP) {}

            std::vector<std::shared_ptr<Node>>& getChildren() {
                return children;
            }

            void addChild(std::shared_ptr<Node> child) {
                children.push_back(child);
            }

        private:
            std::vector<std::shared_ptr<Node>> children;

        };



        class Mesh : public Node {
        public:
            Mesh() : Node(NT_MESH) {};

        public: 
            std::vector<VertexAttributes> vertices;
            std::vector<vtxID> indices; // indices for triangles (every 3 indices define a triangle)
        };


	}
}
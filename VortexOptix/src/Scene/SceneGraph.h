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

            /* Update the transformation given the vector representation*/
            void updateFromVectors() {
				AffineTransform = math::affine3f::translate(translation) * (math::affine3f)math::AffineFromEuler<math::LinearSpace3f>(eulerAngles) * math::affine3f::scale(scale);
			}

            /* Update the vector representation given the affine matrix*/
            void updateFromAffine() {
                math::VectorFromAffine<math::LinearSpace3f>(AffineTransform, translation, scale, eulerAngles);
            }

            

        };

		enum NodeType {
			NT_GROUP,
			NT_INSTANCE,
			NT_MESH,
            NT_MATERIAL,
            NT_TRANSFORM,
            NT_CAMERA,
            NT_RENDERER,

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

            math::vec3f TransformVector(const math::vec3f& vector) {
                return math::TransfomrVector3f(transformationAttribute.AffineTransform, vector);
            }

            math::vec3f TransformNormal(const math::vec3f& vector) {
                return math::TransfomrNormal3f(transformationAttribute.AffineTransform, vector);
            }

            math::vec3f TransformPoint(const math::vec3f& vector) {
                return math::TransfomrPoint3f(transformationAttribute.AffineTransform, vector);
            }

            /* Translation utility given vector */
            void translate(math::vec3f translation) {
                math::affine3f translationMatrix = math::affine3f::translate(translation);
                transformationAttribute.AffineTransform = translationMatrix * transformationAttribute.AffineTransform;
                transformationAttribute.updateFromAffine();
            }

            /* Translation utility given axis and ammount */
            void translate(math::vec3f direction, float ammount) {
                translate(direction * ammount);
            }

            /* Rotation utility for axis angle in radians */
            void rotate(math::vec3f axis, float radian) {
                math::affine3f rotationMatrix = math::affine3f::rotate(axis, radian);
                transformationAttribute.AffineTransform = rotationMatrix * transformationAttribute.AffineTransform;
                transformationAttribute.updateFromAffine();
            }

            /* Rotation utility for axis angle in degree */
            void rotateDegree(math::vec3f axis, float degree) {
                rotate(axis, math::toRadians(degree));
            }

            /* Rotation utility for axis angle around point in radians */
            void rotateAroundPoint(math::vec3f point, math::vec3f axis, float radian) {
                math::affine3f transformation = math::affine3f::rotate(point, axis, radian);
                transformationAttribute.AffineTransform = transformation * transformationAttribute.AffineTransform;
                transformationAttribute.updateFromAffine();
            }

            /* Rotation utility for axis angle around point in degree */
            void rotateAroundPointDegree(math::vec3f point, math::vec3f axis, float degree) {
                rotateAroundPoint(point, axis, math::toRadians(degree));
            }

            void rotateQuaternion(math::Quaternion3f quat) {
                math::LinearSpace3f rotationMatrix = math::LinearSpace3f(quat);
                math::affine3f transformation = math::affine3f(rotationMatrix);
                transformationAttribute.AffineTransform = transformation * transformationAttribute.AffineTransform;
                transformationAttribute.updateFromAffine();
            }

            void rotateOrbit(float pitch, math::vec3f xAxis, float yaw, math::vec3f zAxis) {
                math::affine3f rotationPITCHMatrix = math::affine3f::rotate(xAxis, pitch);
                math::affine3f rotationYAWMatrix = math::affine3f::rotate(zAxis, yaw);
                transformationAttribute.AffineTransform = rotationPITCHMatrix * rotationYAWMatrix * transformationAttribute.AffineTransform;

                transformationAttribute.updateFromAffine();
            }

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
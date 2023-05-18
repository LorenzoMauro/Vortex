#pragma once
#include <memory>
#include "Scene/SIM.h"
#include "Core/Math.h"

namespace vtx::graph
{
    namespace shader {
        class Material;
        class DiffuseReflection;
        class Texture;
        class MaterialSurface;
    }

	struct VertexAttributes;
	class Group;
	class Node;
	class Mesh;
	class TransformAttribute;
}

namespace vtx::ops {

    template<typename T,typename... Ts>
    std::shared_ptr<T> createNode(Ts... optionalArgs) {
        static_assert(std::is_base_of_v<graph::Node, T>, "Pushed type is not subclass of Node!");
        std::shared_ptr<T> node = std::make_shared<T>(optionalArgs...);
        graph::SIM::record(node);
        return node;
    }

    // A simple unit cube built from 12 triangles.
    std::shared_ptr<graph::Mesh> createBox(float sideLength=2.0f);

    void applyTransformation(graph::TransformAttribute& transformation, const math::affine3f& affine);

    std::shared_ptr<graph::Mesh> createPlane(float width=2.0f, float height=2.0f);

    void updateMaterialSlots(std::shared_ptr<graph::Mesh> mesh, int removedSlot);

    float gaussianFilter(const float* rgba,
                         const unsigned int width,
                         const unsigned int height,
                         const unsigned int x,
                         const unsigned int y,
                         const bool isSpherical);

    std::shared_ptr <graph::shader::ImportedNode> createPbsdfGraph();

    void computeFaceAttributes(const std::shared_ptr<graph::Mesh>& mesh);

    void computeVertexNormals(std::shared_ptr<graph::Mesh> mesh);

    void computeVertexTangentSpace(const std::shared_ptr<graph::Mesh>& mesh);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Some Comodity Functions for hard coded scenes /////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<graph::Group> simpleScene01();

    std::shared_ptr<graph::Group> importedScene();
}
#pragma once
#include <memory>
#include "Scene/SIM.h"

namespace vtx::graph
{
	struct VertexAttributes;
	class Group;
	class Node;
	class Mesh;
	class TransformAttribute;
}

namespace vtx::ops {

    template<typename T>
    std::shared_ptr<T> createNode() {
        static_assert(std::is_base_of_v<graph::Node, T>, "Pushed type is not subclass of Node!");
        std::shared_ptr<T> node = std::make_shared<T>();
        graph::SIM::record(node);
        return node;
    }

    // A simple unit cube built from 12 triangles.
    std::shared_ptr<graph::Mesh> createBox();

    void applyTransformation(graph::TransformAttribute& transformation, const math::affine3f& affine);

    std::shared_ptr<graph::Mesh> createPlane();

    void updateMaterialSlots(std::shared_ptr<graph::Mesh> mesh, int removedSlot);

    float gaussianFilter(const float* rgba,
                         const unsigned int width,
                         const unsigned int height,
                         const unsigned int x,
                         const unsigned int y,
                         const bool isSpherical);

    void computeTangents(std::vector<graph::VertexAttributes>& vertices, const std::vector<unsigned int>& indices);


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Some Comodity Functions for hard coded scenes /////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<graph::Group> simpleScene01();

    std::shared_ptr<graph::Group> importedScene();
}
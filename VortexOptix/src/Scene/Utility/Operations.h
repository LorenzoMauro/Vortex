#pragma once
#include <memory>
#include "Scene/Graph.h"
#include "Scene/SIM.h"

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

}
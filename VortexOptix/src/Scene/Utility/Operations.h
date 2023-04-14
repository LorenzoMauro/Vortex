#pragma once
#include <memory>
#include <vector>
#include <algorithm>
#include "Scene/Graph.h"
#include "Core/VortexID.h"
#include "Core/Log.h"
#include "Scene/SIM.h"

namespace vtx {
    using namespace graph;

    namespace ops {


        template<typename T>
        std::shared_ptr<T> createNode() {
            static_assert(std::is_base_of_v<graph::Node, T>, "Pushed type is not subclass of Node!");
            std::shared_ptr<T> node = std::make_shared<T>();
            SIM::Get()->Record(node);
            return node;
        }

        /*static void AddMaterial(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material) {
            auto iter = std::find(mesh->materials.begin(), mesh->materials.end(), material);
            if (iter != mesh->materials.end()) {
                return;
            }
            else {
                mesh->materials.push_back(material);
            }
        }*/

        // A simple unit cube built from 12 triangles.
        static std::shared_ptr<graph::Mesh> createBox()
        {
            VTX_INFO("Creating Box");
            std::shared_ptr<graph::Mesh> mesh = createNode<graph::Mesh>();

            //std::shared_ptr<Mesh> mesh = (std::shared_ptr<Mesh>)nodeRecall;
            {

                mesh->vertices.clear();
                mesh->indices.clear();

                const float left = -1.0f;
                const float right = 1.0f;
                const float bottom = -1.0f;
                const float top = 1.0f;
                const float back = -1.0f;
                const float front = 1.0f;

                graph::VertexAttributes attrib;

                // Left.
                attrib.tangent = math::vec3f(0.0f, 0.0f, 1.0f);
                attrib.normal = math::vec3f(-1.0f, 0.0f, 0.0f);

                attrib.position = math::vec3f(left, bottom, back);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, bottom, front);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, top, front);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, top, back);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                // Right.
                attrib.tangent = math::vec3f(0.0f, 0.0f, -1.0f);
                attrib.normal = math::vec3f(1.0f, 0.0f, 0.0f);

                attrib.position = math::vec3f(right, bottom, front);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, bottom, back);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, back);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, front);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                // Back.  
                attrib.tangent = math::vec3f(-1.0f, 0.0f, 0.0f);
                attrib.normal = math::vec3f(0.0f, 0.0f, -1.0f);

                attrib.position = math::vec3f(right, bottom, back);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, bottom, back);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, top, back);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, back);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                // Front.
                attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
                attrib.normal = math::vec3f(0.0f, 0.0f, 1.0f);

                attrib.position = math::vec3f(left, bottom, front);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, bottom, front);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, front);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, top, front);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                // Bottom.
                attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
                attrib.normal = math::vec3f(0.0f, -1.0f, 0.0f);

                attrib.position = math::vec3f(left, bottom, back);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, bottom, back);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, bottom, front);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, bottom, front);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                // Top.
                attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
                attrib.normal = math::vec3f(0.0f, 1.0f, 0.0f);

                attrib.position = math::vec3f(left, top, front);
                attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, front);
                attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(right, top, back);
                attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                attrib.position = math::vec3f(left, top, back);
                attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
                mesh->vertices.push_back(attrib);

                for (unsigned int i = 0; i < 6; ++i)
                {
                    const unsigned int idx = i * 4; // Four Mesh->vertices per box face.

                    mesh->indices.push_back(idx);
                    mesh->indices.push_back(idx + 1);
                    mesh->indices.push_back(idx + 2);

                    mesh->indices.push_back(idx + 2);
                    mesh->indices.push_back(idx + 3);
                    mesh->indices.push_back(idx);
                }
            }
            return mesh;
        }

        static void applyTransformation(graph::TransformAttribute& transformation, const math::affine3f& affine) {
            transformation.affineTransform = transformation.affineTransform * affine;
            transformation.updateFromAffine();
        }
		
	}
}
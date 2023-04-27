#include "Operations.h"

namespace vtx::ops
{
    using namespace graph;

    // A simple unit cube built from 12 triangles.
    std::shared_ptr<Mesh> ops::createBox()
    {
        VTX_INFO("Creating Box");
        std::shared_ptr<Mesh> mesh = createNode<Mesh>();

        const float left = -1.0f;
        const float right = 1.0f;
        const float bottom = -1.0f;
        const float top = 1.0f;
        const float back = -1.0f;
        const float front = 1.0f;

        VertexAttributes attrib;

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

        FaceAttributes faceAttrib;
        for (unsigned int i = 0; i < 6; ++i)
        {
            const unsigned int idx = i * 4; // Four Mesh->vertices per box face.

            mesh->indices.push_back(idx);
            mesh->indices.push_back(idx + 1);
            mesh->indices.push_back(idx + 2);
            mesh->faceAttributes.push_back(faceAttrib);

            mesh->indices.push_back(idx + 2);
            mesh->indices.push_back(idx + 3);
            mesh->indices.push_back(idx);
            mesh->faceAttributes.push_back(faceAttrib);
        }
        return mesh;
    }

    void applyTransformation(graph::TransformAttribute& transformation, const math::affine3f& affine) {
        transformation.affineTransform = transformation.affineTransform * affine;
        transformation.updateFromAffine();
    }

    std::shared_ptr<graph::Mesh> createPlane()
    {
        VTX_INFO("Creating Plane");
        std::shared_ptr<graph::Mesh> mesh = createNode<graph::Mesh>();

        mesh->vertices.clear();
        mesh->indices.clear();

        math::vec3f corner;

        VertexAttributes attrib;

        // Positive z-axis is the geometry normal, create geometry on the xy-plane.
        corner = math::vec3f(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. texcoord (0.0f, 0.0f).

        attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
        attrib.normal = math::vec3f(0.0f, 0.0f, 1.0f);

        attrib.position = corner + math::vec3f(0.0f, 0.0f, 0.0f);
        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(2.0f, 0.0f, 0.0f);
        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(0.0f, 2.0f, 0.0f);
        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        attrib.position = corner + math::vec3f(2.0f, 2.0f, 0.0f);
        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
        mesh->vertices.push_back(attrib);

        FaceAttributes faceAttrib;
        mesh->indices.push_back(0);
        mesh->indices.push_back(1);
        mesh->indices.push_back(3);
        mesh->faceAttributes.push_back(faceAttrib);

        mesh->indices.push_back(3);
        mesh->indices.push_back(2);
        mesh->indices.push_back(0);
        mesh->faceAttributes.push_back(faceAttrib);

        return mesh;
    }


}


#ifndef GEOMETRY_DATA_H
#define GEOMETRY_DATA_H

#include "Core/VortexID.h"
#include "Scene/DataStructs/VertexAttribute.h"
#include <optix_types.h>

namespace vtx
{

    enum PrimitiveType {
        PT_TRIANGLES,

        NUM_PT
    };

    struct GeometryData {
        PrimitiveType				type;
        OptixTraversableHandle		traversable;
        graph::VertexAttributes* vertexAttributeData;
        graph::FaceAttributes* faceAttributeData;

        vtxID* indicesData;
        size_t						numVertices;
        size_t						numIndices;
        size_t                      numFaces;
    };
}

#endif
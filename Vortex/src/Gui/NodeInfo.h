#pragma once
#include <memory>
#include <string>

#include "Core/VortexID.h"
#include "Core/Math.h"
#include "Scene/Node.h"

namespace vtx::gui
{
    struct LinkInfo
    {
        vtxID linkId;
        vtxID inputSocketId;
        vtxID childOutputSocketId;
        graph::NodeType childNodeType;
    };

    struct NodeInfo
    {
        std::weak_ptr<graph::Node> node;
        graph::NodeType nodeType;
        math::vec2f size{0, 0};
        math::vec2f pos{ 0, 0 };
        int width = -1;
        int depth = -1;
        int         overallWidth;
        bool widthRemapped = false;
        std::vector<LinkInfo> links;
        bool verticalLayout = false;

        std::string title;
        vtxID       id;
    };
}

#pragma once
#include "GuiWindow.h"
#include "GuiElements/SceneGraph.h"

namespace vtx {
    class GraphWindow : public Window {
    public:
        GraphWindow();

        void renderMainContent() override;

        std::shared_ptr<graph::Renderer> renderer;

    };
}

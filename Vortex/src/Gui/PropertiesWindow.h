#pragma once
#include "Gui/GuiWindow.h"
#include "GuiElements/SceneGraph.h"

namespace vtx {
    class PropertiesWindow : public Window {
    public:
        PropertiesWindow();

        void renderMainContent() override;

        std::shared_ptr<graph::Renderer> renderer;
        gui::GuiVisitor                  guiVisitor;
    };
}

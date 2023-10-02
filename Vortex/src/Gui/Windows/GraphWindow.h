#pragma once
#include "Gui/GuiWindow.h"
#include "Gui/NodeEditor.h"

namespace vtx {
    class GraphWindow : public Window {
    public:
        GraphWindow();

        void mainContent() override;

        void menuBarContent() override;

		gui::NodeEditor                 nodeEditor;

    };
}

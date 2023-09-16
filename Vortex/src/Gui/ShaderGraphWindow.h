#pragma once
#include "Gui/GuiWindow.h"
#include "GuiElements/MaterialNodeGui.h"

namespace vtx {
    class ShaderGraphWindow: public Window {
    public:

        ShaderGraphWindow();

        void renderMainContent() override;
    public:
        gui::MaterialGui materialGui;
    };
};

#pragma once
#include "Gui/GuiWindow.h"
namespace vtx {
	class AppWindow : public Window {
    public:

        AppWindow();

        void preRender() override;

        void mainMenuBar();
    };
}

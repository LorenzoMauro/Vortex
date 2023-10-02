#pragma once
#include "Gui/GuiWindow.h"
#include "Gui/GuiVisitor.h"

namespace vtx {
	namespace graph
	{
		class Renderer;
	}

	class PropertiesWindow : public Window {
    public:
        PropertiesWindow();

        void mainContent() override;

        gui::GuiVisitor                  guiVisitor;
    };
}

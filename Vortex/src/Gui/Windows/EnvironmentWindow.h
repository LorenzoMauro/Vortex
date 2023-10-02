#pragma once
#include "Scene/Nodes/Renderer.h"
#include "Gui/GuiWindow.h"

namespace vtx {

    class EnvironmentWindow : public Window {
    public:

        EnvironmentWindow();

        virtual void OnUpdate(float ts) override;

        virtual void mainContent() override;

        virtual void menuBarContent() override;

    public:
        GLuint glEnvironmentTexture;
		std::string    texturePath;
        int width;
        int height;
        bool isReady = false;
	};
}

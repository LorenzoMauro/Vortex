#pragma once
#include "Layers/GuiLayer.h"
#include "GuiElements/MaterialNodeGui.h"

namespace vtx {
    class MaterialEditorLayer : public Layer {
    public:

        MaterialEditorLayer();

        virtual void OnAttach();

        virtual void OnDetach();

        virtual void OnUpdate(float ts);

        virtual void OnUIRender();

    public:
        gui::MaterialGui materialGui;
    };
};

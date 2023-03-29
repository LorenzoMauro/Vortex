#pragma once
#include "Renderer/Renderer.h"
#include "Layers/GuiLayer.h"
#include "Core/Options.h"

namespace vtx {
    class ViewportLayer : public Layer {
    public:

        ViewportLayer(Renderer* _Renderer);

        virtual void OnAttach();

        virtual void OnDetach();

        virtual void OnUpdate(float ts);

        virtual void OnUIRender();

    public:
        Renderer*   renderer;
        uint32_t    m_width = g_option.width;
        uint32_t    m_height = g_option.height;
        bool        m_isResized = false;
    };
}

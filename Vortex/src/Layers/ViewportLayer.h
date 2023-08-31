#pragma once
#include "Scene/Nodes/Renderer.h"
#include "Layers/GuiLayer.h"
#include "Core/Options.h"
#include "Device/DeviceVisitor.h"
#include "GuiElements/MaterialNodeGui.h"
#include "Scene/HostVisitor.h"

namespace vtx {
    class ViewportLayer : public Layer {
    public:

        ViewportLayer();

        virtual void OnAttach();

        virtual void OnDetach();

        virtual void OnUpdate(float ts);

        virtual void OnUIRender();

    public:
        std::shared_ptr<graph::Renderer> renderer;
        uint32_t    m_width = getOptions()->width;
        uint32_t    m_height = getOptions()->height;
        math::vec2i forcedSize = math::vec2i(0, 0);
        bool        m_isResized = false;
        device::DeviceVisitor deviceVisitor;
        HostVisitor hostVisitor;
        //gui::MaterialGui materialGui;
    };
}

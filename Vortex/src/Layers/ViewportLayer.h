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

        ViewportLayer(std::shared_ptr<graph::Renderer> _Renderer);

        virtual void OnAttach();

        virtual void OnDetach();

        virtual void OnUpdate(float ts);

        virtual void OnUIRender();

    public:
        std::shared_ptr<graph::Renderer> renderer;
        uint32_t    m_width = getOptions()->width;
        uint32_t    m_height = getOptions()->height;
        bool        m_isResized = false;
        std::shared_ptr<device::DeviceVisitor> deviceVisitor;
        std::shared_ptr<HostVisitor> hostVisitor;
        gui::MaterialGui materialGui;
    };
}

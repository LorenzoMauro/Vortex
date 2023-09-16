#pragma once
#include "Scene/Nodes/Renderer.h"
#include "Gui/GuiWindow.h"
#include "Core/Options.h"
#include "Device/DeviceVisitor.h"
#include "GuiElements/MaterialNodeGui.h"
#include "Scene/HostVisitor.h"

namespace vtx {
    class ViewportWindow : public Window {
    public:

        ViewportWindow();

        virtual void OnUpdate(float ts) override;

        virtual void renderMainContent() override;

        virtual void preRender() override;

    public:
        std::shared_ptr<graph::Renderer> renderer;
        uint32_t    m_width = getOptions()->width;
        uint32_t    m_height = getOptions()->height;
        math::vec2i forcedSize = math::vec2i(0, 0);
        bool        m_isResized = false;
        device::DeviceVisitor deviceVisitor;
        HostVisitor hostVisitor;
    };
}

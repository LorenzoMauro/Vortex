#pragma once
#include "GuiWindow.h"

namespace vtx {
    class LoadingWindow : public Window {
    public:

        LoadingWindow();

        virtual void preRender() override;

        virtual void OnUpdate(float ts) override;

        virtual void renderMainContent() override;

    public:
    };

    class SavingWindow : public Window {
    public:

        SavingWindow();

        virtual void preRender() override;

        virtual void OnUpdate(float ts) override;

        virtual void renderMainContent() override;

    public:
    };
}

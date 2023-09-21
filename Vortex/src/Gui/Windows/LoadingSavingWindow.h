#pragma once
#include "Gui/GuiWindow.h"

namespace vtx {
    class LoadingWindow : public Window {
    public:

        LoadingWindow();

        virtual void preRender() override;

        virtual void OnUpdate(float ts) override;

        virtual void mainContent() override;

    public:
    };

    class SavingWindow : public Window {
    public:

        SavingWindow();

        virtual void preRender() override;

        virtual void OnUpdate(float ts) override;

        virtual void mainContent() override;

    public:
    };
}

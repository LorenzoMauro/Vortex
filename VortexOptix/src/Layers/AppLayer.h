#pragma once
#include "GuiLayer.h"
#include <functional>

namespace vtx {
	class AppLayer : public Layer {
    public:

        AppLayer();

		virtual void OnAttach();

		virtual void OnDetach();

        virtual void OnUIRender();

    public:
        std::function<void()>   m_MenubarCallback;
    };
}

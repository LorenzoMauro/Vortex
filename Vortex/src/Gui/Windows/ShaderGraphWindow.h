#pragma once
#include "Gui/GuiWindow.h"
#include "Gui/NodeEditor.h"

namespace vtx {
    class ShaderGraphWindow: public Window {
    public:

        ShaderGraphWindow();

        void mainContent() override;

        void materialSelector();

        void menuBarContent() override;

    public:
        gui::NodeEditor                                 nodeEditor;
        std::map<vtxID, std::map<vtxID, gui::NodeInfo>> nodeInfoByMaterial;
        std::map<vtxID, ImNodesEditorContext*>          editorContextByMaterial;
        vtxID                                           materialOpened = 0;
        vtxID                                           previousMaterial = 0;
        std::string                                     openedMaterialName;
        bool                                            materialOpenedChanged = false;

    };
};

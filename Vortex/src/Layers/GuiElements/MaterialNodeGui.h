#pragma once
#include <memory>
#include <vector>

#include "Core/VortexID.h"


namespace vtx::graph
{
	class Material;
}

namespace vtx::gui
{
    class MaterialGui
    {
    public:
        vtxID                                           selectedMaterialId = 0;
        std::vector<std::shared_ptr< graph::Material >> materials;

        void refreshMaterialList();
        bool materialSelector();
        void materialGui();
        static void materialNodeGui(const vtxID materialId);

    };
	
}
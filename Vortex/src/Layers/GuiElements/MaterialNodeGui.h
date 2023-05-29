#pragma once
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "Scene/Node.h"
#include "Core/VortexID.h"
#include "Core/Math.h"


namespace vtx::graph
{
	class Material;
}

namespace vtx::gui
{
    struct NodeInfo
    {
        std::shared_ptr<graph::shader::ShaderNode> shaderNode;
        math::vec2f size{0,0};
        math::vec2i pos{ 0,0 };
        int width = 0;
        int depth = 0;
	};

    class MaterialGui
    {
    public:
        vtxID                                           selectedMaterialId = 0;
        std::vector<std::shared_ptr< graph::Material >> materials;
        std::map<vtxID, std::map<vtxID, NodeInfo>>      nodeInfo;
        vtxID                                           materialOpened;

        void refreshMaterialList();
        bool materialSelector();
        void materialGui();


        bool shaderNodeGui(std::shared_ptr<graph::shader::ShaderNode> shaderNode, std::shared_ptr<graph::Material> material);
        bool socketGui(graph::shader::ShaderNodeSocket& socket, std::shared_ptr<graph::Material> material);
        bool parameterGui(graph::shader::ParameterInfo& param, std::shared_ptr<graph::Material> material);

        bool inputSocketNodeGui(std::shared_ptr<graph::shader::ShaderNode> inputSocketNode, std::shared_ptr<graph::Material> material);

        bool nodeEditorShaderNodeGui(std::shared_ptr<graph::shader::ShaderNode> shaderNode, std::shared_ptr<graph::Material> material);

        void nodeEditorShaderNodeGuiLink(std::shared_ptr<graph::shader::ShaderNode> shaderNode);

        void materialNodeEditorGui(const vtxID materialId);
        void materialNodeGui(const vtxID materialId);

    };
	
}
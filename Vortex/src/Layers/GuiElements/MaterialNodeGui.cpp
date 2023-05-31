#include "MaterialNodeGui.h"

#include "imgui.h"
#include "imnodes.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Material.h"
#include "Scene/Nodes/Renderer.h"
#include "Scene/Nodes/Shader/Texture.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::gui {
    void MaterialGui::refreshMaterialList()
    {
        std::vector<std::shared_ptr<graph::Material>> retMaterials = graph::SIM::getAllNodeOfType<graph::Material>(
            graph::NT_MATERIAL);

        materials.clear();
        //TODO this is just a hack
        for (std::shared_ptr<graph::Material> mat : retMaterials)
        {
            if (!mat->materialGraph->name.empty() && mat->materialGraph->name != "")
            {
                materials.push_back(mat);
            }
        }
    }

    bool MaterialGui::materialSelector()
    {
        refreshMaterialList();
        // My material reference names are unique identifiers.
        std::shared_ptr<graph::Material> selectedMaterial = nullptr;
        if (selectedMaterialId != 0)
        {
            selectedMaterial = graph::SIM::getNode<graph::Material>(selectedMaterialId);
        }
        else if ((selectedMaterialId == 0 || !selectedMaterial) && !materials.empty())
        {
            selectedMaterial = materials[0];
            selectedMaterialId = selectedMaterial->getID();
        }
        if (selectedMaterialId == 0 || selectedMaterial == nullptr)
        {
            return false;
        }

        const std::string labelCombo = selectedMaterial->materialGraph->name;

        if (ImGui::BeginCombo("Reference", labelCombo.c_str()))
        {
            // add selectable materials to the combo box
            for (size_t i = 0; i < materials.size(); ++i)
            {
                const bool isSelected = materials[i]->getID() == selectedMaterialId;
                //const bool isSelected = (i == imguiIndexMaterial);

                std::string label = materials[i]->materialGraph->name;

                if (ImGui::Selectable(label.c_str(), isSelected))
                {
                    selectedMaterialId = materials[i]->getID();
                }
                if (isSelected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        return true;
    }

    void MaterialGui::materialGui()
    {
        ImGui::Begin("Materials Settings");
        if (materialSelector())
        {
            materialNodeGui(selectedMaterialId);
        }

        ImGui::End();
    }

    bool MaterialGui::parameterGui(graph::shader::ParameterInfo& param, std::shared_ptr<graph::Material> material)
    {
        std::string HiddenIdentifier = "##hidden" + param.annotation.displayName;

        //ImGui::Columns(2, "mycolumns"); // Begin 2 columns
        //ImGui::SetColumnWidth(-1, 0); // Set the first column width

        //ImGui::Text(param.annotation.displayName.c_str());
        //ImGui::NextColumn();
        //ImGui::SetColumnWidth(-1, 200); // Set the second column width

        // Ensure unique ID even for parameters with same display names.
        bool changed = false;
        // Choose proper edit control depending on the parameter kind
        switch (param.kind)
        {
        case graph::shader::ParameterInfo::PK_FLOAT:

				changed |= vtxImGui::HalfSpaceWidget(param.annotation.displayName.c_str(), ImGui::SliderFloat, HiddenIdentifier.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1], "%.3f", 0);
                //changed |= vtxImGui::SliderFloat(param.annotation.displayName.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1]);
                break;
            case graph::shader::ParameterInfo::PK_FLOAT2:
                changed |= ImGui::SliderFloat2(param.annotation.displayName.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1]);
                break;
            case graph::shader::ParameterInfo::PK_FLOAT3:
                changed |= ImGui::SliderFloat3(param.annotation.displayName.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1]);
                break;
			case graph::shader::ParameterInfo::PK_COLOR:
                //changed |= vtxImGui::ColorEdit3NoInputs(param.annotation.displayName.c_str(), &param.data<float>());
                changed |= vtx::vtxImGui::HalfSpaceWidget(param.annotation.displayName.c_str(), vtxImGui::colorPicker, HiddenIdentifier.c_str(), &param.data<float>());
                break;
            case graph::shader::ParameterInfo::PK_BOOL:
                changed |= vtxImGui::HalfSpaceWidget(param.annotation.displayName.c_str(), ImGui::Checkbox, HiddenIdentifier.c_str(), &param.data<bool>());
                //changed |= ImGui::Checkbox(param.annotation.displayName.c_str(), &param.data<bool>());
                break;
            case graph::shader::ParameterInfo::PK_INT:
                changed |= vtxImGui::HalfSpaceWidget(param.annotation.displayName.c_str(), ImGui::SliderInt, HiddenIdentifier.c_str(), &param.data<int>(), int(param.annotation.range[0]), int(param.annotation.range[1]), "%d", 0);
                //changed |= ImGui::SliderInt(param.annotation.displayName.c_str(), &param.data<int>(), int(param.annotation.range[0]), int(param.annotation.range[1]));
                break;
            case graph::shader::ParameterInfo::PK_ARRAY:
            {
                ImGui::Text("%s", param.annotation.displayName.c_str());
                ImGui::Indent(16.0f);

                char* ptr = &param.data<char>();

                for (mi::Size i = 0, n = param.arraySize; i < n; ++i)
                {
                    std::string idxStr = std::to_string(i);

                    switch (param.arrayElemKind)
                    {
                        case graph::shader::ParameterInfo::PK_FLOAT:
                            changed |= ImGui::SliderFloat(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
                            break;
                        case graph::shader::ParameterInfo::PK_FLOAT2:
                            changed |= ImGui::SliderFloat2(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
                            break;
                        case graph::shader::ParameterInfo::PK_FLOAT3:
                            changed |= ImGui::SliderFloat3(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
                            break;
                        case graph::shader::ParameterInfo::PK_COLOR:
                            changed |= ImGui::ColorEdit3(idxStr.c_str(), reinterpret_cast<float*>(ptr));
                            break;
                        case graph::shader::ParameterInfo::PK_BOOL:
                            changed |= vtxImGui::HalfSpaceWidget(param.annotation.displayName.c_str(), ImGui::Checkbox, HiddenIdentifier.c_str(), reinterpret_cast<bool*>(ptr));
                            //changed |= ImGui::Checkbox(param.annotation.displayName.c_str(), reinterpret_cast<bool*>(ptr));
                            break;
                        case graph::shader::ParameterInfo::PK_INT:
                            changed |= ImGui::SliderInt(param.annotation.displayName.c_str(), reinterpret_cast<int*>(ptr), int(param.annotation.range[0]), int(param.annotation.range[1]));
                            break;
                        default:
                            std::cerr << "ERROR: guiWindow() Material parameter " << param.annotation.displayName.c_str() << " array element " << idxStr << " type invalid or unhandled.\n";
                    }
                    ptr += param.arrayPitch;
                }
                ImGui::Unindent(16.0f);
            }
            break;
            case graph::shader::ParameterInfo::PK_ENUM:
            {
                const int value = param.data<int>();
                std::string currentValue;

                const graph::shader::EnumTypeInfo* info = param.enumInfo;
                for (size_t i = 0, n = info->values.size(); i < n; ++i)
                {
                    if (info->values[i].value == value)
                    {
                        currentValue = info->values[i].name;
                        break;
                    }
                }

                if (ImGui::BeginCombo(param.annotation.displayName.c_str(), currentValue.c_str()))
                {
                    for (size_t i = 0, n = info->values.size(); i < n; ++i)
                    {
                        const std::string& name = info->values[i].name;

                        const bool isSelected = (currentValue == name);

                        if (ImGui::Selectable(info->values[i].name.c_str(), isSelected))
                        {
                            param.data<int>() = info->values[i].value;
                            changed = true;
                        }
                        if (isSelected)
                        {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
            }
            break;
            case graph::shader::ParameterInfo::PK_STRING:
            case graph::shader::ParameterInfo::PK_TEXTURE:
            {

                int textureId = param.data<int>();
                std::string inputText = material->textures[textureId - 1]->filePath;

            	vtxImGui::HalfSpaceWidget("Texture:", vtxImGui::ClippedText, inputText.c_str());
            }
            case graph::shader::ParameterInfo::PK_LIGHT_PROFILE:
            case graph::shader::ParameterInfo::PK_BSDF_MEASUREMENT:
                // Currently not supported by this example
                break;
            default:
                break;
        }
        //ImGui::Columns(1); // End columns
        return changed;
    }

    bool MaterialGui::inputSocketNodeGui(std::shared_ptr<graph::shader::ShaderNode> inputSocketNode, std::shared_ptr<graph::Material> material)
    {

        ImNodes::BeginOutputAttribute(inputSocketNode->getID());



        ImNodes::EndOutputAttribute();

        return false;
    }

    bool MaterialGui::socketGui(graph::shader::ShaderNodeSocket& socket, std::shared_ptr<graph::Material> material)
    {
        bool changed = false;
        if (socket.node)
        {
            ImGui::Text("%s", socket.parameterInfo.annotation.displayName.c_str());
            changed |= shaderNodeGui(socket.node, material);
        }
        else
        {
            changed |= parameterGui(socket.parameterInfo, material);
        }
        return changed;
    }

    bool MaterialGui::shaderNodeGui(std::shared_ptr<graph::shader::ShaderNode> shaderNode, std::shared_ptr<graph::Material> material)
    {
        bool changed = false;
        int id = 0;

        if (ImGui::CollapsingHeader(("Node: " + shaderNode->name).c_str()))
        {
            ImGui::Indent();
            for (auto [groupName, sockets] : shaderNode->socketsGroupedByGroup)
            {
                ImGui::PushID(shaderNode->getID());
                if (ImGui::CollapsingHeader(groupName.c_str()))
                {
                    for (auto& socketName : sockets)
                    {
                        if (shaderNode->sockets.count(socketName) > 0) {
                            ImGui::PushID(id);
                            graph::shader::ShaderNodeSocket& socket = shaderNode->sockets[socketName];

                            changed |= socketGui(socket, material);
                            ImGui::PopID();
                            id++;
                        }
                        else
                        {
                            VTX_INFO("GUI ERROR - socket not found {}", socketName);
                        }
                    }
                }
                ImGui::PopID();
            }
            ImGui::Unindent();
        }

        return changed;
    }

    bool MaterialGui::nodeEditorShaderNodeGui(std::shared_ptr<graph::shader::ShaderNode> shaderNode, std::shared_ptr<graph::Material> material)
    {
        bool changed = false;
        int id = 0;

        ImNodes::BeginNode(shaderNode->getID());

        NodeInfo& info = nodeInfo[material->getID()][shaderNode->getID()];
        info.size.x = std::max(info.size.x, getOptions()->nodeWidth);  // Enforce a minimum size
        info.size.y = std::max(info.size.y, getOptions()->nodeWidth);

        ImNodes::BeginNodeTitleBar();
        std::string functionSignature = shaderNode->functionInfo.name;
        std::string title = (utl::splitString(functionSignature, "::")).back();

        ImGui::TextUnformatted(title.c_str());
        ImNodes::EndNodeTitleBar();


        //ImGui::Text("Function: %s", functionSignature.c_str());

        ImGui::PushItemWidth(info.size.x); // Set the width of the next widget to 200

        vtxImGui::HalfSpaceWidget("Function Signature", vtxImGui::ClippedText, functionSignature.c_str());
        vtxImGui::HalfSpaceWidget("Node Id:", vtxImGui::ClippedText, std::to_string(shaderNode->getID()).c_str());

        ImNodes::BeginOutputAttribute(shaderNode->outputSocket.Id);
        vtxImGui::HalfSpaceWidget("Output Type", vtxImGui::ClippedText, shaderNode->outputSocket.parameterInfo.annotation.displayName.c_str());
        ImNodes::EndInputAttribute();

        //TODO: Some socket appear with no description
        for(auto& [SocketGroup, socketGroupName] : shaderNode->socketsGroupedByGroup)
        {
            vtxImGui::HalfSpaceWidget(" ", vtxImGui::booleanText, " ");
            vtxImGui::HalfSpaceWidget("Group:", vtxImGui::ClippedText, SocketGroup.c_str());
            for (auto& socketName : socketGroupName)
            {
                std::string name = socketName;
                auto& socket = shaderNode->sockets[name];

                if (socket.node)
                {
                    ImNodes::BeginInputAttribute(socket.Id);
                    ImGui::Text(name.c_str());

                    ImNodes::EndInputAttribute();
                }
                else
                {
                    if (!socket.parameterInfo.annotation.displayName.empty())
                    {
                        ImNodes::BeginInputAttribute(socket.Id);
                        changed |= parameterGui(socket.parameterInfo, material);
                        ImNodes::EndInputAttribute();
                    }
                }
            }
        }


        ImGui::PopItemWidth(); // Resets the width for the widgets that will be declared later

        ImNodes::EndNode();
        // Invisible button for corner dragging
        ImVec2 corner = ImGui::GetItemRectMax();  // Get the maximum point of the last drawn item (should be the last input/output attribute)
        int cornerSize = 20;  // Size of the draggable corner
        corner.x -= cornerSize;  // Make sure the corner does not overlap the node content
        corner.y -= cornerSize;

        ImGui::SetCursorScreenPos(corner);
        ImGui::InvisibleButton(("cornerbutton" + std::to_string(shaderNode->getID())).c_str(), ImVec2((float)cornerSize, (float)cornerSize));

        if (ImGui::IsMouseHoveringRect(corner, ImVec2(corner.x + cornerSize, corner.y + cornerSize))) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        }

        if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
        {
            // Update node size based on mouse drag delta
            info.size.x += (int)ImGui::GetIO().MouseDelta.x;
            info.size.y += (int)ImGui::GetIO().MouseDelta.y;
        }
        //ImGui::GetWindowDrawList()->AddRectFilled(corner, ImVec2(corner.x + cornerSize, corner.y + cornerSize), IM_COL32(255, 0, 0, 255));

        // Dummy item with the size of the node
        ImGui::SetCursorScreenPos(ImGui::GetItemRectMin());  // Set cursor to the minimum point of the last drawn item (should be the last input/output attribute)
        ImGui::Dummy(ImVec2((float)info.size.x, (float)info.size.y));

        return changed;
    }

    void MaterialGui::nodeEditorShaderNodeGuiLink(std::shared_ptr<graph::shader::ShaderNode> shaderNode)
    {
        for (auto& [name, socket] : shaderNode->sockets)
        {
            if (socket.node)
            {
                ImNodes::Link(socket.linkId, socket.node->outputSocket.Id, socket.Id);
            }
        }
    }

    void collectShaderNodeGraph(std::shared_ptr<graph::shader::ShaderNode> shaderNode, std::map<vtxID, NodeInfo>& nodeInfo,int width = 0, int currentDepth = 0)
    {
        if(nodeInfo.count(shaderNode->getID()) == 0)
        {
            nodeInfo.insert({ shaderNode->getID(), NodeInfo{} });
            nodeInfo[shaderNode->getID()].shaderNode = shaderNode;
            nodeInfo[shaderNode->getID()].depth = currentDepth;
            nodeInfo[shaderNode->getID()].width = width;
        }

        for (auto& [name, socket] : shaderNode->sockets)
        {
            if (socket.node != nullptr && nodeInfo.count(socket.node->getID()) == 0)
            {
                collectShaderNodeGraph(socket.node, nodeInfo, width, currentDepth + 1);
            }
            width++;
        }
    }
    void arrangeNodes(std::map<vtxID, NodeInfo>& nodeInfoMap)
    {
        const float padding = 50.0f;  // Some padding to avoid nodes touching each other

        // Find max depth to position nodes from right to left
        int maxDepth = 0;
        for (const auto& [id, nodeInfo] : nodeInfoMap)
        {
            if (nodeInfo.depth > maxDepth)
                maxDepth = nodeInfo.depth;
        }

        // Create a map for each depth, to map old widths to new, continuous widths
        // Also, keep track of the max node dimensions at each depth level
        std::map<int, std::map<int, int>> widthRemapping;
        std::map<int, ImVec2> maxNodeDimensionsAtDepth;
        for (const auto& [id, nodeInfo] : nodeInfoMap)
        {
            widthRemapping[nodeInfo.depth][nodeInfo.width] = 0;
            ImVec2  nodeDim                                = ImNodes::GetNodeDimensions(id);
            ImVec2& maxDimAtDepth                          = maxNodeDimensionsAtDepth[nodeInfo.depth];
            maxDimAtDepth.x                                = std::max(maxDimAtDepth.x, nodeDim.x);
            maxDimAtDepth.y                                = std::max(maxDimAtDepth.y, nodeDim.y);
        }

        // Generate new widths
        for (auto& [depth, widthMap] : widthRemapping)
        {
            int newWidth = 0;
            for (auto& [oldWidth, _] : widthMap)
            {
                widthMap[oldWidth] = newWidth++;
            }
        }

        // Arrange nodes using new widths, considering node dimensions
        for (auto& [id, nodeInfo] : nodeInfoMap)
        {
            int newWidth = widthRemapping[nodeInfo.depth][nodeInfo.width];
            ImVec2 maxDimAtDepth = maxNodeDimensionsAtDepth[nodeInfo.depth];
            float horizontalSpacing = maxDimAtDepth.x + padding;
            float verticalSpacing = maxDimAtDepth.y + padding;
            ImVec2 pos((maxDepth - nodeInfo.depth) * horizontalSpacing, newWidth * verticalSpacing);
            ImNodes::SetNodeGridSpacePos(id, pos);
        }
    }

    void MaterialGui::materialNodeEditorGui(const vtxID materialId)
    {
        const std::shared_ptr<graph::Material> material = graph::SIM::getNode<graph::Material>(materialId);


        bool changed = false;

        std::shared_ptr<graph::shader::ShaderNode> shaderNode = material->materialGraph;

        std::map<vtxID, NodeInfo>& materialNodesInfos = nodeInfo[material->getID()];
        collectShaderNodeGraph(shaderNode, materialNodesInfos, 0, 0);

        for(auto& [id, materialNodesInfo] : materialNodesInfos)
        {
        	changed |= nodeEditorShaderNodeGui(materialNodesInfo.shaderNode, material);
		}


        if (materialOpened != material->getID()) {
            materialOpened = material->getID();
        	arrangeNodes(materialNodesInfos);
        };

        for (auto& [id, materialNodesInfo] : materialNodesInfos)
        {
            nodeEditorShaderNodeGuiLink(materialNodesInfo.shaderNode);
        }

        if (changed)
        {
            material->isUpdated = true;
            auto renderers = graph::SIM::getAllNodeOfType<graph::Renderer>(graph::NT_RENDERER);

            for (auto renderer : renderers)
            {
                renderer->settings.isUpdated = true;
                renderer->settings.iteration = -1;
            }

        }
    }

    void MaterialGui::materialNodeGui(const vtxID materialId)
    {
        const std::shared_ptr<graph::Material> material = graph::SIM::getNode<graph::Material>(materialId);

        bool changed = false;


		std::shared_ptr<graph::shader::ShaderNode> shaderNode = material->materialGraph;

        changed |= shaderNodeGui(shaderNode, material);
        
        if (changed)
        {
            material->isUpdated = true;
            auto renderers = graph::SIM::getAllNodeOfType<graph::Renderer>(graph::NT_RENDERER);

            for (auto renderer : renderers)
            {
                renderer->settings.isUpdated = true;
                renderer->settings.iteration = -1;
            }

        }
    }

}

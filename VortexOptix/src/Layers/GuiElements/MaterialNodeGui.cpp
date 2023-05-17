#include "MaterialNodeGui.h"

#include "imgui.h"
#include "Scene/SIM.h"
#include "Scene/Nodes/Material.h"
#include "Scene/Nodes/Renderer.h"

namespace vtx::gui
{
    void MaterialGui::refreshMaterialList()
    {
        materials = graph::SIM::getAllNodeOfType<graph::Material>(graph::NT_MATERIAL);
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

        const std::string labelCombo = selectedMaterial->shader->name;

        if (ImGui::BeginCombo("Reference", labelCombo.c_str()))
        {
            // add selectable materials to the combo box
            for (size_t i = 0; i < materials.size(); ++i)
            {
                const bool isSelected = materials[i]->getID() == selectedMaterialId;
                //const bool isSelected = (i == imguiIndexMaterial);

                std::string label = materials[i]->shader->name;

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

    void MaterialGui::materialNodeGui(const vtxID materialId)
    {
        const std::shared_ptr<graph::Material> material = graph::SIM::getNode<graph::Material>(materialId);

        const char* groupName = nullptr;

        bool changed = false;

        int id = 0;
        for (auto it = material->params.begin(), end = material->params.end(); it != end; ++it, ++id)
        {
            graph::ParamInfo& param = *it;

            // Ensure unique ID even for parameters with same display names.
            ImGui::PushID(id);

            // Group name changed? Start new group with new header.
            if ((!param.groupName() != !groupName) ||
                (param.groupName() && (!groupName || strcmp(groupName, param.groupName()) != 0)))
            {
                ImGui::Separator();

                if (param.groupName() != nullptr)
                {
                    ImGui::Text("%s", param.groupName());
                }

                groupName = param.groupName();
            }

            // Choose proper edit control depending on the parameter kind
            switch (param.kind())
            {
                case graph::ParamInfo::PK_FLOAT:
                    changed |= ImGui::SliderFloat(param.displayName(), &param.data<float>(), param.rangeMin(), param.rangeMax());
                    break;
                case graph::ParamInfo::PK_FLOAT2:
                    changed |= ImGui::SliderFloat2(param.displayName(), &param.data<float>(), param.rangeMin(), param.rangeMax());
                    break;
                case graph::ParamInfo::PK_FLOAT3:
                    changed |= ImGui::SliderFloat3(param.displayName(), &param.data<float>(), param.rangeMin(), param.rangeMax());
                    break;
                case graph::ParamInfo::PK_COLOR:
                    changed |= ImGui::ColorEdit3(param.displayName(), &param.data<float>());
                    break;
                case graph::ParamInfo::PK_BOOL:
                    changed |= ImGui::Checkbox(param.displayName(), &param.data<bool>());
                    break;
                case graph::ParamInfo::PK_INT:
                    changed |= ImGui::SliderInt(param.displayName(), &param.data<int>(), int(param.rangeMin()), int(param.rangeMax()));
                    break;
                case graph::ParamInfo::PK_ARRAY:
                {
                    ImGui::Text("%s", param.displayName());
                    ImGui::Indent(16.0f);

                    char* ptr = &param.data<char>();

                    for (mi::Size i = 0, n = param.arraySize(); i < n; ++i)
                    {
                        std::string idxStr = std::to_string(i);

                        switch (param.arrayElemKind())
                        {
                            case graph::ParamInfo::PK_FLOAT:
                                changed |= ImGui::SliderFloat(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.rangeMin(), param.rangeMax());
                                break;
                            case graph::ParamInfo::PK_FLOAT2:
                                changed |= ImGui::SliderFloat2(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.rangeMin(), param.rangeMax());
                                break;
                            case graph::ParamInfo::PK_FLOAT3:
                                changed |= ImGui::SliderFloat3(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.rangeMin(), param.rangeMax());
                                break;
                            case graph::ParamInfo::PK_COLOR:
                                changed |= ImGui::ColorEdit3(idxStr.c_str(), reinterpret_cast<float*>(ptr));
                                break;
                            case graph::ParamInfo::PK_BOOL:
                                changed |= ImGui::Checkbox(param.displayName(), reinterpret_cast<bool*>(ptr));
                                break;
                            case graph::ParamInfo::PK_INT:
                                changed |= ImGui::SliderInt(param.displayName(), reinterpret_cast<int*>(ptr), int(param.rangeMin()), int(param.rangeMax()));
                                break;
                            default:
                                std::cerr << "ERROR: guiWindow() Material parameter " << param.displayName() << " array element " << idxStr << " type invalid or unhandled.\n";
                        }
                        ptr += param.arrayPitch();
                    }
                    ImGui::Unindent(16.0f);
                }
                break;
                case graph::ParamInfo::PK_ENUM:
                {
                    const int value = param.data<int>();
                    std::string currentValue;

                    const graph::EnumTypeInfo* info = param.enumInfo();
                    for (size_t i = 0, n = info->values.size(); i < n; ++i)
                    {
                        if (info->values[i].value == value)
                        {
                            currentValue = info->values[i].name;
                            break;
                        }
                    }

                    if (ImGui::BeginCombo(param.displayName(), currentValue.c_str()))
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
                case graph::ParamInfo::PK_STRING:
                case graph::ParamInfo::PK_TEXTURE:
                case graph::ParamInfo::PK_LIGHT_PROFILE:
                case graph::ParamInfo::PK_BSDF_MEASUREMENT:
                    // Currently not supported by this example
                    break;
                default:
                    break;
            }

            ImGui::PopID();
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

}

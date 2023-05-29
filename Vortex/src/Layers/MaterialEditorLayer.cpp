#include "MaterialEditorLayer.h"

#include "imgui.h"
#include "GuiElements/MaterialNodeGui.h"
#include "Scene/Nodes/Material.h"
#include "imnodes.h"

namespace vtx {
    MaterialEditorLayer::MaterialEditorLayer()
    {
        materialGui = gui::MaterialGui();
    }

    void MaterialEditorLayer::OnAttach()
    {
    }

    void MaterialEditorLayer::OnDetach()
    {
    }

    void MaterialEditorLayer::OnUpdate(float ts)
    {
        //graph::computeMaterialsMultiThreadCode();
    }

    void ImNodesStyleEditor()
    {
        static ImNodesStyle& style = ImNodes::GetStyle();

        ImGui::Text("Node Editor Style");

        // Edit style variables
        ImGui::SliderFloat("GridSpacing", &style.GridSpacing, 10.0f, 50.0f);
        ImGui::SliderFloat("NodeCornerRounding", &style.NodeCornerRounding, 0.0f, 10.0f);
        ImGui::SliderFloat2("NodePadding", reinterpret_cast<float*>(&style.NodePadding), 0.0f, 20.0f);
        ImGui::SliderFloat("NodeBorderThickness", &style.NodeBorderThickness, 0.0f, 5.0f);
        ImGui::SliderFloat("LinkThickness", &style.LinkThickness, 0.0f, 5.0f);
        ImGui::SliderFloat("LinkLineSegmentsPerLength", &style.LinkLineSegmentsPerLength, 0.1f, 1.0f);
        ImGui::SliderFloat("LinkHoverDistance", &style.LinkHoverDistance, 0.0f, 20.0f);
        ImGui::SliderFloat("PinCircleRadius", &style.PinCircleRadius, 1.0f, 10.0f);
        ImGui::SliderFloat("PinQuadSideLength", &style.PinQuadSideLength, 1.0f, 10.0f);
        ImGui::SliderFloat("PinTriangleSideLength", &style.PinTriangleSideLength, 1.0f, 10.0f);
        ImGui::SliderFloat("PinLineThickness", &style.PinLineThickness, 0.1f, 5.0f);
        ImGui::SliderFloat("PinHoverRadius", &style.PinHoverRadius, 0.0f, 20.0f);
        ImGui::SliderFloat("PinOffset", &style.PinOffset, 0.0f, 20.0f);
        ImGui::SliderFloat2("MiniMapPadding", reinterpret_cast<float*>(&style.MiniMapPadding), 0.0f, 20.0f);
        ImGui::SliderFloat2("MiniMapOffset", reinterpret_cast<float*>(&style.MiniMapOffset), 0.0f, 100.0f);

        // Edit style flags
        ImGui::Text("Style Flags");
        ImGui::CheckboxFlags("NodeOutline", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_NodeOutline);
        ImGui::CheckboxFlags("GridLines", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLines);
        ImGui::CheckboxFlags("GridLinesPrimary", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLinesPrimary);
        ImGui::CheckboxFlags("GridSnapping", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridSnapping);
    }

    void MaterialEditorLayer::OnUIRender() {

        ImGui::Begin("node editor");

        //ImNodesStyleEditor();
        materialGui.materialSelector();

        ImNodes::BeginNodeEditor();
        materialGui.materialNodeEditorGui(materialGui.selectedMaterialId);
        ImNodes::EndNodeEditor();

        ImGui::End();
    }
}

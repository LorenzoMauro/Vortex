#include "ImGuiOp.h"
#include "Core/Utils.h"
#include "Core/Log.h"
#include "imgui.h"
#include "Core/ImguiBackEnds/imgui_impl_glfw.h"
#include "Core/ImguiBackEnds/imgui_impl_opengl3.h"
#include "Core/Options.h"
#include "glad/glad.h"

namespace vtx {
    void Init_ImGui(GLFWwindow* window) {
        VTX_INFO("Starting ImGui");
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.DeltaTime = 1.0f / 1000000.0f;
        (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;
        io.IniFilename = nullptr;

        ImGui::StyleColorsDark();
        // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
        ImGuiStyle& style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        // Send Vulkan Info to ImGui
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        const char* glsl_version = "#version 130";
        ImGui_ImplOpenGL3_Init(glsl_version);

        // Upload Fonts
        //LoadFonts();

        ImGui::LoadIniSettingsFromDisk(utl::absolutePath(getOptions()->imGuiIniFile).data());

        SetAppStyle();
    }

    void SetAppStyle() {
        ImVec4* colors = ImGui::GetStyle().Colors;
        colors[ImGuiCol_Text] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_Border] = ImVec4(0.31f, 0.30f, 0.30f, 0.50f);
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg] = ImVec4(0.29f, 0.29f, 0.29f, 0.54f);
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.51f, 0.51f, 0.51f, 0.54f);
        colors[ImGuiCol_FrameBgActive] = ImVec4(0.87f, 0.87f, 0.87f, 0.54f);
        colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
        colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
        colors[ImGuiCol_CheckMark] = ImVec4(0.79f, 0.79f, 0.79f, 1.00f);
        colors[ImGuiCol_SliderGrab] = ImVec4(0.88f, 0.50f, 0.24f, 1.00f);
        colors[ImGuiCol_SliderGrabActive] = ImVec4(0.98f, 0.55f, 0.26f, 1.00f);
        colors[ImGuiCol_Button] = ImVec4(0.29f, 0.29f, 0.29f, 0.40f);
        colors[ImGuiCol_ButtonHovered] = ImVec4(0.68f, 0.31f, 0.07f, 1.00f);
        colors[ImGuiCol_ButtonActive] = ImVec4(0.98f, 0.43f, 0.06f, 1.00f);
        colors[ImGuiCol_Header] = ImVec4(0.26f, 0.26f, 0.26f, 0.31f);
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 0.80f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
        colors[ImGuiCol_Separator] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
        colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
        colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
        colors[ImGuiCol_ResizeGrip] = ImVec4(0.88f, 0.00f, 1.00f, 0.20f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.97f, 0.00f, 1.00f, 0.67f);
        colors[ImGuiCol_ResizeGripActive] = ImVec4(0.97f, 0.00f, 1.00f, 0.95f);
        colors[ImGuiCol_Tab] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_TabHovered] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
        colors[ImGuiCol_TabActive] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
        colors[ImGuiCol_TabUnfocused] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        colors[ImGuiCol_DockingPreview] = ImVec4(0.31f, 0.31f, 0.31f, 0.70f);
        colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
        colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
        colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.27f, 0.59f, 1.00f, 0.35f);
        colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
        colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

        ImGui::GetStyle().Alpha = 1.0f;

        ImGui::GetStyle().WindowPadding = ImVec2(20.0f, 4.0f);
        ImGui::GetStyle().FramePadding = ImVec2(4.0f, 2.0f);
        ImGui::GetStyle().CellPadding = ImVec2(4.0f, 2.0f);
        ImGui::GetStyle().ItemSpacing = ImVec2(11.0f, 4.0f);
        ImGui::GetStyle().ItemInnerSpacing = ImVec2(4.0f, 4.0f);
        ImGui::GetStyle().IndentSpacing = 21.0f;
        ImGui::GetStyle().ScrollbarSize = 16.0f;
        ImGui::GetStyle().GrabMinSize = 4.0f;

        ImGui::GetStyle().WindowBorderSize = 1;
        ImGui::GetStyle().ChildBorderSize = 0;
        ImGui::GetStyle().PopupBorderSize = 1;
        ImGui::GetStyle().FrameBorderSize = 1;
        ImGui::GetStyle().TabBorderSize = 0;

        ImGui::GetStyle().WindowRounding = 5;
        ImGui::GetStyle().ChildRounding = 5;
        ImGui::GetStyle().FrameRounding = 5;
        ImGui::GetStyle().PopupRounding = 5;
        ImGui::GetStyle().ScrollbarRounding = 5;
        ImGui::GetStyle().GrabRounding = 5;
        ImGui::GetStyle().TabRounding = 5;

        //ImGui::GetStyle().WindowTitleAlign = ;
        ImGui::GetStyle().WindowMenuButtonPosition = 1;
        ImGui::GetStyle().ColorButtonPosition = 0;
        //ImGui::GetStyle().ButtonTextAlign = ;
    }

    void ImGuiRenderStart() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
    void ImGuiDraw(GLFWwindow* window)
    {
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        ImVec4 clear_color(getOptions()->clearColor[1], getOptions()->clearColor[2], getOptions()->clearColor[2], getOptions()->clearColor[4]);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ImDrawData* main_draw_data = ImGui::GetDrawData();
        const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);


        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }
    
    void shutDownImGui() {
        VTX_INFO("ShutDown: ImGui");
        ImGui::SaveIniSettingsToDisk((utl::absolutePath(getOptions()->imGuiIniFile)).data());
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        IMGUI_API
    }
}

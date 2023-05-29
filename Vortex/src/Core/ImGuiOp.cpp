#include "ImGuiOp.h"
#include "Core/Utils.h"
#include "Core/Log.h"
#include "imgui.h"
#include "imnodes.h"
#include "Core/ImguiBackEnds/imgui_impl_glfw.h"
#include "Core/ImguiBackEnds/imgui_impl_opengl3.h"
#include "Core/Options.h"
#include "glad/glad.h"

namespace vtx {

    // Callback function for GLFW window UI
    void SetWindowTitleBarColor(GLFWwindow* window, float r, float g, float b)
    {
        // Retrieve the GLFW window handle
        GLFWwindow* glfwWindow = static_cast<GLFWwindow*>(ImGui::GetIO().UserData);

        // Set the window UI callback functions
        glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
        glfwSetWindowAttrib(glfwWindow, GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
        glfwSetWindowAttrib(glfwWindow, GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

        // Set the window frame color
        //glfwSetWindowMonitor(glfwWindow, nullptr, 0, 0, 0, 0, GLFW_DONT_CARE);
        //glfwSetWindowPos(glfwWindow, 0, 0);
        //glfwSetWindowSizeLimits(glfwWindow, 0, 0, GLFW_DONT_CARE, GLFW_DONT_CARE);
        //glfwSetWindowPos(glfwWindow, 0, 0);
        //glfwSetWindowSize(glfwWindow, 0, 0);
        //glfwSetWindowOpacity(glfwWindow, 1.0f);
        //glfwSetWindowAttrib(glfwWindow, GLFW_RESIZABLE, GLFW_FALSE);
        //glfwSetWindowAttrib(glfwWindow, GLFW_MAXIMIZABLE, GLFW_FALSE);
        //glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
        //glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

        // Set the window background color
        //glfwSetWindowSizeLimits(glfwWindow, 0, 0, GLFW_DONT_CARE, GLFW_DONT_CARE);
        //glfwSetWindowPos(glfwWindow, 0, 0);
        //glfwSetWindowSize(glfwWindow, 1, 1);
        //glfwSetWindowOpacity(glfwWindow, 1.0f);
        //glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
        //glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

        // Set the color of the window title bar
        GLFWimage image;
        image.width = 1;
        image.height = 1;
        image.pixels = new unsigned char[4];
        image.pixels[0] = static_cast<unsigned char>(r * 255);
        image.pixels[1] = static_cast<unsigned char>(g * 255);
        image.pixels[2] = static_cast<unsigned char>(b * 255);
        image.pixels[3] = 255;
        glfwSetWindowIcon(glfwWindow, 1, &image);
        delete[] image.pixels;
    }

    void Init_ImGui(GLFWwindow* window) {
        VTX_INFO("Starting ImGui");
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImNodes::CreateContext();

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
        //SetWindowTitleBarColor(window, 0.2f, 0.2f, 0.2f);
        // Upload Fonts
        //LoadFonts();

        ImGui::LoadIniSettingsFromDisk(utl::absolutePath(getOptions()->imGuiIniFile).data());

        SetAppStyle();
    }

    uint32_t vec4toImCol32(ImVec4 color)
    {
	    return IM_COL32((uint8_t)(color.x * 255.0f), (uint8_t)(color.y * 255.0f), (uint8_t)(color.z * 255.0f), (uint8_t)(color.w * 255.0f));
    }
    void SetAppStyle() {

        const ImVec4 darkGrey = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
        const ImVec4 lightGrey = ImVec4(0.29f, 0.29f, 0.29f, 0.54f);
        const ImVec4 orange = ImVec4(0.88f, 0.50f, 0.24f, 1.00f);
        const ImVec4 orangeHovered = ImVec4(0.98f, 0.55f, 0.26f, 1.00f);
        const ImVec4 orangeActive = ImVec4(0.98f, 0.43f, 0.06f, 1.00f);
        const ImVec4 transparent = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
        const ImVec4 tabColor = darkGrey;
        const ImVec4 transparentGrey = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);

        ImVec4* colors = ImGui::GetStyle().Colors;
        colors[ImGuiCol_Text] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg] = darkGrey;
        colors[ImGuiCol_ChildBg] = darkGrey;
        colors[ImGuiCol_PopupBg] = darkGrey;
        colors[ImGuiCol_Border] = ImVec4(0.31f, 0.30f, 0.30f, 0.50f);
        colors[ImGuiCol_BorderShadow] = transparent;
        colors[ImGuiCol_FrameBg] = lightGrey;
        colors[ImGuiCol_FrameBgHovered] = lightGrey;
        colors[ImGuiCol_FrameBgActive] = lightGrey;
        colors[ImGuiCol_TitleBg] = darkGrey;
        colors[ImGuiCol_TitleBgActive] = darkGrey;
        colors[ImGuiCol_TitleBgCollapsed] = darkGrey;
        colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
        colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
        colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
        colors[ImGuiCol_CheckMark] = ImVec4(0.79f, 0.79f, 0.79f, 1.00f);
        colors[ImGuiCol_SliderGrab] = orange;
        colors[ImGuiCol_SliderGrabActive] = orangeHovered;
        colors[ImGuiCol_Button] = lightGrey;
        colors[ImGuiCol_ButtonHovered] = orangeHovered;
        colors[ImGuiCol_ButtonActive] = orangeActive;
        colors[ImGuiCol_Header] = lightGrey;
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 0.80f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
        colors[ImGuiCol_Separator] = transparentGrey;
        colors[ImGuiCol_SeparatorHovered] = transparentGrey;
        colors[ImGuiCol_SeparatorActive] = transparentGrey;
        colors[ImGuiCol_ResizeGrip] = ImVec4(0.88f, 0.00f, 1.00f, 0.20f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.97f, 0.00f, 1.00f, 0.67f);
        colors[ImGuiCol_ResizeGripActive] = ImVec4(0.97f, 0.00f, 1.00f, 0.95f);
        colors[ImGuiCol_Tab] = tabColor;
        colors[ImGuiCol_TabHovered] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
        colors[ImGuiCol_TabActive] = ImVec4(0.74f, 0.35f, 0.09f, 1.00f);
        colors[ImGuiCol_TabUnfocused] = tabColor;
        colors[ImGuiCol_TabUnfocusedActive] = tabColor;
        colors[ImGuiCol_DockingPreview] = ImVec4(0.31f, 0.31f, 0.31f, 0.70f);
        colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
        colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
        colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
        colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
        colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
        colors[ImGuiCol_TableRowBg] = transparent;
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

        ImNodesStyle& style = ImNodes::GetStyle();

        // Set colors
        style.Colors[ImNodesCol_GridBackground] = vec4toImCol32(darkGrey);
        style.Colors[ImNodesCol_NodeBackground] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_NodeBackgroundHovered] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_NodeBackgroundSelected] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_NodeOutline] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_TitleBar] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_TitleBarHovered] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_TitleBarSelected] = vec4toImCol32(lightGrey);
        style.Colors[ImNodesCol_Link] = vec4toImCol32(orangeHovered);
        style.Colors[ImNodesCol_LinkHovered] = vec4toImCol32(orangeHovered);
        style.Colors[ImNodesCol_LinkSelected] = vec4toImCol32(orangeHovered);
        style.Colors[ImNodesCol_Pin] = vec4toImCol32(orangeHovered);
        // ... set other colors

        // Set style variables
        style.NodeCornerRounding = 5.0f;
        style.NodePadding = ImVec2(10.0f, 4.0f);
        style.NodeBorderThickness = 1.0f;
        style.LinkThickness = 2.0f;
        style.LinkLineSegmentsPerLength = 0.1f;
        style.LinkHoverDistance = 10.0f;
        // ... set other style variables

        // Set style flags
        style.Flags = ImNodesStyleFlags_NodeOutline;
        // ... set other style flags


        // Edit style flags
        //ImGui::Text("Style Flags");
        //ImGui::CheckboxFlags("NodeOutline", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_NodeOutline);
        //ImGui::CheckboxFlags("GridLines", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLines);
        //ImGui::CheckboxFlags("GridLinesPrimary", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLinesPrimary);
        //ImGui::CheckboxFlags("GridSnapping", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridSnapping);
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

        ImVec4 clear_color(vtx::getOptions()->clearColor[1], vtx::getOptions()->clearColor[2], vtx::getOptions()->clearColor[2], vtx::getOptions()->clearColor[4]);
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

    std::string labelPrefix(const char* const label)
    {
		const float width = ImGui::CalcItemWidth();

		const float x = ImGui::GetCursorPosX();
        ImGui::Text(label);
        ImGui::SameLine();
        ImGui::SetCursorPosX(x + width * 0.5f + ImGui::GetStyle().ItemInnerSpacing.x);
        ImGui::SetNextItemWidth(-1);

        std::string labelID = "##";
        labelID += label;

        return labelID;
    }
    
    void shutDownImGui() {
        VTX_INFO("ShutDown: ImGui");
        ImGui::SaveIniSettingsToDisk((utl::absolutePath(getOptions()->imGuiIniFile)).data());
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImNodes::DestroyContext();
        ImGui::DestroyContext();

        IMGUI_API
    }
}

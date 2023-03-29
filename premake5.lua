include "PremakeScripts/path.lua"

workspace "Vortex"

    configurations { "Debug", "Release" }
    architecture "x64"
    startproject "OptixApp"

    
	filter "system:windows"
    systemversion "latest"
    cppdialect "C++17"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        runtime "Release"
        optimize "on"

    filter "configurations:Dist"
        runtime "Release"
        optimize "on"
        symbols "off"
    
group "Dependencies"
    include "ext/glfw/glfw.lua"
    include "ext/imgui/imgui.lua"
    include "ext/glad/glad.lua"
group ""

group "Core"
    include "VortexOptix/Vortex.lua"
group ""

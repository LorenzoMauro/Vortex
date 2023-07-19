include "PremakeScripts/path.lua"


workspace "Vortex"
    configurations {"Debug", "Release"}
    architecture "x64"
    startproject "OptixApp"
    
	filter "system:windows"
    systemversion "latest"
    cppdialect "C++17"

    filter "configurations:Debug"
        runtime "Debug"
        symbols "on"
        defines { "DEBUG" }

    filter "configurations:Release"
        runtime "Release"
        optimize "on"
        defines { "NDEBUG" }
    
group "Dependencies"
    include "ext/glfw/glfw.lua"
    include "ext/imgui/imgui.lua"
    include "ext/glad/glad.lua"
    include "ext/imnodes/imnode.lua"
    include "ext/implot/implot.lua"
group ""

group "Core"
    include "Vortex/Vortex.lua"
group ""

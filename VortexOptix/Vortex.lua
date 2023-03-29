--include "../PremakeCommon/path.lua"


project "OptixApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir (TARGET_DIR)
    objdir (OBJ_DIR)
    buildcustomizations "BuildCustomizations/CUDA 12.0"

    
    postbuildcommands {
        "{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/"}

    files{
        "src/**.h",
        "src/**.cpp"
    }

    includedirs{
        "src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.GLAD}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.spdlog}",
        "%{IncludeDir.OPTIX}",
        "%{IncludeDir.CUDA}"
    }
    
    libdirs {
        "%{LibDir.CUDA}"
    }

    links {
        "cudart_static",
        "cuda",
        "GLFW",
        "Glad",
        "opengl32.lib",
        "ImGui"
    }

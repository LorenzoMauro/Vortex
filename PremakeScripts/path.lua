
-- Paths For the Build Process
script_dir = path.getabsolute(".")
my_project_dir = path.join(script_dir, "..")

OUTPUT_DIR = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
TARGET_DIR = my_project_dir .. "build/bin/" .. OUTPUT_DIR .. "/%{prj.name}"
OBJ_DIR =   my_project_dir .. "build/bin-int/" .. OUTPUT_DIR .. "/%{prj.name}"

-- Local Project Paths
GLFW_RP = "../ext/glfw/"
VORTEX_RP = "../VortexOptix/"
IMGUI_RP = "../ext/imgui/"

-- External Resources Paths
OPTIX_SDK_PATH = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"

IncludeDir = {}
IncludeDir["GLFW"] = my_project_dir .. "/ext/glfw/include"
IncludeDir["GLAD"] = my_project_dir .. "/ext/glad/include"
IncludeDir["ImGui"] = my_project_dir .. "/ext/imgui"
IncludeDir["spdlog"] = my_project_dir .. "/ext/spdlog/include"
IncludeDir["gdt"] = my_project_dir .. "/ext/gdt"
IncludeDir["OPTIX"] = path.join(OPTIX_SDK_PATH, "include")
IncludeDir["CUDA"] = path.join(CUDA_TOOLKIT_PATH, "include")
IncludeDir["MDL"] = my_project_dir .. "/ext/MDL/include"

LibDir = {}
LibDir["CUDA"] = path.join(CUDA_TOOLKIT_PATH, "lib/x64")
LibDir["DevIL"] = "ext/DevIL/lib/x64/Release"
LibDir["MDL_Debug"] = my_project_dir .. "/ext/MDL/debug/lib"
LibDir["MDL_Release"] = my_project_dir .. "/ext/MDL/release/lib"

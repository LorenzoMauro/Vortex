
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
OPTIX_SDK_PATH = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0"
CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
MDL_SDK_PATH = "E:/Repo/MDLBinaries"


IncludeDir = {}
IncludeDir["GLFW"] = my_project_dir .. "/ext/glfw/include"
IncludeDir["GLAD"] = my_project_dir .. "/ext/glad/include"
IncludeDir["ImGui"] = my_project_dir .. "/ext/imgui"
IncludeDir["spdlog"] = my_project_dir .. "/ext/spdlog/include"
IncludeDir["OPTIX"] = path.join(optix_sdk_path, "include")
IncludeDir["CUDA"] = path.join(cuda_toolkit_path, "include")

LibDir = {}
LibDir["CUDA"] = path.join(cuda_toolkit_path, "lib/x64")
LibDir["DevIL"] = "ext/DevIL/lib/x64/Release"
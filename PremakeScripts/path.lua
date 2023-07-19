
-- Paths For the Build Process
script_dir = path.getabsolute(".")
my_project_dir = path.join(script_dir, "..")

OUTPUT_DIR = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
TARGET_DIR = my_project_dir .. "build/bin/" .. OUTPUT_DIR .. "/%{prj.name}"
OBJ_DIR =   my_project_dir .. "build/bin-int/" .. OUTPUT_DIR .. "/%{prj.name}"

-- Local Project Paths
GLFW_RP = "../ext/glfw/"
VORTEX_RP = "../Vortex/"
IMGUI_RP = "../ext/imgui/"

clang12 = "C:/Program Files (x86)/LLVM/bin/clang.exe"
ccbin_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.36.32532/bin/HostX64/x64"

-- External Resources Paths
OPTIX_SDK_PATH = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"
CUDA_TOOLKIT_PATH_8 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0"
CUDA_TOOLKIT_PATH_12 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
CUDA_TOOLKIT_PATH_11 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7"
ASSIMP_SDK_PATH = "C:/Program Files/Assimp"
LIBTORCH_DEBUG_PATH = "E:/Repo/libtorch/libtorch-win-shared-with-deps-debug-2.0.1+cu117/libtorch"
LIBTORCH_RELEASE_PATH = "E:/Repo/libtorch/libtorch-win-shared-with-deps-2.0.1+cu117/libtorch"
NVTOOLS = "C:/Program Files/NVIDIA Corporation/NvToolsExt"

IncludeDir = {}
IncludeDir["GLFW"] = my_project_dir .. "/ext/glfw/include"
IncludeDir["GLAD"] = my_project_dir .. "/ext/glad/include"
IncludeDir["ImGui"] = my_project_dir .. "/ext/imgui"
IncludeDir["spdlog"] = my_project_dir .. "/ext/spdlog/include"
IncludeDir["gdt"] = my_project_dir .. "/ext/gdt"
IncludeDir["MDL"] = my_project_dir .. "/ext/MDL/include"
IncludeDir["ImNode"] = my_project_dir .. "/ext/imnodes"
IncludeDir["ImPlot"] = my_project_dir .. "/ext/implot"
IncludeDir["OPTIX"] = path.join(OPTIX_SDK_PATH, "include")
IncludeDir["CUDA"] = path.join(CUDA_TOOLKIT_PATH_11, "include")
IncludeDir["ASSIMP"] = path.join(ASSIMP_SDK_PATH, "include")
IncludeDir["NVTOOLS"] = path.join(NVTOOLS, "include")

IncludeDir["LibTorch_Release"] = path.join(LIBTORCH_RELEASE_PATH, "include")
IncludeDir["LibTorch_Release_API"] = path.join(LIBTORCH_RELEASE_PATH, "include/torch/csrc/api/include")

IncludeDir["LibTorch_Debug"] = path.join(LIBTORCH_DEBUG_PATH, "include")
IncludeDir["LibTorch_Debug_API"] = path.join(LIBTORCH_DEBUG_PATH, "include/torch/csrc/api/include")

LibDir = {}
LibDir["CUDA"] = path.join(CUDA_TOOLKIT_PATH_11, "lib/x64")
LibDir["DevIL"] = "ext/DevIL/lib/x64/Release"
LibDir["MDL_Debug"] = my_project_dir .. "/ext/MDL/debug/lib"
LibDir["MDL_Release"] = my_project_dir .. "/ext/MDL/release/lib"
LibDir["ASSIMP"] = path.join(ASSIMP_SDK_PATH, "lib/x64")
LibDir["LibTorch_Release"] = path.join(LIBTORCH_RELEASE_PATH, "lib")
LibDir["LibTorch_Debug"] = path.join(LIBTORCH_DEBUG_PATH, "lib")
LibDir["nvToolsExt64"] = path.join(NVTOOLS, "lib/x64")

DllLocation = {
    ASSIMP = path.join(ASSIMP_SDK_PATH, "bin/x64")
}
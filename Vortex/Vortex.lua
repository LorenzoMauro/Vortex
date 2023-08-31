include "../PremakeScripts/scripts.lua"
include "../PremakeScripts/path.lua"

project "OptixApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir (TARGET_DIR)
    objdir (OBJ_DIR)
    buildcustomizations "BuildCustomizations/CUDA 11.7"
    configurations {"Debug", "Release"}

    files{
        "%{IncludeDir.gdt}/**.h",
        "%{IncludeDir.gdt}/**.cpp",
        "src/**.h",
        "src/**.cpp",
        "src/**.cu"
    }

    local relativeIncludeDirsCommon = {
        "src/",
        IncludeDir["GLFW"],
        IncludeDir["GLAD"],
        IncludeDir["ImGui"],
        IncludeDir["yaml"],
        IncludeDir["spdlog"],
        IncludeDir["OPTIX"],
        IncludeDir["CUDA"],
        IncludeDir["gdt"],
        IncludeDir["MDL"],
        IncludeDir["ASSIMP"],
        IncludeDir["ImNode"],
        IncludeDir["ImPlot"],
        IncludeDir["NVTOOLS"],
        IncludeDir["stbImage"],
        IncludeDir["ImGuiFileDialog"]
    }

    local relativeIncludeDirsDebug = {table.unpack(relativeIncludeDirsCommon)}
    table.insert(relativeIncludeDirsDebug, IncludeDir["LibTorch_Debug"])
    table.insert(relativeIncludeDirsDebug, IncludeDir["LibTorch_Debug_API"])

    local relativeIncludeDirsRelease = {table.unpack(relativeIncludeDirsCommon)}
    table.insert(relativeIncludeDirsRelease, IncludeDir["LibTorch_Release"])
    table.insert(relativeIncludeDirsRelease, IncludeDir["LibTorch_Release_API"])

    local absoluteIncludeDirsDebug = toAbsolutePaths(relativeIncludeDirsDebug)
    local absoluteIncludeDirsRelease = toAbsolutePaths(relativeIncludeDirsRelease)

    local clangInclude = toAbsolutePaths(
        {
            IncludeDir["OPTIX"],
            IncludeDir["spdlog"],
            IncludeDir["MDL"],
            "src/",
            "../ext/gdt/"
        }
    )

    local preprocessorDefinesCommon = {
        "GLFW_INCLUDE_NONE",
        "WIN32",
        "_WINDOWS",
        "NOMINMAX",
        --"C10_MACROS_CMAKE_MACROS_H_",
        "_UNICODE",
        "UNICODE",
        "YAML_CPP_STATIC_DEFINE"
    }

    local preprocessorDefinesDebug = {table.unpack(preprocessorDefinesCommon)}
    table.insert(preprocessorDefinesDebug, "DEBUG")

    local preprocessorDefinesRelease = {table.unpack(preprocessorDefinesCommon)}
    table.insert(preprocessorDefinesRelease, "NDEBUG")

    local postBuildDebug = generatePostBuildCommands(true, useMdlDebug)
    local postBuildRelease = generatePostBuildCommands(false, false)

    links {
        "cudart_static",
        "cuda",
        "GLFW",
        "Glad",
        "opengl32.lib",
        "ImGui",
        "ImNode",
        "ImPlot",
        "ImGuiFileDialog",
        "yaml-cpp",
        "mdl_sdk.lib",
        "mdl_core.lib",
        "nv_freeimage.lib",
        "dds.lib",
        "assimp-vc143-mt",
        -- Links For LibTorch
        "caffe2_nvrtc",
        "c10",
        "c10_cuda",
        "kineto",
        "torch",
        "torch_cpu",
        "torch_cuda",
        "cublas",
        "cudart",
        "cudnn",
        "cufft",
        "curand",
        "nvToolsExt64_1",
        "kernel32",
        "user32",
        "gdi32",
        "winspool",
        "shell32",
        "ole32",
        "oleaut32",
        "uuid",
        "comdlg32",
        "advapi32"
    }

    linkoptions {
        "/NODEFAULTLIB:LIBCMT",
        -- Link Options For LibTorch
        "-INCLUDE:?warp_size@cuda@at@@YAHXZ",
        "/machine:x64"
    }

    useMdlDebug = false

    local ptxOutput     = "%{cfg.targetdir}/ptx/%{file.basename}.ptx"
    local optixIrOutput = "%{cfg.targetdir}/ptx/%{file.basename}.optixir"
    local objOutput     = "%{cfg.objdir}/%{file.basename}.obj"
    local bcOutput      = "%{cfg.targetdir}/bc/%{file.basename}.bc"
    local dOutput       = "%{cfg.targetdir}/bc/%{file.basename}.d"

    local CudaCompileCommandDebug = nvccCompile("%{file.relpath}", objOutput, CUDA_TOOLKIT_PATH_11, ccbin_path, absoluteIncludeDirsDebug,preprocessorDefinesDebug, true)
    local ptxCompileForCudaLinkerDebug = nvccCompileToIntermediate("%{file.relpath}", ptxOutput, "ptx", CUDA_TOOLKIT_PATH_11, absoluteIncludeDirsDebug,preprocessorDefinesDebug, true)
    local optixirCompileCommandDebug = nvccCompileToIntermediate("%{file.relpath}", optixIrOutput, "optix-ir", CUDA_TOOLKIT_PATH_11, absoluteIncludeDirsDebug,preprocessorDefinesDebug, true)
    local clangCompileCommandDebug = clangCompileCu("%{file.relpath}", bcOutput, dOutput, clang12, CUDA_TOOLKIT_PATH_8, clangInclude,preprocessorDefinesDebug, true)
    
    local CudaCompileCommandRelease = nvccCompile("%{file.relpath}", objOutput, CUDA_TOOLKIT_PATH_11, ccbin_path, absoluteIncludeDirsRelease,preprocessorDefinesRelease, false)
    local ptxCompileForCudaLinkerRelease = nvccCompileToIntermediate("%{file.relpath}", ptxOutput, "ptx", CUDA_TOOLKIT_PATH_11, absoluteIncludeDirsRelease,preprocessorDefinesRelease, false)
    local optixirCompileCommandRelease = nvccCompileToIntermediate("%{file.relpath}", optixIrOutput, "optix-ir", CUDA_TOOLKIT_PATH_11, absoluteIncludeDirsRelease,preprocessorDefinesRelease, false)
    local clangCompileCommandRelease = clangCompileCu("%{file.relpath}", bcOutput, dOutput, clang12, CUDA_TOOLKIT_PATH_8,clangInclude ,preprocessorDefinesRelease, false)
    
    local torchBuildInputs = {
        "src/NeuralNetworks/Interface/InferenceQueries.h",
        "src/NeuralNetworks/Interface/NetworkInputs.h",
        "src/NeuralNetworks/Interface/NetworkInterface.h",
        "src/NeuralNetworks/Interface/NpgTrainingData.h",
        "src/NeuralNetworks/Interface/Paths.h",
        "src/NeuralNetworks/Interface/ReplayBuffer.h",

        "src/NeuralNetworks/NetworkImplementation.h",
        "src/NeuralNetworks/NetworkSettings.h",
        "src/NeuralNetworks/NeuralNetworkGraphs.h",
        "src/NeuralNetworks/tools.h",
        "src/NeuralNetworks/Networks/Sac.h",
        "src/NeuralNetworks/Networks/Npg.h",
        "src/NeuralNetworks/Networks/PathGuidingNetwork.h",
        "src/NeuralNetworks/Networks/InputComposer.h",
        "src/NeuralNetworks/Distributions/GaussianToSphere.h",
        "src/NeuralNetworks/Distributions/SphericalGaussian.h",
        "src/NeuralNetworks/Distributions/Nasg.h",
        "src/NeuralNetworks/Distributions/Mixture.h",
        "src/Device/DevicePrograms/LaunchParams.h"
    }

    local cudaBuildInputs = {
        "src/NeuralNetworks/Interface/InferenceQueries.h",
        "src/NeuralNetworks/Interface/NetworkInputs.h",
        "src/NeuralNetworks/Interface/NetworkInterface.h",
        "src/NeuralNetworks/Interface/NpgTrainingData.h",
        "src/NeuralNetworks/Interface/Paths.h",
        "src/NeuralNetworks/Interface/ReplayBuffer.h",
        "src/NeuralNetworks/Distributions/Mixture.h",
        "src/NeuralNetworks/Distributions/SphericalGaussian.h",
        "src/NeuralNetworks/Distributions/Nasg.h",
        "src/Device/DevicePrograms/LaunchParams.h",
        "src/Device/DevicePrograms/rendererFunctions.h"
    }

    filter "configurations:Debug"
        includedirs{
            relativeIncludeDirsDebug
        }
        postbuildcommands {
            generatePostBuildCommands(true, useMdlDebug)
        }
        libdirs {
            getLibDirs(true, useMdlDebug)
        }
        defines{
            preprocessorDefinesDebug
        }
    filter "configurations:Release"
        includedirs{
            relativeIncludeDirsRelease
        }
        postbuildcommands {
            generatePostBuildCommands(false, false)
        }
        libdirs {
            getLibDirs(false, false)
        }
        defines{
            preprocessorDefinesRelease
        }

    filter {"configurations:Release", "files:src/Device/DevicePrograms/ptx/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. ptxCompileForCudaLinkerRelease .. '"',
            ptxCompileForCudaLinkerRelease,
        }
        buildoutputs {
            ptxOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Release", "files:src/Device/DevicePrograms/Cuda/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandRelease .. '"',
            CudaCompileCommandRelease,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

        
    filter {"configurations:Release", "files:src/Device/CudaFunctions/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandRelease .. '"',
            CudaCompileCommandRelease,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Release","files:src/NeuralNetworks/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandRelease .. '"',
            CudaCompileCommandRelease,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
            torchBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Release", "files:src/Device/DevicePrograms/optixIr/*.cu"}

        buildcommands {
            'echo "Running NVCC:"' .. optixirCompileCommandRelease .. '"',
            optixirCompileCommandRelease,
        }
        buildoutputs {
            optixIrOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Release", "files:src/Device/DevicePrograms/Clang/*.cu"}
        buildcommands {
            'echo "Running Clang:"' .. clangCompileCommandRelease .. '"',
            clangCompileCommandRelease,
        }
        buildoutputs {
            bcOutput,
            dOutput
        }
        buildinputs{
            cudaBuildInputs         
        }
        flags { "MultiProcessorCompile" }
    
    
        
    filter {"configurations:Debug", "files:src/Device/DevicePrograms/Cuda/*.cu"}

        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandDebug .. '"',
            CudaCompileCommandDebug,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Debug", "files:src/Device/CudaFunctions/*.cu"}

        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandDebug .. '"',
            CudaCompileCommandDebug,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }
    filter {"configurations:Debug", "files:src/NeuralNetworks/*.cu"}

        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommandDebug .. '"',
            CudaCompileCommandDebug,
        }
        buildoutputs {
            objOutput,
        }
        buildinputs{
           torchBuildInputs
        }
        flags { "MultiProcessorCompile" }
    
    filter {"configurations:Debug", "files:src/Device/DevicePrograms/ptx/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. ptxCompileForCudaLinkerDebug .. '"',
            ptxCompileForCudaLinkerDebug,
        }
        buildoutputs {
            ptxOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Debug", "files:src/Device/DevicePrograms/optixIr/*.cu"}
        buildcommands {
            'echo "Running NVCC:"' .. optixirCompileCommandDebug .. '"',
            optixirCompileCommandDebug,
        }
        buildoutputs {
            optixIrOutput,
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    filter {"configurations:Debug", "files:src/Device/DevicePrograms/Clang/*.cu"}
        buildcommands {
            'echo "Running Clang:"' .. clangCompileCommandDebug .. '"',
            clangCompileCommandDebug,
        }
        buildoutputs {
            bcOutput,
            dOutput
        }
        buildinputs{
            cudaBuildInputs
        }
        flags { "MultiProcessorCompile" }

    --Debug Print
    printTable("\nDEBUG: Defines:\n", preprocessorDefinesDebug)
    printTable("\nDEBUG: Include Dirs:\n", absoluteIncludeDirsDebug)
    printTable("\nDEBUG: Post Build Commands:\n", postBuildDebug)
    printf("\nDEBUG: Compile Command for standard Cuda:\n %s", CudaCompileCommandDebug)
    printf("\nDEBUG: Compile Command for Ptx for cuda Linker:\n %s", ptxCompileForCudaLinkerDebug)
    printf("\nDEBUG: Compile Command for optix pipeline:\n %s", optixirCompileCommandDebug)
    printf("\nRELEASE: Compile Command for clang:\n %s", clangCompileCommandDebug)
    
    printTable("\nRELEASE: Defines:\n", preprocessorDefinesRelease)
    printTable("\nRELEASE: Include Dirs:\n", absoluteIncludeDirsRelease)
    printTable("\nRELEASE: Post Build Commands:\n", postBuildRelease)
    printf("\nRELEASE: Compile Command for standard Cuda:\n %s", CudaCompileCommandRelease)
    printf("\nRELEASE: Compile Command for Ptx for cuda Linker:\n %s", ptxCompileForCudaLinkerRelease)
    printf("\nRELEASE: Compile Command for optix pipeline:\n %s", optixirCompileCommandRelease)
    printf("\nRELEASE: Compile Command for clang:\n %s", clangCompileCommandRelease)

    


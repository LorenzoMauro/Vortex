include "../PremakeScripts/path.lua"
--"C:\Program Files\LLVM-15\bin\clang.exe" -std=c++17 test.cu -O3 -ffast-math -fcuda-flush-denormals-to-zero -fno-vectorize --cuda-gpu-arch=sm_60 --cuda-path="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0" -c -o test.bc
function genNvccCommand(cu_file, ptx_file, cuda_path, include_dirs, debug)
    local nvcc_path = '"' .. path.join(cuda_path, "bin/nvcc.exe") .. '"'
    CudaCompileCommand = nvcc_path
    CudaCompileCommand = CudaCompileCommand .. " " .. cu_file
    CudaCompileCommand = CudaCompileCommand .. " --optix-ir"
    CudaCompileCommand = CudaCompileCommand .. " --use_fast_math"
    CudaCompileCommand = CudaCompileCommand .. " --keep"
    CudaCompileCommand = CudaCompileCommand .. " -Wno-deprecated-gpu-targets"
    CudaCompileCommand = CudaCompileCommand .. " --std=c++17"
    CudaCompileCommand = CudaCompileCommand .. " -m64"
    CudaCompileCommand = CudaCompileCommand .. " --keep-device-functions"
    CudaCompileCommand = CudaCompileCommand .. " --relocatable-device-code=true"
    CudaCompileCommand = CudaCompileCommand .. " --generate-line-info"
    if(debug) then 
        CudaCompileCommand = CudaCompileCommand .. " -G"
    else
        CudaCompileCommand = CudaCompileCommand .. " -O3"
    end
    CudaCompileCommand = CudaCompileCommand .. " -o " .. ptx_file
    
    -- Add additional include directories
    if include_dirs and type(include_dirs) == "table" then
        for _, include_dir in ipairs(include_dirs) do
            -- if path is not an absolute path get absolute
            if path.isabsolute(include_dir) then
                include_path = include_dir
            else
                include_path = path.getabsolute(include_dir)
            end
            include_path = '"' .. include_path .. '"'
            CudaCompileCommand = CudaCompileCommand .. " -I" .. include_path
        end
    end

    return CudaCompileCommand
end

function genClangCommand(cu_file, bc_file,d_file, clang_path, cuda_path, include_dirs, debug)
    ClangCompileCommand = '"' .. clang_path .. '"'
    
    -- Add additional include directories
    if include_dirs and type(include_dirs) == "table" then
        for _, include_dir in ipairs(include_dirs) do
            -- if path is not an absolute path get absolute
            if path.isabsolute(include_dir) then
                include_path = include_dir
            else
                include_path = path.getabsolute(include_dir)
            end
            include_path = '"' .. include_path .. '"'
            ClangCompileCommand = ClangCompileCommand .. " -I" .. include_path
        end
    end

    --ClangCompileCommand = ClangCompileCommand .. " --cuda-gpu-arch=sm_60  --cuda-gpu-arch=sm_75" -- example GPU architecture, replace with your actual architecture
    ClangCompileCommand = ClangCompileCommand .. " --cuda-path=" .. '"' .. cuda_path .. '"'
    ClangCompileCommand = ClangCompileCommand .. " -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ClangCompileCommand = ClangCompileCommand .. " -std=c++17"
    ClangCompileCommand = ClangCompileCommand .. " -emit-llvm"
    ClangCompileCommand = ClangCompileCommand .. " -c"
    ClangCompileCommand = ClangCompileCommand .. " -w"
    ClangCompileCommand = ClangCompileCommand .. " -O3"
    ClangCompileCommand = ClangCompileCommand .. " -ffast-math"
    ClangCompileCommand = ClangCompileCommand .. " -fcuda-flush-denormals-to-zero"
    ClangCompileCommand = ClangCompileCommand .. " -fno-vectorize"
    ClangCompileCommand = ClangCompileCommand .. " --cuda-device-only"
    ClangCompileCommand = ClangCompileCommand .. " " .. cu_file
    ClangCompileCommand = ClangCompileCommand .. " -o" .. bc_file
    ClangCompileCommand = ClangCompileCommand .. " -MD -MT".. bc_file
    ClangCompileCommand = ClangCompileCommand .. " -MP -MF".. d_file


    return ClangCompileCommand
end


project "OptixApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir (TARGET_DIR)
    objdir (OBJ_DIR)
    buildcustomizations "BuildCustomizations/CUDA 12.1"
    buildoptions { "--std=c++17" }

    files{
        "%{IncludeDir.gdt}/**.h",
        "%{IncludeDir.gdt}/**.cpp",
        "src/**.h",
        "src/**.cpp",
        "src/Device/DevicePrograms/**.cu"
        "src/Device/DevicePrograms/Cuda/**.cu"
    }

    includedirs{
        "src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.GLAD}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.spdlog}",
        "%{IncludeDir.OPTIX}",
        "%{IncludeDir.CUDA}",
        "%{IncludeDir.gdt}",
        "%{IncludeDir.MDL}",
        "%{IncludeDir.ASSIMP}"
    }
    
    libdirs {
        "%{LibDir.CUDA}",
        "%{LibDir.ASSIMP}"
    }

    links {
        "cudart_static",
        "cuda",
        "GLFW",
        "Glad",
        "opengl32.lib",
        "ImGui",
        "mdl_sdk.lib",
        "mdl_core.lib",
        "nv_freeimage.lib",
        "dds.lib",
        "assimp-vc143-mt"
    }

    linkoptions {
        "/NODEFAULTLIB:LIBCMT"
    }

    useMdlDebug = true

    -- Add build options to the CUDA configuration
    filter { "action:vs*", "language:Cuda" }
        filter { "CUDA" }
            buildoptions { "/std:c++17" }

    filter "configurations:Debug"
        defines{"GLFW_INCLUDE_NONE"}

        if useMdlDebug then
            postbuildcommands {
                --"{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/",
                "{MKDIR} %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/debug/bin/libmdl_sdk.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/debug/bin/nv_freeimage.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/debug/bin/freeimage.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/debug/bin/dds.dll %{cfg.targetdir}/lib"
            }
            libdirs {
                "%{LibDir.MDL_Debug}"
            }
        else
            postbuildcommands {
                --"{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/",
                "{MKDIR} %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/release/bin/libmdl_sdk.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/release/bin/nv_freeimage.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/release/bin/freeimage.dll %{cfg.targetdir}/lib",
                "{COPY} %{wks.location}ext/MDL/release/bin/dds.dll %{cfg.targetdir}/lib"
            }
    
            libdirs {
                "%{LibDir.ASSIMP}",
                "%{LibDir.MDL_Release}"
            }
        end

    filter "configurations:Release"
        defines{"GLFW_INCLUDE_NONE"}

        postbuildcommands {
            -- "{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/",
            "{MKDIR} %{cfg.targetdir}/lib",
            "{COPY} %{wks.location}ext/MDL/release/bin/libmdl_sdk.dll %{cfg.targetdir}/lib",
            "{COPY} %{wks.location}ext/MDL/release/bin/nv_freeimage.dll %{cfg.targetdir}/lib",
            "{COPY} %{wks.location}ext/MDL/release/bin/freeimage.dll %{cfg.targetdir}/lib",
            "{COPY} %{wks.location}ext/MDL/release/bin/dds.dll %{cfg.targetdir}/lib"
        }

        libdirs {
            "%{LibDir.ASSIMP}",
            "%{LibDir.MDL_Release}"
        }
    
    local cu_file = "%{file.relpath}"
    local ptx_file = "%{cfg.targetdir}/ptx/%{file.basename}.optixir"
    local bc_file = "%{cfg.targetdir}/bc/%{file.basename}.bc"
    local d_file = "%{cfg.targetdir}/bc/%{file.basename}.d"

    -- include dirs for the compilation of ptx files
    local include_dirs = {
        IncludeDir["OPTIX"],
        IncludeDir["spdlog"],
        IncludeDir["MDL"],
        "src/",
        "../ext/gdt/"
    }
    -- Custom build step to compile .cu files to .ptx files
    filter {"configurations:Debug", "files:src/Device/DevicePrograms/Ptx/**.cu"}
        
        cudaCompileCommand = genNvccCommand(cu_file, ptx_file, CUDA_TOOLKIT_PATH, include_dirs, true)
        
        printf("Optix Compile Debug command:\n %s", cudaCompileCommand)

        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommand .. '"',
            CudaCompileCommand,
        }
        buildoutputs {
            ptx_file,
        }

    -- Custom build step to compile .cu files to .ptx files
    filter {"configurations:Release", "files:src/Device/DevicePrograms/Ptx/**.cu"}

        cudaCompileCommand = genNvccCommand(cu_file, ptx_file, CUDA_TOOLKIT_PATH, include_dirs, false)
        
        printf("Optix Compile Release command:\n %s", cudaCompileCommand)

        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommand .. '"',
            CudaCompileCommand,
        }
        buildoutputs {
            ptx_file,
        }

    clang12 = "C:/Program Files (x86)/LLVM/bin/clang.exe"
    clang15 = "C:/Program Files/LLVM-15/bin/clang.exe"
    clangCudaPath = CUDA_TOOLKIT_PATH
    -- Custom build step to compile .cu files to .bc files
    filter {"files:src/Device/DevicePrograms/Clang/**.cu"}

        clangCompileCommand = genClangCommand(cu_file, bc_file,d_file, clang12, CUDA_TOOLKIT_PATH_8, include_dirs, true)

        printf("Clang cu Compile command:\n %s", clangCompileCommand)

        buildcommands {
            'echo "Running Clang:"' .. clangCompileCommand .. '"',
            clangCompileCommand,
        }
        buildoutputs {
            bc_file,
            d_file
        }

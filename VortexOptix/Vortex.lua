include "../PremakeScripts/path.lua"

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

project "OptixApp"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir (TARGET_DIR)
    objdir (OBJ_DIR)
    buildcustomizations "BuildCustomizations/CUDA 12.1"

    files{
        "%{IncludeDir.gdt}/**.h",
        "%{IncludeDir.gdt}/**.cpp",
        "src/**.h",
        "src/**.cpp",
        "src/Device/DevicePrograms/**.cu"
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

    useMdlDebug = false
    
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
            "{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/",
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

    -- include dirs for the compilation of ptx files
    local include_dirs = {
        IncludeDir["OPTIX"],
        IncludeDir["spdlog"],
        IncludeDir["MDL"],
        "src/",
        "../ext/gdt/"
    }
    -- Custom build step to compile .cu files to .ptx files
    filter {"configurations:Debug", "files:src/Device/DevicePrograms/**.cu"}
        
        cudaCompileCommand = genNvccCommand(cu_file, ptx_file, CUDA_TOOLKIT_PATH, include_dirs, true)
        
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommand .. '"',
            CudaCompileCommand,
        }
        buildoutputs {
            ptx_file,
        }

    -- Custom build step to compile .cu files to .ptx files
    filter {"configurations:Release", "files:src/Device/DevicePrograms/**.cu"}

        cudaCompileCommand = genNvccCommand(cu_file, ptx_file, CUDA_TOOLKIT_PATH, include_dirs, false)
        
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommand .. '"',
            CudaCompileCommand,
        }
        buildoutputs {
            ptx_file,
        }

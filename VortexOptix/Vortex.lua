include "../PremakeScripts/path.lua"

function genNvccCommand(cu_file, ptx_file, cuda_path, include_dirs)
    local nvcc_path = '"' .. path.join(cuda_path, "bin/nvcc.exe") .. '"'
    CudaCompileCommand = nvcc_path
    CudaCompileCommand = CudaCompileCommand .. " " .. cu_file
    CudaCompileCommand = CudaCompileCommand .. " --ptx"
    CudaCompileCommand = CudaCompileCommand .. " --generate-line-info"
    CudaCompileCommand = CudaCompileCommand .. " --use_fast_math"
    CudaCompileCommand = CudaCompileCommand .. " --keep"
    CudaCompileCommand = CudaCompileCommand .. " --relocatable-device-code=true"
    CudaCompileCommand = CudaCompileCommand .. " -Wno-deprecated-gpu-targets"
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
    buildcustomizations "BuildCustomizations/CUDA 12.0"

    
    postbuildcommands {
        "{COPY} %{wks.location}/VortexOptix/src/data %{cfg.targetdir}/data/"}

    files{
        "%{IncludeDir.gdt}/**.h",
        "%{IncludeDir.gdt}/**.cpp",
        "src/**.h",
        "src/**.cpp",
        "src/Renderer/DevicePrograms/**.cu"
    }

    includedirs{
        "src",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.GLAD}",
        "%{IncludeDir.ImGui}",
        "%{IncludeDir.spdlog}",
        "%{IncludeDir.OPTIX}",
        "%{IncludeDir.CUDA}",
        "%{IncludeDir.gdt}"
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

    -- Custom build step to compile .cu files to .ptx files
    filter "files:src/Renderer/DevicePrograms/**.cu"

        local cu_file = "%{file.relpath}"
        local ptx_file = "%{cfg.targetdir}/data/ptx/%{file.basename}.ptx"

        -- include dirs for the compilation of ptx files
        local include_dirs = {
            IncludeDir["OPTIX"],
            "src/",
            "../ext/gdt/"
        }
        cudaCompileCommand = genNvccCommand(cu_file, ptx_file, CUDA_TOOLKIT_PATH, include_dirs)
        
        local Output = "%{file.directory}/%{file.basename}.ptx"
        
        -- Build message showing the command being used
        -- buildmessage("Compiling %{file.relpath} with command: " .. Command)
        
        buildcommands {
            'echo "Running NVCC:"' .. CudaCompileCommand .. '"',
            CudaCompileCommand,
        }
        buildoutputs {
            ptx_file,
        }

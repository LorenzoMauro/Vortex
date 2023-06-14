function compileToPtx(cu_file, ptx_file, cuda_path, include_dirs, debug)
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

function compileToCudaObj(cu_file, obj_file, cuda_path, ccbin_path, sdk_dir, include_dirs, debug, macros)
    local nvcc_path = '"' .. path.join(cuda_path, "bin/nvcc.exe") .. '"'
    local CudaCompileCommand = nvcc_path
    
    -- Add each macro to the command
    if macros and type(macros) == "table" then
        for _, macro in ipairs(macros) do
            CudaCompileCommand = CudaCompileCommand .. " -D" .. macro
        end
    end
    --CudaCompileCommand = CudaCompileCommand .. " --use-local-env -ccbin " .. '"' .. ccbin_path .. '"'
    CudaCompileCommand = CudaCompileCommand .. " -x cu --keep-dir x64/" .. (debug and "Debug" or "Release")
    CudaCompileCommand = CudaCompileCommand .. " -maxrregcount=0"
    CudaCompileCommand = CudaCompileCommand .. " --machine 64"
    CudaCompileCommand = CudaCompileCommand .. " --compile"
    CudaCompileCommand = CudaCompileCommand .. " -cudart static"
    CudaCompileCommand = CudaCompileCommand .. " --extended-lambda"
    CudaCompileCommand = CudaCompileCommand .. " --expt-relaxed-constexpr"
    CudaCompileCommand = CudaCompileCommand .. " --std=c++17"
    CudaCompileCommand = CudaCompileCommand .. " --gpu-architecture=sm_70 --gpu-architecture=sm_86"
    
    if os.istarget("windows") then
        CudaCompileCommand = CudaCompileCommand .. " -Xcompiler=/bigobj"
    else
        CudaCompileCommand = CudaCompileCommand .. " -Xcompiler=-Wno-float-conversion"
        CudaCompileCommand = CudaCompileCommand .. " -Xcompiler=-fno-strict-aliasing"
        CudaCompileCommand = CudaCompileCommand .. " -Xcudafe=--diag_suppress=unrecognized_gcc_pragma"
    end
    
    CudaCompileCommand = CudaCompileCommand .. ' -o "' .. obj_file .. '" "' .. cu_file .. '"'

    -- Add additional include directories
    if include_dirs and type(include_dirs) == "table" then
        for _, include_dir in ipairs(include_dirs) do
            if not path.isabsolute(include_dir) then
                include_dir = path.getabsolute(include_dir)
            end
            CudaCompileCommand = CudaCompileCommand .. ' -I"' .. include_dir .. '"'
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

    ClangCompileCommand = ClangCompileCommand .. " --cuda-path=" .. '"' .. cuda_path .. '"'
    ClangCompileCommand = ClangCompileCommand .. " -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ClangCompileCommand = ClangCompileCommand .. " -std=c++17"
    ClangCompileCommand = ClangCompileCommand .. " -emit-llvm"
    ClangCompileCommand = ClangCompileCommand .. " -c"
    ClangCompileCommand = ClangCompileCommand .. " -w"
    ClangCompileCommand = ClangCompileCommand .. " -O3"
    ClangCompileCommand = ClangCompileCommand .. " -m64"
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
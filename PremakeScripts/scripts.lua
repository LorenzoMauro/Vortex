include "../PremakeScripts/path.lua"

function toAbsolutePaths(include_dirs)
    local absolute_paths = {}

    for _, include_dir in ipairs(include_dirs) do
        local absolute_path = '"' .. path.getabsolute(include_dir) .. '"'
        table.insert(absolute_paths, absolute_path)
    end

    return absolute_paths
end

function printTable(message, table)
    print(message)
    for k,v in pairs(table) do
        print(k,v)
    end
end

function nvccCompileToIntermediate(inputCudaFile, outputFile, outputType, cuda_path, include_dirs, defines, isDebug)
    local nvcc_path = '"' .. path.join(cuda_path, "bin/nvcc.exe") .. '"'
    local compileComand = nvcc_path
    compileComand = compileComand .. " " .. inputCudaFile
    compileComand = compileComand .. " -o " .. outputFile
    compileComand = compileComand .. " -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" "
    compileComand = compileComand .. " --use_fast_math"
    compileComand = compileComand .. " --keep"
    compileComand = compileComand .. " -Wno-deprecated-gpu-targets"
    compileComand = compileComand .. " --std=c++17"
    compileComand = compileComand .. " -m64"
    compileComand = compileComand .. " --keep-device-functions"
    compileComand = compileComand .. " --relocatable-device-code=true"
    compileComand = compileComand .. " --generate-line-info"
    compileComand = compileComand .. " -I" .. table.concat(include_dirs, " -I")
    compileComand = compileComand .. " -D " .. table.concat(defines, " -D ")
    --compileComand = compileComand .. " --split-compile=0"
    --compileComand = compileComand .. " --threads=0"
    
    if(outputType == "ptx") then
        compileComand = compileComand .. " -ptx"
    elseif(outputType == "optix-ir") then
        compileComand = compileComand .. " --optix-ir"
    else
        error("Unknown output type: " .. outputType)
    end

    if(isDebug) then 
        compileComand = compileComand .. " -G"
    else
        compileComand = compileComand .. " -O3"
    end

    return compileComand
end

function clangCompileCu(cu_file, bc_file,d_file, clang_path, cuda_path, include_dirs,defines, debug)
    local compileCommand = '"' .. clang_path .. '"'
    compileCommand = compileCommand .. " -I" .. table.concat(include_dirs, " -I")
    compileCommand = compileCommand .. " -D " .. table.concat(defines, " -D ")
    compileCommand = compileCommand .. " --cuda-path=" .. '"' .. cuda_path .. '"'
    compileCommand = compileCommand .. " -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    compileCommand = compileCommand .. " -std=c++17"
    compileCommand = compileCommand .. " -emit-llvm"
    compileCommand = compileCommand .. " -c"
    compileCommand = compileCommand .. " -w"
    compileCommand = compileCommand .. " -O3"
    compileCommand = compileCommand .. " -m64"
    compileCommand = compileCommand .. " -ffast-math"
    compileCommand = compileCommand .. " -fcuda-flush-denormals-to-zero"
    compileCommand = compileCommand .. " -fno-vectorize"
    compileCommand = compileCommand .. " --cuda-device-only"
    compileCommand = compileCommand .. " " .. cu_file
    compileCommand = compileCommand .. " -o" .. bc_file
    compileCommand = compileCommand .. " -MD -MT".. bc_file
    compileCommand = compileCommand .. " -MP -MF".. d_file
    return compileCommand
end

function nvccCompile(inputCudaFile, outputFile, cuda_path, ccbin_path, include_dirs,defines, isDebug)
    local nvcc_path = '"' .. path.join(cuda_path, "bin/nvcc.exe") .. '"'
    local compileCommand = nvcc_path
    compileCommand = compileCommand .. " -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" "
    compileCommand = compileCommand .. " --use-local-env"
    compileCommand = compileCommand .. " -ccbin \"" .. ccbin_path .. "\""
    compileCommand = compileCommand .. " -x cu"
    compileCommand = compileCommand .. " -I " .. table.concat(include_dirs, " -I ")
    compileCommand = compileCommand .. " -D " .. table.concat(defines, " -D ")
    compileCommand = compileCommand .. " --maxrregcount=0"
    compileCommand = compileCommand .. " --machine 64"
    compileCommand = compileCommand .. " --compile"
    compileCommand = compileCommand .. " -cudart static"
    compileCommand = compileCommand .. " -std=c++17"
    compileCommand = compileCommand .. " --extended-lambda"
    compileCommand = compileCommand .. " --use_fast_math"
    --compileCommand = compileCommand .. " --verbose"
    --compileCommand = compileCommand .. " --split-compile=0"
    --compileCommand = compileCommand .. " --threads=0"
    
    local xcompiler_base_options = {
        "/EHsc",
        "/W3",
        "/nologo",
        "/FS",
        string.format('/Fd"%s\\vc143.pdb"', "%{cfg.objectdir}")
    }

    if(isDebug) then
        compileCommand = compileCommand .. " -G -g"
        local xcompiler_debug_options = {
            "/Od",
            "/Zi",
            "/RTC1",
            "/MDd"
        }
        compileCommand = compileCommand .. " -Xcompiler \"" .. table.concat(xcompiler_base_options, " ") .. " " .. table.concat(xcompiler_debug_options, " ") .. "\""
        compileCommand = compileCommand .. " --keep-dir x64\\Debug"
    else
        compileCommand = compileCommand .. " -O3"
        local xcompiler_release_options = {
            "/MD",
            "/Ox"
        }
        compileCommand = compileCommand .. " -Xcompiler \"" .. table.concat(xcompiler_base_options, " ") .. " " .. table.concat(xcompiler_release_options, " ") .. "\""
        compileCommand = compileCommand .. " --keep-dir x64\\Release"
    end

    compileCommand = compileCommand .. " -D " .. table.concat(defines, " -D ")
    compileCommand = compileCommand .. " -o " .. outputFile
    compileCommand = compileCommand .. " " .. inputCudaFile

    return compileCommand
end

function generatePostBuildCommands(isDebug, useMdlDebug)
    local mdlPath  -- Declare mdlPath as a local variable

    if (isDebug and useMdlDebug) then
        mdlPath = '%{wks.location}ext\\MDL\\debug'
    else
        mdlPath = '%{wks.location}ext\\MDL\\release'
    end

    local libTorchPath
    if (isDebug) then
        libTorchPath = LibDir["LibTorch_Debug"]
    else
        libTorchPath = LibDir["LibTorch_Release"]
    end

    local commands = {
        "{MKDIR} %{cfg.targetdir}/lib",
        'xcopy /Y /I /D "' .. mdlPath .. '\\bin\\libmdl_sdk.dll" "%{cfg.targetdir}\\lib\\"',
        'xcopy /Y /I /D "' .. mdlPath .. '\\bin\\nv_freeimage.dll" "%{cfg.targetdir}\\lib\\"',
        'xcopy /Y /I /D "' .. mdlPath .. '\\bin\\freeimage.dll" "%{cfg.targetdir}\\lib\\"',
        'xcopy "' .. libTorchPath .. '" "%{cfg.targetdir}" /D /Y',
        'xcopy /Y /I /D "' .. mdlPath .. '\\bin\\dds.dll" "%{cfg.targetdir}\\lib\\"'
    }


    local echoedCommands = {}

    for i, command in ipairs(commands) do
        table.insert(echoedCommands, 'echo "Running command: ' .. command .. '"')
        table.insert(echoedCommands, command)
    end

    return echoedCommands
end

function getLibDirs(isDebug, useMdlDebug)
    local mdlDir = (isDebug and useMdlDebug) and "%{LibDir.MDL_Debug}" or "%{LibDir.MDL_Release}"
    local libTorch

    local libTorchPath
    if (isDebug) then
        libTorchPath = LibDir["LibTorch_Debug"]
    else
        libTorchPath = LibDir["LibTorch_Release"]
    end

    return {
        mdlDir,
        libTorchPath,
        "%{LibDir.CUDA}",
        "%{LibDir.ASSIMP}",
        "%{LibDir.nvToolsExt64}",
        '%{cfg.targetdir}/lib'
    }
end
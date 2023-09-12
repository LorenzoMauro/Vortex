
#Vortex CXX files
file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

#Vortex CU files to be compiled to obj
file(GLOB CUDA_FUNCTIONS "${CMAKE_CURRENT_SOURCE_DIR}/Device/CudaFunctions/*.cu")
file(GLOB DEVICE_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/Device/DevicePrograms/Cuda/*.cu")
file(GLOB NETWORK_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetworks/*.cu")
list(APPEND STANDARD_CUDA_COMPILE_FILES ${CUDA_FUNCTIONS} ${DEVICE_CUDA} ${NETWORK_CUDA})

#Vortex CU files to be compiled to intermediate representation
file(GLOB PTX_COMPILE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Device/DevicePrograms/ptx/*.cu")
file(GLOB OPTIXIR_COMPILE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Device/DevicePrograms/optixIr/*.cu")
file(GLOB CLANG_COMPILE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/Device/DevicePrograms/Clang/*.cu")

#Dependencies for the custom cuda compilation step with nvcc
list(APPEND CU_DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OPTIX77_INCLUDE_DIR} ${MDL_SDK_INCLUDE_DIRS} ${SPDLOG_INCLUDE_DIRS})

#Intermediate CUDA Compilation Flags
list(APPEND CU_COMPILE_FLAGS "-gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\"")
list(APPEND CU_COMPILE_FLAGS "--use_fast_math")
list(APPEND CU_COMPILE_FLAGS "--keep")
list(APPEND CU_COMPILE_FLAGS "-Wno-deprecated-gpu-targets")
list(APPEND CU_COMPILE_FLAGS "--std=c++17")
list(APPEND CU_COMPILE_FLAGS "--machine=64")
list(APPEND CU_COMPILE_FLAGS "--keep-device-functions")
list(APPEND CU_COMPILE_FLAGS "--relocatable-device-code=true")
list(APPEND CU_COMPILE_FLAGS "--generate-line-info")
list(APPEND CU_COMPILE_FLAGS "$<$<CONFIG:Debug>:-G>$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-O3>")
foreach(dep ${CU_DEPENDENCIES})
  list(APPEND CU_COMPILE_FLAGS "-I \"${dep}\"")
endforeach()

#Files to be used by optix
NVCUDA_COMPILE_MODULE(
  SOURCES ${OPTIXIR_COMPILE_FILES}
  DEPENDENCIES ${CU_DEPENDENCIES}
  TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/ptx"
  EXTENSION ".optixir"
  GENERATED_FILES OPTIXIR_MODULES
  NVCC_OPTIONS "--optix-ir" ${CU_COMPILE_FLAGS}
  COMMAND_EXPAND_LISTS
  )

#files to be used by wavefront kernel with MDL Just in time Linking of material functions
NVCUDA_COMPILE_MODULE(
  SOURCES ${PTX_COMPILE_FILES}
  DEPENDENCIES ${CU_DEPENDENCIES}
  TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/ptx"
  EXTENSION ".ptx"
  GENERATED_FILES PTX_MODULES
  NVCC_OPTIONS "-ptx" ${CU_COMPILE_FLAGS}
  COMMAND_EXPAND_LISTS
)

# Clang Compilation of Cu files for MDL Inlining of functions
list(APPEND CLANG_DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR} "${CUDA_8_PATH}/include" ${OPTIX77_INCLUDE_DIR} ${MDL_SDK_INCLUDE_DIRS} ${SPDLOG_INCLUDE_DIRS})
list(APPEND CLANG_COMPILE_COMMAND "${CLANG_12_PATH}")
list(APPEND CLANG_COMPILE_COMMAND "--cuda-path=${CUDA_8_PATH}")
list(APPEND CLANG_COMPILE_COMMAND "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
list(APPEND CLANG_COMPILE_COMMAND "-std=c++17")
list(APPEND CLANG_COMPILE_COMMAND "-emit-llvm")
list(APPEND CLANG_COMPILE_COMMAND "-c")
list(APPEND CLANG_COMPILE_COMMAND "-w")
list(APPEND CLANG_COMPILE_COMMAND "-O3")
list(APPEND CLANG_COMPILE_COMMAND "-m64")
list(APPEND CLANG_COMPILE_COMMAND "-ffast-math")
list(APPEND CLANG_COMPILE_COMMAND "-fcuda-flush-denormals-to-zero")
list(APPEND CLANG_COMPILE_COMMAND "-fno-vectorize")
list(APPEND CLANG_COMPILE_COMMAND "--cuda-device-only")
list(APPEND CLANG_COMPILE_COMMAND "-w")
foreach(dep ${CLANG_DEPENDENCIES})
  list(APPEND CLANG_COMPILE_COMMAND "-I \"${dep}\"")
  endforeach()

CLANG_COMPILE_MODULE(
  SOURCES ${CLANG_COMPILE_FILES}
  DEPENDENCIES ${CLANG_DEPENDENCIES}
  TARGET_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bc"
  EXTENSION ".bc"
  GENERATED_FILES CLANG_MODULES
  CLANG_OPTIONS ${CLANG_COMPILE_COMMAND}
  COMMAND_EXPAND_LISTS
)

#Organize Files into Folder Like Filters
assign_source_group(
    ${SOURCES}
    ${HEADERS}
    ${STANDARD_CUDA_COMPILE_FILES}
    ${PTX_COMPILE_FILES}
    ${OPTIXIR_COMPILE_FILES}
    ${CLANG_COMPILE_FILES}
    ${OPTIXIR_MODULES}
    ${PTX_MODULES}
    ${CLANG_MODULES}
  )

add_executable(Vortex Core/main.cpp)
target_sources(Vortex PRIVATE 
  ${SOURCES}
  ${HEADERS}
  ${STANDARD_CUDA_COMPILE_FILES}
  ${PTX_COMPILE_FILES}
  ${OPTIXIR_COMPILE_FILES}
  ${CLANG_COMPILE_FILES}
  ${OPTIXIR_MODULES}
  ${PTX_MODULES}
  ${CLANG_MODULES}
)
set_property(TARGET Vortex PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Vortex PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:Vortex>"
)

#Compile Options for Standard CUDA files
target_compile_options(Vortex PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math 
    --relocatable-device-code=true
    -gencode=arch=compute_70,code="sm_70,compute_70"
    --use-local-env
    -x cu
    --maxrregcount=0
    --machine 64
    --compile
    -cudart static
    -std=c++17
    --extended-lambda
  >
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-O3>
)

target_link_directories(Vortex PRIVATE
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib
)

#libraries to be used by Torch
#even though Torch provide a cmake to use, there is no way to have both debug and release in the same VS solution using it
set(TORCH_DEBUG_LIBS 
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/c10.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/kineto.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/caffe2_nvrtc.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/c10_cuda.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/torch.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/torch_cuda.lib
  ${TORCH_INSTALL_PREFIX_DEBUG}/lib/torch_cpu.lib
)

set(TORCH_RELEASE_LIBS 
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/c10.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/kineto.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/caffe2_nvrtc.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/c10_cuda.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/torch.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/torch_cuda.lib
  ${TORCH_INSTALL_PREFIX_RELEASE}/lib/torch_cpu.lib
)

target_link_libraries(Vortex PRIVATE 
  imgui::imgui
  spdlog::spdlog_header_only
  glfw
  implot::implot
  Glad
  yaml-cpp
  imnodes
  ImGuiFileDialog
  assimp::assimp
  CUDA::cudart_static
  CUDA::cuda_driver
  CUDA::nvtx3
  CUDA::cufft
  CUDA::curand
  CUDA::cublas
  CUDA::cublasLt
  $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${TORCH_RELEASE_LIBS}>
  $<$<CONFIG:Debug>:${TORCH_DEBUG_LIBS}>
)
#Options from Torch, not sure what is the need of this
if(MSVC)
  target_link_options(yourTargetName PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
endif()

target_include_directories(Vortex PRIVATE
  ${CMAKE_SOURCE_DIR}/Vortex/src
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${OPTIX77_INCLUDE_DIR}
  ${MDL_SDK_INCLUDE_DIRS}
  ${EXTERNAL}/imnodes
  ${EXTERNAL}/ImGuiFileDialog
  ${EXTERNAL}/stbImage
  ${TORCH_INSTALL_PREFIX_DEBUG}/include
  ${TORCH_INSTALL_PREFIX_DEBUG}/include/torch/csrc/api/include
)

target_compile_definitions(Vortex PRIVATE
     # Disable warnings for file operations fopen etc.
    _CRT_SECURE_NO_WARNINGS
    # DAR HACK Set this when building against the MDL SDK open-source version.
    # There is some case inside the load_plugin() function which special cases the nv_freeimage.dll loading, but that is actually not reached.
    MDL_SOURCE_RELEASE
    NOMINMAX
    $<$<CONFIG:Debug>:DEBUG>
)

#disabling some annoying warning coming from Torch
if(MSVC)
  target_compile_options(Vortex PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/wd4251;/wd4275;/wd4624>
  )
endif()

# Copying of Torch DLLs to exectutable file folder
file(GLOB TORCH_RELEASE_DLLS "${TORCH_INSTALL_PREFIX_RELEASE}/lib/*.dll") 
file(GLOB TORCH_DEBUG_DLLS "${TORCH_INSTALL_PREFIX_DEBUG}/lib/*.dll") 

list(LENGTH TORCH_DEBUG_DLLS len)
math(EXPR len "${len} - 1")

foreach(idx RANGE ${len})
  list(GET TORCH_DEBUG_DLLS ${idx} debug_dll)
  list(GET TORCH_RELEASE_DLLS ${idx} release_dll)

  add_custom_command(TARGET Vortex POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E echo "Copying ${debug_dll} or ${release_dll}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     $<$<CONFIG:Debug>:${debug_dll}>$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${release_dll}>
                     $<TARGET_FILE_DIR:Vortex>)
endforeach()

                   
# Copying of MDL DLLs to exectutable file folder
set(MDL_DLLS 
  "${MDL_SDK_PATH}/build/install/bin/libmdl_sdk.dll"
  "${MDL_SDK_PATH}/build/install/bin/nv_openimageio.dll"
  "${MDL_SDK_PATH}/build/install/bin/dds.dll"
)
add_custom_command(TARGET Vortex POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${MDL_DLLS}
                   $<TARGET_FILE_DIR:Vortex>)
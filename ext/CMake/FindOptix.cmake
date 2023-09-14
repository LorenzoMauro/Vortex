# Looks for the environment variable:
# OPTIX_PATH

# Sets the variables :
# OPTIX_INCLUDE_DIR

# OptiX_FOUND

if ("${OPTIX_PATH}" STREQUAL "")
  set(OPTIX_PATH $ENV{OPTIX_PATH})
endif()

if ("${OPTIX_PATH}" STREQUAL "")
  if (WIN32)
    # Try finding it inside the default installation directory under Windows first.
    set(OPTIX_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
  else()
    # Adjust this if the OptiX SDK 7.0.0 installation is in a different location.
    set(OPTIX_PATH "~/NVIDIA-OptiX-SDK-8.0.0-linux64")
  endif()
endif()

find_path(OPTIX_INCLUDE_DIR optix_host.h ${OPTIX_PATH}/include)

message("OPTIX_INCLUDE_DIR = " "${OPTIX_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX DEFAULT_MSG OPTIX_INCLUDE_DIR)

mark_as_advanced(OPTIX_INCLUDE_DIR)

message("OptiX_FOUND = " "${OptiX_FOUND}")

# Add Glad as a static library
add_library(Glad STATIC "src/glad.c")

set_target_properties(Glad PROPERTIES LINKER_LANGUAGE C)

# Specify the include directories
target_include_directories(Glad PUBLIC "include")

# Optional: Add preprocessor definitions and system version if you're on Windows
if(WIN32)
    target_compile_definitions(Glad PUBLIC "GLFW_INCLUDE_NONE")
endif()
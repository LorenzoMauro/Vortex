project "Glad"
    kind "StaticLib"
    language "C"
    staticruntime "off"
    
    targetdir (TARGET_DIR)
    objdir (OBJ_DIR)
    
    files
    {
        "include/glad/glad.h",
        "include/KHR/khrplatform.h",
        "src/glad.c"
    }

    includedirs{
        "include"
    }
    
    filter "system:windows"
        systemversion "latest"
        defines
        {
            "GLFW_INCLUDE_NONE"
        }
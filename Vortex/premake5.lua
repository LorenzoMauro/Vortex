-- premake5.lua

include "vendor/Walnut/WalnutExternal.lua"

project "Vortex"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files
   {
      "src/**.h",
      "src/**.cpp" 
   }
    
   includedirs
   {
      "vendor/Walnut/vendor/imgui",
      "vendor/Walnut/vendor/glfw/include",
      "vendor/Walnut/vendor/glm",
      "vendor/Walnut/Walnut/src",
      "vendor/spdlog/include",

      "%{IncludeDir.VulkanSDK}",
   }

   links
   {
       "Walnut",
   }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG" }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE" }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST" }
      runtime "Release"
      optimize "On"
      symbols "Off"


group "Dependencies"
    include "vendor/Walnut/vendor/imgui"
    include "vendor/Walnut/vendor/glfw"
    include "vendor/Walnut"
group ""

group "Core"
    include "Vortex"
group ""
-- premake5.lua
workspace "Vortex"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "Vortex"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

include "Vortex"
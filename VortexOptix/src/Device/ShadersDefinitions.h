#pragma once
#include <map>
#include <string>

namespace vtx {
	
	/////////////////////////////////////////////////////////////////
	//////////// Module Definitions /////////////////////////////////
	/////////////////////////////////////////////////////////////////
	enum ModuleIdentifier {
		MODULE_ID_DEVICEPROGRAM,
		MODULE_ID_CAMERAFUNCTIONS,

		NUM_MODULE_IDENTIFIERS
	};

	inline std::map<ModuleIdentifier, std::string> MODULE_FILENAME = {
		{ MODULE_ID_DEVICEPROGRAM,		"./data/ptx/devicePrograms.optixir" },
		{ MODULE_ID_CAMERAFUNCTIONS,	"./data/ptx/CameraFunctions.optixir" }
	};

	/////////////////////////////////////////////////////////////////
	//////////// Function Definitions ///////////////////////////////
	/////////////////////////////////////////////////////////////////
	enum FunctionIdentifier {
		__RAYGEN__RENDERFRAME,
		__MISS__RADIANCE,
		__CLOSESTHIT__RADIANCE,
		__ANYHIT__RADIANCE,
		__EXCEPTION__ALL,

		__CALLABLE__PINHOLECAMERA,

		NUM_FUNCTION_NAMES
	};

	struct functionProperties {
		char* name;
		ModuleIdentifier module;
	};

	inline std::map<FunctionIdentifier, functionProperties> funcPropMap = {
		{ __RAYGEN__RENDERFRAME,	functionProperties{"__raygen__renderFrame",		MODULE_ID_DEVICEPROGRAM} },
		{ __MISS__RADIANCE,			functionProperties{"__miss__radiance",			MODULE_ID_DEVICEPROGRAM} },
		{ __CLOSESTHIT__RADIANCE,	functionProperties{"__closesthit__radiance",	MODULE_ID_DEVICEPROGRAM} },
		{ __ANYHIT__RADIANCE,		functionProperties{"__anyhit__radiance",		MODULE_ID_DEVICEPROGRAM} },
		{ __EXCEPTION__ALL,			functionProperties{"__exception__all",			MODULE_ID_DEVICEPROGRAM} },
		{ __CALLABLE__PINHOLECAMERA,functionProperties{"__direct_callable__pinhole",MODULE_ID_CAMERAFUNCTIONS} }


	};

	/////////////////////////////////////////////////////////////////
	//////////// Program Group Definitions //////////////////////////
	/////////////////////////////////////////////////////////////////

	enum ProgramGroupIdentifier {
		/// Raygen
		PROGRAMGROUP_ID_RAYGEN,
		/// Miss
		PROGRAMGROUP_ID_MISS,
		/// Hit
		PROGRAMGROUP_ID_HIT,
		/// Exception
		PROGRAMGROUP_ID_ECXEPTION,
		/// Camera
		PROGRAMGROUP_ID_CAMERA,

		NUM_PROGRAM_GROUPS
	};

#define FIRST_MISS_PROGRAM PROGRAMGROUP_ID_MISS
#define FIRST_HIT_PROGRAM PROGRAMGROUP_ID_HIT
#define FIRST_EXCEPTION_PROGRAM PROGRAMGROUP_ID_ECXEPTION
#define FIRST_CALLABLE_PROGRAM PROGRAMGROUP_ID_CAMERA

	inline std::map<ProgramGroupIdentifier, unsigned int> SBT_POSITION{
		{PROGRAMGROUP_ID_RAYGEN,		(PROGRAMGROUP_ID_RAYGEN		- PROGRAMGROUP_ID_RAYGEN)},
		{PROGRAMGROUP_ID_MISS,			(PROGRAMGROUP_ID_MISS		- FIRST_MISS_PROGRAM)},
		{PROGRAMGROUP_ID_HIT,			(PROGRAMGROUP_ID_HIT		- FIRST_HIT_PROGRAM)},
		{PROGRAMGROUP_ID_ECXEPTION,		(PROGRAMGROUP_ID_ECXEPTION	- FIRST_EXCEPTION_PROGRAM)},
		{PROGRAMGROUP_ID_CAMERA,		(PROGRAMGROUP_ID_CAMERA		- FIRST_CALLABLE_PROGRAM)}
	};

#define NUM_MISS_PROGRAM 1
#define NUM_HIT_PROGRAM 1
#define NUM_EXCEPTION_PROGRAM 1
#define NUM_CALLABLE_PROGRAM 1

	/////////////////////////////////////////////////////////////////
	//////////// Pipline ////////////////////////////////////////////

	inline std::string LaunchParamName = "optixLaunchParams";

}

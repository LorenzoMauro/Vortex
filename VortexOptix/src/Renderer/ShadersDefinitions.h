#pragma once
#include <map>
#include <string>
#include "Core/Utils.h"

namespace vtx {
	
	/////////////////////////////////////////////////////////////////
	//////////// Module Definitions /////////////////////////////////
	/////////////////////////////////////////////////////////////////
	enum ModuleIdentifier {
		MODULE_ID_DEVICEPROGRAM,
		MODULE_ID_CAMERAFUNCTIONS,

		NUM_MODULE_IDENTIFIERS
	};

	inline std::string modulePath = "./data/ptx/";

	inline std::map<ModuleIdentifier, std::string> MODULE_FILENAME = {
		{ MODULE_ID_DEVICEPROGRAM,		utl::absolutePath(modulePath + "devicePrograms.optixir") },
		{ MODULE_ID_CAMERAFUNCTIONS,	utl::absolutePath(modulePath + "CameraFunctions.optixir") }
	};

	/////////////////////////////////////////////////////////////////
	//////////// Function Definitions ///////////////////////////////
	/////////////////////////////////////////////////////////////////
	enum FunctionIdentifier {
		__RAYGEN__RENDERFRAME,
		__MISS__RADIANCE,
		__CLOSESTHIT__RADIANCE,
		__ANYHIT__RADIANCE,

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
		{ __CLOSESTHIT__RADIANCE,	functionProperties{"__closesthit__radiance",		MODULE_ID_DEVICEPROGRAM} },
		{ __ANYHIT__RADIANCE,		functionProperties{"__anyhit__radiance",			MODULE_ID_DEVICEPROGRAM} },
		{ __CALLABLE__PINHOLECAMERA,functionProperties{"__direct_callable__pinhole",	MODULE_ID_CAMERAFUNCTIONS} }


	};

	/////////////////////////////////////////////////////////////////
	//////////// Program Group Definitions //////////////////////////
	/////////////////////////////////////////////////////////////////

	enum ProgramGroupIdentifier {
		PROGRAMGROUP_ID_RAYGEN,
		PROGRAMGROUP_ID_MISS,
		PROGRAMGROUP_ID_HIT,
		PROGRAMGROUP_ID_CAMERA,

		NUM_PROGRAM_GROUPS
	};

	inline std::map<ProgramGroupIdentifier, int> SBT_POSITION{};

	/////////////////////////////////////////////////////////////////
	//////////// Pipline ////////////////////////////////////////////

	inline std::string LaunchParamName = "optixLaunchParams";

}

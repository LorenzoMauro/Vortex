#include "OptixWrapper.h"
#include "PipelineConfiguration.h"
#include "Core/Utils.h"

namespace vtx {
	void pipelineConfiguration()
	{
		///////////////////////////////////////////////////////////////////
		/////////////////////// Modules ///////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto deviceProgramModule = std::make_shared<optix::ModuleOptix>();
		deviceProgramModule->name = "deviceProgram";
		deviceProgramModule->path = "./data/ptx/devicePrograms.optixir";

		auto cameraFunctionsModule = std::make_shared<optix::ModuleOptix>();
		cameraFunctionsModule->name = "cameraFunctions";
		cameraFunctionsModule->path = "./data/ptx/CameraFunctions.optixir";

		auto lightSamplingModule = std::make_shared<optix::ModuleOptix>();
		lightSamplingModule->name = "lightSampling";
		lightSamplingModule->path = "./data/ptx/lightSampling.optixir";

		///////////////////////////////////////////////////////////////////
		/////////////////////// Functions /////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto renderFrameFunction = std::make_shared<optix::FunctionOptix>();
		renderFrameFunction->name = "__raygen__renderFrame";
		renderFrameFunction->module = deviceProgramModule;
		renderFrameFunction->type = optix::OptixFunctionType::F_Raygen;

		auto exceptionFunction = std::make_shared<optix::FunctionOptix>();
		exceptionFunction->name = "__exception__all";
		exceptionFunction->module = deviceProgramModule;
		exceptionFunction->type = optix::OptixFunctionType::F_Exception;

		auto missFunction = std::make_shared<optix::FunctionOptix>();
		missFunction->name = "__miss__radiance";
		missFunction->module = deviceProgramModule;
		missFunction->type = optix::OptixFunctionType::F_Miss;

		auto closestHitFunction = std::make_shared<optix::FunctionOptix>();
		closestHitFunction->name = "__closesthit__radiance";
		closestHitFunction->module = deviceProgramModule;
		closestHitFunction->type = optix::OptixFunctionType::F_ClosestHit;

		auto anyHitFunction = std::make_shared<optix::FunctionOptix>();
		anyHitFunction->name = "__anyhit__radiance";
		anyHitFunction->module = deviceProgramModule;
		anyHitFunction->type = optix::OptixFunctionType::F_AnyHit;

		//auto pinholeFunction = std::make_shared<optix::FunctionOptix>();
		//pinholeFunction->name = "__direct_callable__pinhole";
		//pinholeFunction->module = cameraFunctionsModule;
		//pinholeFunction->type = optix::OptixFunctionType::F_DirectCallable;

		///////////////////////////////////////////////////////////////////
		/////////////////////// Programs //////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto rayGenProgram = std::make_shared<optix::ProgramOptix>();
		rayGenProgram->name = "raygen";
		rayGenProgram->type = optix::OptixProgramType::P_Raygen;
		rayGenProgram->raygenFunction = renderFrameFunction;

		auto exceptionProgram = std::make_shared<optix::ProgramOptix>();
		exceptionProgram->name = "exception";
		exceptionProgram->type = optix::OptixProgramType::P_Exception;
		exceptionProgram->exceptionFunction = exceptionFunction;

		auto missProgram = std::make_shared<optix::ProgramOptix>();
		missProgram->name = "miss";
		missProgram->type = optix::OptixProgramType::P_Miss;
		missProgram->missFunction = missFunction;

		auto hitProgram = std::make_shared<optix::ProgramOptix>();
		hitProgram->name = "hit";
		hitProgram->type = optix::OptixProgramType::P_Hit;
		hitProgram->closestHitFunction = closestHitFunction;
		hitProgram->anyHitFunction = anyHitFunction;


		//auto pinholeProgram = std::make_shared<optix::ProgramOptix>();
		//pinholeProgram->name = "pinHole";
		//pinholeProgram->type = optix::OptixProgramType::P_DirectCallable;
		//pinholeProgram->directCallableFunction = pinholeFunction;

		optix::PipelineOptix* pipeline = optix::getRenderingPipeline();

		pipeline->registerProgram(rayGenProgram);
		pipeline->registerProgram(exceptionProgram);
		pipeline->registerProgram(missProgram);
		pipeline->registerProgram(hitProgram);
		//pipeline->registerProgram(pinholeProgram);

		optix::createDcProgram(cameraFunctionsModule, "__direct_callable__pinhole");
		optix::createDcProgram(lightSamplingModule, "__direct_callable__meshLightSample");
		optix::createDcProgram(lightSamplingModule, "__direct_callable__envLightSample");
	}

}

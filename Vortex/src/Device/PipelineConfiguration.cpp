#include "OptixWrapper.h"
#include "PipelineConfiguration.h"
#include "Core/Utils.h"
#include "MDL/CudaLinker.h"

namespace vtx {
	void pipelineConfiguration()
	{
		///////////////////////////////////////////////////////////////////
		/////////////////////// Modules ///////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto deviceProgramModule = std::make_shared<optix::ModuleOptix>();
		deviceProgramModule->name = "deviceProgram";
		deviceProgramModule->path = "./ptx/devicePrograms.optixir";

		auto cameraFunctionsModule = std::make_shared<optix::ModuleOptix>();
		cameraFunctionsModule->name = "cameraFunctions";
		cameraFunctionsModule->path = "./ptx/CameraFunctions.optixir";

		auto lightSamplingModule = std::make_shared<optix::ModuleOptix>();
		lightSamplingModule->name = "deviceProgram";
		lightSamplingModule->path = "./ptx/devicePrograms.optixir";
		lightSamplingModule->name = "lightSampling";
		lightSamplingModule->path = "./ptx/lightSampling.optixir";

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

		optix::createDcProgram(deviceProgramModule, "__direct_callable__pinhole");
		optix::createDcProgram(deviceProgramModule, "__direct_callable__meshLightSample");
		optix::createDcProgram(deviceProgramModule, "__direct_callable__envLightSample");

		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Radiance Trace ////////////////////////
		///////////////////////////////////////////////////////////////////

		auto rtRaygenF= std::make_shared<optix::FunctionOptix>();
		rtRaygenF->name = "__raygen__rtRaygen";
		rtRaygenF->module = deviceProgramModule;
		rtRaygenF->type = optix::OptixFunctionType::F_Raygen;

		auto rtMissF = std::make_shared<optix::FunctionOptix>();
		rtMissF->name = "__miss__rtMiss";
		rtMissF->module = deviceProgramModule;
		rtMissF->type = optix::OptixFunctionType::F_Miss;

		auto rtClosestF = std::make_shared<optix::FunctionOptix>();
		rtClosestF->name = "__closesthit__rtClosest";
		rtClosestF->module = deviceProgramModule;
		rtClosestF->type = optix::OptixFunctionType::F_ClosestHit;

		auto rtAnyF = std::make_shared<optix::FunctionOptix>();
		rtAnyF->name = "__anyhit__dummyAnyHit";
		rtAnyF->module = deviceProgramModule;
		rtAnyF->type = optix::OptixFunctionType::F_AnyHit;


		auto rtRaygenP = std::make_shared<optix::ProgramOptix>();
		rtRaygenP->name = "rtRaygen";
		rtRaygenP->type = optix::OptixProgramType::P_Raygen;
		rtRaygenP->raygenFunction = rtRaygenF;

		auto rtMissP = std::make_shared<optix::ProgramOptix>();
		rtMissP->name = "rtMiss";
		rtMissP->type = optix::OptixProgramType::P_Miss;
		rtMissP->missFunction = rtMissF;

		auto rtHitP = std::make_shared<optix::ProgramOptix>();
		rtHitP->name = "rtHit";
		rtHitP->type = optix::OptixProgramType::P_Hit;
		rtHitP->closestHitFunction = rtClosestF;
		rtHitP->anyHitFunction = rtAnyF;

		pipeline->registerProgram(rtRaygenP, { "wfRadianceTrace" });
		pipeline->registerProgram(rtMissP, { "wfRadianceTrace" });
		pipeline->registerProgram(rtHitP, { "wfRadianceTrace" });
		//pipeline->registerProgram(exceptionProgram, { "WaveFront_ClosestHit" }); //TODO Handle Duplicate Programs in pipeline Creation


		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Radiance Trace ////////////////////////
		///////////////////////////////////////////////////////////////////


		mdl::MdlCudaLinker& mdlCudaLinker = mdl::getMdlCudaLinker();
		mdlCudaLinker.ptxFile = "./ptx/shadeKernel.ptx";
		mdlCudaLinker.kernelFunctionName = "shadeKernel";

		auto shadeRaygenF= std::make_shared<optix::FunctionOptix>();
		shadeRaygenF->name = "__raygen__shade";
		shadeRaygenF->module = deviceProgramModule;
		shadeRaygenF->type = optix::OptixFunctionType::F_Raygen;

		auto shadeMissF = std::make_shared<optix::FunctionOptix>();
		shadeMissF->name = "__miss__shadeShadowMiss";
		shadeMissF->module = deviceProgramModule;
		shadeMissF->type = optix::OptixFunctionType::F_Miss;

		auto shadeClosestF = std::make_shared<optix::FunctionOptix>();
		shadeClosestF->name = "__closesthit__shadeDummy";
		shadeClosestF->module = deviceProgramModule;
		shadeClosestF->type = optix::OptixFunctionType::F_ClosestHit;

		auto shadeAnyF = std::make_shared<optix::FunctionOptix>();
		shadeAnyF->name = "__anyhit__shadeShadowHit";
		shadeAnyF->module = deviceProgramModule;
		shadeAnyF->type = optix::OptixFunctionType::F_AnyHit;


		auto shadeRaygenP = std::make_shared<optix::ProgramOptix>();
		shadeRaygenP->name = "shadeRaygen";
		shadeRaygenP->type = optix::OptixProgramType::P_Raygen;
		shadeRaygenP->raygenFunction = shadeRaygenF;

		auto shadeMissP = std::make_shared<optix::ProgramOptix>();
		shadeMissP->name = "shadeMiss";
		shadeMissP->type = optix::OptixProgramType::P_Miss;
		shadeMissP->missFunction = shadeMissF;

		auto shadeHitP = std::make_shared<optix::ProgramOptix>();
		shadeHitP->name = "shadeHit";
		shadeHitP->type = optix::OptixProgramType::P_Hit;
		shadeHitP->closestHitFunction = shadeClosestF;
		shadeHitP->anyHitFunction = shadeAnyF;

		pipeline->registerProgram(shadeRaygenP, { "wfShade" });
		pipeline->registerProgram(shadeMissP, { "wfShade" });
		pipeline->registerProgram(shadeHitP, { "wfShade" });
		//pipeline->registerProgram(exceptionProgram, { "WaveFront_ClosestHit" }); //TODO Handle Duplicate Programs in pipeline Creation
		
	}
}

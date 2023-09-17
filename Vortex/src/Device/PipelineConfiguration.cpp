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
		optix::PipelineOptix* pipeline = optix::getRenderingPipeline();

		//////////////////////////////////////////////////////////////////////
		////////////////// Module ////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////
		
		auto deviceProgramModule = std::make_shared<optix::ModuleOptix>();
		deviceProgramModule->name = "deviceProgram";
		deviceProgramModule->path = getOptions()->executablePath + "ptx/devicePrograms.optixir";




		//////////////////////////////////////////////////////////////////////
		////////////////// Exception /////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////
		
		/*auto exceptionFunction = std::make_shared<optix::FunctionOptix>();
		exceptionFunction->name = "__exception__all";
		exceptionFunction->module = deviceProgramModule;
		exceptionFunction->type = optix::OptixFunctionType::F_Exception;

		auto exceptionProgram = std::make_shared<optix::ProgramOptix>();
		exceptionProgram->name = "exception";
		exceptionProgram->type = optix::OptixProgramType::P_Exception;
		exceptionProgram->exceptionFunction = exceptionFunction;*/


		///////////////////////////////////////////////////////////////////
		/////////////////////// Radiance Trace Functions //////////////////
		///////////////////////////////////////////////////////////////////
		
		auto closestHitFunction = std::make_shared<optix::FunctionOptix>();
		closestHitFunction->name = "__closesthit__radiance";
		closestHitFunction->module = deviceProgramModule;
		closestHitFunction->type = optix::OptixFunctionType::F_ClosestHit;


		auto missFunction = std::make_shared<optix::FunctionOptix>();
		missFunction->name = "__miss__radiance";
		missFunction->module = deviceProgramModule;
		missFunction->type = optix::OptixFunctionType::F_Miss;

		//////////////////////////////////////////////////////////////////////
		////////////////// Shadow Trace //////////////////////////////////////
		//////////////////////////////////////////////////////////////////////


		auto shadowMissF = std::make_shared<optix::FunctionOptix>();
		shadowMissF->name = "__miss__shadowMiss";
		shadowMissF->module = deviceProgramModule;
		shadowMissF->type = optix::OptixFunctionType::F_Miss;

		auto shadowHitF = std::make_shared<optix::FunctionOptix>();
		shadowHitF->name = "__anyhit__shadowHit";
		shadowHitF->module = deviceProgramModule;
		shadowHitF->type = optix::OptixFunctionType::F_AnyHit;


		//////////////////////////////////////////////////////////////////////
		/////////////////// Radiance Trace Programs //////////////////////////
		//////////////////////////////////////////////////////////////////////


		auto radianceMissP = std::make_shared<optix::ProgramOptix>();
		radianceMissP->name = "radianceMiss";
		radianceMissP->type = optix::OptixProgramType::P_Miss;
		radianceMissP->missFunction = missFunction;

		auto radianceHitP = std::make_shared<optix::ProgramOptix>();
		radianceHitP->name = "radianceHit";
		radianceHitP->type = optix::OptixProgramType::P_Hit;
		radianceHitP->closestHitFunction = closestHitFunction;
		radianceHitP->anyHitFunction = shadowHitF;


		//////////////////////////////////////////////////////////////////////
		/////////////////// Shadow Trace Programs ////////////////////////////
		//////////////////////////////////////////////////////////////////////

		auto shadowMissP = std::make_shared<optix::ProgramOptix>();
		shadowMissP->name = "shadowMiss";
		shadowMissP->type = optix::OptixProgramType::P_Miss;
		shadowMissP->missFunction = shadowMissF;

		auto shadowHitP = std::make_shared<optix::ProgramOptix>();
		shadowHitP->name = "shadowHit";
		shadowHitP->type = optix::OptixProgramType::P_Hit;
		shadowHitP->closestHitFunction = closestHitFunction;
		shadowHitP->anyHitFunction = shadowHitF;

		///////////////////////////////////////////////////////////////////
		/////////////////////// Full Optix Pipeline ///////////////////////
		///////////////////////////////////////////////////////////////////

		auto renderFrameFunction = std::make_shared<optix::FunctionOptix>();
		renderFrameFunction->name = "__raygen__renderFrame";
		renderFrameFunction->module = deviceProgramModule;
		renderFrameFunction->type = optix::OptixFunctionType::F_Raygen;


		auto rayGenProgram = std::make_shared<optix::ProgramOptix>();
		rayGenProgram->name = "raygen";
		rayGenProgram->type = optix::OptixProgramType::P_Raygen;
		rayGenProgram->raygenFunction = renderFrameFunction;

		pipeline->registerProgram(rayGenProgram);
		//pipeline->registerProgram(exceptionProgram);
		pipeline->registerProgram(radianceMissP);
		pipeline->registerProgram(radianceHitP);
		pipeline->registerProgram(shadowMissP);
		pipeline->registerProgram(shadowHitP);

		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Trace /////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto wfRadianceTraceF= std::make_shared<optix::FunctionOptix>();
		wfRadianceTraceF->name = "__raygen__trace";
		wfRadianceTraceF->module = deviceProgramModule;
		wfRadianceTraceF->type = optix::OptixFunctionType::F_Raygen;

		auto wfRadianceTraceP = std::make_shared<optix::ProgramOptix>();
		wfRadianceTraceP->name = "wfRadianceTrace";
		wfRadianceTraceP->type = optix::OptixProgramType::P_Raygen;
		wfRadianceTraceP->raygenFunction = wfRadianceTraceF;

		pipeline->registerProgram(wfRadianceTraceP, { "wfRadianceTrace" });
		pipeline->registerProgram(radianceMissP, { "wfRadianceTrace" });
		pipeline->registerProgram(radianceHitP, { "wfRadianceTrace" });
		pipeline->registerProgram(shadowMissP, { "wfRadianceTrace" });
		pipeline->registerProgram(shadowHitP, { "wfRadianceTrace" });


		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Trace /////////////////////////////////
		///////////////////////////////////////////////////////////////////

		auto wfShadowTraceF = std::make_shared<optix::FunctionOptix>();
		wfShadowTraceF->name = "__raygen__shadow";
		wfShadowTraceF->module = deviceProgramModule;
		wfShadowTraceF->type = optix::OptixFunctionType::F_Raygen;

		auto wfShadowTraceP = std::make_shared<optix::ProgramOptix>();
		wfShadowTraceP->name = "wfShadowTrace";
		wfShadowTraceP->type = optix::OptixProgramType::P_Raygen;
		wfShadowTraceP->raygenFunction = wfShadowTraceF;

		pipeline->registerProgram(wfShadowTraceP, { "wfShadowTrace" });
		pipeline->registerProgram(radianceMissP, { "wfShadowTrace" });
		pipeline->registerProgram(radianceHitP, { "wfShadowTrace" });
		pipeline->registerProgram(shadowMissP, { "wfShadowTrace" });
		pipeline->registerProgram(shadowHitP, { "wfShadowTrace" });

		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Shade /////////////////////////////////
		///////////////////////////////////////////////////////////////////
		
		auto shadeRaygenF= std::make_shared<optix::FunctionOptix>();
		shadeRaygenF->name = "__raygen__shade";
		shadeRaygenF->module = deviceProgramModule;
		shadeRaygenF->type = optix::OptixFunctionType::F_Raygen;

		auto shadeRaygenP = std::make_shared<optix::ProgramOptix>();
		shadeRaygenP->name = "shadeRaygen";
		shadeRaygenP->type = optix::OptixProgramType::P_Raygen;
		shadeRaygenP->raygenFunction = shadeRaygenF;

		pipeline->registerProgram(shadeRaygenP, { "wfShade" });
		pipeline->registerProgram(shadowMissP, { "wfShade" });
		pipeline->registerProgram(shadowHitP, { "wfShade" });

		///////////////////////////////////////////////////////////////////
		///////////////// Cuda Mdl Linker /////////////////////////////////
		///////////////////////////////////////////////////////////////////
		mdl::MdlCudaLinker& mdlCudaLinker = mdl::getMdlCudaLinker();
		mdlCudaLinker.ptxFile = getOptions()->executablePath + "ptx/shadeKernel.ptx";
		mdlCudaLinker.kernelFunctionName = "wfShadeEntry";
		
	}
}

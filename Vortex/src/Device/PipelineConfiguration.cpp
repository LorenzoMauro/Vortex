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

		const auto deviceProgramModule = std::make_shared<optix::ModuleOptix>();
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
		/////////////////////// Miss //////////////////////////////////////
		///////////////////////////////////////////////////////////////////

		const auto missFunction = std::make_shared<optix::FunctionOptix>();
		missFunction->name = "__miss__radianceAndShadow";
		missFunction->module = deviceProgramModule;
		missFunction->type = optix::OptixFunctionType::F_Miss;

		const auto missProgram = std::make_shared<optix::ProgramOptix>();
		missProgram->name = "radianceMiss";
		missProgram->type = optix::OptixProgramType::P_Miss;
		missProgram->missFunction = missFunction;

		///////////////////////////////////////////////////////////////////
		/////////////////////// Radiance Trace ////////////////////////////
		///////////////////////////////////////////////////////////////////

		const auto closestHitFunction = std::make_shared<optix::FunctionOptix>();
		closestHitFunction->name = "__closesthit__radiance";
		closestHitFunction->module = deviceProgramModule;
		closestHitFunction->type = optix::OptixFunctionType::F_ClosestHit;

		const auto radianceAnyHitF = std::make_shared<optix::FunctionOptix>();
		radianceAnyHitF->name = "__anyhit__radianceHit";
		radianceAnyHitF->module = deviceProgramModule;
		radianceAnyHitF->type = optix::OptixFunctionType::F_AnyHit;

		const auto radianceHitP = std::make_shared<optix::ProgramOptix>();
		radianceHitP->name = "radianceHit";
		radianceHitP->type = optix::OptixProgramType::P_Hit;
		radianceHitP->closestHitFunction = closestHitFunction;
		radianceHitP->anyHitFunction = radianceAnyHitF;

		//////////////////////////////////////////////////////////////////////
		////////////////// Shadow Trace //////////////////////////////////////
		//////////////////////////////////////////////////////////////////////

		const auto shadowAnyHitF = std::make_shared<optix::FunctionOptix>();
		shadowAnyHitF->name = "__anyhit__shadow";
		shadowAnyHitF->module = deviceProgramModule;
		shadowAnyHitF->type = optix::OptixFunctionType::F_AnyHit;

		const auto shadowHitP = std::make_shared<optix::ProgramOptix>();
		shadowHitP->name = "shadowHit";
		shadowHitP->type = optix::OptixProgramType::P_Hit;
		shadowHitP->closestHitFunction = closestHitFunction; // same as radiance because it's ignored
		shadowHitP->anyHitFunction = shadowAnyHitF;

		///////////////////////////////////////////////////////////////////
		/////////////////////// Full Optix Pipeline ///////////////////////
		///////////////////////////////////////////////////////////////////

		const auto renderFrameFunction = std::make_shared<optix::FunctionOptix>();
		renderFrameFunction->name = "__raygen__renderFrame";
		renderFrameFunction->module = deviceProgramModule;
		renderFrameFunction->type = optix::OptixFunctionType::F_Raygen;


		const auto rayGenProgram = std::make_shared<optix::ProgramOptix>();
		rayGenProgram->name = "raygen";
		rayGenProgram->type = optix::OptixProgramType::P_Raygen;
		rayGenProgram->raygenFunction = renderFrameFunction;

		pipeline->registerProgram(rayGenProgram);
		//pipeline->registerProgram(exceptionProgram);
		pipeline->registerProgram(missProgram);
		pipeline->registerProgram(radianceHitP);
		pipeline->registerProgram(shadowHitP);

		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront  Radiance Trace ///////////////////////
		///////////////////////////////////////////////////////////////////

		const auto wfRadianceTraceF= std::make_shared<optix::FunctionOptix>();
		wfRadianceTraceF->name = "__raygen__trace";
		wfRadianceTraceF->module = deviceProgramModule;
		wfRadianceTraceF->type = optix::OptixFunctionType::F_Raygen;

		const auto wfRadianceTraceP = std::make_shared<optix::ProgramOptix>();
		wfRadianceTraceP->name = "wfRadianceTrace";
		wfRadianceTraceP->type = optix::OptixProgramType::P_Raygen;
		wfRadianceTraceP->raygenFunction = wfRadianceTraceF;

		pipeline->registerProgram(wfRadianceTraceP, { "wfRadianceTrace" });
		pipeline->registerProgram(missProgram, { "wfRadianceTrace" });
		pipeline->registerProgram(radianceHitP, { "wfRadianceTrace" });
		pipeline->registerProgram(shadowHitP, { "wfRadianceTrace" });


		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Shadow Trace //////////////////////////
		///////////////////////////////////////////////////////////////////

		const auto wfShadowTraceF = std::make_shared<optix::FunctionOptix>();
		wfShadowTraceF->name = "__raygen__shadow";
		wfShadowTraceF->module = deviceProgramModule;
		wfShadowTraceF->type = optix::OptixFunctionType::F_Raygen;

		const auto wfShadowTraceP = std::make_shared<optix::ProgramOptix>();
		wfShadowTraceP->name = "wfShadowTrace";
		wfShadowTraceP->type = optix::OptixProgramType::P_Raygen;
		wfShadowTraceP->raygenFunction = wfShadowTraceF;

		pipeline->registerProgram(wfShadowTraceP, { "wfShadowTrace" });
		pipeline->registerProgram(missProgram, { "wfShadowTrace" });
		pipeline->registerProgram(radianceHitP, { "wfShadowTrace" });
		pipeline->registerProgram(shadowHitP, { "wfShadowTrace" });

		///////////////////////////////////////////////////////////////////
		///////////////// WaveFront Optix Shade ///////////////////////////
		///////////////////////////////////////////////////////////////////

		const auto shadeRaygenF= std::make_shared<optix::FunctionOptix>();
		shadeRaygenF->name = "__raygen__shade";
		shadeRaygenF->module = deviceProgramModule;
		shadeRaygenF->type = optix::OptixFunctionType::F_Raygen;

		const auto shadeRaygenP = std::make_shared<optix::ProgramOptix>();
		shadeRaygenP->name = "shadeRaygen";
		shadeRaygenP->type = optix::OptixProgramType::P_Raygen;
		shadeRaygenP->raygenFunction = shadeRaygenF;

		pipeline->registerProgram(shadeRaygenP, { "wfShade" });
		pipeline->registerProgram(missProgram, { "wfShade" });
		pipeline->registerProgram(shadowHitP, { "wfShade" });

		///////////////////////////////////////////////////////////////////
		///////////////// Cuda Mdl Linker /////////////////////////////////
		///////////////////////////////////////////////////////////////////
		mdl::MdlCudaLinker& mdlCudaLinker = mdl::getMdlCudaLinker();
		mdlCudaLinker.ptxFile = getOptions()->executablePath + "ptx/shadeKernel.ptx";
		mdlCudaLinker.kernelFunctionName = "wfShadeEntry";
		
	}
}

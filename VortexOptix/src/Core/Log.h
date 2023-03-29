#pragma once
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>

namespace vtx
{
	class Log
	{
	public:
		static void Init();

		inline static std::shared_ptr<spdlog::logger>& GetVortexLogger() { return s_VortexLogger; }
		inline static std::shared_ptr<spdlog::logger>& GetGlLogger() { return s_GlLogger; }

	private:
		static std::shared_ptr<spdlog::logger> s_VortexLogger;
		static std::shared_ptr<spdlog::logger> s_GlLogger;

	};
}

// Vortex Log Macros
#define VTX_TRACE(...)	::vtx::Log::GetVortexLogger()->trace(__VA_ARGS__)
#define VTX_INFO(...)	::vtx::Log::GetVortexLogger()->info(__VA_ARGS__)
#define VTX_WARN(...)	::vtx::Log::GetVortexLogger()->warn(__VA_ARGS__)
#define VTX_ERROR(...)	::vtx::Log::GetVortexLogger()->error(__VA_ARGS__)

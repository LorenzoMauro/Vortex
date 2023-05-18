#include "Log.h"

namespace vtx
{
	std::shared_ptr<spdlog::logger> Log::s_VortexLogger;

	void Log::Init()
	{
		spdlog::set_pattern("%^[%T] %n: %v%$");
		s_VortexLogger = spdlog::stdout_color_mt("Vortex");
		s_VortexLogger->set_level(spdlog::level::info);
	}
}
#pragma once
#include <memory>
#include <iostream>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

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

inline void waitAndClose() {
    std::cerr << "Press ENTER to exit..." << std::endl;
    std::cin.get();
	__debugbreak();
    std::exit(EXIT_FAILURE);
}

#define PRINT_MESSAGE(...) do { \
	std::stringstream ss; \
	ss << ##__VA_ARGS__ << std::endl; \
	VTX_ERROR("{}", ss.str()); \
	} while(0)

#define VTX_ASSERT_CLOSE(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_ERROR("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		__debugbreak(); \
		waitAndClose(); \
	}\
} while(0)

#define VTX_ASSERT_CONTINUE(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ );\
	}\
} while(0)

#define VTX_ASSERT_RETURN(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		return;\
	}\
} while(0)


#define VTX_ASSERT_BREAK(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		break;\
	}\
} while(0)

#define VTX_ASSERT_RETURN(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		return 0; \
	}\
} while(0)
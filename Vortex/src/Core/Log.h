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

		template <typename... Args>
		static void error(const Args&... args)
		{
			if constexpr (sizeof...(args) != 0)
			{
				GetVortexLogger()->error(args...);
			}
		}

		template <typename... Args>
		static void info(const Args&... args)
		{
			if constexpr (sizeof...(args) != 0)
			{
				GetVortexLogger()->info(args...);
			}
		}

		template <typename... Args>
		static void warn(const Args&... args)
		{
			if constexpr (sizeof...(args) != 0)
			{
				GetVortexLogger()->warn(args...);
			}
		}

		template <typename... Args>
		static void trace(const Args&... args)
		{
			if constexpr (sizeof...(args) != 0)
			{
				GetVortexLogger()->trace(args...);
			}
		}

	private:
		static std::shared_ptr<spdlog::logger> s_VortexLogger;
		static std::shared_ptr<spdlog::logger> s_GlLogger;

	};
}

// Vortex Log Macros
//define VTX_TRACE(...)	::vtx::Log::GetVortexLogger()->trace(__VA_ARGS__)
//define VTX_INFO(...)	::vtx::Log::GetVortexLogger()->info(__VA_ARGS__)
//define VTX_WARN(...)	::vtx::Log::GetVortexLogger()->warn(__VA_ARGS__)
//define VTX_ERROR(...)	::vtx::Log::GetVortexLogger()->error(__VA_ARGS__)
#define VTX_TRACE(...)	::vtx::Log::trace(__VA_ARGS__)
#define VTX_INFO(...)	::vtx::Log::info(__VA_ARGS__)
#define VTX_WARN(...)	::vtx::Log::warn(__VA_ARGS__)
#define VTX_ERROR(...)	::vtx::Log::error(__VA_ARGS__)

inline void waitAndClose() {
#ifdef _DEBUG
	__debugbreak();
#else
	if (IsDebuggerPresent()) {
		__debugbreak();
}
#endif
    std::cerr << "Press ENTER to exit..." << std::endl;
    std::cin.get();
    std::exit(EXIT_FAILURE);
}

#define IS_EMPTY(...) (strcmp(#__VA_ARGS__, "") == 0)

#define VTX_ASSERT_CLOSE(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_ERROR("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		VTX_ERROR(__VA_ARGS__); \
		waitAndClose(); \
	}\
} while(0)

#define VTX_ASSERT_CONTINUE(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ );\
		VTX_WARN(__VA_ARGS__); \
	}\
} while(0)

#define VTX_ASSERT_RETURN(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		VTX_WARN(__VA_ARGS__); \
		return;\
	}\
} while(0)


#define VTX_ASSERT_BREAK(successCondition, ...) do { \
    if (!(successCondition)) { \
		VTX_WARN("Assert {} Failed in File {} at Line {}", #successCondition, __FILE__, __LINE__ ); \
		VTX_WARN(__VA_ARGS__); \
		break;\
	}\
} while(0)

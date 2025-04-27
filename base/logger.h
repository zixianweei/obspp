#ifndef CUTE_BASE_LOGGER_H_
#define CUTE_BASE_LOGGER_H_

#if defined(HAS_CUTE_LOGGER)

#    include <memory>
#    include <string>

#    include <spdlog/logger.h>

class CuteLogger
{
public:
    static CuteLogger& GetInstance();

    CuteLogger(const CuteLogger&) = delete;
    CuteLogger& operator=(const CuteLogger&) = delete;
    CuteLogger(CuteLogger&&) noexcept = delete;
    CuteLogger& operator=(CuteLogger&&) noexcept = delete;

    bool Init(const std::string& fname, size_t max_size = 1024 * 1024 * 10, size_t max_files = 10);

    std::shared_ptr<spdlog::logger> GetLogger()
    {
        return logger_;
    }

private:
    CuteLogger() = default;

    std::shared_ptr<spdlog::logger> logger_;
};

#    define CUTE_LOG_INSTANCE_LOGGER() CuteLogger::GetInstance().GetLogger()
#    define CUTE_LOG_TRACE(...) CUTE_LOG_INSTANCE_LOGGER()->trace(__VA_ARGS__)
#    define CUTE_LOG_DEBUG(...) CUTE_LOG_INSTANCE_LOGGER()->debug(__VA_ARGS__)
#    define CUTE_LOG_INFO(...) CUTE_LOG_INSTANCE_LOGGER()->info(__VA_ARGS__)
#    define CUTE_LOG_WARN(...) CUTE_LOG_INSTANCE_LOGGER()->warn(__VA_ARGS__)
#    define CUTE_LOG_ERROR(...) CUTE_LOG_INSTANCE_LOGGER()->error(__VA_ARGS__)
#    define CUTE_LOG_CRITICAL(...) CUTE_LOG_INSTANCE_LOGGER()->critical(__VA_ARGS__)

#else

#    define CUTE_LOG_TRACE(...)
#    define CUTE_LOG_DEBUG(...)
#    define CUTE_LOG_INFO(...)
#    define CUTE_LOG_WARN(...)
#    define CUTE_LOG_ERROR(...)
#    define CUTE_LOG_CRITICAL(...)

#endif

#endif  // !CUTE_BASE_LOGGER_H_

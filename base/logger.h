#ifndef CUTENN_BASE_LOGGER_H_
#define CUTENN_BASE_LOGGER_H_

#if defined(HAS_CUTE_LOGGER)

#include <memory>
#include <string>

#include <spdlog/logger.h>

class CuteLogger {
public:
  static CuteLogger &GetInstance();

  CuteLogger(const CuteLogger &) = delete;
  CuteLogger &operator=(const CuteLogger &) = delete;
  CuteLogger(CuteLogger &&) noexcept = delete;
  CuteLogger &operator=(CuteLogger &&) noexcept = delete;

  bool Init(const std::string &fname, size_t max_size = 1024 * 1024 * 10,
            size_t max_files = 10);

  std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }

private:
  CuteLogger() = default;

  std::shared_ptr<spdlog::logger> logger_;
};

#define CUTENN_LOG_INSTANCE_LOGGER() CuteLogger::GetInstance().GetLogger()
#define CUTENN_LOG_TRACE(...) CUTENN_LOG_INSTANCE_LOGGER()->trace(__VA_ARGS__)
#define CUTENN_LOG_DEBUG(...) CUTENN_LOG_INSTANCE_LOGGER()->debug(__VA_ARGS__)
#define CUTENN_LOG_INFO(...) CUTENN_LOG_INSTANCE_LOGGER()->info(__VA_ARGS__)
#define CUTENN_LOG_WARN(...) CUTENN_LOG_INSTANCE_LOGGER()->warn(__VA_ARGS__)
#define CUTENN_LOG_ERROR(...) CUTENN_LOG_INSTANCE_LOGGER()->error(__VA_ARGS__)
#define CUTENN_LOG_CRITICAL(...)                                               \
  CUTENN_LOG_INSTANCE_LOGGER()->critical(__VA_ARGS__)

#else

#define CUTENN_LOG_TRACE(...)
#define CUTENN_LOG_DEBUG(...)
#define CUTENN_LOG_INFO(...)
#define CUTENN_LOG_WARN(...)
#define CUTENN_LOG_ERROR(...)
#define CUTENN_LOG_CRITICAL(...)

#endif // HAS_CUTE_LOGGER

#endif // !CUTENN_BASE_LOGGER_H_

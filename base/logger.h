#ifndef CUTENN_BASE_LOGGER_H_
#define CUTENN_BASE_LOGGER_H_

#include <memory>
#include <string>

#include <spdlog/logger.h>

namespace cutenn {

class Logger {
public:
  static Logger &GetInstance();

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;
  Logger(Logger &&) noexcept = delete;
  Logger &operator=(Logger &&) noexcept = delete;

  bool Init(const std::string &fname, size_t max_size = 1024 * 1024 * 10,
            size_t max_files = 10);

  std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }

private:
  Logger() = default;

  std::shared_ptr<spdlog::logger> logger_;
};

} // namespace cutenn

#define CUTENN_LOG_INSTANCE_LOGGER() cutenn::Logger::GetInstance().GetLogger()
#define CUTENN_LOG_TRACE(...) CUTENN_LOG_INSTANCE_LOGGER()->trace(__VA_ARGS__)
#define CUTENN_LOG_DEBUG(...) CUTENN_LOG_INSTANCE_LOGGER()->debug(__VA_ARGS__)
#define CUTENN_LOG_INFO(...) CUTENN_LOG_INSTANCE_LOGGER()->info(__VA_ARGS__)
#define CUTENN_LOG_WARN(...) CUTENN_LOG_INSTANCE_LOGGER()->warn(__VA_ARGS__)
#define CUTENN_LOG_ERROR(...) CUTENN_LOG_INSTANCE_LOGGER()->error(__VA_ARGS__)
#define CUTENN_LOG_CRITICAL(...)                                               \
  CUTENN_LOG_INSTANCE_LOGGER()->critical(__VA_ARGS__)

#endif // !CUTENN_BASE_LOGGER_H_

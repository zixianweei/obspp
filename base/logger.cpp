#include "logger.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace cutenn {

Logger &Logger::GetInstance() {
  static Logger instance;
  return instance;
}

bool Logger::Init(const std::string &fname, size_t max_size, size_t max_files) {
  try {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        fname, max_size, max_files);
    file_sink->set_level(spdlog::level::debug);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");

    logger_ = std::make_shared<spdlog::logger>(
        "cutenn", spdlog::sinks_init_list{console_sink, file_sink});
    logger_->set_level(spdlog::level::debug);
    logger_->flush_on(spdlog::level::debug);

    return true;
  } catch (const spdlog::spdlog_ex &ex) {
    return false;
  }
}

} // namespace cutenn

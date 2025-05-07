#include <gtest/gtest.h>

#include "base/logger.hpp"

int main() {
#if defined(ENABLE_CUTENN_LOGGER)
  cutenn::Logger::GetInstance().Init("cutenn.log", 1024 * 1024 * 10, 10);
#endif

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

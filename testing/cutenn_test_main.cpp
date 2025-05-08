#include <gtest/gtest.h>

#include "base/context.hpp"
#include "base/logger.hpp"

int main() {
  cutenn::Logger::GetInstance().Init("cutenn.log");
  cutenn::Context::GetInstance().MakeCaptureScopeAvailable();

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

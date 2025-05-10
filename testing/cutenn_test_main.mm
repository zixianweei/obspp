#include <gtest/gtest.h>

#include "base/context.h"
#include "base/logger.h"

int main() {
  cutenn::Logger::GetInstance().Init("cutenn.log");
  cutenn::Context::GetInstance().MakeCaptureScopeAvailable();

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>

#include "base/logger.h"
#if defined(CUTENN_METAL_DEBUG)
#include "base/context.h"
#endif // CUTENN_METAL_DEBUG

int main() {
  cutenn::Logger::GetInstance().Init("cutenn.log");
#if defined(CUTENN_METAL_DEBUG)
  cutenn::Context::GetInstance().MakeCaptureScopeAvailable();
#endif // CUTENN_METAL_DEBUG

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

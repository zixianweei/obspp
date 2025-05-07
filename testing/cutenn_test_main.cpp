#include <gtest/gtest.h>

#include "base/logger.hpp"

int main() {
  cutenn::Logger::GetInstance().Init("cutenn.log");

  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}

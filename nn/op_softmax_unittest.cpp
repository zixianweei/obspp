#include <gtest/gtest.h>

#include "base/tensor.hpp"
#include "nn/op_softmax.hpp"

TEST(OpSoftmax, Softmax1DAxis0) {
  std::vector<float> src_data = {1.0, 2.0, 3.0};
  std::vector<float> dst_data = {0.0, 0.0, 0.0};
  cutenn::TShape src_shape = {3};
  cutenn::TShape dst_shape = {3};
  cutenn::Tensor src(src_data.data(), src_shape, cutenn::Format::kFloat32);
  cutenn::Tensor dst(dst_data.data(), dst_shape, cutenn::Format::kFloat32);

  cutenn::OpSoftmax op;
  op.SetAxis(0);
  ASSERT_TRUE(op.Forward(src, dst));

  ASSERT_TRUE(dst_data.data() != nullptr);
  dst.Download(dst_data.data(), dst_shape, cutenn::Format::kFloat32);

  std::vector<float> expected = {0.09003057, 0.24472847, 0.66524096};
  ASSERT_EQ(dst_data.size(), expected.size());
  for (size_t i = 0; i < dst_data.size(); ++i) {
    ASSERT_NEAR(dst_data[i], expected[i], 1e-6);
  }
}

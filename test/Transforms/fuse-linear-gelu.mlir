module {
  func.func @fuse_linear_gelu(
      %input: tensor<2x3xf32>,
      %weight: tensor<3x4xf32>,
      %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %linear = tf.linear %input, %weight, %bias
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    %activated = tf.gelu %linear : tensor<2x4xf32>
    return %activated : tensor<2x4xf32>
  }
}

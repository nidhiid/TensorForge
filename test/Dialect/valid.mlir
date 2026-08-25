module {
  func.func @matmul(
      %lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = tf.matmul %lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @linear_gelu(
      %input: tensor<2x3xf32>,
      %weight: tensor<3x4xf32>,
      %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %linear = tf.linear %input, %weight, %bias
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    %activated = tf.gelu %linear : tensor<2x4xf32>
    return %activated : tensor<2x4xf32>
  }
}

module {
  func.func @lower_matmul(
      %lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = tf.matmul %lhs, %rhs
        : tensor<2x3xf32>, tensor<3x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @lower_linear(
      %input: tensor<2x3xf32>,
      %weight: tensor<3x4xf32>,
      %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %result = tf.linear %input, %weight, %bias
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @lower_gelu(
      %input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = tf.gelu %input : tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @lower_fused_linear_gelu(
      %input: tensor<2x3xf32>,
      %weight: tensor<3x4xf32>,
      %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %result = tf.fused_linear_gelu %input, %weight, %bias
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @lower_reshape(
      %input: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %result = tf.reshape %input to [3, 2] : tensor<2x3xf32>
    return %result : tensor<3x2xf32>
  }

  func.func @lower_transpose(
      %input: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %result = tf.transpose %input : tensor<2x3xf32>
    return %result : tensor<3x2xf32>
  }
}

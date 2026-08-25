module {
  func.func @lower_matmul(
      %lhs: tensor<2x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %result = tf.matmul %lhs, %rhs
        : tensor<2x3xf32>, tensor<3x4xf32>
    return %result : tensor<2x4xf32>
  }

  func.func @lower_mlp(
      %input: tensor<2x3xf32>,
      %weight1: tensor<3x4xf32>,
      %bias1: tensor<4xf32>,
      %weight2: tensor<4x2xf32>,
      %bias2: tensor<2xf32>) -> tensor<2x2xf32> {
    %linear = tf.linear %input, %weight1, %bias1
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
    %gelu = tf.gelu %linear : tensor<2x4xf32>
    %result = tf.linear %gelu, %weight2, %bias2
        : tensor<2x4xf32>, tensor<4x2xf32>, tensor<2xf32>
    return %result : tensor<2x2xf32>
  }

  func.func @lower_shape_operations(
      %input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %transposed = tf.transpose %input : tensor<2x3xf32>
    %result = tf.reshape %transposed to [2, 3] : tensor<3x2xf32>
    return %result : tensor<2x3xf32>
  }
}

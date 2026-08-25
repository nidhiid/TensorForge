module {
  func.func @remove_reshape_chain(
      %input: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %flat = tf.reshape %input to [6] : tensor<2x3xf32>
    %restored = tf.reshape %flat to [2, 3] : tensor<6xf32>
    return %restored : tensor<2x3xf32>
  }

  func.func @remove_double_transpose(
      %input: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %first = tf.transpose %input : tensor<3x4xf32>
    %second = tf.transpose %first : tensor<4x3xf32>
    return %second : tensor<3x4xf32>
  }

  func.func @fold_constant_reshape() -> tensor<2x3xf32> {
    %constant = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]>
        : tensor<6xf32>
    %reshaped = tf.reshape %constant to [2, 3] : tensor<6xf32>
    return %reshaped : tensor<2x3xf32>
  }
}

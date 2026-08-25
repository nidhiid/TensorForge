module {
  func.func @unsupported(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = tf.gelu %input : tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

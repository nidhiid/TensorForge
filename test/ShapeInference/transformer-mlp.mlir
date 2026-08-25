module {
  func.func @bert_mlp(
      %input: tensor<128x768xf32>,
      %weight1: tensor<768x3072xf32>,
      %bias1: tensor<3072xf32>,
      %weight2: tensor<3072x768xf32>,
      %bias2: tensor<768xf32>) -> tensor<128x768xf32> {
    %expanded = tf.linear %input, %weight1, %bias1
        : tensor<128x768xf32>, tensor<768x3072xf32>, tensor<3072xf32>
    %activated = tf.gelu %expanded : tensor<128x3072xf32>
    %output = tf.linear %activated, %weight2, %bias2
        : tensor<128x3072xf32>, tensor<3072x768xf32>, tensor<768xf32>
    return %output : tensor<128x768xf32>
  }
}

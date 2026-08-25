module {
  func.func @incompatible_matmul(
      %lhs: tensor<2x3xf32>,
      %rhs: tensor<5x4xf32>) -> tensor<2x4xf32> {
    // expected-error @+1 {{'tf.matmul' op has incompatible contracting dimensions: 3 and 5}}
    %result = tf.matmul %lhs, %rhs
        : tensor<2x3xf32>, tensor<5x4xf32> -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @wrong_linear_bias(
      %input: tensor<2x3xf32>,
      %weight: tensor<3x4xf32>,
      %bias: tensor<5xf32>) -> tensor<2x4xf32> {
    // expected-error @+1 {{'tf.linear' op bias length must equal the weight output dimension 4, but got 5}}
    %result = tf.linear %input, %weight, %bias
        : tensor<2x3xf32>, tensor<3x4xf32>, tensor<5xf32>
          -> tensor<2x4xf32>
    return %result : tensor<2x4xf32>
  }
}

// -----

module {
  func.func @wrong_gelu_output(
      %input: tensor<2x4xf32>) -> tensor<2x5xf32> {
    // expected-error @+1 {{'tf.gelu' op input and output types must match}}
    %result = tf.gelu %input : tensor<2x4xf32> -> tensor<2x5xf32>
    return %result : tensor<2x5xf32>
  }
}

// -----

module {
  func.func @unsupported_element_type(
      %lhs: tensor<2x3xf64>,
      %rhs: tensor<3x4xf64>) -> tensor<2x4xf64> {
    // expected-error @+1 {{'tf.matmul' op lhs must contain f32 elements}}
    %result = tf.matmul %lhs, %rhs
        : tensor<2x3xf64>, tensor<3x4xf64> -> tensor<2x4xf64>
    return %result : tensor<2x4xf64>
  }
}

// -----

module {
  func.func @unsupported_dynamic_shape(
      %lhs: tensor<?x3xf32>,
      %rhs: tensor<3x4xf32>) -> tensor<?x4xf32> {
    // expected-error @+1 {{'tf.matmul' op lhs must have a static shape}}
    %result = tf.matmul %lhs, %rhs
        : tensor<?x3xf32>, tensor<3x4xf32> -> tensor<?x4xf32>
    return %result : tensor<?x4xf32>
  }
}

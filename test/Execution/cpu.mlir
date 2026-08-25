module {
  func.func @run_matmul() -> f32 attributes {llvm.emit_c_interface} {
    %lhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
        : tensor<2x2xf32>
    %rhs = arith.constant dense<[[5.0, 6.0], [7.0, 8.0]]>
        : tensor<2x2xf32>
    %product = tf.matmul %lhs, %rhs
        : tensor<2x2xf32>, tensor<2x2xf32>
    %c1 = arith.constant 1 : index
    %result = tensor.extract %product[%c1, %c1] : tensor<2x2xf32>
    return %result : f32
  }

  func.func @run_linear() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %weight = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]>
        : tensor<2x2xf32>
    %bias = arith.constant dense<[0.5, -0.5]> : tensor<2xf32>
    %linear = tf.linear %input, %weight, %bias
        : tensor<1x2xf32>, tensor<2x2xf32>, tensor<2xf32>
    %c0 = arith.constant 0 : index
    %result = tensor.extract %linear[%c0, %c0] : tensor<1x2xf32>
    return %result : f32
  }

  func.func @run_gelu() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<1.0> : tensor<1xf32>
    %activated = tf.gelu %input : tensor<1xf32>
    %c0 = arith.constant 0 : index
    %result = tensor.extract %activated[%c0] : tensor<1xf32>
    return %result : f32
  }

  func.func @run_reshape() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
        : tensor<2x3xf32>
    %reshaped = tf.reshape %input to [3, 2] : tensor<2x3xf32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %result = tensor.extract %reshaped[%c2, %c0] : tensor<3x2xf32>
    return %result : f32
  }

  func.func @run_transpose() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
        : tensor<2x3xf32>
    %transposed = tf.transpose %input : tensor<2x3xf32>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = tensor.extract %transposed[%c2, %c1] : tensor<3x2xf32>
    return %result : f32
  }

  func.func @main() -> f32 attributes {llvm.emit_c_interface} {
    %input = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
        : tensor<2x2xf32>
    %weight = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]>
        : tensor<2x2xf32>
    %bias = arith.constant dense<0.0> : tensor<2xf32>

    %linear = tf.linear %input, %weight, %bias
        : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2xf32>
    %activated = tf.gelu %linear : tensor<2x2xf32>

    %c0 = arith.constant 0 : index
    %result = tensor.extract %activated[%c0, %c0] : tensor<2x2xf32>
    return %result : f32
  }
}

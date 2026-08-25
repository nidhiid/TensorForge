module {
  func.func @identity(%value: tensor<2x2xf32>) -> tensor<2x2xf32> {
    return %value : tensor<2x2xf32>
  }
}

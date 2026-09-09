// RUN: enzymexlamlir-opt %s --split-input-file --enzyme-hlo-generate-td="patterns=dynamic_slice_to_static" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Same story as slice_of_dynamic_update: the start index is an operand, and a
// producer may hand it over as arithmetic over constants.

module {
  func.func @sub_index(%operand: tensor<247423xf64>) -> tensor<91xf64> {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %base = stablehlo.constant dense<99994> : tensor<i32>
    %off = stablehlo.subtract %base, %one : tensor<i32>
    %ds = stablehlo.dynamic_slice %operand, %off, sizes = [91] : (tensor<247423xf64>, tensor<i32>) -> tensor<91xf64>
    return %ds : tensor<91xf64>
  }
}

// CHECK-LABEL:   func.func @sub_index(
// CHECK:           %[[STATIC:.*]] = stablehlo.slice %arg0 [99993:100084] : (tensor<247423xf64>) -> tensor<91xf64>
// CHECK:           return %[[STATIC]]

// -----

module {
  func.func @runtime_index(%operand: tensor<1024xf64>, %off: tensor<i32>) -> tensor<8xf64> {
    %ds = stablehlo.dynamic_slice %operand, %off, sizes = [8] : (tensor<1024xf64>, tensor<i32>) -> tensor<8xf64>
    return %ds : tensor<8xf64>
  }
}

// CHECK-LABEL:   func.func @runtime_index(
// CHECK:           stablehlo.dynamic_slice

// RUN: enzymexlamlir-opt %s --split-input-file --enzyme-hlo-generate-td="patterns=slice_of_dynamic_update" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A dynamic_update_slice start index is an *operand*, so producers commonly
// hand one over as a small arithmetic expression over constants rather than as
// a literal `stablehlo.constant`.  (Reactant lowers a 1-based Julia index as
// `subtract %c_start, %c_one`.)  slice_of_dynamic_update must still see through
// it, without depending on a separate integer constant folder having run.

module {
  // The read window is disjoint from the updated window, so the read is
  // forwarded to the DUS operand.
  func.func @disjoint_sub(%operand: tensor<247423xf64>, %update: tensor<91xf64>) -> tensor<91xf64> {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %base = stablehlo.constant dense<106455> : tensor<i32>
    %off = stablehlo.subtract %base, %one : tensor<i32>
    %dus = stablehlo.dynamic_update_slice %operand, %update, %off : (tensor<247423xf64>, tensor<91xf64>, tensor<i32>) -> tensor<247423xf64>
    %s = stablehlo.slice %dus [99993:100084] : (tensor<247423xf64>) -> tensor<91xf64>
    return %s : tensor<91xf64>
  }
}

// CHECK-LABEL:   func.func @disjoint_sub(
// CHECK:           %[[FWD:.*]] = stablehlo.slice %arg0 [99993:100084] : (tensor<247423xf64>) -> tensor<91xf64>
// CHECK:           return %[[FWD]]

// -----

module {
  // The read window lies inside the updated window, so the read is rebased
  // onto the update.
  func.func @contained_sub(%operand: tensor<247423xf64>, %update: tensor<91xf64>) -> tensor<20xf64> {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %base = stablehlo.constant dense<106455> : tensor<i32>
    %off = stablehlo.subtract %base, %one : tensor<i32>
    %dus = stablehlo.dynamic_update_slice %operand, %update, %off : (tensor<247423xf64>, tensor<91xf64>, tensor<i32>) -> tensor<247423xf64>
    %s = stablehlo.slice %dus [106500:106520] : (tensor<247423xf64>) -> tensor<20xf64>
    return %s : tensor<20xf64>
  }
}

// CHECK-LABEL:   func.func @contained_sub(
// CHECK:           %[[REBASED:.*]] = stablehlo.slice %arg1 [46:66] : (tensor<91xf64>) -> tensor<20xf64>
// CHECK:           return %[[REBASED]]

// -----

module {
  // add, and a widening convert, are folded the same way.
  func.func @disjoint_add_convert(%operand: tensor<1024xf64>, %update: tensor<8xf64>) -> tensor<4xf64> {
    %a = stablehlo.constant dense<500> : tensor<i32>
    %b = stablehlo.constant dense<12> : tensor<i32>
    %sum = stablehlo.add %a, %b : tensor<i32>
    %off = stablehlo.convert %sum : (tensor<i32>) -> tensor<i64>
    %dus = stablehlo.dynamic_update_slice %operand, %update, %off : (tensor<1024xf64>, tensor<8xf64>, tensor<i64>) -> tensor<1024xf64>
    %s = stablehlo.slice %dus [0:4] : (tensor<1024xf64>) -> tensor<4xf64>
    return %s : tensor<4xf64>
  }
}

// CHECK-LABEL:   func.func @disjoint_add_convert(
// CHECK:           %[[FWD2:.*]] = stablehlo.slice %arg0 [0:4] : (tensor<1024xf64>) -> tensor<4xf64>
// CHECK:           return %[[FWD2]]

// -----

module {
  // A run of zeroing updates, as reverse mode emits for a chain of in-place
  // writes.  A read disjoint from every zeroed window must be forwarded all the
  // way to the base buffer, which is what leaves each intermediate version with
  // a single consumer.
  func.func @zeroing_chain(%operand: tensor<1024xf64>) -> tensor<8xf64> {
    %one = stablehlo.constant dense<1> : tensor<i32>
    %zeros = stablehlo.constant dense<0.000000e+00> : tensor<8xf64>
    %b0 = stablehlo.constant dense<101> : tensor<i32>
    %b1 = stablehlo.constant dense<201> : tensor<i32>
    %b2 = stablehlo.constant dense<301> : tensor<i32>
    %o0 = stablehlo.subtract %b0, %one : tensor<i32>
    %o1 = stablehlo.subtract %b1, %one : tensor<i32>
    %o2 = stablehlo.subtract %b2, %one : tensor<i32>
    %d0 = stablehlo.dynamic_update_slice %operand, %zeros, %o0 : (tensor<1024xf64>, tensor<8xf64>, tensor<i32>) -> tensor<1024xf64>
    %d1 = stablehlo.dynamic_update_slice %d0, %zeros, %o1 : (tensor<1024xf64>, tensor<8xf64>, tensor<i32>) -> tensor<1024xf64>
    %d2 = stablehlo.dynamic_update_slice %d1, %zeros, %o2 : (tensor<1024xf64>, tensor<8xf64>, tensor<i32>) -> tensor<1024xf64>
    %s = stablehlo.slice %d2 [0:8] : (tensor<1024xf64>) -> tensor<8xf64>
    return %s : tensor<8xf64>
  }
}

// CHECK-LABEL:   func.func @zeroing_chain(
// CHECK:           %[[BASE:.*]] = stablehlo.slice %arg0 [0:8] : (tensor<1024xf64>) -> tensor<8xf64>
// CHECK:           return %[[BASE]]

// -----

module {
  // A genuinely dynamic index is still left alone.
  func.func @runtime_index(%operand: tensor<1024xf64>, %update: tensor<8xf64>, %off: tensor<i32>) -> tensor<4xf64> {
    %dus = stablehlo.dynamic_update_slice %operand, %update, %off : (tensor<1024xf64>, tensor<8xf64>, tensor<i32>) -> tensor<1024xf64>
    %s = stablehlo.slice %dus [0:4] : (tensor<1024xf64>) -> tensor<4xf64>
    return %s : tensor<4xf64>
  }
}

// CHECK-LABEL:   func.func @runtime_index(
// CHECK:           stablehlo.dynamic_update_slice
// CHECK:           stablehlo.slice

// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

// A memref.store guarded by an affine.if raises to a masked scatter: a
// masked-out lane must not write at all — its index expression is
// unconstrained and can collide with a live lane's slot, and scatter
// applies duplicate indices in unspecified order. Dead lanes' indices are
// selected out of bounds, so the scatter drops their updates.

module {
  func.func @main(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    affine.parallel (%i, %j) = (0, 0) to (10, 10) step (1, 1) {
      affine.if affine_set<(d0, d1) : (d0 - d1 >= 0)>(%i, %j) {
        %0 = affine.load %arg1[%i * 10 + %j] : memref<100xf32>
        affine.store %0, %arg0[%i * 10 + %j] : memref<100xf32>
      }
    }
    return
  }
}

// CHECK:    func.func private @main_raised(%[[a1:.+]]: tensor<100xf32>, %[[a2:.+]]: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<10> : tensor<10xi64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<-1> : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<0> : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.iota dim = 0 {enzymexla.non_negative = [#enzymexla<guaranteed GUARANTEED>]} : tensor<10xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.negate %[[a6]] : tensor<10xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.iota dim = 0 : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.broadcast_in_dim %[[a7]], dims = [1] : (tensor<10xi64>) -> tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.add %[[a8]], %[[a9]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.compare GE, %[[a10]], %[[a5]] : (tensor<10x10x1xi64>, tensor<10x10x1xi64>) -> tensor<10x10x1xi1>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.multiply %[[a6]], %[[a3]] : tensor<10xi64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [0] : (tensor<10xi64>) -> tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.iota dim = 1 : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.add %[[a13]], %[[a14]] : tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.reshape %[[a2]] : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.select %[[a11]], %[[a15]], %[[a4]] : tensor<10x10x1xi1>, tensor<10x10x1xi64>
// CHECK-NEXT:    %[[a18:.+]] = "stablehlo.scatter"(%[[a1]], %[[a17]], %[[a16]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%[[a19:.+]]: tensor<f32>, %[[a20:.+]]: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %[[a20]] : tensor<f32>
// CHECK-NEXT:    }) : (tensor<100xf32>, tensor<10x10x1xi64>, tensor<10x10xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %[[a18]], %[[a2]] : tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }

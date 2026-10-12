// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s

// No transpose/pointwise rewrite pattern is registered here. The generic
// language enumerates witnessed TTIR operations and reuses the witnessed
// transpose order. Per-element denotations prove that truncation commutes with
// the index permutation; the structural-width cost prefers the f16 transpose.

// CHECK-LABEL: tt.func @narrow_before_transpose(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8x16xf32>
// CHECK-NOT: tt.trans %[[SOURCE:.*]]
// CHECK: %[[NARROW:.*]] = arith.truncf %[[SOURCE]] : tensor<8x16xf32> to tensor<8x16xf16>
// CHECK-NEXT: %[[RESULT:.*]] = tt.trans %[[NARROW]] {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>
// CHECK-NEXT: tt.return %[[RESULT]] : tensor<16x8xf16>
tt.func @narrow_before_transpose(%source: tensor<8x16xf32>) -> tensor<16x8xf16> {
  %transposed = tt.trans %source {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  %result = arith.truncf %transposed : tensor<16x8xf32> to tensor<16x8xf16>
  tt.return %result : tensor<16x8xf16>
}

// A shared transpose is an island boundary. Replacing only one consumer may
// not claim the transpose disappears, so AGS conservatively leaves it alone.
// CHECK-LABEL: tt.func @shared_transpose_is_not_free(
// CHECK: %[[TRANS:.*]] = tt.trans %{{.*}} {order = array<i32: 1, 0>}
// CHECK: arith.truncf %[[TRANS]]
// CHECK: tt.return %{{.*}}, %[[TRANS]]
tt.func @shared_transpose_is_not_free(%source: tensor<8x16xf32>) -> (tensor<16x8xf16>, tensor<16x8xf32>) {
  %transposed = tt.trans %source {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  %narrow = arith.truncf %transposed : tensor<16x8xf32> to tensor<16x8xf16>
  tt.return %narrow, %transposed : tensor<16x8xf16>, tensor<16x8xf32>
}

// Binary pointwise semantics are registered exactly like unary semantics. The
// generic enumerator discovers one add followed by one transpose instead of
// transposing both inputs independently.
// CHECK-LABEL: tt.func @merge_binary_transposes(
// CHECK-SAME: %[[LHS:.*]]: tensor<8x16xf32>, %[[RHS:.*]]: tensor<8x16xf32>
// CHECK-NOT: tt.trans %[[LHS]]
// CHECK-NOT: tt.trans %[[RHS]]
// CHECK: %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : tensor<8x16xf32>
// CHECK-NEXT: %[[RESULT:.*]] = tt.trans %[[SUM]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
// CHECK-NEXT: tt.return %[[RESULT]] : tensor<16x8xf32>
tt.func @merge_binary_transposes(%lhs: tensor<8x16xf32>, %rhs: tensor<8x16xf32>) -> tensor<16x8xf32> {
  %lhs_t = tt.trans %lhs {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  %rhs_t = tt.trans %rhs {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
  %result = arith.addf %lhs_t, %rhs_t : tensor<16x8xf32>
  tt.return %result : tensor<16x8xf32>
}

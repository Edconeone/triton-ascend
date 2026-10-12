// This file is consumed by the non-default ags-indirect-memory-audit target.
// Rule generation sees only the finite operation/type universe and declarative
// semantics; it does not see this source topology.

// CHECK-LABEL: tt.func @masked_indirect_load(
// CHECK: ascend.indirect_load
// CHECK-NOT: tt.splat
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
tt.func @masked_indirect_load(
    %base: !tt.ptr<f32>,
    %offsets: tensor<4xi64>,
    %mask: tensor<4xi1>,
    %other: tensor<4xf32>) -> tensor<4xf32> {
  %pointers = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %addresses = tt.addptr %pointers, %offsets
    : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
  %loaded = tt.load %addresses, %mask, %other
    : tensor<4x!tt.ptr<f32>>
  tt.return %loaded : tensor<4xf32>
}

// A volatile target form is intentionally outside the equivalence universe:
// its observable read cannot replace a normal TTIR load.
// CHECK-LABEL: tt.func @keep_unrelated_scalar_load(
// CHECK: tt.load
tt.func @keep_unrelated_scalar_load(%pointer: !tt.ptr<f32>) -> f32 {
  %loaded = tt.load %pointer : !tt.ptr<f32>
  tt.return %loaded : f32
}

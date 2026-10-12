// This file is consumed by the non-default ags-store-coalescing-audit target.
// Its database is generated exclusively from BuiltinTTOpSemantics.ttsem and
// the finite C07 universe; the source/replacement topology is not an input to
// rule generation.

// CHECK-LABEL: tt.func @adjacent_stores(
// CHECK-NOT: tt.store
// CHECK: tensor.empty
// CHECK-COUNT-2: tensor.insert_slice
// CHECK: tt.store
// CHECK-NOT: tt.store
tt.func @adjacent_stores(
    %base: !tt.ptr<i32>,
    %low: tensor<4xi32>,
    %high: tensor<4xi32>) {
  %base4 = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %high_range = tt.make_range {start = 4 : i32, end = 8 : i32} : tensor<4xi32>
  %high_ptr = tt.addptr %base4, %high_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  tt.store %high_ptr, %high : tensor<4x!tt.ptr<i32>>
  %low_range = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %low_ptr = tt.addptr %base4, %low_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  tt.store %low_ptr, %low : tensor<4x!tt.ptr<i32>>
  tt.return
}

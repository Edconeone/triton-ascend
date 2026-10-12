// This blind acceptance input is intentionally not part of rule generation.
// The offline universe contains only finite types, OPs and .ttsem semantics.

// CHECK-LABEL: tt.func @rank1_stride_load(
// CHECK: ascend.stride_load
// CHECK-NOT: tt.make_range
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
tt.func @rank1_stride_load(
    %base: !tt.ptr<f32>, %offset: i64, %other: f32,
    %stride: i64, %numel: i64) -> tensor<4xf32> {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32}
    : tensor<4xi32>
  %wide = arith.extsi %range : tensor<4xi32> to tensor<4xi64>
  %stride_tensor = tt.splat %stride : i64 -> tensor<4xi64>
  %scaled = arith.muli %wide, %stride_tensor : tensor<4xi64>
  %offset_tensor = tt.splat %offset : i64 -> tensor<4xi64>
  %offsets = arith.addi %offset_tensor, %scaled : tensor<4xi64>
  %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %addresses = tt.addptr %base_tensor, %offsets
    : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
  %numel_tensor = tt.splat %numel : i64 -> tensor<4xi64>
  %mask = arith.cmpi slt, %wide, %numel_tensor
    : tensor<4xi64>
  %other_tensor = tt.splat %other : f32 -> tensor<4xf32>
  %loaded = tt.load %addresses, %mask, %other_tensor
    : tensor<4x!tt.ptr<f32>>
  tt.return %loaded : tensor<4xf32>
}

// CHECK-LABEL: tt.func @rank1_stride_store(
// CHECK: ascend.stride_store
// CHECK-NOT: tt.make_range
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.store
tt.func @rank1_stride_store(
    %base: !tt.ptr<f32>, %source: tensor<4xf32>,
    %offset: i64, %stride: i64, %numel: i64) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32}
    : tensor<4xi32>
  %wide = arith.extsi %range : tensor<4xi32> to tensor<4xi64>
  %stride_tensor = tt.splat %stride : i64 -> tensor<4xi64>
  %scaled = arith.muli %wide, %stride_tensor : tensor<4xi64>
  %offset_tensor = tt.splat %offset : i64 -> tensor<4xi64>
  %offsets = arith.addi %offset_tensor, %scaled : tensor<4xi64>
  %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %addresses = tt.addptr %base_tensor, %offsets
    : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
  %numel_tensor = tt.splat %numel : i64 -> tensor<4xi64>
  %mask = arith.cmpi slt, %wide, %numel_tensor
    : tensor<4xi64>
  tt.store %addresses, %source, %mask : tensor<4x!tt.ptr<f32>>
  tt.return
}

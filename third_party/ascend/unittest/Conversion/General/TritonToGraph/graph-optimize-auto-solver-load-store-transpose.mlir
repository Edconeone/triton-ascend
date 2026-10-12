// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s

// The solver receives no LoadStoreTranspose pattern.  tt.load and tt.trans
// contribute only their basic denotations.  Import normalization proves that
// applying a permutation and its inverse returns the original loaded tensor;
// the structural cost model then reuses the existing one-node load instead of
// the three-node load/transpose/transpose graph.
// CHECK-LABEL: tt.func @cancel_roundtrip_load_transpose(
// CHECK-SAME: %[[POINTERS:.*]]: tensor<2x3x!tt.ptr<f32>>
// CHECK-NEXT: %[[LOADED:.*]] = tt.load %[[POINTERS]] : tensor<2x3x!tt.ptr<f32>>
// CHECK-NEXT: tt.return %[[LOADED]] : tensor<2x3xf32>
// CHECK-NOT: tt.trans
tt.func @cancel_roundtrip_load_transpose(
    %pointers: tensor<2x3x!tt.ptr<f32>>) -> tensor<2x3xf32> {
  %loaded = tt.load %pointers : tensor<2x3x!tt.ptr<f32>>
  %swapped = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<2x3xf32> -> tensor<3x2xf32>
  %restored = tt.trans %swapped {order = array<i32: 1, 0>} : tensor<3x2xf32> -> tensor<2x3xf32>
  tt.return %restored : tensor<2x3xf32>
}

// A shared intermediate transpose is outside the current single-root island.
// It must not be deleted just because one result happens to round-trip.
// CHECK-LABEL: tt.func @keep_shared_intermediate_transpose(
// CHECK: %[[LOADED:.*]] = tt.load
// CHECK-NEXT: %[[SWAPPED:.*]] = tt.trans %[[LOADED]]
// CHECK-NEXT: %[[RESTORED:.*]] = tt.trans %[[SWAPPED]]
// CHECK-NEXT: tt.return %[[RESTORED]], %[[SWAPPED]]
tt.func @keep_shared_intermediate_transpose(
    %pointers: tensor<2x3x!tt.ptr<f32>>) -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  %loaded = tt.load %pointers : tensor<2x3x!tt.ptr<f32>>
  %swapped = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<2x3xf32> -> tensor<3x2xf32>
  %restored = tt.trans %swapped {order = array<i32: 1, 0>} : tensor<3x2xf32> -> tensor<2x3xf32>
  tt.return %restored, %swapped : tensor<2x3xf32>, tensor<3x2xf32>
}

// This is the closed affine memory component exercised by the handwritten
// LoadStoreTranspose rule, not a transpose-cancellation proxy.  The offline
// rank-2 semantic-family database is generated before this file is read.  Its
// Value+Memory proof discovers the shape-permuted implementation, while the
// generic memory-stride cost selects the unit-stride innermost lane.
// DB-C06-LABEL: tt.func @closed_affine_memory_component(
// DB-C06-NOT: tensor<2x3x!tt.ptr<f32>>
// DB-C06: tt.load {{.*}} : tensor<3x2x!tt.ptr<f32>>
// DB-C06: arith.negf {{.*}} : tensor<3x2xf32>
// DB-C06: tt.store {{.*}} : tensor<3x2x!tt.ptr<f32>>
tt.func @closed_affine_memory_component(
    %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %two = arith.constant 2 : i32
  %axis1_scale = tt.splat %two : i32 -> tensor<1x3xi32>
  %axis1_scaled = arith.muli %axis1_expand, %axis1_scale : tensor<1x3xi32>
  %axis0_full = tt.broadcast %axis0_expand : tensor<2x1xi32> -> tensor<2x3xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x3xi32> -> tensor<2x3xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<2x3xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
  %src_ptr = tt.addptr %src_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
  %value = tt.load %src_ptr : tensor<2x3x!tt.ptr<f32>>
  %result = arith.negf %value : tensor<2x3xf32>
  tt.store %dst_ptr, %result : tensor<2x3x!tt.ptr<f32>>
  tt.return
}

// Opaque pointer tensors carry no physical-stride evidence.  The cost model
// must fail closed rather than inventing a profitable orientation from shape
// alone.  This function is intentionally not a positive DB-C06 check.
tt.func @masked_rank2_memory_opaque_arguments(
    %source: tensor<8x64x!tt.ptr<bf16>>,
    %target: tensor<8x64x!tt.ptr<bf16>>,
    %mask: tensor<8x64xi1>,
    %other: tensor<8x64xbf16>) {
  %value = tt.load %source, %mask, %other : tensor<8x64x!tt.ptr<bf16>>
  tt.store %target, %value, %mask : tensor<8x64x!tt.ptr<bf16>>
  tt.return
}

// The masked production-shape shard is generated without reading this test.
// Its symbolic template binds pointer/value/mask holes to this previously
// unseen dynamic-affine address graph.  The online pass re-imports the bound
// SSA pointer only for cost estimation, recognizes the symbolic inner stride,
// and instantiates the already-proved permutation without online enumeration.
// DB-C06-TEMPLATE-LABEL: tt.func @masked_rank2_dynamic_affine_template(
// DB-C06-TEMPLATE: tt.load {{.*}} : tensor<64x8x!tt.ptr<bf16>>
// DB-C06-TEMPLATE: tt.store {{.*}} : tensor<64x8x!tt.ptr<bf16>>
// DB-C06-TEMPLATE-NOT: tt.load {{.*}} : tensor<8x64x!tt.ptr<bf16>>
// DB-C06-TEMPLATE-NOT: tt.store {{.*}} : tensor<8x64x!tt.ptr<bf16>>
tt.func @masked_rank2_dynamic_affine_template(
    %source_base: !tt.ptr<bf16>,
    %target_base: !tt.ptr<bf16>,
    %stride: i32,
    %mask: tensor<8x64xi1>,
    %other: tensor<8x64xbf16>) {
  %axis0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %axis1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %axis0_expand = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %axis1_expand = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %axis0_full = tt.broadcast %axis0_expand : tensor<8x1xi32> -> tensor<8x64xi32>
  %stride_splat = tt.splat %stride : i32 -> tensor<1x64xi32>
  %axis1_scaled = arith.muli %axis1_expand, %stride_splat : tensor<1x64xi32>
  %axis1_full = tt.broadcast %axis1_scaled : tensor<1x64xi32> -> tensor<8x64xi32>
  %offset = arith.addi %axis0_full, %axis1_full : tensor<8x64xi32>
  %source_splat = tt.splat %source_base : !tt.ptr<bf16> -> tensor<8x64x!tt.ptr<bf16>>
  %target_splat = tt.splat %target_base : !tt.ptr<bf16> -> tensor<8x64x!tt.ptr<bf16>>
  %source = tt.addptr %source_splat, %offset : tensor<8x64x!tt.ptr<bf16>>, tensor<8x64xi32>
  %target = tt.addptr %target_splat, %offset : tensor<8x64x!tt.ptr<bf16>>, tensor<8x64xi32>
  %value = tt.load %source, %mask, %other : tensor<8x64x!tt.ptr<bf16>>
  tt.store %target, %value, %mask : tensor<8x64x!tt.ptr<bf16>>
  tt.return
}

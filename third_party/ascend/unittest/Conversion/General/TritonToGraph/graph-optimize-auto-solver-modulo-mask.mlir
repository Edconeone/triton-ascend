// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s

// C04 is proved at the store observation boundary.  The source load may use
// the linear address only after it carries the same mask as the store; on true
// lanes, 0 <= offset < bound proves signed remainder == offset.
// CHECK-LABEL: tt.func @linearize_wrapped_load(
// CHECK-NOT: arith.remsi
// CHECK: %[[GUARD:.*]] = arith.cmpi slt
// CHECK: %[[LOAD:.*]] = tt.load %{{.*}}, %[[GUARD]]
// CHECK: tt.store %{{.*}}, %[[LOAD]], %[[GUARD]]
// DB-RANK1-LABEL: tt.func @linearize_wrapped_load(
// DB-RANK1-NOT: arith.remsi
// DB-RANK1: %[[DB_GUARD:.*]] = arith.cmpi slt
// DB-RANK1: %[[DB_LOAD:.*]] = tt.load %{{.*}}, %[[DB_GUARD]]
// The database materializer may rebuild the same comparison for the store;
// common-subexpression elimination runs later in the pipeline.
// DB-RANK1: %[[DB_STORE_GUARD:.*]] = arith.cmpi slt
// DB-RANK1: tt.store %{{.*}}, %[[DB_LOAD]], %[[DB_STORE_GUARD]]
tt.func @linearize_wrapped_load(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
  %offset = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %bound = tt.splat %n : i32 -> tensor<16xi32>
  %wrapped = arith.remsi %offset, %bound : tensor<16xi32>
  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %wrapped_ptr = tt.addptr %src_base, %wrapped : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %value = tt.load %wrapped_ptr : tensor<16x!tt.ptr<f32>>
  %guard = arith.cmpi slt, %offset, %bound : tensor<16xi32>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %linear_dst = tt.addptr %dst_base, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %linear_dst, %value, %guard : tensor<16x!tt.ptr<f32>>
  tt.return
}

// The end-to-end Python kernel expands the wrapped column to two dimensions
// and combines it with a row offset.  Keep that shape in the structural guard:
// after AGS selects the linear masked load, the obsolete wrapped-address DAG
// must be removed as generic dead code rather than leaving arith.remsi behind.
// CHECK-LABEL: tt.func @linearize_wrapped_2d_load(
// CHECK-NOT: arith.remsi
// CHECK: %[[GUARD_2D:.*]] = tt.broadcast %{{.*}} : tensor<1x16xi1> -> tensor<8x16xi1>
// CHECK: %[[LOAD_2D:.*]] = tt.load %{{.*}}, %[[GUARD_2D]]
// CHECK: tt.store %{{.*}}, %[[LOAD_2D]], %[[GUARD_2D]]
// DB-RANK2-LABEL: tt.func @linearize_wrapped_2d_load(
// DB-RANK2-NOT: arith.remsi
// DB-RANK2: %[[DB_LOAD_GUARD_2D:.*]] = tt.broadcast
// DB-RANK2: %[[DB_LOAD_2D:.*]] = tt.load %{{.*}}, %[[DB_LOAD_GUARD_2D]]
// DB-RANK2: %[[DB_STORE_GUARD_2D:.*]] = tt.broadcast
// DB-RANK2: tt.store %{{.*}}, %[[DB_LOAD_2D]], %[[DB_STORE_GUARD_2D]]
tt.func @linearize_wrapped_2d_load(
    %src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
  %columns = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %bound_columns = tt.splat %n : i32 -> tensor<16xi32>
  %wrapped = arith.remsi %columns, %bound_columns : tensor<16xi32>
  %wrapped_row = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %wrapped_full = tt.broadcast %wrapped_row : tensor<1x16xi32> -> tensor<8x16xi32>

  %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %rows_column = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %bound_rows = tt.splat %n : i32 -> tensor<8x1xi32>
  %row_offset = arith.muli %rows_column, %bound_rows : tensor<8x1xi32>
  %row_full = tt.broadcast %row_offset : tensor<8x1xi32> -> tensor<8x16xi32>
  %wrapped_offset = arith.addi %row_full, %wrapped_full : tensor<8x16xi32>

  %linear_row = tt.expand_dims %columns {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %linear_full = tt.broadcast %linear_row : tensor<1x16xi32> -> tensor<8x16xi32>
  %linear_offset = arith.addi %row_full, %linear_full : tensor<8x16xi32>
  %bound_guard = tt.splat %n : i32 -> tensor<1x16xi32>
  %guard_row = arith.cmpi slt, %linear_row, %bound_guard : tensor<1x16xi32>
  %guard = tt.broadcast %guard_row : tensor<1x16xi1> -> tensor<8x16xi1>

  %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %wrapped_ptr = tt.addptr %src_base, %wrapped_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
  %value = tt.load %wrapped_ptr : tensor<8x16x!tt.ptr<f32>>
  %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %linear_ptr = tt.addptr %dst_base, %linear_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
  tt.store %linear_ptr, %value, %guard : tensor<8x16x!tt.ptr<f32>>
  tt.return
}


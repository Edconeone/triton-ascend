// Blind acceptance for a guarded tail partition.  The frontend contract says
// only that grid axis 0 is bounded by argument 2; the semantic proof still has
// to find a complete lane set and a guard on every lane-dependent write.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hacc.coalesce_axis = 0 : i32
// CHECK-SAME: hacc.coalesce_factor = 16 : i32
// CHECK-SAME: hacc.coalesce_grid_ceil_div = 1 : i32
// CHECK-LABEL: tt.func @guarded_write_one_value_per_program(
// CHECK-NOT: ags.launch_grid_bound
// CHECK-COUNT-16: tt.store
// CHECK-NOT: ags.kernel_partition
module {
  tt.func @guarded_write_one_value_per_program(
      %base: !tt.ptr<i32>, %value: i32, %bound: i32)
      attributes {ags.launch_grid_bound_axes = array<i32: 0>,
                  ags.launch_grid_bound_arguments = array<i32: 2>} {
    %pid = tt.get_program_id x : i32
    %past_end = arith.cmpi sge, %pid, %bound : i32
    cf.cond_br %past_end, ^return, ^work
  ^return:
    tt.return
  ^work:
    %pointer = tt.addptr %base, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %value : !tt.ptr<i32>
    tt.return
  }
}

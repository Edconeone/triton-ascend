// This file is consumed by ags-launch-partition-audit after the RuleDB has
// already been generated from the finite semantic universe.  The source
// topology is therefore blind acceptance data, not rule-generation input.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hacc.coalesce_axis = 1 : i32
// CHECK-SAME: hacc.coalesce_factor = 16 : i32
// CHECK-SAME: hacc.coalesce_grid_ceil_div = 0 : i32
// CHECK-LABEL: tt.func @write_one_value_per_axis1_program(
// CHECK-COUNT-16: tt.store
// CHECK-NOT: ags.launch_remap
// CHECK-NOT: ags.kernel
// CHECK-NOT: ags.kernel_partition
module {
  tt.func @write_one_value_per_axis1_program(%base: !tt.ptr<i32>, %value: i32) {
    %pid = tt.get_program_id y : i32
    %pointer = tt.addptr %base, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %value : !tt.ptr<i32>
    tt.return
  }
}

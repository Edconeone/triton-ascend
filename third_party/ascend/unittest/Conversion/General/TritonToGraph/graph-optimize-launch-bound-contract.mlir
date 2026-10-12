// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=2048' -o - | FileCheck %s

// Launch-bound attributes are frontend-to-AGS proof inputs.  The graph pass
// must consume them even when no database rule is selected, so they never
// leak into TritonAdapter or target IR.

// CHECK-LABEL: tt.func @launch_bound_contract(
// CHECK-NOT: ags.launch_grid_bound
// CHECK: tt.return
tt.func @launch_bound_contract(%ptr: !tt.ptr<i32>, %rows: i32, %columns: i32)
    attributes {ags.launch_grid_bound_axes = array<i32: 0, 1>,
                ags.launch_grid_bound_arguments = array<i32: 1, 2>} {
  tt.store %ptr, %rows : !tt.ptr<i32>
  tt.return
}

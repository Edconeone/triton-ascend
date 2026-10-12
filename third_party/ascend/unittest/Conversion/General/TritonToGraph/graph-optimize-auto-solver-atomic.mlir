// This file is consumed by the non-default ags-atomic-transaction-audit
// target.  Its database is discovered from the finite atomic universe.  The
// source/replacement topology is not an input to rule generation.

// CHECK-LABEL: tt.func @constant_true_mask_atomic(
// CHECK: tt.atomic_rmw add, relaxed, gpu, %{{.*}}, %{{.*}} : (!tt.ptr<i32>, i32) -> i32
// CHECK-NOT: tt.atomic_rmw {{.*}} : (!tt.ptr<i32>, i32, i1) -> i32
tt.func @constant_true_mask_atomic(%ptr: !tt.ptr<i32>, %value: i32) {
  %true = arith.constant true
  %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %value, %true
    : (!tt.ptr<i32>, i32, i1) -> i32
  tt.return
}

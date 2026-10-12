// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=0 max-rewrites-per-function=1' -o - | FileCheck %s
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize -graph-optimize -o - | FileCheck %s
// RUN: triton-opt %s -graph-optimize='rule-mask=0' -o - | FileCheck %s --check-prefix=ZERO-MASK
// RUN: triton-opt %s -graph-optimize -o - | FileCheck %s --check-prefix=ALL-MASK
// RUN: not triton-opt %s -graph-optimize='rule-mask=4096' -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID-MASK

// ZERO-MASK: hacc.auto_graph_solver_rule_mask = 0 : i16
// ALL-MASK: hacc.auto_graph_solver_rule_mask = 4095 : i16
// INVALID-MASK: graph-optimize rule-mask contains unknown or out-of-range bits: 4096

// CHECK-LABEL: tt.func @graph_optimize_first()
tt.func @graph_optimize_first() {
  // CHECK-NEXT: tt.return
  tt.return
  // CHECK-NEXT: }
}

// -----

// CHECK-LABEL: tt.func @graph_optimize_second(
tt.func @graph_optimize_second(%value: i32) -> i32 {
  // CHECK-NEXT: tt.return %{{.*}} : i32
  tt.return %value : i32
  // CHECK-NEXT: }
}

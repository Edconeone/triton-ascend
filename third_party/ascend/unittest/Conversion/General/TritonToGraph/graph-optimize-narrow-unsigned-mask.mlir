// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=0' -o - | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=1' -o - | FileCheck %s --check-prefix=SELECT
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=2' -o - | FileCheck %s --check-prefix=BITWISE
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=4' -o - | FileCheck %s --check-prefix=SHIFT
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=8' -o - | FileCheck %s --check-prefix=COMPARE
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512 narrow-unsigned-rule-mask=8 compile-on-910-95=true' -o - | FileCheck %s --check-prefix=A5-COMPARE
// RUN: not triton-opt %s -graph-optimize='rule-mask=0 narrow-unsigned-rule-mask=16' -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: graph-optimize narrow-unsigned-rule-mask contains unknown or out-of-range bits: 16

// OFF-LABEL: tt.func @narrow_select(
// OFF: arith.select {{.*}} : tensor<16xi1>, tensor<16xi64>
// SELECT-LABEL: tt.func @narrow_select(
// SELECT: arith.select {{.*}} : tensor<16xi1>, tensor<16xi32>
// SELECT: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
tt.func @narrow_select(%condition: tensor<16xi1>, %lhs: tensor<16xi32>, %rhs: tensor<16xi32>) -> tensor<16xi64> {
  %lhs64 = arith.extui %lhs : tensor<16xi32> to tensor<16xi64>
  %rhs64 = arith.extui %rhs : tensor<16xi32> to tensor<16xi64>
  %result = arith.select %condition, %lhs64, %rhs64 : tensor<16xi1>, tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// OFF-LABEL: tt.func @narrow_bitwise(
// OFF: arith.andi {{.*}} : tensor<16xi64>
// BITWISE-LABEL: tt.func @narrow_bitwise(
// BITWISE-NOT: arith.andi
// BITWISE: arith.trunci {{.*}} : tensor<16xi64> to tensor<16xi32>
// BITWISE: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
tt.func @narrow_bitwise(%input: tensor<16xi64>) -> tensor<16xi64> {
  %mask = arith.constant dense<4294967295> : tensor<16xi64>
  %result = arith.andi %input, %mask : tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// Dead roots are not extraction candidates. Rewriting them would spend the
// global budget and manufacture another dead narrow graph.
// OFF-LABEL: tt.func @keep_dead_bitwise(
// OFF: arith.andi {{.*}} : tensor<16xi64>
// BITWISE-LABEL: tt.func @keep_dead_bitwise(
// BITWISE: arith.andi {{.*}} : tensor<16xi64>
// BITWISE-NOT: arith.andi {{.*}} : tensor<16xi32>
tt.func @keep_dead_bitwise(%input: tensor<16xi64>) {
  %mask = arith.constant dense<4294967295> : tensor<16xi64>
  %dead = arith.andi %input, %mask : tensor<16xi64>
  tt.return
}

// OFF-LABEL: tt.func @narrow_shift(
// OFF: arith.shrui {{.*}} : tensor<16xi64>
// SHIFT-LABEL: tt.func @narrow_shift(
// SHIFT: arith.shrui {{.*}} : tensor<16xi32>
// SHIFT: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
tt.func @narrow_shift(%input: tensor<16xi32>) -> tensor<16xi64> {
  %wide = arith.extui %input : tensor<16xi32> to tensor<16xi64>
  %amount = arith.constant dense<8> : tensor<16xi64>
  %result = arith.shrui %wide, %amount : tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// OFF-LABEL: tt.func @narrow_compare(
// OFF: arith.cmpi eq, {{.*}} : tensor<16xi64>
// COMPARE-LABEL: tt.func @narrow_compare(
// COMPARE: arith.cmpi eq, {{.*}} : tensor<16xi32>
tt.func @narrow_compare(%lhs: tensor<16xi32>, %rhs: tensor<16xi32>) -> tensor<16xi1> {
  %lhs64 = arith.extui %lhs : tensor<16xi32> to tensor<16xi64>
  %rhs64 = arith.extui %rhs : tensor<16xi32> to tensor<16xi64>
  %result = arith.cmpi eq, %lhs64, %rhs64 : tensor<16xi64>
  tt.return %result : tensor<16xi1>
}

// The scalar-threshold alternative is legal but currently costs more because
// it needs an upper-bound guard, splat, and boolean combine. Cost extraction
// must therefore keep the original wide compare instead of rewriting merely
// because the semantic rule matched.
// OFF-LABEL: tt.func @keep_scalar_threshold_compare(
// OFF: arith.cmpi eq, {{.*}} : tensor<16xi64>
// COMPARE-LABEL: tt.func @keep_scalar_threshold_compare(
// COMPARE: arith.cmpi eq, {{.*}} : tensor<16xi64>
// COMPARE-NOT: arith.cmpi eq, {{.*}} : tensor<16xi32>
// A5-COMPARE-LABEL: tt.func @keep_scalar_threshold_compare(
// A5-COMPARE: arith.cmpi ule, {{.*}} : i64
// A5-COMPARE: arith.trunci {{.*}} : i64 to i32
// A5-COMPARE: arith.cmpi eq, {{.*}} : tensor<16xi32>
// A5-COMPARE: tt.splat {{.*}} : i1 -> tensor<16xi1>
// A5-COMPARE: arith.andi {{.*}} : tensor<16xi1>
// A5-COMPARE-NOT: arith.cmpi eq, {{.*}} : tensor<16xi64>
tt.func @keep_scalar_threshold_compare(%lhs: tensor<16xi32>, %threshold: i64) -> tensor<16xi1> {
  %lhs64 = arith.extui %lhs : tensor<16xi32> to tensor<16xi64>
  %rhs64 = tt.splat %threshold : i64 -> tensor<16xi64>
  %result = arith.cmpi eq, %lhs64, %rhs64 : tensor<16xi64>
  tt.return %result : tensor<16xi1>
}

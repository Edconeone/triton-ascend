// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=2048' -o - | FileCheck %s --check-prefix=NO-DB

// AutoGraphSolver is only given primitive integer tensor semantics. It imports
// this graph, enumerates arith.subi(scan, input), and proves the candidate by
// generic finite-sum normalization. No diagonal source pattern is registered.

// CHECK-LABEL: tt.func @forward_unit_shift(
// NO-DB-LABEL: tt.func @forward_unit_shift(
// NO-DB: tt.reduce
// CHECK: %[[SCAN:.*]] = "tt.scan"
// CHECK: %[[RESULT:.*]] = arith.subi %[[SCAN]], %[[INPUT:.*]] : tensor<16xi32>
// CHECK-NOT: tt.reduce
// CHECK: tt.return %[[RESULT]]
tt.func @forward_unit_shift(%input: tensor<16xi32>) -> tensor<16xi32> {
  %scan = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32>) -> tensor<16xi32>
  %expanded = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %values = tt.broadcast %expanded : tensor<1x16xi32> -> tensor<16x16xi32>
  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expanded = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %rows = tt.broadcast %row_expanded : tensor<16x1xi32> -> tensor<16x16xi32>
  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expanded = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %shifted = arith.addi %col_expanded, %one : tensor<1x16xi32>
  %cols = tt.broadcast %shifted : tensor<1x16xi32> -> tensor<16x16xi32>
  %diagonal = arith.cmpi eq, %rows, %cols : tensor<16x16xi32>
  %zero = arith.constant dense<0> : tensor<16x16xi32>
  %selected = arith.select %diagonal, %values, %zero : tensor<16x16xi1>, tensor<16x16xi32>
  %result = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) : (tensor<16x16xi32>) -> tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// The same generic identities derive the reverse-scan form.
// CHECK-LABEL: tt.func @reverse_unit_shift(
// CHECK: %[[REVERSE_SCAN:.*]] = "tt.scan"
// CHECK: %[[REVERSE_RESULT:.*]] = arith.subi %[[REVERSE_SCAN]], %[[REVERSE_INPUT:.*]] : tensor<16xi32>
// CHECK-NOT: tt.reduce
// CHECK: tt.return %[[REVERSE_RESULT]]
tt.func @reverse_unit_shift(%input: tensor<16xi32>) -> tensor<16xi32> {
  %scan = "tt.scan"(%input) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32>) -> tensor<16xi32>
  %expanded = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %values = tt.broadcast %expanded : tensor<1x16xi32> -> tensor<16x16xi32>
  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expanded = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %rows = tt.broadcast %row_expanded : tensor<16x1xi32> -> tensor<16x16xi32>
  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expanded = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %shifted = arith.subi %col_expanded, %one : tensor<1x16xi32>
  %cols = tt.broadcast %shifted : tensor<1x16xi32> -> tensor<16x16xi32>
  %diagonal = arith.cmpi eq, %rows, %cols : tensor<16x16xi32>
  %zero = arith.constant dense<0> : tensor<16x16xi32>
  %selected = arith.select %diagonal, %values, %zero : tensor<16x16xi1>, tensor<16x16xi32>
  %result = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) : (tensor<16x16xi32>) -> tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// A shift of two is not scan - input. The generic equivalence oracle rejects
// the cheaper candidate without a workload-specific negative pattern.
// CHECK-LABEL: tt.func @reject_shift_of_two(
// CHECK: tt.reduce
// CHECK-NOT: arith.subi %{{.*}}, %input : tensor<16xi32>
tt.func @reject_shift_of_two(%input: tensor<16xi32>) -> tensor<16xi32> {
  %scan = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32>) -> tensor<16xi32>
  %expanded = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %values = tt.broadcast %expanded : tensor<1x16xi32> -> tensor<16x16xi32>
  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expanded = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %rows = tt.broadcast %row_expanded : tensor<16x1xi32> -> tensor<16x16xi32>
  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expanded = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %two = arith.constant dense<2> : tensor<1x16xi32>
  %shifted = arith.addi %col_expanded, %two : tensor<1x16xi32>
  %cols = tt.broadcast %shifted : tensor<1x16xi32> -> tensor<16x16xi32>
  %diagonal = arith.cmpi eq, %rows, %cols : tensor<16x16xi32>
  %zero = arith.constant dense<0> : tensor<16x16xi32>
  %selected = arith.select %diagonal, %values, %zero : tensor<16x16xi1>, tensor<16x16xi32>
  %result = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) : (tensor<16x16xi32>) -> tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

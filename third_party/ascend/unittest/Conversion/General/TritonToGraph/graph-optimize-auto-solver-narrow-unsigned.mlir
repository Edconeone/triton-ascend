// RUN: triton-opt %s --verify-each '-graph-optimize=rule-mask=2048 narrow-unsigned-rule-mask=0 allow-online-semantic-enumeration=true' -o - | FileCheck %s

// These checks exercise only the generic AutoGraphSolver.  There is no
// NarrowUnsigned candidate provider: i32 alternatives are enumerated from
// the extui/trunci/bitwise/shift/compare semantics and selected by cost.

// CHECK-LABEL: tt.func @narrow_select(
// CHECK: arith.select {{.*}} : tensor<16xi1>, tensor<16xi32>
// CHECK: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
tt.func @narrow_select(%condition: tensor<16xi1>, %lhs: tensor<16xi32>, %rhs: tensor<16xi32>) -> tensor<16xi64> {
  %lhs64 = arith.extui %lhs : tensor<16xi32> to tensor<16xi64>
  %rhs64 = arith.extui %rhs : tensor<16xi32> to tensor<16xi64>
  %result = arith.select %condition, %lhs64, %rhs64 : tensor<16xi1>, tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// CHECK-LABEL: tt.func @narrow_bitwise(
// CHECK-NOT: arith.andi {{.*}} : tensor<16xi64>
// CHECK: arith.trunci {{.*}} : tensor<16xi64> to tensor<16xi32>
// CHECK: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
tt.func @narrow_bitwise(%input: tensor<16xi64>) -> tensor<16xi64> {
  %mask = arith.constant dense<4294967295> : tensor<16xi64>
  %result = arith.andi %input, %mask : tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// CHECK-LABEL: tt.func @narrow_shift(
// CHECK: arith.shrui {{.*}} : tensor<16xi32>
// CHECK: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
// CHECK-NOT: arith.shrui {{.*}} : tensor<16xi64>
tt.func @narrow_shift(%input: tensor<16xi32>) -> tensor<16xi64> {
  %wide = arith.extui %input : tensor<16xi32> to tensor<16xi64>
  %amount = arith.constant dense<8> : tensor<16xi64>
  %result = arith.shrui %wide, %amount : tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

// CHECK-LABEL: tt.func @narrow_compare(
// CHECK: arith.cmpi eq, {{.*}} : tensor<16xi32>
// CHECK-NOT: arith.cmpi eq, {{.*}} : tensor<16xi64>
tt.func @narrow_compare(%lhs: tensor<16xi32>, %rhs: tensor<16xi32>) -> tensor<16xi1> {
  %lhs64 = arith.extui %lhs : tensor<16xi32> to tensor<16xi64>
  %rhs64 = arith.extui %rhs : tensor<16xi32> to tensor<16xi64>
  %result = arith.cmpi eq, %lhs64, %rhs64 : tensor<16xi64>
  tt.return %result : tensor<16xi1>
}

// The real radix key graph shares its zero-extended low bits between the
// positive and negative branches.  Solve the whole DAG before extracting so
// a partial extui(trunci(input)) graph is never committed.
// CHECK-LABEL: tt.func @narrow_radix_key(
// CHECK: arith.trunci {{.*}} : tensor<16xi64> to tensor<16xi32>
// CHECK: arith.xori {{.*}} : tensor<16xi32>
// CHECK: arith.ori {{.*}} : tensor<16xi32>
// CHECK: arith.select {{.*}} : tensor<16xi1>, tensor<16xi32>
// CHECK: arith.shrui {{.*}} : tensor<16xi32>
// CHECK: arith.extui {{.*}} : tensor<16xi32> to tensor<16xi64>
// CHECK-NOT: arith.shrsi
// CHECK: tt.return
tt.func @narrow_radix_key(%input: tensor<16xi64>) -> tensor<16xi64> {
  %zero = arith.constant dense<0> : tensor<16xi64>
  %low_mask = arith.constant dense<4294967295> : tensor<16xi64>
  %sign = arith.constant dense<2147483648> : tensor<16xi64>
  %minus_one = arith.constant dense<-1> : tensor<16xi64>
  %amount = arith.constant dense<8> : tensor<16xi64>
  %low = arith.andi %input, %low_mask : tensor<16xi64>
  %negative = arith.cmpi slt, %input, %zero : tensor<16xi64>
  %inverted = arith.xori %low, %minus_one : tensor<16xi64>
  %negative_key = arith.andi %inverted, %low_mask : tensor<16xi64>
  %positive_key = arith.ori %low, %sign : tensor<16xi64>
  %key = arith.select %negative, %negative_key, %positive_key : tensor<16xi1>, tensor<16xi64>
  %shifted = arith.shrsi %key, %amount : tensor<16xi64>
  tt.return %shifted : tensor<16xi64>
}

// Signed shift is not a low-unsigned-bits operation and must remain wide.
// CHECK-LABEL: tt.func @keep_signed_shift(
// CHECK: arith.shrsi {{.*}} : tensor<16xi64>
tt.func @keep_signed_shift(%input: tensor<16xi64>) -> tensor<16xi64> {
  %amount = arith.constant dense<8> : tensor<16xi64>
  %result = arith.shrsi %input, %amount : tensor<16xi64>
  tt.return %result : tensor<16xi64>
}

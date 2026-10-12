// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=512' -o - | FileCheck %s --check-prefix=AUTO
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=0' -o - | FileCheck %s --check-prefix=OFF

// The first equality-saturation rule computes a truncating subtraction in the
// result width.  Its operands become explicit low-bit projections.
//
// AUTO-LABEL: tt.func @narrow_low_bits(
// AUTO-NOT: arith.subi {{.*}} : tensor<128xi64>
// AUTO: %[[LHS:.*]] = arith.trunci %{{.*}} : tensor<128xi64> to tensor<128xi32>
// AUTO: %[[RHS:.*]] = arith.trunci %{{.*}} : tensor<128xi64> to tensor<128xi32>
// AUTO: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : tensor<128xi32>
// AUTO: tt.return %[[SUB]] : tensor<128xi32>
// OFF-LABEL: tt.func @narrow_low_bits(
// OFF: %[[WIDE:.*]] = arith.subi {{.*}} : tensor<128xi64>
// OFF: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// OFF: tt.return %[[RESULT]] : tensor<128xi32>
tt.func @narrow_low_bits(%lhs: tensor<128xi64>, %rhs: tensor<128xi64>) -> tensor<128xi32> {
  %wide = arith.subi %lhs, %rhs : tensor<128xi64>
  %result = arith.trunci %wide : tensor<128xi64> to tensor<128xi32>
  tt.return %result : tensor<128xi32>
}

// Saturation discovers trunc(ext(x)) == x on both operands.  Extraction then
// selects a single i32 subtraction with neither casts nor wide arithmetic.
//
// AUTO-LABEL: tt.func @collapse_extend_chain(
// AUTO-NOT: arith.extui
// AUTO-NOT: arith.subi {{.*}} : tensor<128xi64>
// AUTO: %[[SUB:.*]] = arith.subi %{{.*}}, %{{.*}} : tensor<128xi32>
// AUTO: tt.return %[[SUB]] : tensor<128xi32>
// OFF-LABEL: tt.func @collapse_extend_chain(
// OFF: %[[LHS64:.*]] = arith.extui %{{.*}} : tensor<128xi32> to tensor<128xi64>
// OFF: %[[RHS64:.*]] = arith.extui %{{.*}} : tensor<128xi32> to tensor<128xi64>
// OFF: %[[WIDE:.*]] = arith.subi %[[LHS64]], %[[RHS64]] : tensor<128xi64>
// OFF: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// OFF: tt.return %[[RESULT]] : tensor<128xi32>
tt.func @collapse_extend_chain(%lhs: tensor<128xi32>, %rhs: tensor<128xi32>) -> tensor<128xi32> {
  %lhs64 = arith.extui %lhs : tensor<128xi32> to tensor<128xi64>
  %rhs64 = arith.extui %rhs : tensor<128xi32> to tensor<128xi64>
  %wide = arith.subi %lhs64, %rhs64 : tensor<128xi64>
  %result = arith.trunci %wide : tensor<128xi64> to tensor<128xi32>
  tt.return %result : tensor<128xi32>
}

// Shared wide arithmetic is a region boundary: narrowing only one use would
// duplicate the subtraction instead of simplifying the whole graph.
//
// AUTO-LABEL: tt.func @keep_shared_sub(
// AUTO: %[[WIDE:.*]] = arith.subi {{.*}} : tensor<128xi64>
// AUTO: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// AUTO: tt.return %[[RESULT]], %[[WIDE]] : tensor<128xi32>, tensor<128xi64>
// OFF-LABEL: tt.func @keep_shared_sub(
// OFF: %[[WIDE:.*]] = arith.subi {{.*}} : tensor<128xi64>
// OFF: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// OFF: tt.return %[[RESULT]], %[[WIDE]] : tensor<128xi32>, tensor<128xi64>
tt.func @keep_shared_sub(%lhs: tensor<128xi64>, %rhs: tensor<128xi64>) -> (tensor<128xi32>, tensor<128xi64>) {
  %wide = arith.subi %lhs, %rhs : tensor<128xi64>
  %result = arith.trunci %wide : tensor<128xi64> to tensor<128xi32>
  tt.return %result, %wide : tensor<128xi32>, tensor<128xi64>
}

// The width at which poison is defined is observable, so overflow promises
// deliberately exclude the subtraction from the imported solver region.
//
// AUTO-LABEL: tt.func @keep_overflow_contract(
// AUTO: %[[WIDE:.*]] = arith.subi {{.*}} overflow<nuw> : tensor<128xi64>
// AUTO: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// AUTO: tt.return %[[RESULT]] : tensor<128xi32>
// OFF-LABEL: tt.func @keep_overflow_contract(
// OFF: %[[WIDE:.*]] = arith.subi {{.*}} overflow<nuw> : tensor<128xi64>
// OFF: %[[RESULT:.*]] = arith.trunci %[[WIDE]] : tensor<128xi64> to tensor<128xi32>
// OFF: tt.return %[[RESULT]] : tensor<128xi32>
tt.func @keep_overflow_contract(%lhs: tensor<128xi64>, %rhs: tensor<128xi64>) -> tensor<128xi32> {
  %wide = arith.subi %lhs, %rhs overflow<nuw> : tensor<128xi64>
  %result = arith.trunci %wide : tensor<128xi64> to tensor<128xi32>
  tt.return %result : tensor<128xi32>
}

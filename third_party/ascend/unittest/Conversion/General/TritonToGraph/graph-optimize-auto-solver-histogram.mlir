// RUN: triton-opt %s --verify-each -graph-optimize='compile-on-910-95=true rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s --check-prefix=AGS
// RUN: triton-opt %s --verify-each -graph-optimize='compile-on-910-95=false rule-mask=2048 allow-online-semantic-enumeration=true' -o - | FileCheck %s --check-prefix=OTHER

// This is the histogram semantic-search POC baseline. The solver is not given
// graph as a rewrite pattern. It imports the TTIR graph, enumerates well-typed
// masked tt.histogram candidates, proves equality from per-op denotations, and
// extracts the lower structural-cost representative.
//
// AGS-LABEL: tt.func @fold_zero_parking(
// AGS-NOT: arith.select
// AGS-NOT: tt.reduce
// AGS: %[[RESULT:.*]] = tt.histogram %[[SOURCE:.*]], %[[MASK:.*]] : tensor<64xi32> -> tensor<16xi32>
// AGS: tt.return %[[RESULT]] : tensor<16xi32>
// OTHER-LABEL: tt.func @fold_zero_parking(
// OTHER: tt.histogram {{.*}} : tensor<64xi32> -> tensor<16xi32>
// OTHER: tt.reduce
// OTHER: arith.subi
tt.func @fold_zero_parking(%source: tensor<64xi32>, %live: tensor<64xi1>) -> tensor<16xi32> {
  %zero_source = arith.constant dense<0> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %parked = arith.select %live, %source, %zero_source : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %live, %true : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_parking_bin = arith.cmpi eq, %bins, %zero_bins : tensor<16xi32>
  %correction = arith.select %is_parking_bin, %invalid_splat, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction : tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// Holdout shape: parking bin 7, reversed xor operands, and reversed compare
// operands. A topology matcher for the baseline normally misses at least one
// of these changes; the base semantics and commutative normalization do not.
//
// AGS-LABEL: tt.func @fold_nonzero_parking_holdout(
// AGS-NOT: arith.select
// AGS-NOT: tt.reduce
// AGS: %[[RESULT:.*]] = tt.histogram %[[SOURCE:.*]], %[[MASK:.*]] : tensor<64xi32> -> tensor<16xi32>
// AGS: tt.return %[[RESULT]] : tensor<16xi32>
tt.func @fold_nonzero_parking_holdout(%source: tensor<64xi32>, %live: tensor<64xi1>) -> tensor<16xi32> {
  %park = arith.constant dense<7> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %park_bins = arith.constant dense<7> : tensor<16xi32>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %parked = arith.select %live, %source, %park : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %true, %live : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_parking_bin = arith.cmpi eq, %park_bins, %bins : tensor<16xi32>
  %correction = arith.select %is_parking_bin, %invalid_splat, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction : tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// The correction targets bin 6 while parking uses bin 7. The denotations are
// different, so the equivalence oracle must reject the masked candidate.
//
// AGS-LABEL: tt.func @reject_wrong_correction(
// AGS: tt.histogram {{.*}} : tensor<64xi32> -> tensor<16xi32>
// AGS: tt.reduce
// AGS: arith.subi
tt.func @reject_wrong_correction(%source: tensor<64xi32>, %live: tensor<64xi1>) -> tensor<16xi32> {
  %park = arith.constant dense<7> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %wrong_bin = arith.constant dense<6> : tensor<16xi32>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %parked = arith.select %live, %source, %park : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %live, %true : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_wrong_bin = arith.cmpi eq, %bins, %wrong_bin : tensor<16xi32>
  %correction = arith.select %is_wrong_bin, %invalid_splat, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction : tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// The correction count is deliberately off by one. A topology matcher could
// still accept the surrounding shape, but semantic equality must fail.
//
// AGS-LABEL: tt.func @reject_wrong_correction_amount(
// AGS: tt.histogram {{.*}} : tensor<64xi32> -> tensor<16xi32>
// AGS: tt.reduce
// AGS: arith.addi
// AGS: arith.subi
tt.func @reject_wrong_correction_amount(%source: tensor<64xi32>, %live: tensor<64xi1>) -> tensor<16xi32> {
  %park = arith.constant dense<7> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %park_bins = arith.constant dense<7> : tensor<16xi32>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %one_bins = arith.constant dense<1> : tensor<16xi32>
  %parked = arith.select %live, %source, %park : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %live, %true : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %wrong_count = arith.addi %invalid_splat, %one_bins : tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_parking_bin = arith.cmpi eq, %bins, %park_bins : tensor<16xi32>
  %correction = arith.select %is_parking_bin, %wrong_count, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction : tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

// The raw histogram escapes the candidate region. Replacing the subtraction
// is not enough to remove that histogram, so the POC keeps this DAG unchanged
// until shared-subgraph costing and multi-result materialization are present.
//
// AGS-LABEL: tt.func @reject_shared_histogram(
// AGS: %[[RAW:.*]] = tt.histogram {{.*}} : tensor<64xi32> -> tensor<16xi32>
// AGS: %[[RESULT:.*]] = arith.subi %[[RAW]],
// AGS: tt.return %[[RESULT]], %[[RAW]]
tt.func @reject_shared_histogram(%source: tensor<64xi32>, %live: tensor<64xi1>) -> (tensor<16xi32>, tensor<16xi32>) {
  %zero_source = arith.constant dense<0> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %parked = arith.select %live, %source, %zero_source : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %live, %true : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_parking_bin = arith.cmpi eq, %bins, %zero_bins : tensor<16xi32>
  %correction = arith.select %is_parking_bin, %invalid_splat, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction : tensor<16xi32>
  tt.return %result, %histogram : tensor<16xi32>, tensor<16xi32>
}

// Overflow promises affect poison semantics. The first POC conservatively
// leaves such a root opaque instead of proving it with modular arithmetic.
//
// AGS-LABEL: tt.func @reject_overflow_promise(
// AGS: tt.histogram {{.*}} : tensor<64xi32> -> tensor<16xi32>
// AGS: arith.subi {{.*}} overflow<nuw>
tt.func @reject_overflow_promise(%source: tensor<64xi32>, %live: tensor<64xi1>) -> tensor<16xi32> {
  %zero_source = arith.constant dense<0> : tensor<64xi32>
  %true = arith.constant dense<true> : tensor<64xi1>
  %zero_bins = arith.constant dense<0> : tensor<16xi32>
  %parked = arith.select %live, %source, %zero_source : tensor<64xi1>, tensor<64xi32>
  %histogram = tt.histogram %parked : tensor<64xi32> -> tensor<16xi32>
  %invalid = arith.xori %live, %true : tensor<64xi1>
  %invalid_i32 = arith.extui %invalid : tensor<64xi1> to tensor<64xi32>
  %invalid_count = "tt.reduce"(%invalid_i32) ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<64xi32>) -> i32
  %invalid_splat = tt.splat %invalid_count : i32 -> tensor<16xi32>
  %bins = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
  %is_parking_bin = arith.cmpi eq, %bins, %zero_bins : tensor<16xi32>
  %correction = arith.select %is_parking_bin, %invalid_splat, %zero_bins : tensor<16xi1>, tensor<16xi32>
  %result = arith.subi %histogram, %correction overflow<nuw> : tensor<16xi32>
  tt.return %result : tensor<16xi32>
}

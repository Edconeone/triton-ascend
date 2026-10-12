/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/AutoGraphSolver.h"
#include "TritonToGraph/OfflineGraphEnumerator.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <limits>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static TensorDenotation makeDenotation(MLIRContext &context, int64_t extent,
                                       unsigned indexId) {
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation result;
  result.shape.push_back(extent);
  result.resultIndices.push_back(
      TensorSemanticExpr::boundIndex(&context, indexId, i32));
  result.value = TensorSemanticExpr::symbol(&context, "value", i32);
  result.defined = TensorSemanticExpr::boolean(&context, true);
  return result;
}

static bool hasDomain(ArrayRef<GraphSemanticDomain> domains,
                      GraphSemanticDomain expected) {
  return llvm::is_contained(domains, expected);
}

TEST(TensorSemanticExprTest, StructuralHashAgreesWithStructuralEquality) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr first = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {TensorSemanticExpr::symbol(&context, "x", i32),
       TensorSemanticExpr::integer(&context, 1, i32)});
  TensorSemanticExpr rebuilt = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {TensorSemanticExpr::symbol(&context, "x", i32),
       TensorSemanticExpr::integer(&context, 1, i32)});

  ASSERT_TRUE(first.structurallyEquals(rebuilt));
  EXPECT_EQ(first.getStructuralHash(), rebuilt.getStructuralHash());
  EXPECT_EQ(TensorSemanticExpr().getStructuralHash(), 0u);
}

TEST(TensorSemanticExprTest,
     AbstractSemanticSignatureSeparatesObservableDomains) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation first = makeDenotation(context, 16, 0);
  TensorDenotation rebuilt = first;
  AbstractSemanticSignature firstSignature =
      buildAbstractSemanticSignature(i32, first);
  AbstractSemanticSignature rebuiltSignature =
      buildAbstractSemanticSignature(i32, rebuilt);
  EXPECT_TRUE(firstSignature.isExact());
  EXPECT_TRUE(firstSignature.structurallyEquals(rebuiltSignature));
  EXPECT_EQ(firstSignature.getBucketHash(), rebuiltSignature.getBucketHash());

  rebuilt.defined = TensorSemanticExpr::boolean(&context, false);
  AbstractSemanticSignature invalidSignature =
      buildAbstractSemanticSignature(i32, rebuilt);
  EXPECT_FALSE(firstSignature.structurallyEquals(invalidSignature));
  EXPECT_NE(firstSignature.definednessHash, invalidSignature.definednessHash);
}

TEST(TensorSemanticExprTest, StableDenotationKeyHashesSharedDAGCompactly) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  auto buildSharedDAG = [&](StringRef symbol) {
    TensorSemanticExpr value =
        TensorSemanticExpr::symbol(&context, symbol, i32);
    for (unsigned depth = 0; depth < 24; ++depth)
      value = TensorSemanticExpr::operation(TensorSemanticKind::Add, i32,
                                            {value, value});
    return value;
  };

  TensorDenotation first;
  first.value = buildSharedDAG("x");
  first.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation rebuilt;
  rebuilt.value = buildSharedDAG("x");
  rebuilt.defined = TensorSemanticExpr::boolean(&context, true);

  std::string firstKey = getStableTensorDenotationKey(first);
  std::string rebuiltKey = getStableTensorDenotationKey(rebuilt);
  EXPECT_EQ(firstKey, rebuiltKey);
  EXPECT_TRUE(StringRef(firstKey).starts_with("sha256:"));
  EXPECT_EQ(firstKey.size(), 71u);
}

TEST(TensorSemanticNormalizerTest, ReusesAnAlreadyNormalizedExpressionDAG) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation input = makeDenotation(context, 16, 0);
  input.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i32,
      {TensorSemanticExpr::symbol(&context, "lhs", i32),
       TensorSemanticExpr::symbol(&context, "rhs", i32)});

  TensorSemanticNormalizer normalizer;
  TensorDenotation first = normalizer.normalize(input).denotation;
  ASSERT_EQ(first.value.getOperands().size(), 2u);
  const TensorSemanticExpr *firstOperands = first.value.getOperands().data();

  TensorDenotation second = normalizer.normalize(first).denotation;
  ASSERT_TRUE(second.value.structurallyEquals(first.value));
  EXPECT_EQ(second.value.getOperands().data(), firstOperands);
}

TEST(TensorSemanticNormalizerTest, CooperativelyStopsBeforeDeepNormalization) {
  MLIRContext context;
  TensorDenotation input = makeDenotation(context, /*extent=*/16,
                                          /*indexId=*/0);
  unsigned checks = 0;
  TensorSemanticNormalizer normalizer([&]() {
    ++checks;
    return true;
  });

  SemanticNormalizationResult result = normalizer.normalize(input);

  EXPECT_TRUE(normalizer.wasInterrupted());
  EXPECT_EQ(checks, 1u);
  EXPECT_TRUE(result.denotation.value.structurallyEquals(input.value));
}

TEST(TensorSemanticNormalizerTest, PushesPureBinaryFunctionsThroughSelect) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr condition =
      TensorSemanticExpr::symbol(&context, "condition", i1);
  TensorSemanticExpr x = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr y = TensorSemanticExpr::symbol(&context, "y", i32);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(&context, 0, i32);
  TensorSemanticExpr selected = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {condition, x, zero});

  TensorDenotation factored = makeDenotation(context, 16, 0);
  factored.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {selected, y});
  TensorDenotation distributed = factored;
  distributed.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32,
      {condition,
       TensorSemanticExpr::operation(TensorSemanticKind::Multiply, i32,
                                     {x, y}),
       zero});

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_EQ(oracle.compare(factored, distributed).status,
            SemanticProofStatus::Proved);
}

TEST(TensorSemanticNormalizerTest,
     CombinesBinarySelectOperandsWithTheSameCondition) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr condition =
      TensorSemanticExpr::symbol(&context, "condition", i1);
  TensorSemanticExpr x = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr minusOne =
      TensorSemanticExpr::integer(&context, -1, i32);
  TensorSemanticExpr positive = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {condition, x, one});
  TensorSemanticExpr signedValue = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {condition, x, minusOne});

  TensorDenotation lhs = makeDenotation(context, 16, 0);
  lhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {positive, positive});
  TensorDenotation rhs = lhs;
  rhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {signedValue, signedValue});

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, RecordsEveryComparedDenotationDomain) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  TensorDenotation lhs = makeDenotation(context, 16, 0);
  TensorDenotation rhs = makeDenotation(context, 16, 0);
  std::optional<SemanticEquivalenceProof> proof = oracle.prove(lhs, rhs);

  ASSERT_TRUE(proof);
  EXPECT_FALSE(proof->getStableHash().empty());
  SemanticProofResult replay = oracle.replay(*proof);
  EXPECT_EQ(replay.status, SemanticProofStatus::Proved);

  std::string serialized = proof->toJSON();
  MLIRContext independentContext;
  std::string decodeError;
  std::optional<SemanticEquivalenceProof> decoded =
      SemanticEquivalenceProof::fromJSON(serialized, independentContext,
                                         decodeError);
  ASSERT_TRUE(decoded) << decodeError;
  TensorSemanticNormalizer independentNormalizer;
  SemanticEquivalenceOracle independentOracle(independentNormalizer);
  SemanticProofResult independentReplay = independentOracle.replay(*decoded);
  EXPECT_EQ(independentReplay.status, SemanticProofStatus::Proved);
  EXPECT_EQ(decoded->getStableHash(), proof->getStableHash());
  EXPECT_TRUE(hasDomain(proof->verifiedDomains, GraphSemanticDomain::Shape));
  EXPECT_TRUE(hasDomain(proof->verifiedDomains, GraphSemanticDomain::Index));
  EXPECT_TRUE(hasDomain(proof->verifiedDomains, GraphSemanticDomain::Layout));
  EXPECT_TRUE(hasDomain(proof->verifiedDomains, GraphSemanticDomain::Value));
  EXPECT_TRUE(
      hasDomain(proof->verifiedDomains, GraphSemanticDomain::Definedness));
}

TEST(SemanticEquivalenceOracleTest,
     BuildsReplayableReflexiveCertificateFromOneDenotation) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  TensorDenotation denotation = makeDenotation(context, 16, 0);
  Type i32 = IntegerType::get(&context, 32);
  denotation.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {denotation.value, TensorSemanticExpr::integer(&context, 0, i32)});

  SemanticProofResult result = oracle.compare(denotation, denotation);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved);
  ASSERT_TRUE(result.proof);
  EXPECT_EQ(result.reason, "reflexive normalized tensor denotation");
  EXPECT_FALSE(result.proof->steps.empty());
  EXPECT_EQ(oracle.replay(*result.proof).status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, RejectsDifferentShapes) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  SemanticProofResult result =
      oracle.compare(makeDenotation(context, 16, 0),
                     makeDenotation(context, 32, 0));
  EXPECT_EQ(result.status, SemanticProofStatus::Refuted);
  EXPECT_EQ(result.reason, "shape/rank mismatch");
}

TEST(SemanticEquivalenceOracleTest, RejectsDifferentIndexBinders) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  SemanticProofResult result =
      oracle.compare(makeDenotation(context, 16, 0),
                     makeDenotation(context, 16, 1));
  EXPECT_EQ(result.status, SemanticProofStatus::Refuted);
  EXPECT_EQ(result.reason, "result index domain mismatch");
}

TEST(SemanticEquivalenceOracleTest, UnequalNormalFormsRemainUnknown) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  TensorDenotation lhs = makeDenotation(context, 16, 0);
  TensorDenotation rhs = makeDenotation(context, 16, 0);
  rhs.value = TensorSemanticExpr::integer(
      &context, 7, IntegerType::get(&context, 32));
  SemanticProofResult result = oracle.compare(lhs, rhs);
  EXPECT_EQ(result.status, SemanticProofStatus::Unknown);
  EXPECT_EQ(result.reason, "normal forms differ; counterexample required");
}

TEST(SemanticEquivalenceOracleTest, ProvesBooleanCompareTruthTables) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  TensorSemanticExpr x = TensorSemanticExpr::symbol(&context, "x", i1);
  TensorSemanticExpr zero = TensorSemanticExpr::boolean(&context, false);
  TensorSemanticExpr one = TensorSemanticExpr::boolean(&context, true);

  TensorDenotation direct;
  direct.value = x;
  direct.defined = one;
  TensorDenotation equalTrue = direct;
  equalTrue.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {one, x});
  TensorDenotation lessThanZero = direct;
  lessThanZero.value = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {x, zero});
  TensorDenotation subtractFalse = direct;
  subtractFalse.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i1, {x, zero});

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_EQ(oracle.compare(direct, equalTrue).status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(direct, lessThanZero).status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(direct, subtractFalse).status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     ProvesBranchLocalFalseConditionSubstitution) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i1);
  TensorSemanticExpr zero = TensorSemanticExpr::boolean(&context, false);

  TensorDenotation lhs;
  lhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1, {mask, value, mask});
  lhs.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation rhs = lhs;
  rhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1, {mask, value, zero});

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     PreservesLazySelectWhenAnArmIsUnobserved) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr one = TensorSemanticExpr::boolean(&context, true);
  TensorSemanticExpr unobserved = TensorSemanticExpr::operation(
      TensorSemanticKind::Unobserved, i1, {});

  TensorDenotation input;
  input.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1, {mask, one, unobserved});
  input.defined = mask;

  TensorSemanticNormalizer normalizer;
  TensorDenotation normalized = normalizer.normalize(input).denotation;
  EXPECT_EQ(normalized.value.getKind(), TensorSemanticKind::Select);
  ASSERT_EQ(normalized.value.getOperands().size(), 3u);
  EXPECT_EQ(normalized.value.getOperands()[2].getKind(),
            TensorSemanticKind::Unobserved);
}

TEST(SemanticEquivalenceOracleTest,
     IgnoresValuesWhenDefinednessIsStaticallyFalse) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation lhs;
  lhs.value = TensorSemanticExpr::symbol(&context, "lhs", i32);
  lhs.defined = TensorSemanticExpr::boolean(&context, false);
  TensorDenotation rhs = lhs;
  rhs.value = TensorSemanticExpr::symbol(&context, "rhs", i32);

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, ProvesSubtractZeroIdentity) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);

  TensorDenotation lhs = makeDenotation(context, 16, 0);
  TensorDenotation rhs = lhs;
  rhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, lhs.value.getType(),
      {lhs.value, TensorSemanticExpr::integer(
                      &context, 0, IntegerType::get(&context, 32))});

  SemanticProofResult result = oracle.compare(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved);
  ASSERT_TRUE(result.proof);
  EXPECT_FALSE(result.proof->getStableHash().empty());
}

TEST(SemanticEquivalenceOracleTest,
     ProvesArithmeticAndLogicalShiftEqualForNonNegativeConstant) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr amount =
      TensorSemanticExpr::symbol(&context, "amount", i32);

  TensorDenotation arithmetic;
  arithmetic.value = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i32, {one, amount});
  arithmetic.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation logical = arithmetic;
  logical.value = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i32, {one, amount});

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(arithmetic, logical);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(SemanticEquivalenceOracleTest, ProvesI1AdditionIsXor) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);

  TensorDenotation zero;
  zero.shape = {16};
  zero.resultIndices.push_back(
      TensorSemanticExpr::boundIndex(&context, 0, IntegerType::get(&context, 32)));
  zero.value = TensorSemanticExpr::integer(&context, 0, i1);
  zero.defined = TensorSemanticExpr::boolean(&context, true);

  TensorDenotation doubled = zero;
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i1);
  doubled.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i1, {value, value});

  EXPECT_EQ(oracle.compare(zero, doubled).status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, ProvesSingletonFiniteBinderIdentities) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);

  TensorSemanticExpr binder = TensorSemanticExpr::boundIndex(&context, 7, i32);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(&context, 0, i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr body = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32, {value, binder});

  auto makeScalar = [&](const TensorSemanticExpr &expression) {
    TensorDenotation result;
    result.value = expression;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  for (TensorSemanticKind kind :
       {TensorSemanticKind::Sum, TensorSemanticKind::Product}) {
    TensorSemanticExpr fold = TensorSemanticExpr::operation(
        kind, i32, {binder, zero, one, body});
    EXPECT_EQ(oracle.compare(makeScalar(value), makeScalar(fold)).status,
              SemanticProofStatus::Proved);
  }

  TensorSemanticExpr predicate = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {binder, zero});
  TensorSemanticExpr forAll = TensorSemanticExpr::operation(
      TensorSemanticKind::ForAll, i1, {binder, zero, one, predicate});
  EXPECT_EQ(oracle.compare(
                          makeScalar(TensorSemanticExpr::boolean(&context, true)),
                          makeScalar(forAll))
                .status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, ProvesProductIdentityForConstantOne) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);

  TensorSemanticExpr binder = TensorSemanticExpr::boundIndex(&context, 8, i32);
  TensorSemanticExpr lower = TensorSemanticExpr::integer(&context, 0, i32);
  TensorSemanticExpr dynamicUpper =
      TensorSemanticExpr::symbol(&context, "extent-x", i32);
  TensorSemanticExpr product = TensorSemanticExpr::operation(
      TensorSemanticKind::Product, i1,
      {binder, lower, dynamicUpper,
       TensorSemanticExpr::boolean(&context, true)});

  TensorDenotation lhs;
  lhs.value = TensorSemanticExpr::boolean(&context, true);
  lhs.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation rhs = lhs;
  rhs.value = product;
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, FoldsSmallConstantFiniteBinders) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr binder = TensorSemanticExpr::boundIndex(&context, 9, i32);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(&context, 0, i32);
  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };
  auto fold = [&](TensorSemanticKind kind, Type type, int64_t upper,
                  int64_t body) {
    return TensorSemanticExpr::operation(
        kind, type,
        {binder, zero, TensorSemanticExpr::integer(&context, upper, i32),
         TensorSemanticExpr::integer(&context, body, type)});
  };

  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(fold(TensorSemanticKind::Sum, i1, 16, 1)))
                .status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::integer(&context, 6, i32)),
                          scalar(fold(TensorSemanticKind::Sum, i32, 2, 3)))
                .status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::integer(&context, 8, i32)),
                          scalar(fold(TensorSemanticKind::Product, i32, 3, 2)))
                .status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, ProvesSignedLessThanIsIrreflexive) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorDenotation lhs;
  lhs.value = TensorSemanticExpr::boolean(&context, false);
  lhs.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation rhs = lhs;
  rhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, IntegerType::get(&context, 1),
      {value, value});
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, FoldsSignedAndUnsignedMultiplyHigh) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i64 = IntegerType::get(&context, 64);
  TensorSemanticExpr minusOne =
      TensorSemanticExpr::integer(&context, -1, i64);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i64);
  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  TensorSemanticExpr unsignedHigh = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedMultiplyHigh, i64, {minusOne, two});
  EXPECT_EQ(oracle.compare(
                          scalar(unsignedHigh),
                          scalar(TensorSemanticExpr::integer(&context, 1, i64)))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr signedHigh = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedMultiplyHigh, i64, {minusOne, two});
  EXPECT_EQ(
      oracle.compare(
                scalar(signedHigh),
                scalar(TensorSemanticExpr::integer(&context, -1, i64)))
          .status,
      SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     CanonicalizesFoundationalDepthOneBitVectorLaws) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = builder.getI1Type();
  Type i8 = builder.getI8Type();
  Type i16 = builder.getI16Type();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto scalar = [&](const TensorSemanticExpr &value,
                    TensorSemanticExpr defined = {}) {
    TensorDenotation result;
    result.value = value;
    result.defined = defined ? defined
                             : TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  TensorSemanticExpr wide =
      TensorSemanticExpr::symbol(&context, "wide", i16);
  TensorSemanticExpr four = TensorSemanticExpr::integer(&context, 4, i16);
  TensorSemanticExpr mulHighLhs = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedMultiplyHigh, i16, {wide, four});
  TensorSemanticExpr mulHighRhs = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedMultiplyHigh, i16, {four, wide});
  EXPECT_EQ(oracle.compare(scalar(mulHighLhs), scalar(mulHighRhs)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr bit = TensorSemanticExpr::symbol(&context, "bit", i1);
  TensorSemanticExpr bitProduct = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i1, {bit, bit});
  EXPECT_EQ(oracle.compare(scalar(bit), scalar(bitProduct)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr source =
      TensorSemanticExpr::symbol(&context, "source", i32);
  TensorSemanticExpr even = TensorSemanticExpr::integer(&context, 2, i32);
  TensorSemanticExpr parityProduct = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i1, {source, even});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(parityProduct))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr sourceLowBit = TensorSemanticExpr::operation(
      TensorSemanticKind::Truncate, i1, {source});
  for (TensorSemanticKind kind :
       {TensorSemanticKind::Add, TensorSemanticKind::Subtract}) {
    TensorSemanticExpr parityArithmetic =
        TensorSemanticExpr::operation(kind, i1, {source, even});
    EXPECT_EQ(oracle.compare(scalar(sourceLowBit),
                             scalar(parityArithmetic))
                  .status,
              SemanticProofStatus::Proved);
  }
  TensorSemanticExpr parityAnd = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseAnd, i1, {source, even});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(parityAnd))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr zeroShift =
      TensorSemanticExpr::boolean(&context, false);
  TensorSemanticExpr shifted = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i1, {bit, zeroShift});
  DictionaryAttr shiftGuardPayload = builder.getDictionaryAttr(
      {builder.getNamedAttr("semantic_role",
                            builder.getStringAttr("shift_amount_valid"))});
  TensorSemanticExpr shiftValid = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {zeroShift}, shiftGuardPayload);
  EXPECT_EQ(oracle.compare(scalar(bit), scalar(shifted, shiftValid)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr wideZero = TensorSemanticExpr::integer(&context, 0, i32);
  TensorSemanticExpr narrowedShift = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i1, {source, wideZero});
  EXPECT_EQ(oracle.compare(scalar(sourceLowBit), scalar(narrowedShift)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr leftShiftZero = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftLeft, i32, {source, wideZero});
  EXPECT_EQ(oracle.compare(scalar(source), scalar(leftShiftZero)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr allTrue = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1,
      {TensorSemanticExpr::boolean(&context, true),
       TensorSemanticExpr::boolean(&context, true),
       TensorSemanticExpr::boolean(&context, true),
       TensorSemanticExpr::boolean(&context, true)});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, true)),
                          scalar(allTrue))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr notBit = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1, {bit});
  TensorSemanticExpr contradiction = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1, {bit, notBit});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(contradiction))
                .status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr one32 = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two32 = TensorSemanticExpr::integer(&context, 2, i32);
  TensorSemanticExpr equalOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {source, one32});
  TensorSemanticExpr equalTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {source, two32});
  TensorSemanticExpr incompatibleEqualities = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1, {equalOne, equalTwo});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(incompatibleEqualities))
                .status,
            SemanticProofStatus::Proved);

  Type pointer = triton::PointerType::get(i1, /*addressSpace=*/1);
  TensorSemanticExpr base =
      TensorSemanticExpr::symbol(&context, "pointer", pointer);
  TensorSemanticExpr offset8 = TensorSemanticExpr::integer(&context, 2, i8);
  TensorSemanticExpr offset16 = TensorSemanticExpr::integer(&context, 2, i16);
  TensorSemanticExpr pointer8 = TensorSemanticExpr::operation(
      TensorSemanticKind::PointerAdd, pointer, {base, offset8});
  TensorSemanticExpr pointer16 = TensorSemanticExpr::operation(
      TensorSemanticKind::PointerAdd, pointer, {base, offset16});
  EXPECT_EQ(oracle.compare(scalar(pointer8), scalar(pointer16)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr pointerZero = TensorSemanticExpr::operation(
      TensorSemanticKind::PointerAdd, pointer,
      {base, TensorSemanticExpr::integer(&context, 0, i8)});
  EXPECT_EQ(oracle.compare(scalar(base), scalar(pointerZero)).status,
            SemanticProofStatus::Proved);

  Type index = builder.getIndexType();
  TensorSemanticExpr zero8 = TensorSemanticExpr::integer(&context, 0, i8);
  TensorSemanticExpr indexZero =
      TensorSemanticExpr::integer(&context, 0, index);
  TensorSemanticExpr one8 = TensorSemanticExpr::integer(&context, 1, i8);
  TensorSemanticExpr indexOne =
      TensorSemanticExpr::integer(&context, 1, index);
  for (TensorSemanticKind kind :
       {TensorSemanticKind::ZeroExtend, TensorSemanticKind::SignExtend}) {
    for (auto [sourceValue, expected] :
         {std::pair{zero8, indexZero}, std::pair{one8, indexOne}}) {
      TensorSemanticExpr extended =
          TensorSemanticExpr::operation(kind, index, {sourceValue});
      EXPECT_EQ(oracle.compare(scalar(expected), scalar(extended)).status,
                SemanticProofStatus::Proved);
    }
  }

  TensorSemanticExpr one16 = TensorSemanticExpr::integer(&context, 1, i16);
  TensorSemanticExpr zero16 = TensorSemanticExpr::integer(&context, 0, i16);
  TensorSemanticExpr unsignedHighOne = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedMultiplyHigh, i16, {wide, one16});
  EXPECT_EQ(oracle.compare(scalar(zero16), scalar(unsignedHighOne)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr sum = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i16, {wide, one16});
  TensorSemanticExpr carryFromWide = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {sum, wide});
  TensorSemanticExpr carryFromOne = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {sum, one16});
  EXPECT_EQ(oracle.compare(scalar(carryFromWide), scalar(carryFromOne)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr bitDivideSelf = TensorSemanticExpr::operation(
      TensorSemanticKind::IntegerDivide, i1, {bit, bit});
  TensorSemanticExpr bitDivideOne = TensorSemanticExpr::operation(
      TensorSemanticKind::IntegerDivide, i1,
      {TensorSemanticExpr::boolean(&context, true), bit});
  EXPECT_EQ(oracle.compare(scalar(bitDivideSelf, bit),
                           scalar(bitDivideOne, bit))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr nonZeroSource = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1,
      {TensorSemanticExpr::operation(
          TensorSemanticKind::Equal, i1,
          {source, TensorSemanticExpr::integer(&context, 0, i32)})});
  TensorSemanticExpr remainderSelf = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedRemainder, i32, {source, source});
  TensorSemanticExpr remainderZero = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedRemainder, i32,
      {TensorSemanticExpr::integer(&context, 0, i32), source});
  EXPECT_EQ(oracle.compare(scalar(remainderSelf, nonZeroSource),
                           scalar(remainderZero, nonZeroSource))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr two16 = TensorSemanticExpr::integer(&context, 2, i16);
  TensorSemanticExpr four16 = TensorSemanticExpr::integer(&context, 4, i16);
  TensorSemanticExpr shiftOne = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftLeft, i16, {wide, one16});
  TensorSemanticExpr doubled = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i16, {wide, wide});
  TensorSemanticExpr multipliedTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i16, {wide, two16});
  EXPECT_EQ(oracle.compare(scalar(shiftOne), scalar(doubled)).status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(scalar(shiftOne), scalar(multipliedTwo)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr shiftTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftLeft, i16, {wide, two16});
  TensorSemanticExpr multipliedFour = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i16, {wide, four16});
  EXPECT_EQ(oracle.compare(scalar(shiftTwo), scalar(multipliedFour)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr remainderOne = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedRemainder, i16, {wide, one16});
  EXPECT_EQ(oracle.compare(scalar(zero16), scalar(remainderOne)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr nonZeroWide = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1,
      {TensorSemanticExpr::operation(TensorSemanticKind::Equal, i1,
                                     {wide, zero16})});
  TensorSemanticExpr zeroDivideWide = TensorSemanticExpr::operation(
      TensorSemanticKind::IntegerDivide, i16, {zero16, wide});
  EXPECT_EQ(oracle.compare(scalar(zero16, nonZeroWide),
                           scalar(zeroDivideWide, nonZeroWide))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr allOnes16 =
      TensorSemanticExpr::integer(&context, -1, i16);
  TensorSemanticExpr orAllOnes = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseOr, i16, {wide, allOnes16});
  EXPECT_EQ(oracle.compare(scalar(allOnes16), scalar(orAllOnes)).status,
            SemanticProofStatus::Proved);

  for (TensorSemanticKind kind :
       {TensorSemanticKind::SignedMultiplyHigh,
        TensorSemanticKind::UnsignedMultiplyHigh}) {
    TensorSemanticExpr high =
        TensorSemanticExpr::operation(kind, i1, {bit, bit});
    EXPECT_EQ(oracle.compare(
                            scalar(TensorSemanticExpr::boolean(&context,
                                                               false)),
                            scalar(high))
                  .status,
              SemanticProofStatus::Proved);
  }
  TensorSemanticExpr signedHighOne = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedMultiplyHigh, i16, {wide, one16});
  TensorSemanticExpr signedHighTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedMultiplyHigh, i16, {wide, two16});
  EXPECT_EQ(oracle.compare(scalar(signedHighOne), scalar(signedHighTwo)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr signBit = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {wide, zero16});
  TensorSemanticExpr unsignedHighTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedMultiplyHigh, i16, {wide, two16});
  TensorSemanticExpr extendedSignBit = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, i16, {signBit});
  EXPECT_EQ(
      oracle.compare(scalar(unsignedHighTwo), scalar(extendedSignBit)).status,
      SemanticProofStatus::Proved);

  TensorSemanticExpr symbolicAmount =
      TensorSemanticExpr::symbol(&context, "amount", i1);
  TensorSemanticExpr symbolicShiftValid = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {symbolicAmount},
      shiftGuardPayload);
  TensorSemanticExpr symbolicAmountIsZero = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1, {symbolicAmount});
  EXPECT_EQ(oracle.compare(scalar(symbolicShiftValid),
                           scalar(symbolicAmountIsZero))
                .status,
            SemanticProofStatus::Proved);
  for (TensorSemanticKind kind :
       {TensorSemanticKind::ShiftLeft,
        TensorSemanticKind::ShiftRightLogical,
        TensorSemanticKind::ShiftRightArithmetic}) {
    TensorSemanticExpr symbolicShift = TensorSemanticExpr::operation(
        kind, i1, {bit, symbolicAmount});
    EXPECT_EQ(oracle.compare(scalar(bit, symbolicShiftValid),
                             scalar(symbolicShift, symbolicShiftValid))
                  .status,
              SemanticProofStatus::Proved);
  }
  TensorSemanticExpr arithmeticSelf = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i16, {wide, wide});
  TensorSemanticExpr logicalSelf = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i16, {wide, wide});
  DictionaryAttr wideShiftGuardPayload = builder.getDictionaryAttr(
      {builder.getNamedAttr("semantic_role",
                            builder.getStringAttr("shift_amount_valid"))});
  TensorSemanticExpr wideShiftValid = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {wide}, wideShiftGuardPayload);
  EXPECT_EQ(oracle.compare(scalar(arithmeticSelf, wideShiftValid),
                           scalar(logicalSelf, wideShiftValid))
                .status,
            SemanticProofStatus::Proved);
  for (TensorSemanticExpr selfShift : {arithmeticSelf, logicalSelf})
    EXPECT_EQ(oracle.compare(scalar(zero16, wideShiftValid),
                             scalar(selfShift, wideShiftValid))
                  .status,
              SemanticProofStatus::Proved);
  TensorSemanticExpr zeroLeft = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftLeft, i16, {zero16, wide});
  TensorSemanticExpr zeroRight = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i16, {zero16, wide});
  EXPECT_EQ(oracle.compare(scalar(zeroLeft, wideShiftValid),
                           scalar(zeroRight, wideShiftValid))
                .status,
            SemanticProofStatus::Proved);
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false),
                                 symbolicShiftValid),
                          scalar(symbolicAmount, symbolicShiftValid))
                .status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr numberedAmount = TensorSemanticExpr::operation(
      TensorSemanticKind::Symbol, i1, {}, builder.getI64IntegerAttr(0));
  TensorSemanticExpr numberedShiftValid = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {numberedAmount},
      shiftGuardPayload);
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false),
                                 numberedShiftValid),
                          scalar(numberedAmount, numberedShiftValid))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr negatedProduct = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i16, {wide, allOnes16});
  TensorSemanticExpr subtractedFromZero = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i16, {zero16, wide});
  EXPECT_EQ(oracle.compare(scalar(negatedProduct),
                           scalar(subtractedFromZero))
                .status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr divideByFour = TensorSemanticExpr::operation(
      TensorSemanticKind::IntegerDivide, i16, {wide, four16});
  TensorSemanticExpr logicalShiftTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i16, {wide, two16});
  EXPECT_EQ(oracle.compare(scalar(divideByFour), scalar(logicalShiftTwo)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr unsignedBelowZero = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {wide, zero16});
  EXPECT_EQ(oracle.compare(
                          scalar(TensorSemanticExpr::boolean(&context, false)),
                          scalar(unsignedBelowZero))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr xorPredicates = TensorSemanticExpr::operation(
      TensorSemanticKind::Xor, i1, {bit, symbolicAmount});
  TensorSemanticExpr equalPredicates = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1, {xorPredicates});
  TensorSemanticExpr guardedEquality = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1, {bit, equalPredicates});
  TensorSemanticExpr bothPredicates = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1, {bit, symbolicAmount});
  EXPECT_EQ(oracle.compare(scalar(guardedEquality), scalar(bothPredicates))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr wide64 =
      TensorSemanticExpr::symbol(&context, "wide64", i64);
  TensorSemanticExpr signedToIndex = TensorSemanticExpr::operation(
      TensorSemanticKind::SignExtend, index, {wide64});
  TensorSemanticExpr unsignedToIndex = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, index, {wide64});
  EXPECT_EQ(oracle.compare(scalar(signedToIndex), scalar(unsignedToIndex)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr four64 = TensorSemanticExpr::integer(&context, 4, i64);
  TensorSemanticExpr fourIndex = TensorSemanticExpr::integer(&context, 4, index);
  for (TensorSemanticKind kind :
       {TensorSemanticKind::ZeroExtend, TensorSemanticKind::SignExtend}) {
    TensorSemanticExpr extended =
        TensorSemanticExpr::operation(kind, index, {four64});
    EXPECT_EQ(oracle.compare(scalar(fourIndex), scalar(extended)).status,
              SemanticProofStatus::Proved);
  }

  TensorSemanticExpr increment = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i16, {wide, one16});
  TensorSemanticExpr incrementCarry = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {increment, wide});
  TensorSemanticExpr isAllOnes = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {wide, allOnes16});
  EXPECT_EQ(oracle.compare(scalar(incrementCarry), scalar(isAllOnes)).status,
            SemanticProofStatus::Proved);
  TensorSemanticExpr doubleWide = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i16, {wide, wide});
  TensorSemanticExpr doubleCarry = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {doubleWide, wide});
  TensorSemanticExpr isNegative = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {wide, zero16});
  EXPECT_EQ(oracle.compare(scalar(doubleCarry), scalar(isNegative)).status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest, CanonicalizesGenericBitVectorIdentities) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i8 = IntegerType::get(&context, 8);
  TensorSemanticExpr x = TensorSemanticExpr::symbol(&context, "x", i8);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i8);
  TensorSemanticExpr allOnes = TensorSemanticExpr::integer(&context, -1, i8);
  TensorSemanticExpr predicate =
      TensorSemanticExpr::symbol(&context, "predicate", i1);
  TensorSemanticExpr otherPredicate =
      TensorSemanticExpr::symbol(&context, "other-predicate", i1);
  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };
  TensorSemanticExpr signedLessThan = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {predicate, otherPredicate});
  TensorSemanticExpr notOther = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1, {otherPredicate});
  TensorSemanticExpr booleanForm = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1, {predicate, notOther});
  EXPECT_EQ(oracle.compare(scalar(signedLessThan), scalar(booleanForm)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr addMinusOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i8, {allOnes, x});
  TensorSemanticExpr subtractOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i8, {x, one});
  EXPECT_EQ(oracle.compare(scalar(addMinusOne), scalar(subtractOne)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr subtractFromAllOnes = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i8, {allOnes, x});
  TensorSemanticExpr xorAllOnes = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseXor, i8, {allOnes, x});
  EXPECT_EQ(
      oracle.compare(scalar(subtractFromAllOnes), scalar(xorAllOnes)).status,
      SemanticProofStatus::Proved);

  TensorSemanticExpr bit = TensorSemanticExpr::symbol(&context, "bit", i1);
  TensorSemanticExpr signExtended = TensorSemanticExpr::operation(
      TensorSemanticKind::SignExtend, i8, {bit});
  TensorSemanticExpr selected = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i8,
      {bit, allOnes, TensorSemanticExpr::integer(&context, 0, i8)});
  EXPECT_EQ(oracle.compare(scalar(signExtended), scalar(selected)).status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     CanonicalizesSelectAndTwosComplementFrontierLaws) {
  MLIRContext context;
  Builder builder(&context);
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  auto scalar = [&](const TensorSemanticExpr &value,
                    TensorSemanticExpr defined = {}) {
    TensorDenotation result;
    result.value = value;
    result.defined = defined ? defined
                             : TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  TensorSemanticExpr x32 =
      TensorSemanticExpr::symbol(&context, "x32", i32);
  TensorSemanticExpr allOnes32 =
      TensorSemanticExpr::integer(&context, -1, i32);
  TensorSemanticExpr complement32 = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseXor, i32, {x32, allOnes32});
  TensorSemanticExpr one32 = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two32 = TensorSemanticExpr::integer(&context, 2, i32);
  TensorSemanticExpr subtractComplement = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, i32, {one32, complement32});
  TensorSemanticExpr addTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32, {x32, two32});
  EXPECT_EQ(oracle.compare(scalar(subtractComplement), scalar(addTwo)).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr x64 =
      TensorSemanticExpr::symbol(&context, "x64", i64);
  TensorSemanticExpr allOnes64 =
      TensorSemanticExpr::integer(&context, -1, i64);
  TensorSemanticExpr one64 = TensorSemanticExpr::integer(&context, 1, i64);
  TensorSemanticExpr complement64 = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseXor, i64, {x64, allOnes64});
  TensorSemanticExpr negativeOdd = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseOr, i64, {complement64, one64});
  TensorSemanticExpr positiveOdd = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseOr, i64, {x64, one64});
  TensorSemanticExpr negativeSquare = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i64, {negativeOdd, negativeOdd});
  TensorSemanticExpr positiveSquare = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i64, {positiveOdd, positiveOdd});
  EXPECT_EQ(oracle.compare(scalar(negativeSquare), scalar(positiveSquare))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr outer =
      TensorSemanticExpr::symbol(&context, "outer", i1);
  TensorSemanticExpr inner =
      TensorSemanticExpr::symbol(&context, "inner", i1);
  TensorSemanticExpr nested = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {inner, x32, one32});
  TensorSemanticExpr guarded = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32,
      {outer, nested, TensorSemanticExpr::integer(&context, 0, i32)});
  TensorSemanticExpr guardedSquare = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {guarded, guarded});
  TensorSemanticExpr guardedTimesNested = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {guarded, nested});
  EXPECT_EQ(oracle.compare(scalar(guardedSquare),
                           scalar(guardedTimesNested))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr amountSource =
      TensorSemanticExpr::symbol(&context, "amount-source", i64);
  TensorSemanticExpr largePower =
      TensorSemanticExpr::integer(&context, 2147483648LL, i64);
  TensorSemanticExpr amount = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i64, {amountSource, largePower});
  DictionaryAttr shiftGuardPayload = builder.getDictionaryAttr(
      {builder.getNamedAttr("semantic_role",
                            builder.getStringAttr("shift_amount_valid"))});
  TensorSemanticExpr shiftValid = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {amount}, shiftGuardPayload);
  TensorSemanticExpr shifted = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i64, {x64, amount});
  EXPECT_EQ(oracle.compare(scalar(x64, shiftValid),
                           scalar(shifted, shiftValid))
                .status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     ProvesPositiveUpperBoundDefinednessInsideGuardedObservation) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr lane =
      TensorSemanticExpr::boundIndex(&context, 0, i32);
  TensorSemanticExpr bound =
      TensorSemanticExpr::symbol(&context, "bound", i32);
  TensorSemanticExpr mask = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {lane, bound});
  TensorSemanticExpr nonZero = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1,
      {TensorSemanticExpr::operation(
          TensorSemanticKind::Equal, i1,
          {TensorSemanticExpr::integer(&context, 0, i32), bound})});
  TensorSemanticExpr accessible =
      TensorSemanticExpr::symbol(&context, "accessible", i1);
  TensorSemanticExpr guardedWithDefinedness = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1,
      {mask,
       TensorSemanticExpr::operation(TensorSemanticKind::And, i1,
                                     {nonZero, accessible}),
       TensorSemanticExpr::boolean(&context, true)});
  TensorSemanticExpr guardedWithoutDefinedness =
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i1,
          {mask, accessible, TensorSemanticExpr::boolean(&context, true)});

  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  SemanticProofResult result =
      oracle.compare(scalar(guardedWithDefinedness),
                     scalar(guardedWithoutDefinedness));
  EXPECT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
}

TEST(SemanticEquivalenceOracleTest,
     ProvesNonNegativeIndexCannotTriggerSignedRemainderOverflow) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr lane =
      TensorSemanticExpr::boundIndex(&context, 0, i32);
  TensorSemanticExpr bound =
      TensorSemanticExpr::symbol(&context, "bound", i32);
  TensorSemanticExpr mask = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {lane, bound});
  TensorSemanticExpr laneIsSignedMin = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1,
      {lane, TensorSemanticExpr::integer(
                 &context, std::numeric_limits<int32_t>::min(), i32)});
  TensorSemanticExpr boundIsMinusOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1,
      {bound, TensorSemanticExpr::integer(&context, -1, i32)});
  TensorSemanticExpr noSignedOverflow = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, i1,
      {TensorSemanticExpr::operation(TensorSemanticKind::And, i1,
                                     {laneIsSignedMin, boundIsMinusOne})});
  TensorSemanticExpr accessible =
      TensorSemanticExpr::symbol(&context, "accessible", i1);
  TensorSemanticExpr guardedWithOverflowCheck = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1,
      {mask,
       TensorSemanticExpr::operation(TensorSemanticKind::And, i1,
                                     {noSignedOverflow, accessible}),
       TensorSemanticExpr::boolean(&context, true)});
  TensorSemanticExpr guardedWithoutOverflowCheck =
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i1,
          {mask, accessible, TensorSemanticExpr::boolean(&context, true)});

  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  SemanticProofResult result =
      oracle.compare(scalar(guardedWithOverflowCheck),
                     scalar(guardedWithoutOverflowCheck));
  EXPECT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
}

TEST(SemanticEquivalenceOracleTest,
     DoesNotAssumeFixedWidthAdditionPreservesNonNegativity) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr lane =
      TensorSemanticExpr::boundIndex(&context, 0, i32);
  TensorSemanticExpr wrapped = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {lane, TensorSemanticExpr::integer(&context, 1, i32)});
  TensorSemanticExpr equalsSignedMin = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1,
      {wrapped, TensorSemanticExpr::integer(
                    &context, std::numeric_limits<int32_t>::min(), i32)});
  TensorSemanticExpr condition =
      TensorSemanticExpr::symbol(&context, "condition", i1);
  TensorSemanticExpr guardedNoWrap = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1,
      {condition,
       TensorSemanticExpr::operation(TensorSemanticKind::Not, i1,
                                     {equalsSignedMin}),
       TensorSemanticExpr::boolean(&context, true)});

  auto scalar = [&](const TensorSemanticExpr &value) {
    TensorDenotation result;
    result.value = value;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  SemanticProofResult result =
      oracle.compare(scalar(guardedNoWrap),
                     scalar(TensorSemanticExpr::boolean(&context, true)));
  EXPECT_NE(result.status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     ProvesFiniteBufferReadAfterCompleteFunctionalWrite) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  Type i64 = IntegerType::get(&context, 64);
  TensorSemanticExpr index = TensorSemanticExpr::boundIndex(&context, 0, i32);
  Attribute sourceId = IntegerAttr::get(i64, 0);
  Attribute targetId = IntegerAttr::get(i64, 1);
  auto element = [&](Attribute id, TensorSemanticExpr coordinate) {
    return TensorSemanticExpr::operation(TensorSemanticKind::Element, i32,
                                         {coordinate}, id);
  };

  TensorSemanticExpr initial =
      TensorSemanticExpr::symbol(&context, "buffer", i1);
  TensorSemanticExpr state = initial;
  for (int64_t lane = 0; lane < 4; ++lane) {
    TensorSemanticExpr coordinate =
        TensorSemanticExpr::integer(&context, lane, i32);
    state = TensorSemanticExpr::operation(
        TensorSemanticKind::BufferWrite, i1,
        {state, element(targetId, coordinate),
         element(sourceId, coordinate)});
  }

  TensorDenotation copied;
  copied.shape = {4};
  copied.resultIndices = {index};
  copied.value = TensorSemanticExpr::operation(
      TensorSemanticKind::BufferRead, i32,
      {state, element(targetId, index)});
  copied.defined = TensorSemanticExpr::boolean(&context, true);
  copied.buffer = state;
  copied.bufferDefined = TensorSemanticExpr::boolean(&context, true);

  TensorDenotation source = copied;
  source.value = element(sourceId, index);
  source.buffer = initial;

  SemanticProofResult result = oracle.compare(copied, source);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  ASSERT_TRUE(result.proof);
  EXPECT_TRUE(llvm::any_of(result.proof->steps, [](const auto &step) {
    return step.reason == "finite-tensor-extensionality";
  }));
  // A value rewrite may discharge a private BufferState suffix online, so
  // the offline value proof intentionally does not claim Buffer equality.
  EXPECT_FALSE(
      hasDomain(result.proof->verifiedDomains, GraphSemanticDomain::Buffer));
}

TEST(SemanticEquivalenceOracleTest, KeepsBufferStateOutputsObservable) {
  MLIRContext context;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr initial =
      TensorSemanticExpr::symbol(&context, "buffer", i1);
  TensorSemanticExpr address =
      TensorSemanticExpr::symbol(&context, "address", i32);
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "value", i32);

  TensorDenotation before;
  before.value = initial;
  before.defined = TensorSemanticExpr::boolean(&context, true);
  before.buffer = initial;
  before.bufferDefined = TensorSemanticExpr::boolean(&context, true);
  before.isBufferStateToken = true;
  TensorDenotation after = before;
  after.value = TensorSemanticExpr::operation(
      TensorSemanticKind::BufferWrite, i1, {initial, address, value});
  after.buffer = after.value;

  SemanticProofResult result = oracle.compare(before, after);
  EXPECT_NE(result.status, SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     ProvesExactFactorTwoProgramIdPartitionAndRejectsMissingLane) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr initialMemory =
      TensorSemanticExpr::symbol(&context, "memory", i1);
  TensorSemanticExpr baseLaunch =
      TensorSemanticExpr::symbol(&context, "launch", i1);
  DictionaryAttr mapping = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("factor", builder.getI32IntegerAttr(2)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(0))});
  TensorSemanticExpr groupedLaunch = TensorSemanticExpr::operation(
      TensorSemanticKind::LaunchRemap, i1, {baseLaunch}, mapping);
  Attribute axis = builder.getI32IntegerAttr(0);
  TensorSemanticExpr originalPid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {baseLaunch}, axis);
  TensorSemanticExpr groupedPid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {groupedLaunch}, axis);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "value", i32);
  TensorSemanticExpr laneZero = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {groupedPid, two});
  TensorSemanticExpr laneOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32, {laneZero, one});

  TensorSemanticExpr originalMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {initialMemory, originalPid, value});
  TensorSemanticExpr groupedLaneZero = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {initialMemory, laneZero, value});
  TensorSemanticExpr groupedMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {groupedLaneZero, laneOne, value});
  auto kernelEffect = [&](TensorSemanticExpr memory,
                          TensorSemanticExpr launch) {
    return TensorSemanticExpr::operation(TensorSemanticKind::KernelEffect,
                                         i1, {memory, launch});
  };
  auto makeKernel = [&](TensorSemanticExpr memory,
                        TensorSemanticExpr launch) {
    TensorDenotation result;
    TensorSemanticExpr observed = kernelEffect(memory, launch);
    result.value = observed;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    result.effect = TensorSemanticExpr::boolean(&context, true);
    result.memory = observed;
    result.memoryDefined = TensorSemanticExpr::boolean(&context, true);
    result.launch = observed;
    result.launchDefined = TensorSemanticExpr::boolean(&context, true);
    result.isMemoryStateToken = true;
    return result;
  };

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  TensorDenotation original = makeKernel(originalMemory, baseLaunch);
  TensorDenotation grouped = makeKernel(groupedMemory, groupedLaunch);
  SemanticProofResult proof = oracle.compare(original, grouped);
  ASSERT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
  ASSERT_TRUE(proof.proof);
  EXPECT_TRUE(llvm::any_of(proof.proof->steps, [](const auto &step) {
    return step.reason == "exact-launch-partition";
  }));

  TensorDenotation missingLane =
      makeKernel(groupedLaneZero, groupedLaunch);
  EXPECT_NE(oracle.compare(original, missingLane).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr originalGridX = TensorSemanticExpr::operation(
      TensorSemanticKind::NumPrograms, i32, {baseLaunch}, axis);
  TensorSemanticExpr groupedGridX = TensorSemanticExpr::operation(
      TensorSemanticKind::NumPrograms, i32, {groupedLaunch}, axis);
  EXPECT_NE(oracle.compare(makeKernel(originalGridX, baseLaunch),
                           makeKernel(groupedGridX, groupedLaunch))
                .status,
            SemanticProofStatus::Proved);

  Attribute axisY = builder.getI32IntegerAttr(1);
  TensorSemanticExpr originalGridY = TensorSemanticExpr::operation(
      TensorSemanticKind::NumPrograms, i32, {baseLaunch}, axisY);
  TensorSemanticExpr groupedGridY = TensorSemanticExpr::operation(
      TensorSemanticKind::NumPrograms, i32, {groupedLaunch}, axisY);
  EXPECT_EQ(oracle.compare(makeKernel(originalGridY, baseLaunch),
                           makeKernel(groupedGridY, groupedLaunch))
                .status,
            SemanticProofStatus::Proved);
}

TEST(SemanticEquivalenceOracleTest,
     ProvesParameterizedExactPartitionWithoutLaneExpansion) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr memory =
      TensorSemanticExpr::symbol(&context, "memory", i1);
  TensorSemanticExpr launch =
      TensorSemanticExpr::symbol(&context, "launch", i1);
  Attribute axis = builder.getI32IntegerAttr(0);
  TensorSemanticExpr pid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {launch}, axis);
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "value", i32);
  TensorSemanticExpr body = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1, {memory, pid, value});
  auto mapping = [&](int64_t factor, int64_t ceilDiv) {
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
         builder.getNamedAttr("factor", builder.getI32IntegerAttr(factor)),
         builder.getNamedAttr("ceil_div",
                              builder.getI32IntegerAttr(ceilDiv))});
  };
  auto make = [&](TensorSemanticKind kind, TensorSemanticExpr state,
                  DictionaryAttr payload = {}) {
    TensorDenotation denotation;
    TensorSemanticExpr observed = TensorSemanticExpr::operation(
        kind, i1, {state, launch}, payload);
    denotation.value = observed;
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    denotation.effect = observed;
    denotation.memory = observed;
    denotation.memoryDefined = TensorSemanticExpr::boolean(&context, true);
    denotation.launch = observed;
    denotation.launchDefined = TensorSemanticExpr::boolean(&context, true);
    denotation.isMemoryStateToken = true;
    return denotation;
  };

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  TensorDenotation original =
      make(TensorSemanticKind::KernelEffect, body);
  for (int64_t factor : {4, 16}) {
    TensorDenotation partitioned = make(
        TensorSemanticKind::KernelPartition, body, mapping(factor, 0));
    SemanticProofResult proof = oracle.compare(original, partitioned);
    ASSERT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
    ASSERT_TRUE(proof.proof);
    EXPECT_TRUE(llvm::any_of(proof.proof->steps, [](const auto &step) {
      return step.reason == "quantified-exact-launch-partition";
    }));
  }

  TensorSemanticExpr unconstrainedMemory =
      TensorSemanticExpr::symbol(&context, "memory-state-0", i1);
  EXPECT_NE(
      oracle
          .compare(make(TensorSemanticKind::KernelEffect,
                        unconstrainedMemory),
                   make(TensorSemanticKind::KernelPartition,
                        unconstrainedMemory, mapping(16, 0)))
          .status,
      SemanticProofStatus::Proved);
  DictionaryAttr exactPremise = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(0))});
  TensorSemanticExpr partitionableMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::PartitionableMemory, i1, {}, exactPremise);
  EXPECT_EQ(oracle
                .compare(make(TensorSemanticKind::KernelEffect,
                              partitionableMemory),
                         make(TensorSemanticKind::KernelPartition,
                              partitionableMemory, mapping(16, 0)))
                .status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr grid = TensorSemanticExpr::operation(
      TensorSemanticKind::NumPrograms, i32, {launch}, axis);
  TensorDenotation observesGrid = make(
      TensorSemanticKind::KernelPartition, grid, mapping(4, 0));
  EXPECT_NE(oracle.compare(make(TensorSemanticKind::KernelEffect, grid),
                           observesGrid)
                .status,
            SemanticProofStatus::Proved);
  TensorDenotation unprovedTail = make(
      TensorSemanticKind::KernelPartition, body, mapping(4, 1));
  EXPECT_NE(oracle.compare(original, unprovedTail).status,
            SemanticProofStatus::Proved);

  TensorSemanticExpr bound =
      TensorSemanticExpr::symbol(&context, "bound", i32);
  TensorSemanticExpr boundedLaunch = TensorSemanticExpr::operation(
      TensorSemanticKind::LaunchBound, i1, {launch, bound}, axis);
  TensorSemanticExpr boundedPid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {boundedLaunch}, axis);
  TensorSemanticExpr mask = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1, {boundedPid, bound});
  TensorSemanticExpr guardedBody = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {memory, boundedPid, value, mask});
  auto makeBounded = [&](TensorSemanticKind kind, DictionaryAttr payload = {}) {
    TensorDenotation denotation;
    TensorSemanticExpr observed = TensorSemanticExpr::operation(
        kind, i1, {guardedBody, boundedLaunch}, payload);
    denotation.value = observed;
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    denotation.effect = observed;
    denotation.memory = observed;
    denotation.memoryDefined = TensorSemanticExpr::boolean(&context, true);
    denotation.launch = observed;
    denotation.launchDefined = TensorSemanticExpr::boolean(&context, true);
    denotation.isMemoryStateToken = true;
    return denotation;
  };
  SemanticProofResult guardedProof = oracle.compare(
      makeBounded(TensorSemanticKind::KernelEffect),
      makeBounded(TensorSemanticKind::KernelPartition, mapping(16, 1)));
  ASSERT_EQ(guardedProof.status, SemanticProofStatus::Proved)
      << guardedProof.reason;
  ASSERT_TRUE(guardedProof.proof);
  EXPECT_TRUE(llvm::any_of(guardedProof.proof->steps, [](const auto &step) {
    return step.reason == "quantified-guarded-tail-launch-partition";
  }));
}

TEST(SemanticEquivalenceOracleTest,
     ProvesGuardedTailPartitionOnlyWithExplicitLaunchBound) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr memory =
      TensorSemanticExpr::symbol(&context, "memory", i1);
  TensorSemanticExpr launch =
      TensorSemanticExpr::symbol(&context, "launch", i1);
  TensorSemanticExpr bound =
      TensorSemanticExpr::symbol(&context, "grid_bound", i32);
  TensorSemanticExpr otherBound =
      TensorSemanticExpr::symbol(&context, "other_grid_bound", i32);
  Attribute axis = builder.getI32IntegerAttr(0);
  TensorSemanticExpr axisZeroBound = TensorSemanticExpr::operation(
      TensorSemanticKind::LaunchBound, i1, {launch, bound}, axis);
  TensorSemanticExpr boundedLaunch = TensorSemanticExpr::operation(
      TensorSemanticKind::LaunchBound, i1, {axisZeroBound, otherBound},
      builder.getI32IntegerAttr(1));
  DictionaryAttr mapping = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("factor", builder.getI32IntegerAttr(2)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(1))});
  TensorSemanticExpr groupedLaunch = TensorSemanticExpr::operation(
      TensorSemanticKind::LaunchRemap, i1, {boundedLaunch}, mapping);
  TensorSemanticExpr originalPid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {boundedLaunch}, axis);
  TensorSemanticExpr groupedPid = TensorSemanticExpr::operation(
      TensorSemanticKind::ProgramId, i32, {groupedLaunch}, axis);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "value", i32);
  TensorSemanticExpr laneZero = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i32, {groupedPid, two});
  TensorSemanticExpr laneOne = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32, {laneZero, one});
  auto lessThanBound = [&](const TensorSemanticExpr &pid) {
    return TensorSemanticExpr::operation(TensorSemanticKind::SignedLessThan,
                                         i1, {pid, bound});
  };
  TensorSemanticExpr originalMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {memory, originalPid, value, lessThanBound(originalPid)});
  TensorSemanticExpr laneZeroMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {memory, laneZero, value, lessThanBound(laneZero)});
  TensorSemanticExpr groupedMemory = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {laneZeroMemory, laneOne, value, lessThanBound(laneOne)});
  auto guardedValidity = [&](const TensorSemanticExpr &pid) {
    TensorSemanticExpr accessible = TensorSemanticExpr::operation(
        TensorSemanticKind::MemoryValid, i1, {memory, pid});
    return TensorSemanticExpr::operation(
        TensorSemanticKind::Select, i1,
        {lessThanBound(pid), accessible,
         TensorSemanticExpr::boolean(&context, true)});
  };
  TensorSemanticExpr originalValidity = guardedValidity(originalPid);
  TensorSemanticExpr groupedValidity = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1,
      {guardedValidity(laneZero), guardedValidity(laneOne)});
  auto makeKernel = [&](const TensorSemanticExpr &state,
                        const TensorSemanticExpr &launchState,
                        const TensorSemanticExpr &validity) {
    TensorDenotation result;
    TensorSemanticExpr observed = TensorSemanticExpr::operation(
        TensorSemanticKind::KernelEffect, i1, {state, launchState});
    result.value = observed;
    result.defined = TensorSemanticExpr::boolean(&context, true);
    result.effect = observed;
    result.memory = observed;
    result.memoryDefined = TensorSemanticExpr::operation(
        TensorSemanticKind::KernelEffect, i1, {validity, launchState});
    result.launch = observed;
    result.launchDefined = TensorSemanticExpr::boolean(&context, true);
    result.isMemoryStateToken = true;
    return result;
  };

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  TensorDenotation original =
      makeKernel(originalMemory, boundedLaunch, originalValidity);
  TensorDenotation grouped =
      makeKernel(groupedMemory, groupedLaunch, groupedValidity);
  SemanticProofResult proof = oracle.compare(original, grouped);
  ASSERT_EQ(proof.status, SemanticProofStatus::Proved)
      << proof.reason << "\noriginal: "
      << getStableTensorDenotationKey(normalizer.normalize(original).denotation)
      << "\ngrouped: "
      << getStableTensorDenotationKey(normalizer.normalize(grouped).denotation);
  ASSERT_TRUE(proof.proof);
  EXPECT_TRUE(llvm::any_of(proof.proof->steps, [](const auto &step) {
    return step.reason == "guarded-tail-launch-partition";
  }));

  TensorSemanticExpr unguardedZero = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {memory, laneZero, value});
  TensorSemanticExpr unguarded = TensorSemanticExpr::operation(
      TensorSemanticKind::MemoryWrite, i1,
      {unguardedZero, laneOne, value});
  EXPECT_NE(oracle.compare(
                           makeKernel(originalMemory, boundedLaunch,
                                      originalValidity),
                           makeKernel(unguarded, groupedLaunch,
                                      groupedValidity))
                .status,
            SemanticProofStatus::Proved);
}

} // namespace

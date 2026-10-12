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

#if defined(TRITON_GRAPH_HAVE_Z3)

#include "TritonToGraph/Z3SemanticSolver.h"

#include "TritonToGraph/OfflineEquivalenceProver.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "gtest/gtest.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static TensorDenotation makeScalar(TensorSemanticExpr value,
                                   MLIRContext &context) {
  TensorDenotation denotation;
  denotation.value = std::move(value);
  denotation.defined = TensorSemanticExpr::boolean(&context, true);
  return denotation;
}

TEST(Z3SemanticSolverTest, ProvesSelectShiftIdentityAndReplaysCertificate) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);

  TensorSemanticExpr shiftedValue = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i32, {value, one});
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::operation(TensorSemanticKind::Select, i32,
                                               {mask, shiftedValue, one}),
                 context);
  TensorSemanticExpr selected = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {mask, value, two});
  TensorDenotation rhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::ShiftRightArithmetic,
                                    i32, {selected, one}),
      context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  // Exercise the backend directly.  The generic normalizer now proves this
  // identity before fallback, but this test owns Z3 translation and
  // certificate replay rather than normalizer coverage.
  SemanticProofResult result = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  ASSERT_TRUE(result.proof);
  EXPECT_EQ(result.proof->proofBackend, "z3-bv-v1");
  EXPECT_EQ(solver.getStatistics().unsupported, 0u);

  std::string serialized = result.proof->toJSON();
  MLIRContext replayContext;
  std::string error;
  std::optional<SemanticEquivalenceProof> decoded =
      SemanticEquivalenceProof::fromJSON(serialized, replayContext, error);
  ASSERT_TRUE(decoded) << error;
  Z3SemanticSolver replaySolver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult replay = replaySolver.replay(*decoded);
  ASSERT_EQ(replay.status, SemanticProofStatus::Proved) << replay.reason;
  ASSERT_TRUE(replay.proof);
  EXPECT_EQ(replay.proof->getStableHash(), decoded->getStableHash());
}

TEST(Z3SemanticSolverTest, DefersFallbackUntilAfterConcreteRefinement) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i32,
          {mask,
           TensorSemanticExpr::operation(
               TensorSemanticKind::ShiftRightArithmetic, i32, {value, one}),
           one}),
      context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ShiftRightArithmetic, i32,
                     {TensorSemanticExpr::operation(TensorSemanticKind::Select,
                                                    i32, {mask, value, two}),
                      one}),
                 context);

  std::vector<OfflineTypedGraph> graphs(2);
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "z3-deferred-" + std::to_string(graphId);
  }
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"z3-deferred-collision", {0, 1}});
  fingerprints.denotations.push_back(lhs);
  fingerprints.denotations.push_back(rhs);
  fingerprints.exactDenotationGroupIds = {0, 1};
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;
  fingerprints.modelCount = 16;
  fingerprints.denotationsNormalized = true;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  options.additionalCounterexampleModels = 1;
  options.counterexampleJobs = 2;
  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, oracle, options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.symbolicPremergeComparisons, 1u);
  EXPECT_EQ(proofs.symbolicPremergeMergedGroups, 0u);
  EXPECT_EQ(proofs.counterexampleModelEvaluations, 2u);
  EXPECT_EQ(proofs.exactPostRefinementComparisons, 1u);
  EXPECT_EQ(proofs.exactPostRefinementProved, 1u);
  EXPECT_EQ(proofs.provedPairs, 1u);
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  // The fallback is called once for the retained pair and once to replay the
  // serialized certificate. It is not called by symbolic premerge.
  EXPECT_EQ(statistics.queries, 2u);
  EXPECT_EQ(statistics.proved, 2u);

  proofs.resolvedProofCache.universeHash = "z3-cache-test";
  std::string cacheJSON = proofs.resolvedProofCache.toJSON();
  std::string cacheError;
  std::optional<OfflineResolvedProofCache> cache =
      OfflineResolvedProofCache::fromJSON(cacheJSON, cacheError);
  ASSERT_TRUE(cache) << cacheError;
  ASSERT_EQ(cache->pairs.size(), 1u);

  OfflineEquivalenceProofOptions resumedOptions = options;
  resumedOptions.resolvedProofCache = &*cache;
  Z3SemanticSolver resumedSolver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle resumedOracle(normalizer, &resumedSolver);
  OfflineEquivalenceProofResult resumed = OfflineEquivalenceProver().prove(
      graphs, fingerprints, resumedOracle, resumedOptions);
  EXPECT_TRUE(resumed.isComplete());
  EXPECT_EQ(resumed.resolvedProofCacheHits, 1u);
  EXPECT_EQ(resumed.resolvedProofCacheRejected, 0u);
  EXPECT_EQ(resumed.resolvedProofCache.pairs.size(), 1u);
  // The expensive exact probe is reused. The one remaining query is the
  // mandatory replay of the serialized certificate before it enters the
  // persistent proof forest.
  EXPECT_EQ(resumedSolver.getStatistics().queries, 1u);

  OfflineResolvedProofCache pendingCache;
  pendingCache.universeHash = "z3-pending-cache-test";
  pendingCache.graphCount = graphs.size();
  pendingCache.pendingPairs.push_back(
      {0, 1, graphs[0].canonicalKey, graphs[1].canonicalKey, 1,
       "Z3 returned unknown: timeout"});
  std::string pendingCacheError;
  std::optional<OfflineResolvedProofCache> decodedPendingCache =
      OfflineResolvedProofCache::fromJSON(pendingCache.toJSON(),
                                          pendingCacheError);
  ASSERT_TRUE(decodedPendingCache) << pendingCacheError;
  ASSERT_EQ(decodedPendingCache->pendingPairs.size(), 1u);

  OfflineEquivalenceProofOptions pendingOptions = options;
  pendingOptions.resolvedProofCache = &*decodedPendingCache;
  Z3SemanticSolver pendingSolver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle pendingOracle(normalizer, &pendingSolver);
  OfflineEquivalenceProofResult pendingResult = OfflineEquivalenceProver().prove(
      graphs, fingerprints, pendingOracle, pendingOptions);
  EXPECT_TRUE(pendingResult.isComplete());
  EXPECT_EQ(pendingResult.pendingProofCacheAttempts, 1u);
  EXPECT_EQ(pendingResult.pendingProofCacheResolved, 1u);
  EXPECT_EQ(pendingResult.pendingProofCacheRemaining, 0u);
  EXPECT_EQ(pendingResult.pendingProofCacheRejected, 0u);
  EXPECT_EQ(pendingResult.resolvedProofCache.pairs.size(), 1u);
}

TEST(Z3SemanticSolverTest, RetriesOnlyPrimaryTimeoutsWithLargerBudget) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i32,
          {mask,
           TensorSemanticExpr::operation(
               TensorSemanticKind::ShiftRightArithmetic, i32, {value, one}),
           one}),
      context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ShiftRightArithmetic, i32,
                     {TensorSemanticExpr::operation(TensorSemanticKind::Select,
                                                    i32, {mask, value, two}),
                      one}),
                 context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1,
                          /*maxQueriesPerWorkerContext=*/4096,
                          /*retryTimeoutMilliseconds=*/1000);
  SemanticProofResult result = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, 1u);
  EXPECT_EQ(statistics.retryChecks, 1u);
  EXPECT_EQ(statistics.retryUnsatChecks, 1u);
  EXPECT_EQ(statistics.retrySatChecks, 0u);
  EXPECT_EQ(statistics.retryUnknown, 0u);

  Z3SemanticSolver bitBlastSolver(
      /*timeoutMilliseconds=*/1,
      /*maxQueriesPerWorkerContext=*/4096,
      /*retryTimeoutMilliseconds=*/0,
      /*bitBlastTimeoutMilliseconds=*/1000);
  SemanticProofResult bitBlastResult =
      bitBlastSolver.compareNormalized(lhs, rhs);
  ASSERT_EQ(bitBlastResult.status, SemanticProofStatus::Proved)
      << bitBlastResult.reason;
  Z3SemanticSolverStatistics bitBlastStatistics =
      bitBlastSolver.getStatistics();
  EXPECT_EQ(bitBlastStatistics.bitBlastChecks, 1u);
  EXPECT_EQ(bitBlastStatistics.bitBlastUnsatChecks, 1u);
  EXPECT_EQ(bitBlastStatistics.bitBlastSatChecks, 0u);
  EXPECT_EQ(bitBlastStatistics.bitBlastUnknown, 0u);
}

TEST(Z3SemanticSolverTest, ReturnsCounterexampleForUnequalBitVectors) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::Add, i32,
                     {value, TensorSemanticExpr::integer(&context, 1, i32)}),
                 context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::Add, i32,
                     {value, TensorSemanticExpr::integer(&context, 2, i32)}),
                 context);

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  EXPECT_EQ(result.status, SemanticProofStatus::Refuted) << result.reason;
  EXPECT_NE(result.reason.find("counterexample"), std::string::npos);
}

TEST(Z3SemanticSolverTest, DischargesLiteralQueriesWithExactSimplifier) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(&context, 0, i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1);
  SemanticProofResult proved = solver.compareNormalized(
      makeScalar(TensorSemanticExpr::operation(TensorSemanticKind::BitwiseXor,
                                               i32, {value, value}),
                 context),
      makeScalar(zero, context));
  ASSERT_EQ(proved.status, SemanticProofStatus::Proved) << proved.reason;

  SemanticProofResult refuted = solver.compareNormalized(
      makeScalar(zero, context), makeScalar(one, context));
  ASSERT_EQ(refuted.status, SemanticProofStatus::Refuted) << refuted.reason;

  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, 2u);
  EXPECT_EQ(statistics.simplifierProved, 1u);
  EXPECT_EQ(statistics.simplifierRefuted, 1u);
  EXPECT_EQ(statistics.timedOut, 0u);
}

TEST(Z3SemanticSolverTest, IgnoresUnobservableStaticShapeMetadata) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(&context, 0, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::BitwiseXor, i32,
                                    {value, value}),
      context);
  TensorDenotation rhs = makeScalar(zero, context);
  lhs.shape = {4};
  rhs.shape = {4};
  lhs.shapeSymbols = {"lhs-static-spelling"};
  rhs.shapeSymbols = {"rhs-static-spelling"};
  lhs.buffer = TensorSemanticExpr::symbol(&context, "lhs-buffer", i32);
  rhs.buffer = TensorSemanticExpr::symbol(&context, "rhs-buffer", i32);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult result = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.unsupported, 0u);
}

TEST(Z3SemanticSolverTest, TotalizesLazyUnobservedArmUnderDefinedness) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr unobserved = TensorSemanticExpr::operation(
      TensorSemanticKind::Unobserved, i32, {});

  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::Select, i32,
                                    {mask, value, unobserved}),
      context);
  lhs.defined = mask;
  TensorDenotation rhs = makeScalar(value, context);
  rhs.defined = mask;

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
}

TEST(Z3SemanticSolverTest, ReusesOneIncrementalContextAcrossQueries) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i32,
          {mask,
           TensorSemanticExpr::operation(
               TensorSemanticKind::ShiftRightArithmetic, i32, {value, one}),
           one}),
      context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ShiftRightArithmetic, i32,
                     {TensorSemanticExpr::operation(TensorSemanticKind::Select,
                                                    i32, {mask, value, two}),
                      one}),
                 context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  constexpr unsigned kQueryCount = 1000;
  for (unsigned query = 0; query < kQueryCount; ++query)
    ASSERT_EQ(solver.compareNormalized(lhs, rhs).status,
              SemanticProofStatus::Proved);
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, kQueryCount);
  EXPECT_EQ(statistics.proved, kQueryCount);
  EXPECT_EQ(statistics.workerContexts, 1u);
  EXPECT_EQ(statistics.workerContextCreations, 1u);
  EXPECT_EQ(statistics.workerContextRecycles, 0u);
}

TEST(Z3SemanticSolverTest, RecyclesAgedWorkerContexts) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i32,
          {mask,
           TensorSemanticExpr::operation(
               TensorSemanticKind::ShiftRightArithmetic, i32, {value, one}),
           one}),
      context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ShiftRightArithmetic, i32,
                     {TensorSemanticExpr::operation(TensorSemanticKind::Select,
                                                    i32, {mask, value, two}),
                      one}),
                 context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000,
                          /*maxQueriesPerWorkerContext=*/8);
  constexpr unsigned kQueryCount = 65;
  for (unsigned query = 0; query < kQueryCount; ++query)
    ASSERT_EQ(solver.compareNormalized(lhs, rhs).status,
              SemanticProofStatus::Proved);

  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, kQueryCount);
  EXPECT_EQ(statistics.proved, kQueryCount);
  EXPECT_EQ(statistics.workerContexts, 1u);
  EXPECT_EQ(statistics.workerContextCreations, 9u);
  EXPECT_EQ(statistics.workerContextRecycles, 8u);
}

TEST(Z3SemanticSolverTest, UsesIndependentContextsForParallelProofWorkers) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr two = TensorSemanticExpr::integer(&context, 2, i32);
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(
          TensorSemanticKind::Select, i32,
          {mask,
           TensorSemanticExpr::operation(
               TensorSemanticKind::ShiftRightArithmetic, i32, {value, one}),
           one}),
      context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ShiftRightArithmetic, i32,
                     {TensorSemanticExpr::operation(TensorSemanticKind::Select,
                                                    i32, {mask, value, two}),
                      one}),
                 context);

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  constexpr unsigned kThreadCount = 8;
  constexpr unsigned kQueriesPerThread = 100;
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  std::atomic<unsigned> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (unsigned thread = 0; thread < kThreadCount; ++thread)
    workers.emplace_back([&]() {
      ++ready;
      while (!start.load(std::memory_order_acquire))
        std::this_thread::yield();
      for (unsigned query = 0; query < kQueriesPerThread; ++query)
        if (oracle.compareNormalized(lhs, rhs).status !=
            SemanticProofStatus::Proved)
          ++failures;
    });
  while (ready.load(std::memory_order_acquire) != kThreadCount)
    std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (std::thread &worker : workers)
    worker.join();

  EXPECT_EQ(failures.load(), 0u);
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, kThreadCount * kQueriesPerThread);
  EXPECT_EQ(statistics.proved, kThreadCount * kQueriesPerThread);
  EXPECT_GE(statistics.workerContexts, 2u);
  EXPECT_LE(statistics.workerContexts, kThreadCount);
}

TEST(Z3SemanticSolverTest, ComparesDefinednessBeforeValue) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::integer(&context, 1, i32), context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::integer(&context, 2, i32), context);
  lhs.defined = TensorSemanticExpr::boolean(&context, false);
  rhs.defined = TensorSemanticExpr::boolean(&context, false);

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Proved);

  rhs.defined = TensorSemanticExpr::boolean(&context, true);
  EXPECT_EQ(oracle.compare(lhs, rhs).status, SemanticProofStatus::Refuted);
}

TEST(Z3SemanticSolverTest, ModelsI1BitwiseAndIntegerCastsExactly) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr inverted = TensorSemanticExpr::operation(
      TensorSemanticKind::BitwiseXor, i1,
      {mask, TensorSemanticExpr::integer(&context, 1, i1)});
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::operation(TensorSemanticKind::ZeroExtend,
                                               i32, {inverted}),
                 context);
  TensorDenotation rhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::Select, i32,
                     {mask, TensorSemanticExpr::integer(&context, 0, i32),
                      TensorSemanticExpr::integer(&context, 1, i32)}),
                 context);

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;

  TensorSemanticExpr widened = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, i32, {mask});
  TensorDenotation roundTrip =
      makeScalar(TensorSemanticExpr::operation(TensorSemanticKind::Truncate, i1,
                                               {widened}),
                 context);
  EXPECT_EQ(oracle.compare(roundTrip, makeScalar(mask, context)).status,
            SemanticProofStatus::Proved);
}

TEST(Z3SemanticSolverTest,
     ModelsPointerAddWithMixedWidthOffsetsAndReplaysCertificate) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i8 = builder.getI8Type();
  Type i64 = builder.getI64Type();
  Type pointer = triton::PointerType::get(builder.getF32Type(),
                                          /*addressSpace=*/1);
  TensorSemanticExpr base =
      TensorSemanticExpr::symbol(&context, "base", pointer);
  TensorSemanticExpr offset =
      TensorSemanticExpr::symbol(&context, "offset", i8);
  TensorSemanticExpr widened = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, i64, {offset});
  TensorDenotation lhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::PointerAdd, pointer,
                                    {base, offset}),
      context);
  TensorDenotation rhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::PointerAdd, pointer,
                                    {base, widened}),
      context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult result = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  ASSERT_TRUE(result.proof);
  EXPECT_EQ(result.proof->proofBackend, "z3-bv-v1");
  EXPECT_EQ(solver.getStatistics().unsupported, 0u);

  std::string serialized = result.proof->toJSON();
  MLIRContext replayContext;
  replayContext.getOrLoadDialect<triton::TritonDialect>();
  std::string error;
  std::optional<SemanticEquivalenceProof> decoded =
      SemanticEquivalenceProof::fromJSON(serialized, replayContext, error);
  ASSERT_TRUE(decoded) << error;
  Z3SemanticSolver replaySolver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult replay = replaySolver.replay(*decoded);
  ASSERT_EQ(replay.status, SemanticProofStatus::Proved) << replay.reason;
  ASSERT_TRUE(replay.proof);
  EXPECT_EQ(replay.proof->getStableHash(), decoded->getStableHash());
}

TEST(Z3SemanticSolverTest,
     ProvesReorderedPointerAddsAndRefutesDifferentOffsets) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  Type index = builder.getIndexType();
  Type pointer = triton::PointerType::get(builder.getF32Type(),
                                          /*addressSpace=*/1);
  TensorSemanticExpr base =
      TensorSemanticExpr::symbol(&context, "base", pointer);
  TensorSemanticExpr first =
      TensorSemanticExpr::symbol(&context, "first", i32);
  TensorSemanticExpr second =
      TensorSemanticExpr::symbol(&context, "second", index);
  auto addPointer = [&](TensorSemanticExpr pointerValue,
                        TensorSemanticExpr offset) {
    return TensorSemanticExpr::operation(TensorSemanticKind::PointerAdd,
                                         pointer, {pointerValue, offset});
  };
  TensorDenotation lhs =
      makeScalar(addPointer(addPointer(base, first), second), context);
  TensorDenotation rhs =
      makeScalar(addPointer(addPointer(base, second), first), context);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult reordered = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(reordered.status, SemanticProofStatus::Proved)
      << reordered.reason;

  TensorDenotation zero = makeScalar(
      addPointer(base, TensorSemanticExpr::integer(&context, 0, i1)), context);
  TensorDenotation one = makeScalar(
      addPointer(base, TensorSemanticExpr::integer(&context, 1, i1)), context);
  SemanticProofResult different = solver.compareNormalized(zero, one);
  EXPECT_EQ(different.status, SemanticProofStatus::Refuted)
      << different.reason;
  EXPECT_EQ(solver.getStatistics().unsupported, 0u);
}

TEST(Z3SemanticSolverTest,
     ModelsMemoryReadAndValidityAsCongruentStateAddressFunctions) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  Type pointer = triton::PointerType::get(f32, /*addressSpace=*/1);
  TensorSemanticExpr memory =
      TensorSemanticExpr::symbol(&context, "memory", i1);
  TensorSemanticExpr base =
      TensorSemanticExpr::symbol(&context, "base", pointer);
  TensorSemanticExpr offset =
      TensorSemanticExpr::symbol(&context, "offset", i32);
  TensorSemanticExpr widened = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, builder.getI64Type(), {offset});
  TensorSemanticExpr narrowAddress = TensorSemanticExpr::operation(
      TensorSemanticKind::PointerAdd, pointer, {base, offset});
  TensorSemanticExpr wideAddress = TensorSemanticExpr::operation(
      TensorSemanticKind::PointerAdd, pointer, {base, widened});
  auto memoryRead = [&](TensorSemanticExpr address) {
    return TensorSemanticExpr::operation(TensorSemanticKind::MemoryRead, f32,
                                         {memory, address});
  };
  auto memoryValid = [&](TensorSemanticExpr address) {
    return TensorSemanticExpr::operation(TensorSemanticKind::MemoryValid, i1,
                                         {memory, address});
  };

  TensorDenotation lhs = makeScalar(memoryRead(narrowAddress), context);
  TensorDenotation rhs = makeScalar(memoryRead(wideAddress), context);
  lhs.defined = memoryValid(narrowAddress);
  rhs.defined = memoryValid(wideAddress);

  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticProofResult result = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
  ASSERT_TRUE(result.proof);
  EXPECT_EQ(result.proof->proofBackend, "z3-bv-v1");
  EXPECT_EQ(solver.getStatistics().unsupported, 0u);

  TensorSemanticExpr mask =
      TensorSemanticExpr::symbol(&context, "mask", i1);
  TensorSemanticExpr unobserved = TensorSemanticExpr::operation(
      TensorSemanticKind::Unobserved, f32, {});
  lhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, f32,
      {mask, memoryRead(narrowAddress), unobserved});
  rhs.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, f32,
      {mask, memoryRead(wideAddress), unobserved});
  TensorSemanticExpr falseValue =
      TensorSemanticExpr::boolean(&context, false);
  lhs.defined = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1,
      {mask, memoryValid(narrowAddress), falseValue});
  rhs.defined = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i1,
      {mask, memoryValid(wideAddress), falseValue});
  SemanticProofResult masked = solver.compareNormalized(lhs, rhs);
  ASSERT_EQ(masked.status, SemanticProofStatus::Proved) << masked.reason;
  EXPECT_EQ(solver.getStatistics().unsupported, 0u);
}

TEST(Z3SemanticSolverTest, ModelsShiftAmountDefinedness) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr mask = TensorSemanticExpr::symbol(&context, "m", i1);
  TensorSemanticExpr lhsValue = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorSemanticExpr rhsValue = TensorSemanticExpr::symbol(&context, "y", i32);
  TensorSemanticExpr amount = TensorSemanticExpr::symbol(&context, "s", i32);
  TensorSemanticExpr selected = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, i32, {mask, lhsValue, rhsValue});
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::Select, i32,
                     {mask,
                      TensorSemanticExpr::operation(
                          TensorSemanticKind::ShiftRightArithmetic, i32,
                          {lhsValue, amount}),
                      TensorSemanticExpr::operation(
                          TensorSemanticKind::ShiftRightArithmetic, i32,
                          {rhsValue, amount})}),
                 context);
  TensorDenotation rhs = makeScalar(
      TensorSemanticExpr::operation(TensorSemanticKind::ShiftRightArithmetic,
                                    i32, {selected, amount}),
      context);
  DictionaryAttr shiftPredicate =
      builder.getDictionaryAttr({builder.getNamedAttr(
          "semantic_role", builder.getStringAttr("shift_amount_valid"))});
  TensorSemanticExpr shiftDefined = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {amount}, shiftPredicate);
  lhs.defined = shiftDefined;
  rhs.defined = shiftDefined;

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Proved) << result.reason;
}

TEST(Z3SemanticSolverTest,
     RefutesNonlinearShiftIdentityBySplittingTheValidAmountDomain) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i64 = builder.getI64Type();
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "x", i64);
  TensorSemanticExpr amount =
      TensorSemanticExpr::symbol(&context, "amount", i64);
  TensorSemanticExpr shifted = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i64, {value, amount});
  TensorSemanticExpr product = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, i64, {value, shifted});
  TensorSemanticExpr lhsValue = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1,
      {TensorSemanticExpr::integer(&context, 4294967295LL, i64), product});
  TensorSemanticExpr shiftedAllOnes = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightArithmetic, i64,
      {TensorSemanticExpr::integer(&context, -1, i64), amount});
  TensorSemanticExpr rhsValue = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal, i1, {amount, shiftedAllOnes});
  DictionaryAttr shiftPredicate =
      builder.getDictionaryAttr({builder.getNamedAttr(
          "semantic_role", builder.getStringAttr("shift_amount_valid"))});
  TensorSemanticExpr shiftDefined = TensorSemanticExpr::operation(
      TensorSemanticKind::ApplyUnary, i1, {amount}, shiftPredicate);
  TensorDenotation lhs = makeScalar(lhsValue, context);
  TensorDenotation rhs = makeScalar(rhsValue, context);
  lhs.defined = shiftDefined;
  rhs.defined = shiftDefined;

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1,
                          /*maxQueriesPerWorkerContext=*/4096,
                          /*retryTimeoutMilliseconds=*/10);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  ASSERT_EQ(result.status, SemanticProofStatus::Refuted) << result.reason;
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_GT(statistics.finiteShiftDomainChecks, 0u);
  EXPECT_EQ(statistics.finiteShiftDomainSatChecks, 1u);
  EXPECT_EQ(statistics.unknown, 0u);
}

TEST(Z3SemanticSolverTest, LeavesUnsupportedSemanticKindsUnknown) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value = TensorSemanticExpr::symbol(&context, "x", i32);
  TensorDenotation lhs =
      makeScalar(TensorSemanticExpr::operation(
                     TensorSemanticKind::ApplyUnary, i32, {value},
                     StringAttr::get(&context, "test.opaque")),
                 context);
  TensorDenotation rhs = makeScalar(value, context);

  TensorSemanticNormalizer normalizer;
  Z3SemanticSolver solver(/*timeoutMilliseconds=*/1000);
  SemanticEquivalenceOracle oracle(normalizer, &solver);
  SemanticProofResult result = oracle.compare(lhs, rhs);
  EXPECT_EQ(result.status, SemanticProofStatus::Unknown);
  EXPECT_NE(result.reason.find("unsupported"), std::string::npos);
  Z3SemanticSolverStatistics statistics = solver.getStatistics();
  EXPECT_EQ(statistics.queries, 1u);
  EXPECT_EQ(statistics.unknown, 1u);
  EXPECT_EQ(statistics.unsupported, 1u);
}

} // namespace

#endif // TRITON_GRAPH_HAVE_Z3

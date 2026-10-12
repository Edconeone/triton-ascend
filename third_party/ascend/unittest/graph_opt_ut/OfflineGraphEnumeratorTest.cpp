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

#include "TritonToGraph/CanonicalRegion.h"
#include "TritonToGraph/OfflineGraphEnumerator.h"
#include "TritonToGraph/OfflineEquivalenceProver.h"
#include "TritonToGraph/SemanticFingerprint.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <climits>
#include <iterator>
#include <map>
#include <set>
#include <vector>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static GraphUniverseSpec makeScalarUniverse(unsigned maxNodes) {
  GraphUniverseSpec spec;
  spec.name = "scalar-enumerator-test";
  spec.target = "generic";
  spec.bounds.maxNodes = maxNodes;
  spec.bounds.maxRank = 0;
  spec.bounds.maxOperands = 2;
  spec.bounds.maxResults = 1;
  spec.operations = {"ags.input", "arith.constant", "arith.subi"};
  spec.catalogDialects = {"arith"};
  spec.integerBits = {32};
  spec.integerConstants = {0, 1};
  spec.semanticDomains = {GraphSemanticDomain::Signature,
                          GraphSemanticDomain::Shape,
                          GraphSemanticDomain::Index,
                          GraphSemanticDomain::Value,
                          GraphSemanticDomain::Definedness};
  return spec;
}

TEST(OfflineGraphEnumeratorTest, ExhaustsOneLayerDeterministically) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraphEnumerator enumerator;

  OfflineEnumerationResult first =
      enumerator.enumerate(makeScalarUniverse(1), schemas, {i32});
  OfflineEnumerationResult second =
      enumerator.enumerate(makeScalarUniverse(1), schemas, {i32});

  ASSERT_TRUE(first.isComplete());
  ASSERT_EQ(first.graphCountBySize.size(), 2u);
  EXPECT_EQ(first.graphCountBySize[0], 4u);
  EXPECT_EQ(first.graphCountBySize[1], 16u);
  EXPECT_EQ(first.graphs.size(), 20u);
  ASSERT_EQ(first.graphs.size(), second.graphs.size());
  for (auto [lhs, rhs] : llvm::zip_equal(first.graphs, second.graphs))
    EXPECT_EQ(lhs.canonicalKey, rhs.canonicalKey);
}

TEST(OfflineGraphEnumeratorTest,
     DepthOneQuotientsTypedBoundaryInputAlphaRenaming) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions options;
  options.semantics = &semantics;
  options.normalizer = &normalizer;
  options.quotientEquivalentOperands = true;

  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator(options).enumerate(
          makeScalarUniverse(/*maxNodes=*/1), schemas,
          {IntegerType::get(&context, 32)});

  ASSERT_TRUE(result.isComplete()) << result.toJSON("test-universe");
  ASSERT_EQ(result.graphCountBySize.size(), 2u);
  EXPECT_EQ(result.graphCountBySize[0], 4u);
  // All two-input equality partitions remain: (x,x) and (x,y). Input names
  // above those canonical representatives are typed alpha-renamings.
  EXPECT_EQ(result.graphCountBySize[1], 10u);
  EXPECT_GT(result.inputRenamingQuotientPruned, 0u);
}

TEST(OfflineGraphEnumeratorTest,
     DeclarativeTypingAcceptsArbitraryPartialOperandAssignments) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions,
                                          /*maxCandidateOperands=*/6);
  auto advance = llvm::find_if(
      schemas, [](const TTIRCandidateOpSchema &schema) {
        return schema.operationName == "tt.advance" &&
               schema.operandCount == 3;
      });
  ASSERT_NE(advance, schemas.end());
  ASSERT_TRUE(advance->mayBeWellTypedAssignment);

  Type i32 = IntegerType::get(&context, 32);
  Type f32 = Float32Type::get(&context);
  Type block = RankedTensorType::get({4, 2}, i32);
  Type pointer = triton::PointerType::get(block, /*addressSpace=*/1);
  Builder builder(&context);
  DictionaryAttr noAttributes = builder.getDictionaryAttr({});

  // The last offset can be tested before the middle offset is assigned.
  EXPECT_FALSE(advance->mayBeWellTypedAssignment(
      pointer, {pointer, Type{}, f32}, noAttributes));
  EXPECT_TRUE(advance->mayBeWellTypedAssignment(
      pointer, {pointer, Type{}, i32}, noAttributes));
  EXPECT_TRUE(advance->mayBeWellTypedAssignment(
      pointer, {pointer, i32, i32}, noAttributes));
}

TEST(OfflineGraphEnumeratorTest, RejectsSelectedOperationWithNoRealizedGraph) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec universe = makeScalarUniverse(/*maxNodes=*/0);

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      universe, schemas, {IntegerType::get(&context, 32)});

  EXPECT_FALSE(result.isComplete());
  EXPECT_EQ(result.unrealizedCandidateSchemaOperations,
            std::vector<std::string>({"arith.subi"}));
  ASSERT_EQ(result.graphCountByOperation.size(), 3u);
  EXPECT_EQ(result.graphCountByOperation[2],
            std::make_pair(std::string("arith.subi"), uint64_t{0}));
}

TEST(OfflineGraphEnumeratorTest,
     ClassifiesTypeInapplicableSiblingOverloadWithoutFailing) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTIRCandidateOpSchema impossibleOverload;
  impossibleOverload.operationName = "arith.subi";
  impossibleOverload.semanticClauseIndex = 999;
  impossibleOverload.operandCount = 2;
  impossibleOverload.constructible = true;
  impossibleOverload.canProduceResult =
      [](Type, DictionaryAttr) { return false; };
  schemas.push_back(std::move(impossibleOverload));

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      makeScalarUniverse(/*maxNodes=*/1), schemas,
      {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.isComplete());
  EXPECT_TRUE(result.unrealizedCandidateSchemaOperations.empty());
  EXPECT_TRUE(llvm::is_contained(
      result.inapplicableCandidateSchemas,
      "arith.subi/2#result0@clause999"));
}

TEST(OfflineGraphEnumeratorTest,
     SchemaReachabilityDoesNotClaimExhaustiveEnumeration) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  OfflineEnumerationOptions options;
  options.schemaReachabilityOnly = true;

  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator(options).enumerate(
          makeScalarUniverse(/*maxNodes=*/1), schemas,
          {IntegerType::get(&context, 32)});

  EXPECT_FALSE(result.isComplete());
  EXPECT_TRUE(result.isSchemaReachabilityComplete());
  EXPECT_TRUE(result.schemaReachabilityOnly);
  EXPECT_TRUE(result.unrealizedCandidateSchemas.empty());
  ASSERT_EQ(result.graphCountBySchema.size(), 1u);
  EXPECT_EQ(result.graphCountBySchema.front().second, 1u);
  // Two input terminals, two constants and one witness for arith.subi.
  EXPECT_EQ(result.graphs.size(), 5u);
}

TEST(OfflineGraphEnumeratorTest,
     ReachabilityReportsTypeDomainInapplicableSchemaWithoutFailing) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTIRCandidateOpSchema impossible;
  impossible.operationName = "test.impossible";
  impossible.semanticClauseIndex = 999;
  impossible.operandCount = 1;
  impossible.constructible = true;
  impossible.canProduceResult = [](Type, DictionaryAttr) { return true; };
  impossible.mayBeWellTyped =
      [](Type, ArrayRef<Type> operands, DictionaryAttr) {
        return operands.empty();
      };
  impossible.isWellTyped =
      [](Type, ArrayRef<Type>, DictionaryAttr) { return false; };
  schemas.push_back(std::move(impossible));

  GraphUniverseSpec universe = makeScalarUniverse(/*maxNodes=*/1);
  universe.operations = {"ags.input", "test.impossible"};
  universe.bounds.maxOperands = 1;
  OfflineEnumerationOptions options;
  options.schemaReachabilityOnly = true;
  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator(options).enumerate(
          universe, schemas, {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.isSchemaReachabilityComplete());
  EXPECT_TRUE(result.unrealizedCandidateSchemas.empty());
  EXPECT_TRUE(result.unrealizedCandidateSchemaOperations.empty());
  ASSERT_EQ(result.inapplicableCandidateSchemas.size(), 1u);
  EXPECT_EQ(result.inapplicableCandidateSchemas.front(),
            "test.impossible/1#result0@clause999");
}

TEST(OfflineGraphEnumeratorTest, ExhaustsEveryExactSizePartition) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraphEnumerator enumerator;

  OfflineEnumerationResult result =
      enumerator.enumerate(makeScalarUniverse(2), schemas, {i32});

  ASSERT_TRUE(result.isComplete());
  ASSERT_EQ(result.graphCountBySize.size(), 3u);
  EXPECT_EQ(result.graphCountBySize[0], 4u);
  EXPECT_EQ(result.graphCountBySize[1], 16u);
  // subi has two ordered operands. Exact DAG size 2 contains (terminal,
  // size-1), (size-1, terminal), and the 16 cases that reuse the same size-1
  // DAG as both operands: 4*16 + 16*4 + 16 = 144.
  EXPECT_EQ(result.graphCountBySize[2], 144u);
  EXPECT_EQ(result.graphs.size(), 164u);
  EXPECT_GT(result.saturatedReachabilityDomains, 0u);
  EXPECT_GT(result.saturatedReachabilityCandidatesSkipped, 0u);
}

TEST(OfflineGraphEnumeratorTest,
     FinalLayerPartitionsExactlyCoverTheUnpartitionedUniverse) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraphEnumerator enumerator;

  GraphUniverseSpec fullSpec = makeScalarUniverse(/*maxNodes=*/2);
  fullSpec.operations = {"ags.input", "arith.addi", "arith.constant",
                         "arith.subi"};
  OfflineEnumerationResult full =
      enumerator.enumerate(fullSpec, schemas, {i32});
  ASSERT_TRUE(full.isComplete());
  ASSERT_FALSE(full.completePrefixHash.empty());
  std::set<std::string> fullPrefix;
  for (const OfflineTypedGraph &graph : full.graphs)
    if (graph.operatorSize < 2)
      fullPrefix.insert(graph.canonicalKey);

  std::set<std::string> unionOfFinalLayers;
  std::set<std::string> previousPartition;
  std::string partitionFamilyHash;
  for (unsigned partition = 0; partition < 2; ++partition) {
    GraphUniverseSpec shardSpec = fullSpec;
    shardSpec.bounds.finalLayerPartitionCount = 2;
    shardSpec.bounds.finalLayerPartitionIndex = partition;
    OfflineEnumerationResult shard =
        enumerator.enumerate(shardSpec, schemas, {i32});
    ASSERT_TRUE(shard.isComplete());
    EXPECT_EQ(shard.finalLayerPartitionCount, 2u);
    EXPECT_EQ(shard.finalLayerPartitionIndex, partition);
    EXPECT_GT(shard.finalLayerPartitionPrefixesPruned, 0u);
    ASSERT_FALSE(shard.finalLayerPartitionFamilyHash.empty());
    if (partition == 0)
      partitionFamilyHash = shard.finalLayerPartitionFamilyHash;
    else
      EXPECT_EQ(shard.finalLayerPartitionFamilyHash, partitionFamilyHash);
    EXPECT_EQ(shard.completePrefixGraphCount, full.completePrefixGraphCount);
    EXPECT_EQ(shard.completePrefixHash, full.completePrefixHash);

    std::set<std::string> shardPrefix;
    for (const OfflineTypedGraph &graph : shard.graphs)
      if (graph.operatorSize < 2)
        shardPrefix.insert(graph.canonicalKey);
    EXPECT_EQ(shardPrefix, fullPrefix);

    std::set<std::string> finalLayer;
    for (const OfflineTypedGraph &graph : shard.graphs)
      if (graph.operatorSize == 2)
        finalLayer.insert(graph.canonicalKey);
    EXPECT_FALSE(finalLayer.empty());
    if (partition != 0) {
      std::vector<std::string> overlap;
      std::set_intersection(previousPartition.begin(), previousPartition.end(),
                            finalLayer.begin(), finalLayer.end(),
                            std::back_inserter(overlap));
      EXPECT_TRUE(overlap.empty());
    }
    previousPartition = finalLayer;
    unionOfFinalLayers.insert(finalLayer.begin(), finalLayer.end());
  }

  std::set<std::string> fullFinalLayer;
  for (const OfflineTypedGraph &graph : full.graphs)
    if (graph.operatorSize == 2)
      fullFinalLayer.insert(graph.canonicalKey);
  EXPECT_EQ(unionOfFinalLayers, fullFinalLayer);
}

TEST(OfflineGraphEnumeratorTest,
     FinalLayerPartitionsExactlyCoverTheQuotientedUniverse) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions options;
  options.semantics = &semantics;
  options.normalizer = &normalizer;
  options.quotientEquivalentOperands = true;
  OfflineTypedGraphEnumerator enumerator(options);
  Type i32 = IntegerType::get(&context, 32);

  GraphUniverseSpec fullSpec = makeScalarUniverse(/*maxNodes=*/2);
  OfflineEnumerationResult full =
      enumerator.enumerate(fullSpec, schemas, {i32});
  ASSERT_TRUE(full.isComplete());
  ASSERT_GT(full.operandQuotientPruned, 0u);
  ASSERT_FALSE(full.completePrefixHash.empty());

  std::set<std::string> fullFinalLayer;
  for (const OfflineTypedGraph &graph : full.graphs)
    if (graph.operatorSize == 2)
      fullFinalLayer.insert(graph.canonicalKey);
  ASSERT_FALSE(fullFinalLayer.empty());

  std::set<std::string> partitionUnion;
  std::set<std::string> previousPartition;
  std::string partitionFamilyHash;
  for (unsigned partition = 0; partition < 2; ++partition) {
    GraphUniverseSpec shardSpec = fullSpec;
    shardSpec.bounds.finalLayerPartitionCount = 2;
    shardSpec.bounds.finalLayerPartitionIndex = partition;
    OfflineEnumerationResult shard =
        enumerator.enumerate(shardSpec, schemas, {i32});
    ASSERT_TRUE(shard.isComplete());
    ASSERT_FALSE(shard.finalLayerPartitionFamilyHash.empty());
    if (partition == 0)
      partitionFamilyHash = shard.finalLayerPartitionFamilyHash;
    else
      EXPECT_EQ(shard.finalLayerPartitionFamilyHash, partitionFamilyHash);
    EXPECT_EQ(shard.completePrefixGraphCount, full.completePrefixGraphCount);
    EXPECT_EQ(shard.completePrefixHash, full.completePrefixHash);

    std::set<std::string> finalLayer;
    for (const OfflineTypedGraph &graph : shard.graphs)
      if (graph.operatorSize == 2)
        finalLayer.insert(graph.canonicalKey);
    EXPECT_FALSE(finalLayer.empty());
    if (partition != 0) {
      std::vector<std::string> overlap;
      std::set_intersection(previousPartition.begin(), previousPartition.end(),
                            finalLayer.begin(), finalLayer.end(),
                            std::back_inserter(overlap));
      EXPECT_TRUE(overlap.empty());
    }
    previousPartition = finalLayer;
    partitionUnion.insert(finalLayer.begin(), finalLayer.end());
  }
  EXPECT_EQ(partitionUnion, fullFinalLayer);
}

TEST(OfflineGraphEnumeratorTest,
     WideFinalLayerSchemasSplitBeyondLowCardinalityFirstOperand) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraphEnumerator enumerator;

  GraphUniverseSpec fullSpec = makeScalarUniverse(/*maxNodes=*/1);
  fullSpec.bounds.maxOperands = 3;
  fullSpec.operations = {"ags.input", "arith.constant", "arith.select"};
  fullSpec.integerBits = {1, 32};
  OfflineEnumerationResult full =
      enumerator.enumerate(fullSpec, schemas, {i1, i32});
  ASSERT_TRUE(full.isComplete());

  std::set<std::string> fullFinalLayer;
  for (const OfflineTypedGraph &graph : full.graphs)
    if (graph.operatorSize == 1)
      fullFinalLayer.insert(graph.canonicalKey);

  std::set<std::string> partitionUnion;
  std::map<std::string, std::set<unsigned>> partitionsByFirstOperand;
  for (unsigned partition = 0; partition < 4; ++partition) {
    GraphUniverseSpec shardSpec = fullSpec;
    shardSpec.bounds.finalLayerPartitionCount = 4;
    shardSpec.bounds.finalLayerPartitionIndex = partition;
    OfflineEnumerationResult shard =
        enumerator.enumerate(shardSpec, schemas, {i1, i32});
    ASSERT_TRUE(shard.isComplete());
    EXPECT_GT(shard.finalLayerPartitionPrefixesPruned, 0u);
    for (const OfflineTypedGraph &graph : shard.graphs) {
      if (graph.operatorSize != 1)
        continue;
      partitionUnion.insert(graph.canonicalKey);
      if (graph.operationName == "arith.select") {
        ASSERT_EQ(graph.operands.size(), 3u);
        partitionsByFirstOperand[shard.graphs[graph.operands[0]].canonicalKey]
            .insert(partition);
      }
    }
  }

  EXPECT_EQ(partitionUnion, fullFinalLayer);
  for (const auto &entry : partitionsByFirstOperand)
    EXPECT_LE(entry.second.size(), 2u);
  EXPECT_TRUE(llvm::any_of(partitionsByFirstOperand, [](const auto &entry) {
    return entry.second.size() > 1;
  }));
}

TEST(OfflineGraphEnumeratorTest,
     SharedPrefixCacheReproducesFreshFinalLayerPartition) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions options;
  options.semantics = &semantics;
  options.normalizer = &normalizer;
  options.quotientEquivalentOperands = true;
  Type i32 = IntegerType::get(&context, 32);

  GraphUniverseSpec firstSpec = makeScalarUniverse(/*maxNodes=*/2);
  firstSpec.bounds.finalLayerPartitionCount = 2;
  firstSpec.bounds.finalLayerPartitionIndex = 0;
  OfflineEnumerationResult first =
      OfflineTypedGraphEnumerator(options).enumerate(firstSpec, schemas,
                                                      {i32});
  ASSERT_TRUE(first.isComplete());
  std::string error;
  std::optional<OfflineEnumerationPrefixCache> cache =
      OfflineEnumerationPrefixCache::fromEnumeration(
          first, /*semanticsHash=*/"test-semantics", /*maximumOperatorSize=*/1,
          error);
  ASSERT_TRUE(cache) << error;
  const std::string serializedCache = cache->toJSON();
  EXPECT_TRUE(llvm::StringRef(serializedCache)
                  .starts_with("AGS_ENUMERATION_PREFIX_JSONL_V2\n"));
  ASSERT_EQ(cache->operandFrontier.size(), cache->graphs.size());
  std::string boundedError;
  std::optional<OfflineEnumerationPrefixCache> overBudget =
      OfflineEnumerationPrefixCache::fromJSON(
          serializedCache, context, boundedError, cache->graphs.size() - 1);
  EXPECT_FALSE(overBudget);
  EXPECT_FALSE(boundedError.empty());
  std::optional<OfflineEnumerationPrefixCache> reloaded =
      OfflineEnumerationPrefixCache::fromJSON(serializedCache, context, error);
  ASSERT_TRUE(reloaded) << error;
  EXPECT_EQ(reloaded->getStableHash(), cache->getStableHash());
  EXPECT_EQ(reloaded->validatedStableHash, cache->getStableHash());
  EXPECT_EQ(reloaded->operandFrontier, cache->operandFrontier);

  GraphUniverseSpec secondSpec = firstSpec;
  secondSpec.bounds.finalLayerPartitionIndex = 1;
  OfflineEnumerationResult fresh =
      OfflineTypedGraphEnumerator(options).enumerate(secondSpec, schemas,
                                                      {i32});
  ASSERT_TRUE(fresh.isComplete());
  const size_t cachedGraphCount = reloaded->graphs.size();
  const size_t cachedFrontierCount =
      llvm::count(reloaded->operandFrontier, 1);
  const std::string cachedHash = reloaded->getStableHash();
  options.prefixCache = &*reloaded;
  options.consumePrefixCache = true;
  OfflineEnumerationResult reused =
      OfflineTypedGraphEnumerator(options).enumerate(secondSpec, schemas,
                                                      {i32});
  ASSERT_TRUE(reused.isComplete());
  EXPECT_EQ(reused.reusedPrefixGraphCount, cachedGraphCount);
  EXPECT_EQ(reused.reusedPrefixOperandFrontierGraphCount,
            cachedFrontierCount);
  EXPECT_EQ(reused.reusedPrefixCacheHash, cachedHash);
  EXPECT_TRUE(reloaded->graphs.empty());
  ASSERT_EQ(reused.graphs.size(), fresh.graphs.size());
  for (auto [cachedGraph, freshGraph] :
       llvm::zip(reused.graphs, fresh.graphs))
    EXPECT_EQ(cachedGraph.canonicalKey, freshGraph.canonicalKey);
  EXPECT_EQ(reused.completePrefixHash, fresh.completePrefixHash);
  EXPECT_EQ(reused.unrealizedCandidateSchemas,
            fresh.unrealizedCandidateSchemas);
}

TEST(OfflineGraphEnumeratorTest,
     StandaloneFamilyPrefixReproducesFreshFinalLayerPartition) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions prefixOptions;
  prefixOptions.semantics = &semantics;
  prefixOptions.normalizer = &normalizer;
  prefixOptions.quotientEquivalentOperands = true;
  prefixOptions.alphaNormalizeFinalLayerInputs = false;
  prefixOptions.recordFinalLayerOperandFrontier = true;
  Type i32 = IntegerType::get(&context, 32);

  GraphUniverseSpec prefixSpec = makeScalarUniverse(/*maxNodes=*/1);
  OfflineEnumerationResult prefix =
      OfflineTypedGraphEnumerator(prefixOptions).enumerate(prefixSpec, schemas,
                                                           {i32});
  ASSERT_TRUE(prefix.isComplete());

  GraphUniverseSpec familySpec = makeScalarUniverse(/*maxNodes=*/2);
  familySpec.bounds.finalLayerPartitionCount = 2;
  familySpec.bounds.finalLayerPartitionIndex = 0;
  GraphUniverseSpec normalizedFamily = familySpec;
  normalizedFamily.name = "ags-final-layer-partition-family";
  normalizedFamily.bounds.finalLayerPartitionIndex = 0;
  prefix.finalLayerPartitionFamilyHash = normalizedFamily.getStableHash();

  std::string error;
  std::optional<OfflineEnumerationPrefixCache> cache =
      OfflineEnumerationPrefixCache::fromEnumeration(
          prefix, /*semanticsHash=*/"test-semantics",
          /*maximumOperatorSize=*/1, error);
  ASSERT_TRUE(cache) << error;
  EXPECT_EQ(cache->graphs.size(), prefix.graphs.size());
  EXPECT_EQ(cache->operandFrontier.size(), cache->graphs.size());
  EXPECT_LT(llvm::count(cache->operandFrontier, 1), cache->graphs.size());
  EXPECT_FALSE(cache->realizedCandidateSchemas.empty());

  familySpec.bounds.finalLayerPartitionIndex = 1;
  OfflineEnumerationOptions fullOptions = prefixOptions;
  fullOptions.alphaNormalizeFinalLayerInputs = true;
  fullOptions.recordFinalLayerOperandFrontier = false;
  fullOptions.semanticValidationJobs = 4;
  OfflineEnumerationResult fresh =
      OfflineTypedGraphEnumerator(fullOptions).enumerate(familySpec, schemas,
                                                         {i32});
  ASSERT_TRUE(fresh.isComplete());
  fullOptions.prefixCache = &*cache;
  OfflineEnumerationResult reused =
      OfflineTypedGraphEnumerator(fullOptions).enumerate(familySpec, schemas,
                                                         {i32});
  ASSERT_TRUE(reused.isComplete());
  ASSERT_EQ(reused.graphs.size(), fresh.graphs.size());
  for (auto [cachedGraph, freshGraph] :
       llvm::zip(reused.graphs, fresh.graphs))
    EXPECT_EQ(cachedGraph.canonicalKey, freshGraph.canonicalKey);
  EXPECT_EQ(reused.completePrefixHash, fresh.completePrefixHash);

  // A later normalizer revision may discover that two representatives in an
  // older cache are equivalent.  Such a cache is an over-approximation and
  // remains complete, so replay must keep the extra representative rather
  // than rejecting or renumbering the cached graph ids.
  auto redundant = llvm::find(cache->operandFrontier, uint8_t{0});
  ASSERT_NE(redundant, cache->operandFrontier.end());
  *redundant = 1;
  fullOptions.prefixCache = &*cache;
  OfflineEnumerationResult overApproximated =
      OfflineTypedGraphEnumerator(fullOptions).enumerate(familySpec, schemas,
                                                         {i32});
  EXPECT_TRUE(overApproximated.isComplete());
  EXPECT_TRUE(overApproximated.unsupportedUniverseFeatures.empty());
  EXPECT_GE(overApproximated.graphs.size(), fresh.graphs.size());
}

TEST(OfflineGraphEnumeratorTest, CountsSharedSubgraphOnceInDagBound) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);

  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator().enumerate(makeScalarUniverse(2), schemas,
                                               {i32});
  ASSERT_TRUE(result.isComplete());
  auto shared = llvm::find_if(result.graphs, [&](const auto &graph) {
    if (graph.operationName != "arith.subi" || graph.operatorSize != 2 ||
        graph.operands.size() != 2 || graph.operands[0] != graph.operands[1])
      return false;
    return result.graphs[graph.operands[0]].operatorSize == 1;
  });
  ASSERT_NE(shared, result.graphs.end());
  EXPECT_EQ(shared->reachableOperatorIds.size(), 2u);
}

TEST(OfflineGraphEnumeratorTest,
     ProofQuotientRetainsLhsButStopsEquivalentOperandExpansion) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions options;
  options.maxGraphs = 1000;
  options.semantics = &semantics;
  options.normalizer = &normalizer;
  options.quotientEquivalentOperands = true;

  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator(options).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_TRUE(result.isComplete());
  EXPECT_TRUE(result.quotientEquivalentOperands);
  EXPECT_GT(result.operandQuotientPruned, 0u);
  ASSERT_EQ(result.exactQuotientClassesBySize.size(),
            result.graphCountBySize.size());
  EXPECT_GT(result.exactQuotientClassesBySize[1], 0u);
  const std::string report = result.toJSON("test-universe");
  EXPECT_NE(report.find("\"frontier_representatives\""), std::string::npos);
  EXPECT_NE(report.find("\"exact_quotient_classes\""), std::string::npos);
  EXPECT_NE(report.find("\"frontier_pruned\""), std::string::npos);
  // Every size-1 LHS is retained for direct RuleDB lookup, while the size-2
  // layer is smaller than the 144 raw DAGs because equivalent size-1 syntax
  // is not recursively re-expanded.
  EXPECT_EQ(result.graphCountBySize[1], 16u);
  EXPECT_LT(result.graphCountBySize[2], 144u);

  OfflineEnumerationOptions parallelOptions = options;
  parallelOptions.semanticValidationJobs = 4;
  OfflineEnumerationResult parallelEnumeration =
      OfflineTypedGraphEnumerator(parallelOptions).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_TRUE(parallelEnumeration.isComplete());
  ASSERT_EQ(parallelEnumeration.graphs.size(), result.graphs.size());
  EXPECT_EQ(parallelEnumeration.rejectedBySemantics,
            result.rejectedBySemantics);
  EXPECT_GT(parallelEnumeration.semanticValidationGraphCount, 0u);
  EXPECT_EQ(parallelEnumeration.semanticValidationJobs, 4u);
  for (auto [index, graph] : llvm::enumerate(result.graphs)) {
    EXPECT_EQ(parallelEnumeration.graphs[index].id, graph.id);
    EXPECT_EQ(parallelEnumeration.graphs[index].canonicalKey,
              graph.canonicalKey);
    EXPECT_EQ(parallelEnumeration.graphs[index].reachableOperatorIds,
              graph.reachableOperatorIds);
  }

  OfflineSemanticFingerprintResult serialFingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/4, /*parallelJobs=*/1)
          .fingerprint(result.graphs, semantics, normalizer);
  OfflineSemanticFingerprintResult parallelFingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/4, /*parallelJobs=*/4)
          .fingerprint(result.graphs, semantics, normalizer);
  ASSERT_TRUE(serialFingerprints.isComplete());
  ASSERT_TRUE(parallelFingerprints.isComplete());
  ASSERT_EQ(parallelFingerprints.buckets.size(),
            serialFingerprints.buckets.size());
  EXPECT_EQ(parallelFingerprints.exactDenotationGroupIds,
            serialFingerprints.exactDenotationGroupIds);
  EXPECT_EQ(parallelFingerprints.proofCandidatePairsAfter,
            serialFingerprints.proofCandidatePairsAfter);
  for (auto [index, bucket] : llvm::enumerate(serialFingerprints.buckets)) {
    EXPECT_EQ(parallelFingerprints.buckets[index].fingerprint,
              bucket.fingerprint);
    EXPECT_EQ(parallelFingerprints.buckets[index].graphIds, bucket.graphIds);
  }
}

TEST(OfflineGraphEnumeratorTest,
     FinalLayerSemanticQuotientCompressesRootsWithoutChangingCoverage) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;

  OfflineEnumerationOptions baselineOptions;
  baselineOptions.maxGraphs = 1000;
  baselineOptions.semantics = &semantics;
  baselineOptions.normalizer = &normalizer;
  baselineOptions.quotientEquivalentOperands = true;
  OfflineEnumerationResult baseline =
      OfflineTypedGraphEnumerator(baselineOptions).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_TRUE(baseline.isComplete()) << baseline.toJSON("test-universe");

  OfflineEnumerationOptions compressedOptions = baselineOptions;
  compressedOptions.finalLayerSemanticQuotient = true;
  compressedOptions.semanticValidationJobs = 4;
  OfflineEnumerationResult compressed =
      OfflineTypedGraphEnumerator(compressedOptions).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_FALSE(compressed.isComplete());
  ASSERT_TRUE(compressed.isSemanticCoverageComplete())
      << compressed.toJSON("test-universe");
  EXPECT_TRUE(compressed.semanticCatalogCompressed);
  EXPECT_GT(compressed.finalLayerSemanticQuotientPruned, 0u);
  EXPECT_GT(compressed.finalLayerNormalizedStructuralPruned, 0u);
  EXPECT_LT(compressed.graphs.size(), baseline.graphs.size());
}

TEST(OfflineGraphEnumeratorTest,
     SemanticFrontierCatalogDropsOnlyProvedPureIntermediateDuplicates) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;

  OfflineEnumerationOptions baselineOptions;
  baselineOptions.maxGraphs = 10000;
  baselineOptions.semantics = &semantics;
  baselineOptions.normalizer = &normalizer;
  baselineOptions.quotientEquivalentOperands = true;
  OfflineEnumerationResult baseline =
      OfflineTypedGraphEnumerator(baselineOptions).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_TRUE(baseline.isComplete()) << baseline.toJSON("test-universe");

  OfflineEnumerationOptions frontierOptions = baselineOptions;
  frontierOptions.quotientEquivalentOperands = false;
  frontierOptions.semanticFrontierCatalogOnly = true;
  OfflineEnumerationResult frontier =
      OfflineTypedGraphEnumerator(frontierOptions).enumerate(
          makeScalarUniverse(2), schemas,
          {IntegerType::get(&context, 32)});
  ASSERT_FALSE(frontier.isComplete());
  ASSERT_TRUE(frontier.isSemanticCoverageComplete())
      << frontier.toJSON("test-universe");
  EXPECT_TRUE(frontier.semanticCatalogCompressed);
  EXPECT_TRUE(frontier.semanticFrontierCatalogOnly);
  EXPECT_GT(frontier.semanticFrontierCatalogPruned, 0u);
  EXPECT_LT(frontier.graphs.size(), baseline.graphs.size());
  EXPECT_EQ(frontier.hitGraphLimit, false);
  EXPECT_EQ(frontier.interrupted, false);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsAcrossDenotationBatchBoundaryWithoutDroppingGraphs) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"ags.input"};
  // SemanticFingerprint.cpp deliberately keeps at most 8192 printed
  // denotations alive at once. One extra graph exercises the final partial
  // batch and guards against boundary loss in the production memory path.
  spec.bounds.maxInputSymbols = 8193;

  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(
          spec, schemas, {IntegerType::get(&context, 32)});
  ASSERT_TRUE(enumeration.isComplete());
  ASSERT_EQ(enumeration.graphs.size(), 8193u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/2, /*parallelJobs=*/4)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 8193u);
  EXPECT_EQ(fingerprints.uniqueDenotationsFingerprinted, 8193u);
  ASSERT_EQ(fingerprints.exactDenotationGroupIds.size(), 8193u);
  EXPECT_TRUE(llvm::all_of(fingerprints.exactDenotationGroupIds,
                           [](unsigned groupId) {
                             return groupId != UINT_MAX;
                           }));
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());
}

TEST(OfflineGraphEnumeratorTest, ScalarSemanticFingerprintPrunesProofPairs) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult enumeration =
      enumerator.enumerate(makeScalarUniverse(1), schemas, {i32});
  ASSERT_TRUE(enumeration.isComplete());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprinter fingerprinter;
  OfflineSemanticFingerprintResult fingerprints = fingerprinter.fingerprint(
      enumeration.graphs, semantics, normalizer);

  ASSERT_TRUE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.modelCount, 4u);
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 20u);
  EXPECT_EQ(fingerprints.wildcardGraphIds.size(), 0u);
  EXPECT_EQ(fingerprints.comparablePairsBefore, 190u);
  EXPECT_LT(fingerprints.proofCandidatePairsAfter,
            fingerprints.comparablePairsBefore);

  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  // Terminal order is input#0, input#1, constant#0, constant#1. Find the
  // graph for input#0 - constant#0 without relying on candidate order.
  auto identity = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        return graph.operationName == "arith.subi" &&
               graph.operands == SmallVector<unsigned, 3>({0, 2});
      });
  ASSERT_NE(identity, enumeration.graphs.end());
  EXPECT_EQ(fingerprintOf(/*input#0=*/0), fingerprintOf(identity->id));
}

TEST(OfflineGraphEnumeratorTest,
     WildcardDiagnosticsAreBoundedWithoutDroppingCompletenessIds) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs;
  constexpr unsigned graphCount =
      OfflineSemanticFingerprintResult::MaxWildcardGraphDetails + 44;
  graphs.reserve(graphCount);
  for (unsigned id = 0; id < graphCount; ++id) {
    OfflineTypedGraph graph;
    graph.id = id;
    graph.operationName = "test.missing_semantics";
    graph.resultType = i32;
    graph.canonicalKey = (llvm::Twine("missing|") + llvm::Twine(id)).str();
    graphs.push_back(std::move(graph));
  }

  TTOpSemanticsRegistry semantics;
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(graphs, semantics,
                                                   normalizer);

  EXPECT_FALSE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.wildcardGraphIds.size(), graphCount);
  EXPECT_EQ(fingerprints.wildcardGraphs.size(),
            OfflineSemanticFingerprintResult::MaxWildcardGraphDetails);
  std::string report = fingerprints.toJSON("bounded-diagnostics", true);
  EXPECT_NE(report.find("\"wildcard_graph_detail_count\": 256"),
            std::string::npos);
  EXPECT_NE(report.find("\"wildcard_graph_details_truncated\": true"),
            std::string::npos);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsShiftDefinednessWithoutEvaluatingInvalidAmounts) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"ags.input", "arith.shrui"};
  spec.integerConstants.clear();
  OfflineTypeDomain types = buildOfflineTypeDomain(context, spec);

  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, types.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);

  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsNarrowBitVectorBindersModuloTheirWidth) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i8 = IntegerType::get(&context, 8);

  TensorSemanticExpr resultIndex =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/0, i1);
  TensorDenotation tensor;
  tensor.shape = {4};
  tensor.resultIndices = {resultIndex};
  tensor.value = resultIndex;
  tensor.defined = TensorSemanticExpr::boolean(&context, true);

  std::string reason;
  EXPECT_TRUE(OfflineSemanticFingerprinter::fingerprintDenotationRange(
      tensor, /*firstModel=*/0, /*modelCount=*/16, reason))
      << reason;

  TensorSemanticExpr reductionIndex =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/1, i1);
  TensorSemanticExpr widenedIndex = TensorSemanticExpr::operation(
      TensorSemanticKind::ZeroExtend, i8, {reductionIndex});
  TensorDenotation reduction;
  reduction.value = TensorSemanticExpr::operation(
      TensorSemanticKind::Sum, i8,
      {reductionIndex, TensorSemanticExpr::integer(&context, 0, i8),
       TensorSemanticExpr::integer(&context, 4, i8), widenedIndex});
  reduction.defined = TensorSemanticExpr::boolean(&context, true);

  reason.clear();
  EXPECT_TRUE(OfflineSemanticFingerprinter::fingerprintDenotationRange(
      reduction, /*firstModel=*/0, /*modelCount=*/16, reason))
      << reason;
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsSeparateSmallSignedComparisonBoundaries) {
  MLIRContext context;
  Type i1 = IntegerType::get(&context, 1);
  Type i32 = IntegerType::get(&context, 32);
  TensorSemanticExpr value =
      TensorSemanticExpr::symbol(&context, "boundary", i32);
  TensorDenotation lessThanTwo;
  lessThanTwo.value = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1,
      {TensorSemanticExpr::integer(&context, 2, i32), value});
  lessThanTwo.defined = TensorSemanticExpr::boolean(&context, true);
  TensorDenotation lessThanFour = lessThanTwo;
  lessThanFour.value = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, i1,
      {TensorSemanticExpr::integer(&context, 4, i32), value});

  std::string lhsReason;
  std::string rhsReason;
  std::optional<std::string> lhs =
      OfflineSemanticFingerprinter::fingerprintDenotationRange(
          lessThanTwo, /*firstModel=*/0, /*modelCount=*/32, lhsReason);
  std::optional<std::string> rhs =
      OfflineSemanticFingerprinter::fingerprintDenotationRange(
          lessThanFour, /*firstModel=*/0, /*modelCount=*/32, rhsReason);
  ASSERT_TRUE(lhs) << lhsReason;
  ASSERT_TRUE(rhs) << rhsReason;
  EXPECT_NE(*lhs, *rhs);

  TensorSemanticExpr addTwo = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {value, TensorSemanticExpr::integer(&context, 2, i32)});
  TensorSemanticExpr addFour = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32,
      {value, TensorSemanticExpr::integer(&context, 4, i32)});
  TensorDenotation carryTwo = lessThanTwo;
  carryTwo.value = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1,
      {addTwo, TensorSemanticExpr::integer(&context, 2, i32)});
  TensorDenotation carryFour = lessThanTwo;
  carryFour.value = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1,
      {addFour, TensorSemanticExpr::integer(&context, 4, i32)});
  lhsReason.clear();
  rhsReason.clear();
  lhs = OfflineSemanticFingerprinter::fingerprintDenotationRange(
      carryTwo, /*firstModel=*/0, /*modelCount=*/32, lhsReason);
  rhs = OfflineSemanticFingerprinter::fingerprintDenotationRange(
      carryFour, /*firstModel=*/0, /*modelCount=*/32, rhsReason);
  ASSERT_TRUE(lhs) << lhsReason;
  ASSERT_TRUE(rhs) << rhsReason;
  EXPECT_NE(*lhs, *rhs);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsIndependentPrivateBufferShiftOperands) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr index =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/0, i32);
  auto privateElement = [&](int64_t symbol) {
    DictionaryAttr payload = builder.getDictionaryAttr(
        {builder.getNamedAttr("private_buffer", builder.getBoolAttr(true)),
         builder.getNamedAttr("symbol_id",
                              builder.getI64IntegerAttr(symbol))});
    return TensorSemanticExpr::operation(TensorSemanticKind::Element, i32,
                                         {index}, payload);
  };
  auto shifted = [&](int64_t sourceSymbol, int64_t amountSymbol) {
    TensorDenotation result;
    result.shape = {8};
    result.resultIndices = {index};
    TensorSemanticExpr source = privateElement(sourceSymbol);
    TensorSemanticExpr amount = privateElement(amountSymbol);
    result.value = TensorSemanticExpr::operation(
        TensorSemanticKind::ShiftLeft, i32, {source, amount});
    DictionaryAttr guardPayload = builder.getDictionaryAttr(
        {builder.getNamedAttr("semantic_role",
                              builder.getStringAttr("shift_amount_valid"))});
    result.defined = TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyUnary, i1, {amount}, guardPayload);
    return result;
  };

  std::string reason;
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      shifted(/*sourceSymbol=*/1, /*amountSymbol=*/1),
      shifted(/*sourceSymbol=*/1, /*amountSymbol=*/5),
      /*firstModel=*/16, /*additionalModelCount=*/64, reason))
      << reason;
  reason.clear();
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      shifted(/*sourceSymbol=*/1, /*amountSymbol=*/1),
      shifted(/*sourceSymbol=*/5, /*amountSymbol=*/1),
      /*firstModel=*/16, /*additionalModelCount=*/64, reason))
      << reason;
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsNestedInvalidShiftAsUndefined) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  auto shiftAmountValid = [&](TensorSemanticExpr amount) {
    DictionaryAttr payload = builder.getDictionaryAttr(
        {builder.getNamedAttr("semantic_role",
                              builder.getStringAttr("shift_amount_valid"))});
    return TensorSemanticExpr::operation(TensorSemanticKind::ApplyUnary, i1,
                                         {amount}, payload);
  };

  TensorSemanticExpr one = TensorSemanticExpr::integer(&context, 1, i32);
  TensorSemanticExpr invalidAmount =
      TensorSemanticExpr::integer(&context, 32, i32);
  TensorSemanticExpr inner = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i32, {one, invalidAmount});
  TensorSemanticExpr outer = TensorSemanticExpr::operation(
      TensorSemanticKind::ShiftRightLogical, i32, {one, inner});

  TensorDenotation denotation;
  denotation.value = outer;
  denotation.defined = TensorSemanticExpr::operation(
      TensorSemanticKind::And, i1,
      {shiftAmountValid(invalidAmount), shiftAmountValid(inner)});

  std::string reason;
  std::optional<std::string> fingerprint =
      OfflineSemanticFingerprinter::fingerprintDenotationRange(
          denotation, /*firstModel=*/0, /*modelCount=*/1, reason);
  EXPECT_TRUE(fingerprint) << reason;
}

TEST(OfflineGraphEnumeratorTest,
     AdaptiveTailSeparatesExternalAndPrivateHistogramSources) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  TensorSemanticExpr resultIndex =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/0, i32);
  TensorSemanticExpr reductionIndex =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/1, i32);
  Attribute externalPayload = builder.getI64IntegerAttr(0);
  DictionaryAttr privatePayload = builder.getDictionaryAttr(
      {builder.getNamedAttr("private_buffer", builder.getBoolAttr(true)),
       builder.getNamedAttr("symbol_id", builder.getI64IntegerAttr(1))});
  auto histogram = [&](Attribute payload) {
    TensorSemanticExpr element = TensorSemanticExpr::operation(
        TensorSemanticKind::Element, i32, {reductionIndex}, payload);
    TensorSemanticExpr matches = TensorSemanticExpr::operation(
        TensorSemanticKind::Equal, i1, {element, resultIndex});
    TensorSemanticExpr contribution = TensorSemanticExpr::operation(
        TensorSemanticKind::Select, i32,
        {matches, TensorSemanticExpr::integer(&context, 1, i32),
         TensorSemanticExpr::integer(&context, 0, i32)});
    TensorDenotation result;
    result.shape = {8};
    result.resultIndices = {resultIndex};
    result.value = TensorSemanticExpr::operation(
        TensorSemanticKind::Sum, i32,
        {reductionIndex, TensorSemanticExpr::integer(&context, 0, i32),
         TensorSemanticExpr::integer(&context, 4, i32), contribution});
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  TensorDenotation external = histogram(externalPayload);
  TensorDenotation privateBuffer = histogram(privatePayload);
  std::string reason;
  // The first 64 adaptive models deliberately preserve the old six-column
  // collision. The extended de-correlation tail must find the distinction.
  EXPECT_FALSE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      external, privateBuffer, /*firstModel=*/16,
      /*additionalModelCount=*/64, reason));
  reason.clear();
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      external, privateBuffer, /*firstModel=*/16,
      /*additionalModelCount=*/96, reason))
      << reason;
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsCrossTypeSmallConstantEqualities) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i8 = builder.getI8Type();
  Type i16 = builder.getI16Type();
  auto equality = [&](Type type) {
    TensorSemanticExpr input = TensorSemanticExpr::operation(
        TensorSemanticKind::Symbol, type, {}, builder.getI64IntegerAttr(0));
    TensorSemanticExpr four = TensorSemanticExpr::integer(&context, 4, type);
    TensorDenotation result;
    result.value = TensorSemanticExpr::operation(
        TensorSemanticKind::Equal, i1, {input, four});
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  std::string reason;
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      equality(i8), equality(i16), /*firstModel=*/16,
      /*additionalModelCount=*/96, reason))
      << reason;
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsCrossTypeTensorEqualitiesForEverySmallConstant) {
  MLIRContext context;
  Builder builder(&context);
  Type i1 = builder.getI1Type();
  Type i32 = builder.getI32Type();
  SmallVector<Type, 3> inputTypes = {builder.getI16Type(), i32,
                                     builder.getI64Type()};
  TensorSemanticExpr resultIndex =
      TensorSemanticExpr::boundIndex(&context, /*indexId=*/0, i32);
  auto equality = [&](Type inputType, int64_t constant) {
    TensorSemanticExpr input = TensorSemanticExpr::operation(
        TensorSemanticKind::Element, inputType, {resultIndex},
        builder.getI64IntegerAttr(0));
    TensorDenotation result;
    result.shape = {4};
    result.resultIndices = {resultIndex};
    result.value = TensorSemanticExpr::operation(
        TensorSemanticKind::Equal, i1,
        {input, TensorSemanticExpr::integer(&context, constant, inputType)});
    result.defined = TensorSemanticExpr::boolean(&context, true);
    return result;
  };

  for (int64_t constant : {2, 4}) {
    for (unsigned lhs = 0; lhs < inputTypes.size(); ++lhs) {
      for (unsigned rhs = lhs + 1; rhs < inputTypes.size(); ++rhs) {
        std::string reason;
        EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
            equality(inputTypes[lhs], constant),
            equality(inputTypes[rhs], constant), /*firstModel=*/16,
            /*additionalModelCount=*/96, reason))
            << "constant=" << constant << " lhs_type_index=" << lhs
            << " rhs_type_index=" << rhs << " reason=" << reason;
      }
    }
  }
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsNaryDefinednessConjunctionsForGather) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 2;
  spec.bounds.maxInputSymbols = 2;
  spec.catalogDialects = {"tt"};
  spec.operations = {"ags.input", "tt.gather"};
  spec.integerConstants.clear();
  spec.tensorShapes = {{1}, {4}};
  GraphAttributeDomain axis;
  axis.name = "axis";
  GraphAttributeValue zero;
  zero.kind = GraphAttributeValue::Kind::Integer;
  zero.integer = 0;
  axis.values = {zero};
  spec.attributeDomains = {axis};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  ASSERT_TRUE(llvm::any_of(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "tt.gather";
  }));

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/4, /*parallelJobs=*/4)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintModelsCoverSymbolEqualityPartitions) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxInputSymbols = 3;
  spec.operations = {"ags.input", "arith.constant", "arith.cmpi"};
  spec.integerBits = {1, 8, 16, 32, 64};
  spec.integerConstants = {-1, 1};
  GraphAttributeDomain predicate;
  predicate.name = "cmpi_predicate";
  GraphAttributeValue eq;
  eq.kind = GraphAttributeValue::Kind::Integer;
  eq.integer = 0;
  predicate.values = {eq};
  spec.attributeDomains = {predicate};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto input = [&](unsigned width, int64_t symbol) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto id = graph.attributes
                    ? dyn_cast_or_null<IntegerAttr>(
                          graph.attributes.get("ags.symbol_id"))
                    : IntegerAttr{};
      return graph.operationName == "ags.input" &&
             graph.resultType.isInteger(width) && id && id.getInt() == symbol;
    });
  };
  auto x0 = input(/*width=*/8, /*symbol=*/0);
  auto x1 = input(/*width=*/8, /*symbol=*/1);
  auto x2 = input(/*width=*/8, /*symbol=*/2);
  auto y0 = input(/*width=*/16, /*symbol=*/0);
  auto y1 = input(/*width=*/16, /*symbol=*/1);
  auto z0 = input(/*width=*/32, /*symbol=*/0);
  auto z1 = input(/*width=*/32, /*symbol=*/1);
  ASSERT_NE(x0, enumeration.graphs.end());
  ASSERT_NE(x1, enumeration.graphs.end());
  ASSERT_NE(x2, enumeration.graphs.end());
  ASSERT_NE(y0, enumeration.graphs.end());
  ASSERT_NE(y1, enumeration.graphs.end());
  ASSERT_NE(z0, enumeration.graphs.end());
  ASSERT_NE(z1, enumeration.graphs.end());
  auto compare = [&](unsigned lhs, unsigned rhs) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      return graph.operationName == "arith.cmpi" &&
             graph.operands == SmallVector<unsigned, 3>({lhs, rhs});
    });
  };
  auto x0EqX1 = compare(x0->id, x1->id);
  auto x0EqX2 = compare(x0->id, x2->id);
  auto y0EqY1 = compare(y0->id, y1->id);
  ASSERT_NE(x0EqX1, enumeration.graphs.end());
  ASSERT_NE(x0EqX2, enumeration.graphs.end());
  ASSERT_NE(y0EqY1, enumeration.graphs.end());
  auto constant = [&](unsigned width, int64_t expected) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto value = graph.attributes
                       ? dyn_cast_or_null<IntegerAttr>(
                             graph.attributes.get("value"))
                       : IntegerAttr{};
      return graph.operationName == "arith.constant" &&
             graph.resultType.isInteger(width) && value &&
             value.getValue().sextOrTrunc(64).getSExtValue() == expected;
    });
  };
  auto minusOneI8 = constant(/*width=*/8, /*expected=*/-1);
  auto minusOneI16 = constant(/*width=*/16, /*expected=*/-1);
  auto minusOneI32 = constant(/*width=*/32, /*expected=*/-1);
  auto plusOneI16 = constant(/*width=*/16, /*expected=*/1);
  auto plusOneI32 = constant(/*width=*/32, /*expected=*/1);
  ASSERT_NE(minusOneI8, enumeration.graphs.end());
  ASSERT_NE(minusOneI16, enumeration.graphs.end());
  ASSERT_NE(minusOneI32, enumeration.graphs.end());
  ASSERT_NE(plusOneI16, enumeration.graphs.end());
  ASSERT_NE(plusOneI32, enumeration.graphs.end());
  auto minusOneI8EqX0 = compare(minusOneI8->id, x0->id);
  auto minusOneI16EqY0 = compare(minusOneI16->id, y0->id);
  auto minusOneI8EqX1 = compare(minusOneI8->id, x1->id);
  auto minusOneI8EqX2 = compare(minusOneI8->id, x2->id);
  auto minusOneI32EqZ0 = compare(minusOneI32->id, z0->id);
  auto plusOneI16EqY1 = compare(plusOneI16->id, y1->id);
  auto plusOneI32EqZ1 = compare(plusOneI32->id, z1->id);
  ASSERT_NE(minusOneI8EqX0, enumeration.graphs.end());
  ASSERT_NE(minusOneI16EqY0, enumeration.graphs.end());
  ASSERT_NE(minusOneI8EqX1, enumeration.graphs.end());
  ASSERT_NE(minusOneI8EqX2, enumeration.graphs.end());
  ASSERT_NE(minusOneI32EqZ0, enumeration.graphs.end());
  ASSERT_NE(plusOneI16EqY1, enumeration.graphs.end());
  ASSERT_NE(plusOneI32EqZ1, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(x0EqX1->id), fingerprintOf(x0EqX2->id));
  EXPECT_NE(fingerprintOf(x0EqX1->id), fingerprintOf(y0EqY1->id));
  EXPECT_NE(fingerprintOf(minusOneI8EqX0->id),
            fingerprintOf(minusOneI16EqY0->id));
  EXPECT_NE(fingerprintOf(minusOneI8EqX0->id),
            fingerprintOf(minusOneI32EqZ0->id));
  EXPECT_NE(fingerprintOf(minusOneI8EqX1->id),
            fingerprintOf(minusOneI8EqX2->id));
  EXPECT_NE(fingerprintOf(plusOneI16EqY1->id),
            fingerprintOf(plusOneI32EqZ1->id));
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsDistinguishSameOrdinalInputsWithDifferentTypes) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxOperands = 1;
  spec.bounds.maxInputSymbols = 2;
  spec.operations = {"ags.input", "arith.trunci"};
  spec.integerBits = {32, 64};
  spec.integerConstants.clear();

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto input = [&](unsigned width, int64_t symbol) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto id = graph.attributes
                    ? dyn_cast_or_null<IntegerAttr>(
                          graph.attributes.get("ags.symbol_id"))
                    : IntegerAttr{};
      return graph.operationName == "ags.input" &&
             graph.resultType.isInteger(width) && id &&
             id.getInt() == symbol;
    });
  };
  auto inputI32 = input(/*width=*/32, /*symbol=*/1);
  auto inputI64 = input(/*width=*/64, /*symbol=*/1);
  ASSERT_NE(inputI32, enumeration.graphs.end());
  ASSERT_NE(inputI64, enumeration.graphs.end());
  auto truncI64 = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "arith.trunci" &&
           graph.resultType.isInteger(32) &&
           graph.operands == SmallVector<unsigned, 3>({inputI64->id});
  });
  ASSERT_NE(truncI64, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(inputI32->id), fingerprintOf(truncI64->id));
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsCoverIndependentBooleanTruthAssignments) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxOperands = 3;
  spec.bounds.maxInputSymbols = 3;
  spec.operations = {"ags.input", "arith.select"};
  spec.integerBits = {1};
  spec.integerConstants.clear();

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto input = [&](int64_t symbol) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto id = graph.attributes
                    ? dyn_cast_or_null<IntegerAttr>(
                          graph.attributes.get("ags.symbol_id"))
                    : IntegerAttr{};
      return graph.operationName == "ags.input" &&
             graph.resultType.isInteger(1) && id && id.getInt() == symbol;
    });
  };
  auto x0 = input(0);
  auto x1 = input(1);
  auto x2 = input(2);
  ASSERT_NE(x0, enumeration.graphs.end());
  ASSERT_NE(x1, enumeration.graphs.end());
  ASSERT_NE(x2, enumeration.graphs.end());
  auto selected = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "arith.select" &&
           graph.operands ==
               SmallVector<unsigned, 3>({x1->id, x2->id, x0->id});
  });
  ASSERT_NE(selected, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(x0->id), fingerprintOf(selected->id));
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsCoverIndependentBooleanTensorTruthAssignments) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxRank = 2;
  spec.bounds.maxOperands = 3;
  spec.bounds.maxInputSymbols = 3;
  spec.operations = {"ags.input", "arith.select"};
  spec.integerBits = {1};
  spec.integerConstants.clear();
  spec.tensorShapes = {{1, 16}};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto input = [&](int64_t symbol) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto type = dyn_cast<RankedTensorType>(graph.resultType);
      auto id = graph.attributes
                    ? dyn_cast_or_null<IntegerAttr>(
                          graph.attributes.get("ags.symbol_id"))
                    : IntegerAttr{};
      return graph.operationName == "ags.input" && type &&
             type.getShape() == ArrayRef<int64_t>({1, 16}) &&
             type.getElementType().isInteger(1) && id &&
             id.getInt() == symbol;
    });
  };
  auto x0 = input(0);
  auto x1 = input(1);
  auto x2 = input(2);
  ASSERT_NE(x0, enumeration.graphs.end());
  ASSERT_NE(x1, enumeration.graphs.end());
  ASSERT_NE(x2, enumeration.graphs.end());
  auto selected = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "arith.select" &&
           graph.resultType == x0->resultType &&
           graph.operands ==
               SmallVector<unsigned, 3>({x1->id, x2->id, x0->id});
  });
  ASSERT_NE(selected, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/32)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(x0->id), fingerprintOf(selected->id));
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsDistinguishScalarFromSingletonTensorElement) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 1;
  spec.bounds.maxInputSymbols = 2;
  spec.bounds.allowControlFlow = true;
  spec.operations = {"ags.input", "tt.reduce"};
  spec.integerBits = {1};
  spec.integerConstants.clear();
  spec.tensorShapes = {{1}};
  GraphAttributeDomain axis;
  axis.name = "axis";
  GraphAttributeValue zero;
  zero.kind = GraphAttributeValue::Kind::Integer;
  zero.integer = 0;
  axis.values = {zero};
  spec.attributeDomains = {axis};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto scalar = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    auto id = graph.attributes
                  ? dyn_cast_or_null<IntegerAttr>(
                        graph.attributes.get("ags.symbol_id"))
                  : IntegerAttr{};
    return graph.operationName == "ags.input" &&
           graph.resultType.isInteger(1) && id && id.getInt() == 1;
  });
  auto tensor = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    auto type = dyn_cast<RankedTensorType>(graph.resultType);
    auto id = graph.attributes
                  ? dyn_cast_or_null<IntegerAttr>(
                        graph.attributes.get("ags.symbol_id"))
                  : IntegerAttr{};
    return graph.operationName == "ags.input" && type &&
           type.getRank() == 1 && type.getDimSize(0) == 1 &&
           type.getElementType().isInteger(1) && id && id.getInt() == 1;
  });
  ASSERT_NE(scalar, enumeration.graphs.end());
  ASSERT_NE(tensor, enumeration.graphs.end());
  auto reduced = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "tt.reduce" &&
           graph.resultType.isInteger(1) &&
           graph.operands == SmallVector<unsigned, 3>({tensor->id});
  });
  ASSERT_NE(reduced, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(scalar->id), fingerprintOf(reduced->id));
}

TEST(OfflineGraphEnumeratorTest, ProvesEveryFingerprintCandidateTriState) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      makeScalarUniverse(1), schemas, {i32});
  ASSERT_TRUE(enumeration.isComplete());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                 normalizer);
  OfflineSemanticFingerprintResult parallelFingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/4, /*parallelJobs=*/4)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_EQ(parallelFingerprints.buckets.size(), fingerprints.buckets.size());
  EXPECT_EQ(parallelFingerprints.wildcardGraphIds,
            fingerprints.wildcardGraphIds);
  EXPECT_EQ(parallelFingerprints.proofCandidatePairsAfter,
            fingerprints.proofCandidatePairsAfter);
  EXPECT_EQ(parallelFingerprints.denotationModelEvaluations,
            fingerprints.denotationModelEvaluations);
  EXPECT_EQ(parallelFingerprints.denotationModelEvaluationsAvoided,
            fingerprints.denotationModelEvaluationsAvoided);
  EXPECT_EQ(parallelFingerprints.exactDenotationGroupIds,
            fingerprints.exactDenotationGroupIds);
  EXPECT_EQ(parallelFingerprints.parallelJobs, 4u);
  for (auto [index, bucket] : llvm::enumerate(fingerprints.buckets)) {
    EXPECT_EQ(parallelFingerprints.buckets[index].fingerprint,
              bucket.fingerprint);
    EXPECT_EQ(parallelFingerprints.buckets[index].graphIds, bucket.graphIds);
  }
  EXPECT_EQ(fingerprints.uniqueDenotationsFingerprinted +
                fingerprints.denotationFingerprintCacheHits,
            enumeration.graphs.size());
  EXPECT_GT(fingerprints.denotationFingerprintCacheHits, 0u);
  EXPECT_GT(fingerprints.denotationModelEvaluationsAvoided, 0u);
  EXPECT_LT(fingerprints.denotationModelEvaluations,
            fingerprints.uniqueDenotationsFingerprinted *
                fingerprints.modelCount);

  // Adaptive evaluation must produce exactly the same partition as eagerly
  // evaluating the full configured model range. A differing concrete model is
  // a sound negative edge; the adaptive tree may stop only after that edge has
  // already separated the two denotations.
  std::vector<unsigned> bucketOf(enumeration.graphs.size(), UINT_MAX);
  for (auto [bucketId, bucket] : llvm::enumerate(fingerprints.buckets))
    for (unsigned graphId : bucket.graphIds)
      bucketOf[graphId] = bucketId;
  for (unsigned lhs = 0; lhs < enumeration.graphs.size(); ++lhs) {
    ASSERT_TRUE(fingerprints.denotations[lhs]);
    ASSERT_NE(bucketOf[lhs], UINT_MAX);
    for (unsigned rhs = lhs + 1; rhs < enumeration.graphs.size(); ++rhs) {
      ASSERT_TRUE(fingerprints.denotations[rhs]);
      ASSERT_NE(bucketOf[rhs], UINT_MAX);
      std::string reason;
      bool eagerModelsSeparate =
          OfflineSemanticFingerprinter::hasConcreteCounterexample(
              *fingerprints.denotations[lhs],
              *fingerprints.denotations[rhs], /*firstModel=*/0,
              fingerprints.modelCount, reason);
      EXPECT_EQ(bucketOf[lhs] != bucketOf[rhs], eagerModelsSeparate)
          << "lhs=" << lhs << " rhs=" << rhs;
    }
  }
  SemanticEquivalenceOracle oracle(normalizer);
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      enumeration.graphs, fingerprints, oracle);

  EXPECT_EQ(proofs.pairs.size(), fingerprints.proofCandidatePairsAfter);
  EXPECT_EQ(proofs.candidatePairs, fingerprints.proofCandidatePairsAfter);
  EXPECT_EQ(proofs.processedPairs, fingerprints.proofCandidatePairsAfter);
  EXPECT_EQ(proofs.provedPairs + proofs.refutedPairs + proofs.unknownPairs,
            proofs.pairs.size());
  EXPECT_GT(proofs.provedPairs, 0u);
  for (const OfflineSemanticProofPair &pair : proofs.pairs)
    if (pair.status == SemanticProofStatus::Proved)
      EXPECT_FALSE(pair.proofHash.empty());
}

TEST(OfflineGraphEnumeratorTest,
     PrunesProofPairsAlreadyConnectedByReplayableEdges) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      makeScalarUniverse(1), schemas, {i32});
  ASSERT_TRUE(enumeration.isComplete());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  SemanticEquivalenceOracle oracle(normalizer);
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      enumeration.graphs, fingerprints, oracle, options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.candidatePairs, fingerprints.proofCandidatePairsAfter);
  EXPECT_EQ(proofs.processedPairs, fingerprints.proofCandidatePairsAfter);
  EXPECT_GT(proofs.transitivelyProvedPairs, 0u);
  EXPECT_EQ(proofs.provedPairs + proofs.transitivelyProvedPairs +
                proofs.refutedPairs + proofs.unknownPairs,
            proofs.processedPairs);
  EXPECT_EQ(proofs.pairs.size(),
            proofs.provedPairs + proofs.refutedPairs + proofs.unknownPairs);
  for (const OfflineSemanticProofPair &pair : proofs.pairs) {
    if (pair.status != SemanticProofStatus::Proved)
      continue;
    std::string decodeError;
    std::optional<SemanticEquivalenceProof> decoded =
        SemanticEquivalenceProof::fromJSON(pair.getProofCertificate(), context,
                                           decodeError);
    ASSERT_TRUE(decoded) << decodeError;
    SemanticProofResult replay = oracle.replay(*decoded);
    ASSERT_TRUE(replay.isProved()) << replay.reason;
    ASSERT_TRUE(replay.proof);
    EXPECT_EQ(replay.proof->getStableHash(), pair.proofHash);
  }
}

TEST(OfflineGraphEnumeratorTest,
     ProofCancellationCannotMasqueradeAsComplete) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  Type i32 = IntegerType::get(&context, 32);
  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      makeScalarUniverse(1), schemas, {i32});
  ASSERT_TRUE(enumeration.isComplete());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  OfflineEquivalenceProofOptions options;
  unsigned polls = 0;
  options.shouldStop = [&]() { return ++polls == 1; };
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      enumeration.graphs, fingerprints,
      SemanticEquivalenceOracle(normalizer), options);

  EXPECT_TRUE(proofs.interrupted);
  EXPECT_FALSE(proofs.isComplete());
  EXPECT_EQ(polls, 1u);
  EXPECT_NE(proofs.toJSON("test-universe", true, true)
                .find("\"interrupted\": true"),
            std::string::npos);
}

TEST(OfflineGraphEnumeratorTest,
     CachedCounterexampleFingerprintRunsBeforeSymbolicOracle) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  OfflineTypedGraph lhsGraph;
  lhsGraph.id = 0;
  lhsGraph.resultType = i32;
  lhsGraph.canonicalKey = "lhs";
  OfflineTypedGraph rhsGraph = lhsGraph;
  rhsGraph.id = 1;
  rhsGraph.canonicalKey = "rhs";
  std::vector<OfflineTypedGraph> graphs = {lhsGraph, rhsGraph};

  auto constantDenotation = [&](int64_t value) {
    TensorDenotation denotation;
    denotation.value =
        TensorSemanticExpr::integer(&context, value, i32);
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    return denotation;
  };
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"primary-collision", {0, 1}});
  fingerprints.denotations = {constantDenotation(0), constantDenotation(1)};
  fingerprints.exactDenotationGroupIds = {0, 1};
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;
  fingerprints.modelCount = 0;

  OfflineEquivalenceProofOptions options;
  options.additionalCounterexampleModels = 1;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.counterexampleRefutedPairs, 1u);
  EXPECT_EQ(proofs.refutedPairs, 1u);
  EXPECT_EQ(proofs.oracleComparisons, 0u);
}

TEST(OfflineGraphEnumeratorTest,
     SemanticQuotientPartitionsClassProductsByConcreteSignature) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs(4);
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"primary-collision", {0, 1, 2, 3}});
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "constant-" + std::to_string(graphId);
    TensorDenotation denotation;
    denotation.value =
        TensorSemanticExpr::integer(&context, graphId, i32);
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    fingerprints.denotations.push_back(std::move(denotation));
    fingerprints.exactDenotationGroupIds.push_back(graphId);
  }
  fingerprints.comparablePairsBefore = 6;
  fingerprints.proofCandidatePairsAfter = 6;
  fingerprints.modelCount = 0;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  options.additionalCounterexampleModels = 8;
  options.counterexampleJobs = 4;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);
  OfflineEquivalenceProofOptions serialOptions = options;
  serialOptions.counterexampleJobs = 1;
  OfflineEquivalenceProofResult serialProofs =
      OfflineEquivalenceProver().prove(
          graphs, fingerprints, SemanticEquivalenceOracle(normalizer),
          serialOptions);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_TRUE(serialProofs.isComplete());
  EXPECT_TRUE(proofs.usedSemanticQuotient);
  EXPECT_EQ(proofs.semanticQuotientGroups, 4u);
  EXPECT_EQ(proofs.semanticQuotientClasses, 4u);
  EXPECT_EQ(proofs.counterexampleJobs, 4u);
  EXPECT_EQ(proofs.counterexampleModelEvaluations, 4u);
  EXPECT_EQ(proofs.counterexampleModelEvaluationsAvoided, 28u);
  EXPECT_EQ(proofs.processedPairs, 6u);
  EXPECT_EQ(proofs.counterexampleRefutedPairs, 6u);
  EXPECT_EQ(proofs.refutedPairs, 6u);
  EXPECT_EQ(proofs.oracleComparisons, 0u);
  EXPECT_TRUE(proofs.pairs.empty());
  EXPECT_EQ(proofs.processedPairs, serialProofs.processedPairs);
  EXPECT_EQ(proofs.refutedPairs, serialProofs.refutedPairs);
  EXPECT_EQ(proofs.counterexampleRefutedPairs,
            serialProofs.counterexampleRefutedPairs);
  EXPECT_EQ(proofs.counterexampleModelEvaluations,
            serialProofs.counterexampleModelEvaluations);
  EXPECT_EQ(proofs.counterexampleModelEvaluationsAvoided,
            serialProofs.counterexampleModelEvaluationsAvoided);
}

TEST(OfflineGraphEnumeratorTest,
     SemanticQuotientSkipsCounterexampleModelsForOneExactGroup) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs(2);
  TensorDenotation denotation;
  denotation.value = TensorSemanticExpr::integer(&context, 7, i32);
  denotation.defined = TensorSemanticExpr::boolean(&context, true);

  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"one-exact-group", {0, 1}});
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "same-" + std::to_string(graphId);
    fingerprints.denotations.push_back(denotation);
    fingerprints.exactDenotationGroupIds.push_back(0);
  }
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;
  fingerprints.modelCount = 16;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  options.additionalCounterexampleModels = 96;
  options.counterexampleJobs = 4;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.provedPairs, 1u);
  EXPECT_EQ(proofs.counterexampleModelEvaluations, 0u);
  EXPECT_EQ(proofs.counterexampleModelEvaluationsAvoided, 0u);
}

TEST(OfflineGraphEnumeratorTest,
     SemanticQuotientMemoizesFinalUnknownClassification) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs(2);
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"unknown-collision", {0, 1}});
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "unknown-" + std::to_string(graphId);
    TensorDenotation denotation;
    denotation.value =
        TensorSemanticExpr::integer(&context, graphId, i32);
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    fingerprints.denotations.push_back(std::move(denotation));
    fingerprints.exactDenotationGroupIds.push_back(graphId);
  }
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);

  EXPECT_FALSE(proofs.isComplete());
  EXPECT_EQ(proofs.processedPairs, 1u);
  EXPECT_EQ(proofs.unknownPairs, 1u);
  EXPECT_EQ(proofs.pairs.size(), 1u);
  EXPECT_EQ(proofs.oracleComparisons, 1u);
}

TEST(OfflineGraphEnumeratorTest,
     SymbolicPremergeSkipsModelsForProvedEquivalentExactGroups) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs(2);
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"symbolic-collision", {0, 1}});
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "equivalent-" + std::to_string(graphId);
    TensorDenotation denotation;
    denotation.value = TensorSemanticExpr::integer(&context, 7, i32);
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    fingerprints.denotations.push_back(std::move(denotation));
    // Deliberately model two exact-normal-form groups. The symbolic oracle is
    // the authority that may coalesce them before concrete refinement.
    fingerprints.exactDenotationGroupIds.push_back(graphId);
  }
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;
  fingerprints.modelCount = 16;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  options.additionalCounterexampleModels = 96;
  options.counterexampleJobs = 4;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.symbolicPremergeComparisons, 1u);
  EXPECT_EQ(proofs.symbolicPremergeMergedGroups, 1u);
  EXPECT_EQ(proofs.counterexampleModelEvaluations, 0u);
  EXPECT_EQ(proofs.provedPairs, 1u);
}

TEST(OfflineGraphEnumeratorTest,
     SymbolicPremergeSkipsModelsForFullyRefutedBucket) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  std::vector<OfflineTypedGraph> graphs(2);
  OfflineSemanticFingerprintResult fingerprints;
  fingerprints.buckets.push_back({"refuted-collision", {0, 1}});
  for (unsigned graphId = 0; graphId < graphs.size(); ++graphId) {
    graphs[graphId].id = graphId;
    graphs[graphId].resultType = i32;
    graphs[graphId].canonicalKey = "refuted-" + std::to_string(graphId);
    TensorDenotation denotation;
    if (graphId == 1)
      denotation.shape.push_back(1);
    denotation.value = TensorSemanticExpr::integer(&context, 7, i32);
    denotation.defined = TensorSemanticExpr::boolean(&context, true);
    fingerprints.denotations.push_back(std::move(denotation));
    fingerprints.exactDenotationGroupIds.push_back(graphId);
  }
  fingerprints.comparablePairsBefore = 1;
  fingerprints.proofCandidatePairsAfter = 1;
  fingerprints.modelCount = 16;

  OfflineEquivalenceProofOptions options;
  options.pruneTransitivelyImpliedPairs = true;
  options.retainRefutedPairs = false;
  options.additionalCounterexampleModels = 96;
  options.counterexampleJobs = 4;
  TensorSemanticNormalizer normalizer;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      graphs, fingerprints, SemanticEquivalenceOracle(normalizer), options);

  EXPECT_TRUE(proofs.isComplete());
  EXPECT_EQ(proofs.symbolicPremergeComparisons, 1u);
  EXPECT_EQ(proofs.symbolicPremergeMergedGroups, 0u);
  EXPECT_EQ(proofs.counterexampleModelEvaluations, 0u);
  EXPECT_EQ(proofs.refutedPairs, 1u);
  EXPECT_EQ(proofs.counterexampleRefutedPairs, 0u);
  EXPECT_EQ(proofs.oracleComparisons, 0u);
}

TEST(OfflineGraphEnumeratorTest, FingerprintsFloatingBoundaryInputsByBits) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations = {"ags.input"};
  spec.integerBits.clear();
  spec.integerConstants.clear();
  spec.floatingBits = {32};
  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult enumeration =
      enumerator.enumerate(spec, /*schemas=*/{}, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete());
  ASSERT_EQ(enumeration.graphs.size(), 2u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 2u);
  EXPECT_EQ(fingerprints.comparablePairsBefore, 1u);
  EXPECT_EQ(fingerprints.proofCandidatePairsAfter, 0u);
}

TEST(OfflineGraphEnumeratorTest, FingerprintsDistinctI1BoundaryInputs) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(0);
  spec.bounds.maxInputSymbols = 3;
  spec.operations = {"ags.input"};
  spec.integerBits = {1};
  spec.integerConstants.clear();

  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      spec, /*schemas=*/{}, {IntegerType::get(&context, 1)});
  ASSERT_TRUE(enumeration.isComplete());
  ASSERT_EQ(enumeration.graphs.size(), 3u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.comparablePairsBefore, 3u);
  EXPECT_EQ(fingerprints.proofCandidatePairsAfter, 0u);
}

TEST(OfflineGraphEnumeratorTest, ExhaustsFiniteStaticTensorTypeDomain) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 2;
  spec.tensorShapes = {{4}};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(typeDomain.unsupportedUniverseFeatures.empty());
  ASSERT_EQ(typeDomain.types.size(), 2u);
  EXPECT_TRUE(isa<IntegerType>(typeDomain.types[0]));
  EXPECT_EQ(typeDomain.types[1],
            RankedTensorType::get({4}, IntegerType::get(&context, 32)));

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult result =
      enumerator.enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(result.isComplete());
  ASSERT_EQ(result.graphCountBySize.size(), 2u);
  EXPECT_EQ(result.graphCountBySize[0], 8u);
  EXPECT_EQ(result.graphCountBySize[1], 32u);
  EXPECT_EQ(result.graphs.size(), 40u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(result.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 40u);
  EXPECT_EQ(fingerprints.comparablePairsBefore, 380u);
  EXPECT_LT(fingerprints.proofCandidatePairsAfter,
            fingerprints.comparablePairsBefore);
}

TEST(OfflineGraphEnumeratorTest, UsesExactBoundaryInputSignature) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"ags.input"};
  spec.integerBits = {32, 64};
  spec.integerConstants.clear();
  spec.inputSymbolTypes = {"i64", "i32"};
  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, /*schemas=*/{}, typeDomain.types);
  ASSERT_TRUE(result.isComplete()) << result.toJSON(spec.getStableHash());
  ASSERT_EQ(result.graphs.size(), 2u);
  EXPECT_TRUE(result.graphs[0].resultType.isInteger(64));
  EXPECT_EQ(result.graphs[0].attributes.getAs<IntegerAttr>("ags.symbol_id")
                .getInt(),
            0);
  EXPECT_TRUE(result.graphs[1].resultType.isInteger(32));
  EXPECT_EQ(result.graphs[1].attributes.getAs<IntegerAttr>("ags.symbol_id")
                .getInt(),
            1);
}

TEST(OfflineGraphEnumeratorTest, RejectsInputSymbolTypeOutsideFiniteDomain) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"ags.input"};
  spec.inputSymbolTypes = {"f32"};
  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, /*schemas=*/{}, typeDomain.types);
  EXPECT_FALSE(result.isComplete());
  EXPECT_TRUE(llvm::is_contained(
      result.unsupportedUniverseFeatures,
      std::string(
          "input symbol type is outside the finite type domain: f32")));
}

TEST(OfflineGraphEnumeratorTest, BoundsDistinctEffectfulNodesInCandidateDag) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/2);
  spec.bounds.allowEffects = true;
  spec.bounds.maxEffectfulNodes = 1;
  spec.operations = {"ags.input", "test.read"};
  spec.integerConstants.clear();

  TTIRCandidateOpSchema read;
  read.operationName = "test.read";
  read.operandCount = 1;
  read.effectKind = TTIRCandidateEffectKind::Read;
  read.constructible = true;
  read.canProduceResult =
      [](Type resultType, DictionaryAttr) { return resultType.isInteger(32); };
  read.isWellTyped = [](Type resultType, ArrayRef<Type> operandTypes,
                        DictionaryAttr) {
    return resultType.isInteger(32) && operandTypes.size() == 1 &&
           operandTypes.front().isInteger(32);
  };

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, {read}, {i32});
  ASSERT_TRUE(result.isComplete()) << result.toJSON(spec.getStableHash());
  ASSERT_EQ(result.graphCountBySize.size(), 3u);
  EXPECT_EQ(result.graphCountBySize[0], 2u);
  EXPECT_EQ(result.graphCountBySize[1], 2u);
  EXPECT_EQ(result.graphCountBySize[2], 0u);
}

TEST(OfflineGraphEnumeratorTest, ExhaustsExplicitFiniteTensorEncodings) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxInputSymbols = 3;
  spec.tensorShapes = {{4}};
  spec.tensorEncodings = {"\"layout-a\"", "\"layout-b\""};
  spec.semanticDomains.push_back(GraphSemanticDomain::Layout);
  llvm::sort(spec.semanticDomains);
  std::string error;
  ASSERT_TRUE(spec.verify(error)) << error;

  OfflineTypeDomain domain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(domain.unsupportedUniverseFeatures.empty());
  // One scalar i32 plus the two explicitly encoded tensor<4xi32> types.
  ASSERT_EQ(domain.types.size(), 3u);
  auto first = dyn_cast<RankedTensorType>(domain.types[1]);
  auto second = dyn_cast<RankedTensorType>(domain.types[2]);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(cast<StringAttr>(first.getEncoding()).getValue(), "layout-a");
  EXPECT_EQ(cast<StringAttr>(second.getEncoding()).getValue(), "layout-b");
}

TEST(OfflineGraphEnumeratorTest, BuildsExplicitFiniteIndexTypeDomain) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.includeIndexType = true;
  spec.bounds.maxRank = 1;
  spec.tensorShapes = {{4}};

  OfflineTypeDomain domain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(domain.unsupportedUniverseFeatures.empty());
  ASSERT_EQ(domain.types.size(), 4u);
  EXPECT_TRUE(isa<IntegerType>(domain.types[0]));
  EXPECT_TRUE(isa<IndexType>(domain.types[1]));
  auto integerTensor = dyn_cast<RankedTensorType>(domain.types[2]);
  auto indexTensor = dyn_cast<RankedTensorType>(domain.types[3]);
  ASSERT_TRUE(integerTensor);
  ASSERT_TRUE(indexTensor);
  EXPECT_TRUE(isa<IntegerType>(integerTensor.getElementType()));
  EXPECT_TRUE(isa<IndexType>(indexTensor.getElementType()));
}

TEST(OfflineGraphEnumeratorTest, ParsesExplicitStageAndTupleTypes) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.explicitTypes = {"memref<4xi32>", "tuple<i32, i1>"};

  OfflineTypeDomain domain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(domain.unsupportedUniverseFeatures.empty());
  ASSERT_EQ(domain.types.size(), 3u);
  EXPECT_TRUE(isa<IntegerType>(domain.types[0]));
  EXPECT_TRUE(isa<MemRefType>(domain.types[1]));
  EXPECT_TRUE(isa<TupleType>(domain.types[2]));
}

TEST(OfflineGraphEnumeratorTest, ReportsInvalidExplicitType) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.explicitTypes = {"not-a-type"};

  OfflineTypeDomain domain = buildOfflineTypeDomain(context, spec);
  ASSERT_EQ(domain.unsupportedUniverseFeatures.size(), 1u);
  EXPECT_NE(domain.unsupportedUniverseFeatures.front().find(
                "cannot parse explicit MLIR type"),
            std::string::npos);
}

TEST(OfflineGraphEnumeratorTest, ExhaustsFiniteDynamicTensorTypeDomain) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxShapeSymbols = 1;
  spec.bounds.allowDynamicShapes = true;
  spec.tensorShapes = {{GraphUniverseSpec::kDynamicDimension}};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(typeDomain.unsupportedUniverseFeatures.empty());
  ASSERT_EQ(typeDomain.types.size(), 2u);
  EXPECT_EQ(typeDomain.types[1],
            RankedTensorType::get({ShapedType::kDynamic},
                                  IntegerType::get(&context, 32)));

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult result =
      enumerator.enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(result.isComplete());
  ASSERT_EQ(result.graphCountBySize.size(), 2u);
  // The dynamic tensor contributes two boundary inputs but no DenseElements
  // constants, because arith.constant tensors require a static shape.
  EXPECT_EQ(result.graphCountBySize[0], 6u);
  EXPECT_EQ(result.graphCountBySize[1], 20u);
  EXPECT_EQ(result.graphs.size(), 26u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(result.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 26u);
  EXPECT_LT(fingerprints.proofCandidatePairsAfter,
            fingerprints.comparablePairsBefore);
}

TEST(OfflineGraphEnumeratorTest, BuildsExplicitFinitePointerTypeDomain) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.tensorShapes = {{4}};
  spec.pointerAddressSpaces = {1};

  OfflineTypeDomain domain = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(domain.unsupportedUniverseFeatures.empty());
  ASSERT_EQ(domain.types.size(), 4u);
  Type i32 = IntegerType::get(&context, 32);
  Type pointer = triton::PointerType::get(i32, 1);
  EXPECT_EQ(domain.types[0], i32);
  EXPECT_EQ(domain.types[1], pointer);
  EXPECT_EQ(domain.types[2], RankedTensorType::get({4}, i32));
  EXPECT_EQ(domain.types[3], RankedTensorType::get({4}, pointer));
}

TEST(OfflineGraphEnumeratorTest, EnumeratesLoadFromDeclarativeSchema) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 4;
  spec.bounds.allowEffects = true;
  spec.operations = {"ags.input", "ags.memory", "tt.load", "tt.store"};
  spec.integerBits = {1, 32};
  spec.integerConstants.clear();
  spec.tensorShapes = {{4}};
  spec.pointerAddressSpaces = {1};
  auto segmentDomain = [](StringRef name, ArrayRef<int32_t> segments) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::IntegerArray;
    value.integerArray.assign(segments.begin(), segments.end());
    domain.values.push_back(std::move(value));
    return domain;
  };
  spec.attributeDomains = {
      segmentDomain("load_segments_unmasked", {1, 0, 0}),
      segmentDomain("load_segments_masked", {1, 1, 0}),
      segmentDomain("load_segments_other", {1, 1, 1})};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, typeDomain.types);
  ASSERT_TRUE(result.isComplete());
  EXPECT_TRUE(llvm::any_of(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "tt.load" && graph.operands.size() == 2;
  }));
  EXPECT_TRUE(llvm::any_of(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "tt.load" && graph.operands.size() == 3;
  }));
  EXPECT_TRUE(llvm::any_of(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "tt.store" && isa<NoneType>(graph.resultType) &&
           (graph.operands.size() == 3 || graph.operands.size() == 4);
  }));

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(result.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
}

TEST(OfflineGraphEnumeratorTest, MemoryEffectsCannotSilentlyLackStateDomain) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations = {"ags.input", "tt.load"};
  spec.pointerAddressSpaces = {1};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, typeDomain.types);
  EXPECT_FALSE(result.isComplete());
  EXPECT_TRUE(llvm::any_of(
      result.unsupportedUniverseFeatures, [](const std::string &message) {
        return message.find("allow_effects=true") != std::string::npos;
      }));
  EXPECT_TRUE(llvm::any_of(
      result.unsupportedUniverseFeatures, [](const std::string &message) {
        return message.find("ags.memory terminal") != std::string::npos;
      }));
}

TEST(OfflineGraphEnumeratorTest,
     MemoryProjectionOnlyEnumeratesStatefulValueProducers) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(2);
  spec.bounds.maxOperands = 4;
  spec.bounds.allowEffects = true;
  spec.operations = {"ags.input", "ags.memory", "ags.memory_after",
                     "tt.atomic_rmw"};
  spec.integerBits = {1, 32};
  spec.integerConstants.clear();
  spec.pointerAddressSpaces = {1};
  auto integerDomain = [](StringRef name, int64_t integer) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::Integer;
    value.integer = integer;
    domain.values = {value};
    return domain;
  };
  spec.attributeDomains = {integerDomain("atomic_rmw_op", 4),
                           integerDomain("memory_scope", 1),
                           integerDomain("memory_semantic", 1)};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, typeDomain.types);
  ASSERT_TRUE(result.isComplete());
  bool foundProjection = false;
  for (const OfflineTypedGraph &graph : result.graphs) {
    if (graph.operationName != "ags.memory_after")
      continue;
    foundProjection = true;
    ASSERT_EQ(graph.operands.size(), 1u);
    EXPECT_EQ(result.graphs[graph.operands.front()].effectKind,
              TTIRCandidateEffectKind::Atomic);
  }
  EXPECT_TRUE(foundProjection);
}

TEST(OfflineGraphEnumeratorTest,
     AtomicEffectHistoryIsSeparatedBeforeExactProof) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/2);
  spec.bounds.maxOperands = 4;
  spec.bounds.maxInputSymbols = 3;
  spec.bounds.allowEffects = true;
  spec.operations = {"ags.input", "ags.memory", "ags.memory_after",
                     "arith.constant", "tt.atomic_cas", "tt.atomic_rmw"};
  spec.integerBits.clear();
  spec.integerConstants = {0, 1};
  spec.explicitTypes = {"i1", "i32", "!tt.ptr<i32, 1>"};
  auto integerDomain = [](StringRef name, int64_t integer) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::Integer;
    value.integer = integer;
    domain.values = {value};
    return domain;
  };
  spec.attributeDomains = {integerDomain("atomic_rmw_op", 4),
                           integerDomain("memory_scope", 1),
                           integerDomain("memory_semantic", 4)};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  SemanticEquivalenceOracle oracle(normalizer);
  OfflineEquivalenceProofOptions proofOptions;
  proofOptions.retainRefutedPairs = false;
  proofOptions.retainGraphKeys = false;
  proofOptions.pruneTransitivelyImpliedPairs = true;
  proofOptions.additionalCounterexampleModels = 32;
  proofOptions.counterexampleJobs = 4;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      enumeration.graphs, fingerprints, oracle, proofOptions);

  EXPECT_TRUE(proofs.isComplete()) << proofs.toJSON(
      spec.getStableHash(), enumeration.isComplete(),
      fingerprints.isComplete());
  EXPECT_EQ(proofs.unknownPairs, 0u);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsHistogramSumAndForAllDenotations) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  graphOptions.compileOn91095 = true;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);

  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxInputSymbols = 3;
  spec.operations = {"ags.input", "arith.constant", "tt.histogram"};
  spec.integerBits = {1, 32};
  spec.integerConstants = {-1, 1};
  spec.tensorShapes = {{1}, {4}, {16}};
  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult enumeration =
      enumerator.enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete());

  auto source = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto symbol = graph.attributes
                          ? dyn_cast_or_null<IntegerAttr>(
                                graph.attributes.get("ags.symbol_id"))
                          : IntegerAttr{};
        return graph.operationName == "ags.input" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 4 &&
               tensor.getElementType().isInteger(32) && symbol &&
               symbol.getInt() == 0;
      });
  auto otherSource = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto symbol = graph.attributes
                          ? dyn_cast_or_null<IntegerAttr>(
                                graph.attributes.get("ags.symbol_id"))
                          : IntegerAttr{};
        return graph.operationName == "ags.input" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 4 &&
               tensor.getElementType().isInteger(32) && symbol &&
               symbol.getInt() == 1;
      });
  auto unitSource = [&](int64_t symbolId) {
    return llvm::find_if(
        enumeration.graphs, [&](const OfflineTypedGraph &graph) {
          auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
          auto symbol = graph.attributes
                            ? dyn_cast_or_null<IntegerAttr>(
                                  graph.attributes.get("ags.symbol_id"))
                            : IntegerAttr{};
          return graph.operationName == "ags.input" && tensor &&
                 tensor.getRank() == 1 && tensor.getDimSize(0) == 1 &&
                 tensor.getElementType().isInteger(32) && symbol &&
                 symbol.getInt() == symbolId;
        });
  };
  auto unitSource0 = unitSource(0);
  auto unitSource1 = unitSource(1);
  auto trueMask = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto dense = graph.attributes
                         ? dyn_cast_or_null<DenseElementsAttr>(
                               graph.attributes.get("value"))
                         : DenseElementsAttr{};
        return graph.operationName == "arith.constant" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 4 &&
               tensor.getElementType().isInteger(1) && dense &&
               dense.isSplat() &&
               !dense.getSplatValue<IntegerAttr>().getValue().isZero();
      });
  auto inputMask = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto symbol = graph.attributes
                          ? dyn_cast_or_null<IntegerAttr>(
                                graph.attributes.get("ags.symbol_id"))
                          : IntegerAttr{};
        return graph.operationName == "ags.input" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 4 &&
               tensor.getElementType().isInteger(1) && symbol &&
               symbol.getInt() == 0;
      });
  auto findUnitMask = [&](int64_t symbolId) {
    return llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto symbol = graph.attributes
                          ? dyn_cast_or_null<IntegerAttr>(
                                graph.attributes.get("ags.symbol_id"))
                          : IntegerAttr{};
        return graph.operationName == "ags.input" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 1 &&
               tensor.getElementType().isInteger(1) && symbol &&
               symbol.getInt() == symbolId;
      });
  };
  auto unitMask = findUnitMask(0);
  auto unitMask2 = findUnitMask(2);
  auto unitConstant = [&](bool allOnes) {
    return llvm::find_if(
        enumeration.graphs, [&](const OfflineTypedGraph &graph) {
          auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
          auto dense = graph.attributes
                           ? dyn_cast_or_null<DenseElementsAttr>(
                                 graph.attributes.get("value"))
                           : DenseElementsAttr{};
          if (graph.operationName != "arith.constant" || !tensor ||
              tensor.getRank() != 1 || tensor.getDimSize(0) != 1 ||
              !tensor.getElementType().isInteger(32) || !dense ||
              !dense.isSplat())
            return false;
          APInt value = dense.getSplatValue<IntegerAttr>().getValue();
          return allOnes ? value.isAllOnes() : value.isOne();
        });
  };
  auto unitOne = unitConstant(/*allOnes=*/false);
  auto unitMinusOne = unitConstant(/*allOnes=*/true);
  auto negativeSource = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        auto dense = graph.attributes
                         ? dyn_cast_or_null<DenseElementsAttr>(
                               graph.attributes.get("value"))
                         : DenseElementsAttr{};
        return graph.operationName == "arith.constant" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 4 &&
               tensor.getElementType().isInteger(32) && dense &&
               dense.isSplat() &&
               dense.getSplatValue<IntegerAttr>().getValue().isAllOnes();
      });
  ASSERT_NE(source, enumeration.graphs.end());
  ASSERT_NE(otherSource, enumeration.graphs.end());
  ASSERT_NE(unitSource0, enumeration.graphs.end());
  ASSERT_NE(unitSource1, enumeration.graphs.end());
  ASSERT_NE(trueMask, enumeration.graphs.end());
  ASSERT_NE(inputMask, enumeration.graphs.end());
  ASSERT_NE(unitMask, enumeration.graphs.end());
  ASSERT_NE(unitMask2, enumeration.graphs.end());
  ASSERT_NE(unitOne, enumeration.graphs.end());
  ASSERT_NE(unitMinusOne, enumeration.graphs.end());
  ASSERT_NE(negativeSource, enumeration.graphs.end());
  auto unmasked = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        return graph.operationName == "tt.histogram" &&
               graph.operands ==
                   SmallVector<unsigned, 3>({source->id});
      });
  auto masked = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        return graph.operationName == "tt.histogram" &&
               graph.operands == SmallVector<unsigned, 3>(
                                     {source->id, trueMask->id});
      });
  ASSERT_NE(unmasked, enumeration.graphs.end());
  ASSERT_NE(masked, enumeration.graphs.end());
  auto maskedInput = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        return graph.operationName == "tt.histogram" &&
               graph.resultType == masked->resultType &&
               graph.operands == SmallVector<unsigned, 3>(
                                     {source->id, inputMask->id});
      });
  auto alwaysZero = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        return graph.operationName == "tt.histogram" &&
               graph.resultType == masked->resultType &&
               graph.operands ==
                   SmallVector<unsigned, 3>({negativeSource->id});
      });
  ASSERT_NE(maskedInput, enumeration.graphs.end());
  ASSERT_NE(alwaysZero, enumeration.graphs.end());
  auto otherUnmasked = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        return graph.operationName == "tt.histogram" &&
               graph.resultType == unmasked->resultType &&
               graph.operands ==
                   SmallVector<unsigned, 3>({otherSource->id});
      });
  ASSERT_NE(otherUnmasked, enumeration.graphs.end());
  auto histogram16 = [&](unsigned operand) {
    return llvm::find_if(
        enumeration.graphs, [&](const OfflineTypedGraph &graph) {
          auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
          return graph.operationName == "tt.histogram" && tensor &&
                 tensor.getRank() == 1 && tensor.getDimSize(0) == 16 &&
                 graph.operands == SmallVector<unsigned, 3>({operand});
        });
  };
  auto unitHistogram0 = histogram16(unitSource0->id);
  auto unitHistogram1 = histogram16(unitSource1->id);
  auto unitAlwaysZero = histogram16(unitMinusOne->id);
  auto unitMaskedOne = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        return graph.operationName == "tt.histogram" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 16 &&
               graph.operands == SmallVector<unsigned, 3>(
                                     {unitOne->id, unitMask->id});
      });
  auto unitMaskedOther = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        return graph.operationName == "tt.histogram" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 1 &&
               graph.operands == SmallVector<unsigned, 3>(
                                     {unitSource0->id, unitMask2->id});
      });
  ASSERT_NE(unitHistogram0, enumeration.graphs.end());
  ASSERT_NE(unitHistogram1, enumeration.graphs.end());
  ASSERT_NE(unitAlwaysZero, enumeration.graphs.end());
  ASSERT_NE(unitMaskedOne, enumeration.graphs.end());
  ASSERT_NE(unitMaskedOther, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/32)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_EQ(fingerprintOf(unmasked->id), fingerprintOf(masked->id));
  EXPECT_NE(fingerprintOf(unmasked->id),
            fingerprintOf(otherUnmasked->id));
  EXPECT_NE(fingerprintOf(maskedInput->id), fingerprintOf(alwaysZero->id));
  EXPECT_NE(fingerprintOf(unitHistogram0->id),
            fingerprintOf(unitHistogram1->id));
  EXPECT_NE(fingerprintOf(unitAlwaysZero->id),
            fingerprintOf(unitMaskedOne->id));
  auto unitMaskedInput = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
        return graph.operationName == "tt.histogram" && tensor &&
               tensor.getRank() == 1 && tensor.getDimSize(0) == 1 &&
               graph.operands == SmallVector<unsigned, 3>(
                                     {unitSource0->id, unitMask->id});
      });
  ASSERT_NE(unitMaskedInput, enumeration.graphs.end());
  EXPECT_NE(fingerprintOf(unitMaskedInput->id),
            fingerprintOf(unitMaskedOther->id));
  std::string counterexampleReason;
  ASSERT_TRUE(fingerprints.denotations[unitMaskedInput->id]);
  ASSERT_TRUE(fingerprints.denotations[unitMaskedOther->id]);
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      *fingerprints.denotations[unitMaskedInput->id],
      *fingerprints.denotations[unitMaskedOther->id],
      /*firstModel=*/32, /*additionalModelCount=*/64,
      counterexampleReason));
  EXPECT_FALSE(counterexampleReason.empty());
  auto maskedHistogram16 = [&](unsigned maskId) {
    return llvm::find_if(
        enumeration.graphs, [&](const OfflineTypedGraph &graph) {
          auto tensor = dyn_cast<RankedTensorType>(graph.resultType);
          return graph.operationName == "tt.histogram" && tensor &&
                 tensor.getRank() == 1 && tensor.getDimSize(0) == 16 &&
                 graph.operands == SmallVector<unsigned, 3>(
                                       {unitSource0->id, maskId});
        });
  };
  auto maskedHistogram16Input = maskedHistogram16(unitMask->id);
  auto maskedHistogram16Other = maskedHistogram16(unitMask2->id);
  ASSERT_NE(maskedHistogram16Input, enumeration.graphs.end());
  ASSERT_NE(maskedHistogram16Other, enumeration.graphs.end());
  ASSERT_TRUE(fingerprints.denotations[maskedHistogram16Input->id]);
  ASSERT_TRUE(fingerprints.denotations[maskedHistogram16Other->id]);
  counterexampleReason.clear();
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      *fingerprints.denotations[maskedHistogram16Input->id],
      *fingerprints.denotations[maskedHistogram16Other->id],
      /*firstModel=*/32, /*additionalModelCount=*/64,
      counterexampleReason));
}

TEST(OfflineGraphEnumeratorTest, ReportsMissingConstructionKnowledge) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations.push_back("test.no_candidate_schema");
  spec.operations.push_back("arith.cmpi");

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult result = enumerator.enumerate(
      spec, schemas, {IntegerType::get(&context, 32)});

  EXPECT_FALSE(result.isComplete());
  EXPECT_EQ(result.missingCandidateSchemaOperations,
            std::vector<std::string>({"test.no_candidate_schema"}));
  EXPECT_EQ(result.missingAttributeDomainOperations,
            std::vector<std::string>(
                {"arith.cmpi/2:cmpi_predicate"}));
}

TEST(OfflineGraphEnumeratorTest, EnumeratesDeclarativeAttributeDomain) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations = {"ags.input", "arith.cmpi"};
  GraphAttributeDomain predicates;
  predicates.name = "cmpi_predicate";
  GraphAttributeValue equal;
  equal.kind = GraphAttributeValue::Kind::Integer;
  equal.integer = 0;
  GraphAttributeValue signedLessThan;
  signedLessThan.kind = GraphAttributeValue::Kind::Integer;
  signedLessThan.integer = 2;
  predicates.values = {equal, signedLessThan};
  spec.attributeDomains = {predicates};

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult result = enumerator.enumerate(
      spec, schemas,
      {IntegerType::get(&context, 1), IntegerType::get(&context, 32)});

  ASSERT_TRUE(result.isComplete())
      << result.toJSON(spec.getStableHash());
  ASSERT_EQ(result.graphCountBySize.size(), 2u);
  EXPECT_EQ(result.graphCountBySize[0], 4u);
  EXPECT_EQ(result.graphCountBySize[1], 16u);
  EXPECT_EQ(result.graphs.size(), 20u);
  EXPECT_TRUE(llvm::any_of(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "arith.cmpi" && graph.attributes &&
           graph.attributes.get("predicate");
  }));
}

TEST(OfflineGraphEnumeratorTest, MaterializesI64ArrayAttributeDomain) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions,
                                          /*maxCandidateOperands=*/2);
  GraphUniverseSpec spec;
  spec.name = "static-slice-enumerator-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 2;
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 2;
  spec.bounds.maxResults = 1;
  spec.operations = {"ags.input", "tensor.empty", "tensor.insert_slice"};
  spec.integerBits = {32};
  spec.semanticDomains = {GraphSemanticDomain::Signature,
                          GraphSemanticDomain::Shape,
                          GraphSemanticDomain::Index,
                          GraphSemanticDomain::Layout,
                          GraphSemanticDomain::Value,
                          GraphSemanticDomain::Definedness};
  auto arrayDomain = [](StringRef name, ArrayRef<int32_t> values) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::IntegerArray;
    value.integerArray.assign(values.begin(), values.end());
    domain.values = {value};
    return domain;
  };
  spec.attributeDomains = {
      arrayDomain("insert_slice_segments", {1, 1, 0, 0, 0}),
      arrayDomain("slice_offsets", {0}),
      arrayDomain("slice_sizes", {4}),
      arrayDomain("slice_strides", {1})};
  Type tensor4 = RankedTensorType::get({4}, IntegerType::get(&context, 32));

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, {tensor4});
  ASSERT_TRUE(result.isComplete()) << result.toJSON(spec.getStableHash());
  auto slice = llvm::find_if(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "tensor.insert_slice";
  });
  ASSERT_NE(slice, result.graphs.end());
  EXPECT_TRUE(isa<DenseI32ArrayAttr>(
      slice->attributes.get("operandSegmentSizes")));
  EXPECT_TRUE(isa<DenseI64ArrayAttr>(
      slice->attributes.get("static_offsets")));
  EXPECT_TRUE(isa<DenseI64ArrayAttr>(slice->attributes.get("static_sizes")));
  EXPECT_TRUE(isa<DenseI64ArrayAttr>(
      slice->attributes.get("static_strides")));
}

TEST(OfflineGraphEnumeratorTest, MaterializesTTIRIntegerAttributesAsI32) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxRank = 1;
  spec.operations = {"tt.make_range"};
  spec.tensorShapes = {{4}};
  auto integerDomain = [](StringRef name, int64_t integer) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::Integer;
    value.integer = integer;
    domain.values = {value};
    return domain;
  };
  spec.attributeDomains = {integerDomain("range_end", 4),
                           integerDomain("range_start", 0)};
  Type tensor4 = RankedTensorType::get({4}, IntegerType::get(&context, 32));

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, {tensor4});
  ASSERT_TRUE(result.isComplete());
  auto range = llvm::find_if(result.graphs, [](const OfflineTypedGraph &graph) {
    return graph.operationName == "tt.make_range";
  });
  ASSERT_NE(range, result.graphs.end());
  auto start = dyn_cast<IntegerAttr>(range->attributes.get("start"));
  auto end = dyn_cast<IntegerAttr>(range->attributes.get("end"));
  ASSERT_TRUE(start);
  ASSERT_TRUE(end);
  EXPECT_TRUE(start.getType().isInteger(32));
  EXPECT_TRUE(end.getType().isInteger(32));
}

TEST(OfflineGraphEnumeratorTest, AuditsCandidateCoveragePerOverload) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations = {"tt.histogram"};

  TTIRCandidateOpSchema unary;
  unary.operationName = "tt.histogram";
  unary.operandCount = 1;
  TTIRCandidateOpSchema binary;
  binary.operationName = "tt.histogram";
  binary.operandCount = 2;
  binary.constructible = true;
  binary.canProduceResult = [](Type, DictionaryAttr) { return false; };
  SmallVector<TTIRCandidateOpSchema, 2> schemas;
  schemas.push_back(std::move(unary));
  schemas.push_back(std::move(binary));

  OfflineTypedGraphEnumerator enumerator;
  OfflineEnumerationResult result = enumerator.enumerate(
      spec, schemas, {IntegerType::get(&context, 32)});

  EXPECT_FALSE(result.isComplete());
  EXPECT_EQ(result.missingCandidateSchemaOperations,
            std::vector<std::string>({"tt.histogram/1"}));
}

TEST(OfflineGraphEnumeratorTest, GraphLimitCannotMasqueradeAsComplete) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  OfflineTypedGraphEnumerator enumerator(
      OfflineEnumerationOptions{/*maxGraphs=*/5});

  OfflineEnumerationResult result = enumerator.enumerate(
      makeScalarUniverse(1), schemas, {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.hitGraphLimit);
  EXPECT_FALSE(result.isComplete());
  EXPECT_EQ(result.graphs.size(), 5u);
}

TEST(OfflineGraphEnumeratorTest,
     GraphLimitStillValidatesDeferredFinalLayerSemantics) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineEnumerationOptions options;
  options.maxGraphs = 30;
  options.semantics = &semantics;
  options.normalizer = &normalizer;
  options.quotientEquivalentOperands = true;
  options.semanticValidationJobs = 4;

  OfflineEnumerationResult result =
      OfflineTypedGraphEnumerator(options).enumerate(
          makeScalarUniverse(/*maxNodes=*/2), schemas,
          {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.hitGraphLimit);
  EXPECT_FALSE(result.isComplete());
  EXPECT_GT(result.semanticValidationGraphCount, 0u);
  EXPECT_EQ(result.semanticValidationJobs, 4u);
  EXPECT_LE(result.graphs.size(), options.maxGraphs);

  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/4, /*parallelJobs=*/4)
          .fingerprint(result.graphs, semantics, normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());
}

TEST(OfflineGraphEnumeratorTest, CooperativeCancellationCannotMasqueradeAsComplete) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  unsigned polls = 0;
  OfflineEnumerationOptions options;
  options.shouldStop = [&]() { return ++polls == 1; };

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator(options).enumerate(
      makeScalarUniverse(/*maxNodes=*/3), schemas,
      {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.interrupted);
  EXPECT_FALSE(result.hitGraphLimit);
  EXPECT_FALSE(result.isComplete());
  EXPECT_EQ(polls, 1u);
  EXPECT_NE(result.toJSON("test-universe").find("\"interrupted\": true"),
            std::string::npos);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsGenericTernaryPointwiseSemanticsWithoutWildcard) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.operations = {"ags.input", "math.fma"};
  spec.bounds.maxOperands = 3;
  spec.integerBits.clear();
  spec.integerConstants.clear();
  spec.floatingBits = {32};
  OfflineTypeDomain types = buildOfflineTypeDomain(context, spec);

  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, types.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  EXPECT_TRUE(llvm::any_of(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "math.fma" && graph.operands.size() == 3;
  }));

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(),
            enumeration.graphs.size());
}

TEST(OfflineGraphEnumeratorTest,
     EnumeratesVariadicDynamicSliceFromDeclarativeSegments) {
  static constexpr StringLiteral source = R"json(
  {
    "schema_version": 2,
    "name": "dynamic-slice-enumerator-test",
    "target": "generic",
    "bounds": {
      "max_nodes": 1,
      "max_rank": 1,
      "max_operands": 2,
      "max_results": 1,
      "max_shape_symbols": 1,
      "allow_dynamic_shapes": true,
      "allow_control_flow": false,
      "allow_effects": false
    },
    "catalog_dialects": ["tensor"],
    "operations": ["ags.input", "tensor.extract_slice"],
    "integer_bits": [32],
    "floating_bits": [],
    "include_index_type": true,
    "pointer_address_spaces": [],
    "integer_constants": [],
    "tensor_shapes": [[4], [8]],
    "attribute_domains": {
      "slice_segments": [[1, 1, 0, 0]],
      "slice_offsets": [[-1]],
      "slice_sizes": [[4]],
      "slice_strides": [[1]]
    },
    "semantic_domains": ["definedness", "effect", "index", "layout",
                         "memory", "shape", "signature", "value"]
  }
  )json";
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;

  MLIRContext context;
  OfflineTypeDomain types = buildOfflineTypeDomain(context, *spec);
  ASSERT_TRUE(types.unsupportedUniverseFeatures.empty());
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options,
                                          spec->bounds.maxOperands);
  auto dynamicSchema = llvm::find_if(
      schemas, [](const TTIRCandidateOpSchema &schema) {
        return schema.operationName == "tensor.extract_slice" &&
               schema.operandCount == 2;
      });
  ASSERT_NE(dynamicSchema, schemas.end());
  Builder builder(&context);
  DictionaryAttr dynamicAttributes = builder.getDictionaryAttr({
      builder.getNamedAttr("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr({1, 1, 0, 0})),
      builder.getNamedAttr("static_offsets",
                           builder.getDenseI64ArrayAttr({-1})),
      builder.getNamedAttr("static_sizes",
                           builder.getDenseI64ArrayAttr({4})),
      builder.getNamedAttr("static_strides",
                           builder.getDenseI64ArrayAttr({1}))});
  Type tensor4 = RankedTensorType::get(
      {4}, IntegerType::get(&context, 32));
  Type tensor8 = RankedTensorType::get(
      {8}, IntegerType::get(&context, 32));
  Type index = builder.getIndexType();
  ASSERT_TRUE(dynamicSchema->canProduceResult(tensor4, dynamicAttributes));
  ASSERT_TRUE(dynamicSchema->mayBeWellTyped(tensor4, {tensor8},
                                             dynamicAttributes));
  ASSERT_TRUE(dynamicSchema->isWellTyped(tensor4, {tensor8, index},
                                          dynamicAttributes));
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(*spec, schemas, types.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec->getStableHash());
  auto dynamicSlice = llvm::find_if(
      enumeration.graphs, [&](const OfflineTypedGraph &graph) {
        if (graph.operationName != "tensor.extract_slice" ||
            graph.operands.size() != 2 ||
            graph.resultType !=
                RankedTensorType::get({4}, IntegerType::get(&context, 32)))
          return false;
        return enumeration.graphs[graph.operands[1]].resultType.isIndex();
      });
  ASSERT_NE(dynamicSlice, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec->getStableHash(), enumeration.isComplete());
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsQuantifiedDynamicBufferFillWithoutFreeIndex) {
  MLIRContext context;
  GraphUniverseSpec spec;
  spec.name = "dynamic-buffer-fill-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 1;
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 3;
  spec.bounds.maxResults = 1;
  spec.bounds.maxShapeSymbols = 1;
  spec.bounds.allowDynamicShapes = true;
  spec.bounds.allowEffects = true;
  spec.catalogDialects = {"linalg"};
  spec.operations = {"ags.buffer", "ags.input", "linalg.fill"};
  spec.integerBits = {32};
  spec.tensorShapes = {{GraphUniverseSpec::kDynamicDimension}};
  spec.explicitTypes = {"memref<?xi32>"};
  spec.semanticDomains = {
      GraphSemanticDomain::Signature, GraphSemanticDomain::Value,
      GraphSemanticDomain::Definedness, GraphSemanticDomain::Shape,
      GraphSemanticDomain::Index, GraphSemanticDomain::Layout,
      GraphSemanticDomain::Effect, GraphSemanticDomain::Buffer};
  std::string error;
  ASSERT_TRUE(spec.verify(error)) << error;

  OfflineTypeDomain types = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(types.isComplete());
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions,
                                          spec.bounds.maxOperands);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, types.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto fill = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "linalg.fill";
  });
  ASSERT_NE(fill, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  ASSERT_LT(fill->id, fingerprints.denotations.size());
  ASSERT_TRUE(fingerprints.denotations[fill->id].has_value());
  EXPECT_EQ(fingerprints.denotations[fill->id]->value.getKind(),
            TensorSemanticKind::BufferFill);
}

TEST(OfflineGraphEnumeratorTest,
     DiscoversCompleteBufferCopyForwardingFromGenericSemantics) {
  MLIRContext context;
  GraphUniverseSpec spec;
  spec.name = "buffer-copy-forwarding-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 2;
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 3;
  spec.bounds.maxResults = 1;
  spec.bounds.allowEffects = true;
  spec.catalogDialects = {"bufferization", "memref"};
  spec.operations = {"ags.buffer", "ags.input", "bufferization.to_tensor",
                     "memref.copy"};
  spec.integerBits = {32};
  spec.tensorShapes = {{4}};
  spec.explicitTypes = {"memref<4xi32>"};
  spec.semanticDomains = {
      GraphSemanticDomain::Signature, GraphSemanticDomain::Value,
      GraphSemanticDomain::Definedness, GraphSemanticDomain::Shape,
      GraphSemanticDomain::Index, GraphSemanticDomain::Layout,
      GraphSemanticDomain::Effect, GraphSemanticDomain::Buffer};
  std::string error;
  ASSERT_TRUE(spec.verify(error)) << error;

  OfflineTypeDomain types = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(types.isComplete());
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, types.types);
  ASSERT_TRUE(enumeration.isComplete());

  auto buffer = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "ags.buffer";
  });
  ASSERT_NE(buffer, enumeration.graphs.end());
  auto memrefInputs = llvm::make_filter_range(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
        return graph.operationName == "ags.input" &&
               isa<MemRefType>(graph.resultType);
      });
  SmallVector<unsigned, 3> inputs;
  for (const OfflineTypedGraph &input : memrefInputs)
    inputs.push_back(input.id);
  ASSERT_GE(inputs.size(), 2u);

  auto copy = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "memref.copy" &&
           graph.operands == SmallVector<unsigned, 3>(
                                 {inputs[0], inputs[1], buffer->id});
  });
  ASSERT_NE(copy, enumeration.graphs.end());
  auto copiedRead = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "bufferization.to_tensor" &&
           graph.operands ==
               SmallVector<unsigned, 2>({inputs[1], copy->id});
  });
  auto sourceRead = llvm::find_if(enumeration.graphs, [&](const auto &graph) {
    return graph.operationName == "bufferization.to_tensor" &&
           graph.operands ==
               SmallVector<unsigned, 2>({inputs[0], buffer->id});
  });
  ASSERT_NE(copiedRead, enumeration.graphs.end());
  ASSERT_NE(sourceRead, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  ASSERT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  ASSERT_TRUE(fingerprints.denotations[copiedRead->id].has_value());
  ASSERT_TRUE(fingerprints.denotations[sourceRead->id].has_value());
  std::optional<SemanticEquivalenceProof> directProof =
      SemanticEquivalenceOracle(normalizer).prove(
          *fingerprints.denotations[copiedRead->id],
          *fingerprints.denotations[sourceRead->id]);
  ASSERT_TRUE(directProof)
      << "copied="
      << getStableTensorDenotationKey(
             *fingerprints.denotations[copiedRead->id])
      << "\nsource="
      << getStableTensorDenotationKey(
             *fingerprints.denotations[sourceRead->id]);
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      enumeration.graphs, fingerprints, SemanticEquivalenceOracle(normalizer));
  auto proof = llvm::find_if(proofs.pairs, [&](const auto &pair) {
    return ((pair.lhsGraphId == copiedRead->id &&
             pair.rhsGraphId == sourceRead->id) ||
            (pair.rhsGraphId == copiedRead->id &&
             pair.lhsGraphId == sourceRead->id));
  });
  ASSERT_NE(proof, proofs.pairs.end())
      << "copied_id=" << copiedRead->id << " source_id=" << sourceRead->id
      << "\ncopied="
      << getStableTensorDenotationKey(
             *fingerprints.denotations[copiedRead->id])
      << "\nsource="
      << getStableTensorDenotationKey(
             *fingerprints.denotations[sourceRead->id]) << "\n"
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  EXPECT_EQ(proof->status, SemanticProofStatus::Proved) << proof->reason;
  EXPECT_FALSE(proof->getProofCertificate().empty());
}

TEST(OfflineGraphEnumeratorTest, FingerprintsPrivateBufferTerminals) {
  MLIRContext context;
  GraphUniverseSpec spec = makeScalarUniverse(0);
  spec.bounds.maxRank = 1;
  spec.operations = {"ags.private_buffer"};
  spec.integerConstants.clear();
  spec.tensorShapes.clear();
  spec.explicitTypes = {"memref<4xi32>"};

  OfflineTypeDomain types = buildOfflineTypeDomain(context, spec);
  ASSERT_TRUE(types.isComplete());
  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      spec, /*schemas=*/{}, types.types);
  ASSERT_TRUE(enumeration.isComplete());
  ASSERT_EQ(enumeration.graphs.size(), 2u);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(enumeration.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());
  EXPECT_EQ(fingerprints.getFingerprintedGraphCount(), 2u);
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsOpaqueRegisteredOperationsWithoutWildcards) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"ags.input", "math.exp", "math.log"};
  spec.integerBits.clear();
  spec.floatingBits = {32};
  spec.integerConstants.clear();

  Type f32 = Float32Type::get(&context);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, {f32});
  ASSERT_TRUE(enumeration.isComplete());

  auto exp = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "math.exp";
  });
  auto log = llvm::find_if(enumeration.graphs, [](const auto &graph) {
    return graph.operationName == "math.log";
  });
  ASSERT_NE(exp, enumeration.graphs.end());
  ASSERT_NE(log, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphIds.empty());

  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_FALSE(fingerprintOf(exp->id).empty());
  EXPECT_FALSE(fingerprintOf(log->id).empty());
  EXPECT_NE(fingerprintOf(exp->id), fingerprintOf(log->id));
}

TEST(OfflineGraphEnumeratorTest,
     OpaqueLowBitFingerprintsDoNotKeepOneHashCollisionAcrossModels) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.operations = {"arith.constant", "arith.minsi", "arith.minui"};
  spec.integerBits = {8};
  spec.integerConstants = {-1, 0, 1, 2};

  Type i8 = IntegerType::get(&context, 8);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, {i8});
  ASSERT_TRUE(enumeration.isComplete());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());

  for (const OfflineSemanticFingerprintBucket &bucket : fingerprints.buckets) {
    bool hasSignedMin = false;
    bool hasUnsignedMin = false;
    for (unsigned graphId : bucket.graphIds) {
      hasSignedMin |=
          enumeration.graphs[graphId].operationName == "arith.minsi";
      hasUnsignedMin |=
          enumeration.graphs[graphId].operationName == "arith.minui";
    }
    EXPECT_FALSE(hasSignedMin && hasUnsignedMin)
        << "typed opaque operations retained a low-bit hash collision";
  }
}

TEST(OfflineGraphEnumeratorTest, EnumeratesEveryMultiResultProjection) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxResults = 2;
  spec.operations = {"ags.input", "arith.mului_extended"};
  spec.integerConstants.clear();

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, {IntegerType::get(&context, 32)});

  ASSERT_TRUE(result.isComplete())
      << result.toJSON(spec.getStableHash());
  bool sawLow = false;
  bool sawHigh = false;
  for (const OfflineTypedGraph &graph : result.graphs) {
    if (graph.operationName != "arith.mului_extended")
      continue;
    auto index = dyn_cast_or_null<IntegerAttr>(
        graph.attributes.get("ags.result_index"));
    auto types = dyn_cast_or_null<ArrayAttr>(
        graph.attributes.get("ags.operation_result_types"));
    ASSERT_TRUE(index);
    ASSERT_TRUE(types);
    ASSERT_EQ(types.size(), 2u);
    sawLow |= index.getInt() == 0;
    sawHigh |= index.getInt() == 1;
  }
  EXPECT_TRUE(sawLow);
  EXPECT_TRUE(sawHigh);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(result.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsDynamicScanWithoutWildcard) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxRank = 1;
  spec.bounds.maxOperands = 1;
  spec.bounds.maxInputSymbols = 1;
  spec.bounds.allowDynamicShapes = true;
  spec.bounds.allowControlFlow = true;
  spec.operations = {"ags.input", "tt.scan"};
  spec.integerBits = {1, 8, 16, 32, 64};
  spec.integerConstants.clear();
  spec.tensorShapes = {{ShapedType::kDynamic},
                       {ShapedType::kDynamic, ShapedType::kDynamic},
                       {ShapedType::kDynamic, 16},
                       {1},
                       {1, 16},
                       {16},
                       {16, ShapedType::kDynamic},
                       {16, 1},
                       {16, 16}};
  auto integerDomain = [](StringRef name, int64_t integer) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::Integer;
    value.integer = integer;
    domain.values = {value};
    return domain;
  };
  GraphAttributeDomain reverse;
  reverse.name = "reverse";
  GraphAttributeValue falseValue;
  falseValue.kind = GraphAttributeValue::Kind::Boolean;
  falseValue.boolean = false;
  GraphAttributeValue trueValue = falseValue;
  trueValue.boolean = true;
  reverse.values = {falseValue, trueValue};
  GraphAttributeDomain axis = integerDomain("axis", 0);
  axis.values.push_back(integerDomain("axis", 1).values.front());
  spec.attributeDomains = {std::move(axis), std::move(reverse)};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/16)
          .fingerprint(enumeration.graphs, semantics, normalizer);

  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec.getStableHash(), enumeration.isComplete());
  EXPECT_TRUE(fingerprints.wildcardGraphs.empty());
}

TEST(OfflineGraphEnumeratorTest,
     FingerprintsSeparateScanAxesAndDirectionsOnSquareTensors) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(/*maxNodes=*/1);
  spec.bounds.maxRank = 2;
  spec.bounds.maxOperands = 1;
  spec.bounds.maxInputSymbols = 1;
  spec.bounds.allowControlFlow = true;
  spec.operations = {"ags.input", "arith.constant", "tt.scan"};
  spec.integerBits = {1};
  spec.integerConstants = {1};
  spec.tensorShapes = {{16, 16}};
  auto integerDomain = [](StringRef name, int64_t integer) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::Integer;
    value.integer = integer;
    domain.values = {value};
    return domain;
  };
  GraphAttributeDomain reverse;
  reverse.name = "reverse";
  GraphAttributeValue falseValue;
  falseValue.kind = GraphAttributeValue::Kind::Boolean;
  falseValue.boolean = false;
  GraphAttributeValue trueValue = falseValue;
  trueValue.boolean = true;
  reverse.values = {falseValue, trueValue};
  GraphAttributeDomain axis = integerDomain("axis", 0);
  axis.values.push_back(integerDomain("axis", 1).values.front());
  spec.attributeDomains = {std::move(axis), std::move(reverse)};

  OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, spec);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, typeDomain.types);
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  auto constant = llvm::find_if(
      enumeration.graphs, [](const OfflineTypedGraph &graph) {
    auto type = dyn_cast<RankedTensorType>(graph.resultType);
    auto dense = graph.attributes
                     ? dyn_cast_or_null<DenseElementsAttr>(
                           graph.attributes.get("value"))
                     : DenseElementsAttr{};
    return graph.operationName == "arith.constant" && type &&
           type.getShape() == ArrayRef<int64_t>({16, 16}) &&
           type.getElementType().isInteger(1) && dense && dense.isSplat() &&
           !dense.getSplatValue<IntegerAttr>().getValue().isZero();
  });
  ASSERT_NE(constant, enumeration.graphs.end());
  auto scan = [&](int64_t expectedAxis, bool expectedReverse) {
    return llvm::find_if(enumeration.graphs, [&](const auto &graph) {
      auto axisAttr = graph.attributes
                          ? dyn_cast_or_null<IntegerAttr>(
                                graph.attributes.get("axis"))
                          : IntegerAttr{};
      auto reverseAttr = graph.attributes
                             ? dyn_cast_or_null<BoolAttr>(
                                   graph.attributes.get("reverse"))
                             : BoolAttr{};
      return graph.operationName == "tt.scan" && axisAttr && reverseAttr &&
             axisAttr.getInt() == expectedAxis &&
             reverseAttr.getValue() == expectedReverse &&
             graph.operands ==
                 SmallVector<unsigned, 3>({constant->id});
    });
  };
  auto axis0Forward = scan(/*expectedAxis=*/0, /*expectedReverse=*/false);
  auto axis1Forward = scan(/*expectedAxis=*/1, /*expectedReverse=*/false);
  auto axis1Reverse = scan(/*expectedAxis=*/1, /*expectedReverse=*/true);
  ASSERT_NE(axis0Forward, enumeration.graphs.end());
  ASSERT_NE(axis1Forward, enumeration.graphs.end());
  ASSERT_NE(axis1Reverse, enumeration.graphs.end());

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter(/*modelCount=*/32)
          .fingerprint(enumeration.graphs, semantics, normalizer);
  ASSERT_TRUE(fingerprints.isComplete());
  auto fingerprintOf = [&](unsigned graphId) -> StringRef {
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets)
      if (llvm::is_contained(bucket.graphIds, graphId))
        return bucket.fingerprint;
    return {};
  };
  EXPECT_NE(fingerprintOf(axis0Forward->id),
            fingerprintOf(axis1Reverse->id));
  ASSERT_TRUE(fingerprints.denotations[axis0Forward->id]);
  ASSERT_TRUE(fingerprints.denotations[axis1Forward->id]);
  std::string counterexampleReason;
  EXPECT_TRUE(OfflineSemanticFingerprinter::hasConcreteCounterexample(
      *fingerprints.denotations[axis0Forward->id],
      *fingerprints.denotations[axis1Forward->id],
      /*firstModel=*/32, /*additionalModelCount=*/64,
      counterexampleReason));
}

TEST(OfflineGraphEnumeratorTest, ControlFlowPermissionDoesNotInventCFGs) {
  MLIRContext context;
  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  GraphUniverseSpec spec = makeScalarUniverse(1);
  spec.bounds.maxResults = 2;
  spec.bounds.allowControlFlow = true;

  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, {IntegerType::get(&context, 32)});

  EXPECT_TRUE(result.isComplete());
}

TEST(OfflineGraphEnumeratorTest, EnumeratesDeclarativeRegionRecipes) {
  static constexpr StringLiteral source = R"json(
  {
    "schema_version": 2,
    "name": "region-enumerator-test",
    "target": "generic",
    "bounds": {
      "max_nodes": 1,
      "max_rank": 1,
      "max_operands": 1,
      "max_results": 1,
      "max_shape_symbols": 0,
      "allow_dynamic_shapes": false,
      "allow_control_flow": true,
      "allow_effects": false
    },
    "catalog_dialects": ["tt", "arith"],
    "operations": ["ags.input", "tt.scan", "tt.reduce"],
    "integer_bits": [32],
    "floating_bits": [],
    "pointer_address_spaces": [],
    "integer_constants": [],
    "tensor_shapes": [[4]],
    "attribute_domains": {"axis": [0], "reverse": [false, true]},
    "semantic_domains": ["definedness", "effect", "index", "layout",
                         "memory", "shape", "signature", "value"]
  }
  )json";
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  MLIRContext context;
  OfflineTypeDomain types = buildOfflineTypeDomain(context, *spec);
  ASSERT_TRUE(types.isComplete());
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options,
                                          spec->bounds.maxOperands);
  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      *spec, schemas, types.types);
  ASSERT_TRUE(result.isComplete()) << result.toJSON(spec->getStableHash());
  unsigned regionGraphCount = 0;
  bool sawAddCombiner = false;
  bool sawMultiplyCombiner = false;
  for (const OfflineTypedGraph &graph : result.graphs) {
    if (graph.operationName != "tt.scan" &&
        graph.operationName != "tt.reduce")
      continue;
    ++regionGraphCount;
    auto regions = dyn_cast_or_null<ArrayAttr>(
        graph.attributes.get(kCanonicalRegionsAttr));
    ASSERT_TRUE(regions);
    ASSERT_TRUE(succeeded(validateCanonicalRegions(regions, error))) << error;
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    stream << regions;
    stream.flush();
    sawAddCombiner |= StringRef(printed).contains("arith.addi");
    sawMultiplyCombiner |= StringRef(printed).contains("arith.muli");
  }
  EXPECT_GT(regionGraphCount, 0u);
  EXPECT_TRUE(sawAddCombiner);
  EXPECT_TRUE(sawMultiplyCombiner);
}

TEST(OfflineGraphEnumeratorTest,
     EnumeratesDeclarativeProofOnlyKernelLaunchRoots) {
  static constexpr StringLiteral source = R"json(
  {
    "schema_version": 2,
    "name": "kernel-launch-enumerator-test",
    "target": "generic",
    "bounds": {
      "max_nodes": 2,
      "max_rank": 0,
      "max_operands": 2,
      "max_results": 1,
      "max_shape_symbols": 0,
      "allow_dynamic_shapes": false,
      "allow_control_flow": false,
      "allow_effects": false
    },
    "catalog_dialects": ["tt"],
    "operations": ["ags.input", "ags.memory", "ags.partitionable_memory",
                   "ags.launch",
                   "ags.launch_bound", "ags.launch_remap", "ags.kernel",
                   "ags.kernel_partition"],
    "integer_bits": [32],
    "floating_bits": [],
    "pointer_address_spaces": [],
    "integer_constants": [],
    "tensor_shapes": [],
    "attribute_domains": {
      "program_axis": [0],
      "launch_axis": [0],
      "launch_factor": [1, 2, 16],
      "launch_ceil_div": [0, 1]
    },
    "semantic_domains": ["definedness", "effect", "index", "launch",
                         "layout", "memory", "shape", "signature", "value"]
  }
  )json";
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  MLIRContext context;
  OfflineTypeDomain types = buildOfflineTypeDomain(context, *spec);
  ASSERT_TRUE(types.isComplete());
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options,
                                          spec->bounds.maxOperands);
  OfflineEnumerationResult result = OfflineTypedGraphEnumerator().enumerate(
      *spec, schemas, types.types);
  ASSERT_TRUE(result.isComplete()) << result.toJSON(spec->getStableHash());

  bool sawOriginalKernel = false;
  bool sawBoundedKernel = false;
  bool sawRemappedKernel = false;
  bool sawParameterizedFactor16 = false;
  for (const OfflineTypedGraph &graph : result.graphs) {
    if (graph.operationName == "ags.kernel_partition") {
      auto factor = dyn_cast_or_null<IntegerAttr>(
          graph.attributes.get("factor"));
      auto action = dyn_cast_or_null<DictionaryAttr>(
          graph.attributes.get("ags.compiler_action"));
      sawParameterizedFactor16 |=
          factor && factor.getInt() == 16 && action &&
          cast<StringAttr>(action.get("kind")).getValue() ==
              "launch_partition";
      continue;
    }
    if (graph.operationName != "ags.kernel")
      continue;
    EXPECT_EQ(graph.attributes.get("ags.proof_only"),
              Builder(&context).getBoolAttr(true));
    ASSERT_EQ(graph.operands.size(), 2u);
    const OfflineTypedGraph &launch = result.graphs[graph.operands[1]];
    if (launch.operationName == "ags.launch_remap")
      EXPECT_EQ(launch.attributes.get("ags.proof_only"),
                Builder(&context).getBoolAttr(true));
    sawOriginalKernel |= launch.operationName == "ags.launch";
    sawBoundedKernel |= launch.operationName == "ags.launch_bound";
    sawRemappedKernel |= launch.operationName == "ags.launch_remap";
  }
  EXPECT_TRUE(sawOriginalKernel);
  EXPECT_TRUE(sawBoundedKernel);
  EXPECT_TRUE(sawRemappedKernel);
  EXPECT_TRUE(sawParameterizedFactor16);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(result.graphs, semantics,
                                                  normalizer);
  EXPECT_TRUE(fingerprints.isComplete())
      << fingerprints.toJSON(spec->getStableHash(), result.isComplete());
}

} // namespace

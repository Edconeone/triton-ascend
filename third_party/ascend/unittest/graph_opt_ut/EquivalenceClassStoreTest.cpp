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

#include "TritonToGraph/EquivalenceClassStore.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static GraphUniverseSpec makeUniverse() {
  GraphUniverseSpec spec;
  spec.name = "eclass-store-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 1;
  spec.bounds.maxRank = 0;
  spec.bounds.maxOperands = 2;
  spec.bounds.maxResults = 1;
  spec.operations = {"ags.input", "arith.constant", "arith.subi"};
  spec.integerBits = {32};
  spec.integerConstants = {0, 1};
  spec.semanticDomains = {GraphSemanticDomain::Signature,
                          GraphSemanticDomain::Shape,
                          GraphSemanticDomain::Index,
                          GraphSemanticDomain::Value,
                          GraphSemanticDomain::Definedness};
  return spec;
}

struct OfflineRun {
  OfflineEnumerationResult enumeration;
  OfflineSemanticFingerprintResult fingerprints;
  OfflineEquivalenceProofResult proofs;
};

static OfflineRun runOfflinePipeline(MLIRContext &context,
                                     bool pruneTransitivePairs = false) {
  GraphUniverseSpec spec = makeUniverse();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  OfflineRun run;
  run.enumeration = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas, {IntegerType::get(&context, 32)});
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  run.fingerprints = OfflineSemanticFingerprinter().fingerprint(
      run.enumeration.graphs, semantics, normalizer);
  OfflineEquivalenceProofOptions proofOptions;
  proofOptions.pruneTransitivelyImpliedPairs = pruneTransitivePairs;
  proofOptions.retainRefutedPairs = !pruneTransitivePairs;
  run.proofs = OfflineEquivalenceProver().prove(
      run.enumeration.graphs, run.fingerprints,
      SemanticEquivalenceOracle(normalizer), proofOptions);
  return run;
}

TEST(EquivalenceClassStoreTest,
     ReplayableProofForestPreservesTheCompleteClassPartition) {
  MLIRContext context;
  GraphUniverseSpec spec = makeUniverse();
  OfflineRun exhaustive = runOfflinePipeline(context);
  OfflineRun pruned = runOfflinePipeline(context, /*pruneTransitivePairs=*/true);
  ASSERT_TRUE(exhaustive.proofs.isComplete());
  ASSERT_TRUE(pruned.proofs.isComplete());
  ASSERT_TRUE(pruned.proofs.usedSemanticQuotient);
  ASSERT_GT(pruned.proofs.semanticQuotientGroups, 0u);
  ASSERT_GT(pruned.proofs.semanticQuotientClasses, 0u);
  ASSERT_EQ(pruned.proofs.processedPairs, pruned.proofs.candidatePairs);
  ASSERT_EQ(pruned.proofs.provedPairs +
                pruned.proofs.transitivelyProvedPairs +
                pruned.proofs.refutedPairs + pruned.proofs.unknownPairs,
            pruned.proofs.candidatePairs);
  ASSERT_GT(pruned.proofs.transitivelyProvedPairs, 0u);
  ASSERT_LT(pruned.proofs.provedPairs, exhaustive.proofs.provedPairs);

  std::string error;
  std::optional<PersistentEquivalenceClassStore> exhaustiveStore =
      PersistentEquivalenceClassStore::build(
          exhaustive.enumeration.graphs, exhaustive.proofs,
          spec.getStableHash(), getBuiltinTTOpSemanticsStableHash(), true,
          true, nullptr, error);
  ASSERT_TRUE(exhaustiveStore) << error;
  std::optional<PersistentEquivalenceClassStore> prunedStore =
      PersistentEquivalenceClassStore::build(
          pruned.enumeration.graphs, pruned.proofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(), true, true, nullptr, error);
  ASSERT_TRUE(prunedStore) << error;

  ASSERT_EQ(prunedStore->classes.size(), exhaustiveStore->classes.size());
  for (auto [prunedClass, exhaustiveClass] :
       llvm::zip_equal(prunedStore->classes, exhaustiveStore->classes))
    EXPECT_EQ(prunedClass.memberGraphHashes,
              exhaustiveClass.memberGraphHashes);
}

TEST(EquivalenceClassStoreTest, RoundTripsAndReplaysEveryCertificate) {
  MLIRContext context;
  GraphUniverseSpec spec = makeUniverse();
  OfflineRun run = runOfflinePipeline(context);
  ASSERT_TRUE(run.enumeration.isComplete());
  ASSERT_TRUE(run.fingerprints.isComplete());
  ASSERT_TRUE(run.proofs.isComplete());

  std::string error;
  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          run.enumeration.graphs, run.proofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(), /*enumerationComplete=*/true,
          /*fingerprintComplete=*/true, /*base=*/nullptr, error);
  ASSERT_TRUE(store) << error;
  EXPECT_TRUE(store->isComplete());
  EXPECT_EQ(store->graphs.size(), run.enumeration.graphs.size());
  EXPECT_EQ(store->proofEdges.size(), run.proofs.provedPairs);
  EXPECT_LT(store->classes.size(), store->graphs.size());

  std::string serialized = store->toJSON();
  MLIRContext reloadContext;
  TensorSemanticNormalizer reloadNormalizer;
  SemanticEquivalenceOracle reloadOracle(reloadNormalizer);
  std::optional<PersistentEquivalenceClassStore> reloaded =
      PersistentEquivalenceClassStore::fromJSON(
          serialized, reloadContext, reloadOracle, error);
  ASSERT_TRUE(reloaded) << error;
  EXPECT_EQ(reloaded->getStableHash(), store->getStableHash());
  EXPECT_EQ(reloaded->toJSON(), serialized);
}

TEST(EquivalenceClassStoreTest,
     ReconstructsOfflineGraphsFromPersistentHashesForReplay) {
  MLIRContext context;
  OfflineRun run = runOfflinePipeline(context);
  std::string error;
  std::optional<std::vector<PersistentGraphRecord>> records =
      buildPersistentGraphRecords(run.enumeration.graphs, error);
  ASSERT_TRUE(records) << error;
  std::optional<std::vector<OfflineTypedGraph>> reconstructed =
      reconstructOfflineGraphs(*records, context, error);
  ASSERT_TRUE(reconstructed) << error;
  ASSERT_EQ(reconstructed->size(), records->size());
  for (const PersistentGraphRecord &record : *records) {
    auto found = llvm::find_if(
        *reconstructed, [&](const OfflineTypedGraph &graph) {
          return graph.canonicalKey == record.graphHash;
        });
    ASSERT_NE(found, reconstructed->end());
    EXPECT_EQ(found->operationName, record.operationName);
    EXPECT_EQ(found->operatorSize, record.operatorSize);
  }
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprintResult fingerprints =
      OfflineSemanticFingerprinter().fingerprint(*reconstructed, semantics,
                                                 normalizer);
  EXPECT_TRUE(fingerprints.isComplete());
}

TEST(EquivalenceClassStoreTest,
     ReconstructsPersistentEffectKindWithoutTreatingItAsPure) {
  MLIRContext context;
  Builder builder(&context);
  OfflineTypedGraph graph;
  graph.id = 0;
  graph.operatorSize = 1;
  graph.operationName = "tt.atomic_rmw";
  graph.effectKind = TTIRCandidateEffectKind::Atomic;
  graph.resultType = IntegerType::get(&context, 32);
  graph.attributes = builder.getDictionaryAttr({});

  std::string error;
  std::optional<std::vector<PersistentGraphRecord>> records =
      buildPersistentGraphRecords({graph}, error);
  ASSERT_TRUE(records) << error;
  ASSERT_EQ(records->size(), 1u);
  EXPECT_EQ((*records)[0].effectKind, TTIRCandidateEffectKind::Atomic);

  std::optional<std::vector<OfflineTypedGraph>> reconstructed =
      reconstructOfflineGraphs(*records, context, error);
  ASSERT_TRUE(reconstructed) << error;
  ASSERT_EQ(reconstructed->size(), 1u);
  EXPECT_EQ((*reconstructed)[0].effectKind, TTIRCandidateEffectKind::Atomic);

  PersistentGraphRecord pure = (*records)[0];
  pure.effectKind = TTIRCandidateEffectKind::Pure;
  EXPECT_EQ(getPersistentGraphHash((*records)[0]),
            getPersistentGraphHash(pure));
}

TEST(EquivalenceClassStoreTest, IncrementalExtensionMatchesCleanBuild) {
  MLIRContext context;
  GraphUniverseSpec spec = makeUniverse();
  OfflineRun run = runOfflinePipeline(context);
  ASSERT_GE(run.enumeration.graphs.size(), 4u);

  OfflineEquivalenceProofResult noProofs;
  std::string error;
  std::optional<PersistentEquivalenceClassStore> base =
      PersistentEquivalenceClassStore::build(
          ArrayRef<OfflineTypedGraph>(run.enumeration.graphs).take_front(4),
          noProofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(),
          /*enumerationComplete=*/false, /*fingerprintComplete=*/true,
          /*base=*/nullptr, error);
  ASSERT_TRUE(base) << error;

  std::optional<PersistentEquivalenceClassStore> incremental =
      PersistentEquivalenceClassStore::build(
          run.enumeration.graphs, run.proofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(), /*enumerationComplete=*/true,
          /*fingerprintComplete=*/true, &*base, error);
  ASSERT_TRUE(incremental) << error;
  std::optional<PersistentEquivalenceClassStore> clean =
      PersistentEquivalenceClassStore::build(
          run.enumeration.graphs, run.proofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(), /*enumerationComplete=*/true,
          /*fingerprintComplete=*/true, /*base=*/nullptr, error);
  ASSERT_TRUE(clean) << error;
  EXPECT_EQ(incremental->toJSON(), clean->toJSON());
}

TEST(EquivalenceClassStoreTest, RejectsTamperedProofHash) {
  MLIRContext context;
  GraphUniverseSpec spec = makeUniverse();
  OfflineRun run = runOfflinePipeline(context);
  std::string error;
  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          run.enumeration.graphs, run.proofs, spec.getStableHash(),
          getBuiltinTTOpSemanticsStableHash(), /*enumerationComplete=*/true,
          /*fingerprintComplete=*/true, /*base=*/nullptr, error);
  ASSERT_TRUE(store) << error;
  ASSERT_FALSE(store->proofEdges.empty());

  std::string serialized = store->toJSON();
  const std::string &hash = store->proofEdges.front().proofHash;
  std::string needle = "\"proof_hash\":\"" + hash + "\"";
  size_t position = serialized.find(needle);
  ASSERT_NE(position, std::string::npos);
  serialized[position + needle.size() - hash.size() - 1] =
      hash.front() == '0' ? '1' : '0';

  MLIRContext reloadContext;
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  EXPECT_FALSE(PersistentEquivalenceClassStore::fromJSON(
      serialized, reloadContext, oracle, error));
  EXPECT_FALSE(error.empty());
}

TEST(EquivalenceClassStoreTest, DeduplicatesNormalizedStructuralGraphs) {
  MLIRContext context;
  Builder builder(&context);
  OfflineTypedGraph first;
  first.id = 0;
  first.operationName = "arith.constant";
  first.resultType = builder.getI1Type();
  first.attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("value", builder.getBoolAttr(true))});
  OfflineTypedGraph second = first;
  second.id = 1;
  second.canonicalKey = "different-enumeration-id";

  OfflineEquivalenceProofResult proofs;
  OfflineSemanticProofPair identity;
  identity.lhsGraphId = 0;
  identity.rhsGraphId = 1;
  identity.status = SemanticProofStatus::Proved;
  proofs.pairs.push_back(std::move(identity));
  proofs.provedPairs = 1;
  SmallVector<OfflineTypedGraph, 2> graphs = {first, second};

  std::string error;
  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          graphs, proofs, "universe", "semantics",
          /*enumerationComplete=*/true, /*fingerprintComplete=*/true,
          /*base=*/nullptr, error);
  ASSERT_TRUE(store) << error;
  EXPECT_EQ(store->graphs.size(), 1u);
  EXPECT_TRUE(store->proofEdges.empty());
  EXPECT_EQ(store->classes.size(), 1u);
}

} // namespace

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

#include "TritonToGraph/RuleDatabase.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static PersistentGraphRecord makeGraph(
    llvm::StringRef operationName, llvm::StringRef resultType,
    llvm::StringRef attributes, unsigned operatorSize,
    std::vector<std::string> operandHashes = {}) {
  PersistentGraphRecord graph{"", operationName.str(), resultType.str(),
                              attributes.str(), operatorSize,
                              std::move(operandHashes)};
  graph.graphHash = getPersistentGraphHash(graph);
  return graph;
}

static PersistentEquivalenceClassStore
makeStore(llvm::StringRef resultType = "i32",
          llvm::StringRef universeHash = "universe-v1") {
  PersistentEquivalenceClassStore store;
  store.universeHash = universeHash.str();
  store.semanticsHash = "semantics-v1";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  PersistentGraphRecord input{
      "", "ags.input", resultType.str(), "{}", 0, {}};
  input.graphHash = getPersistentGraphHash(input);
  PersistentGraphRecord zero{
      "", "arith.constant", resultType.str(),
      ("{value = 0 : " + resultType + "}").str(), 0, {}};
  zero.graphHash = getPersistentGraphHash(zero);
  PersistentGraphRecord sub{
      "", "arith.subi", resultType.str(), "{}", 1,
      {input.graphHash, zero.graphHash}};
  sub.graphHash = getPersistentGraphHash(sub);
  store.graphs = {input, zero, sub};
  store.proofEdges = {
      {input.graphHash, sub.graphHash, "proof-v1", "certificate-v1"},
  };
  store.classes = {
      {("identity-class-" + resultType).str(), resultType.str(),
       {input.graphHash, sub.graphHash}},
      {("zero-class-" + resultType).str(), resultType.str(),
       {zero.graphHash}},
  };
  return store;
}

static ParetoExtractionResult makePareto(
    const PersistentEquivalenceClassStore &store) {
  ParetoExtractionResult result;
  result.eclassStoreHash = store.getStableHash();
  result.costContextKey = "costctx-v1";
  result.costModelName = "test-model";
  result.costModelHash = "test-model-v1";
  ParetoCandidateCost inputCost;
  ParetoCandidateCost zeroCost;
  zeroCost.structural.operatorCount = 1;
  ParetoCandidateCost subCost;
  subCost.structural.operatorCount = 2;
  result.classes = {
      {store.classes[0].classHash, true,
       {{store.graphs[0].graphHash, inputCost},
        {store.graphs[2].graphHash, subCost}},
       {{store.graphs[0].graphHash, inputCost}}},
      {store.classes[1].classHash, true,
       {{store.graphs[1].graphHash, zeroCost}},
       {{store.graphs[1].graphHash, zeroCost}}},
  };
  result.candidateCountBefore = 3;
  result.candidateCountAfter = 2;
  result.structuralFallbackClassCount = 2;
  return result;
}

TEST(RuleDatabaseTest, CompilesRoundTripsAndLooksUpWithoutSearch) {
  PersistentEquivalenceClassStore store = makeStore();
  ParetoExtractionResult pareto = makePareto(store);
  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  EXPECT_EQ(database->entries.size(), 1u);
  EXPECT_EQ(database->lookup(store.graphs[0].graphHash), nullptr);
  EXPECT_EQ(database->lookup(store.graphs[1].graphHash), nullptr);
  const RuleDatabaseEntry *sub =
      database->lookup(store.graphs[2].graphHash);
  ASSERT_NE(sub, nullptr);
  ASSERT_EQ(sub->candidates.size(), 1u);
  EXPECT_EQ(sub->candidates.front().rootGraphHash,
            store.graphs[0].graphHash);
  EXPECT_EQ(sub->proofHashes, std::vector<std::string>{"proof-v1"});

  std::string serialized = database->toJSON();
  std::optional<RuleDatabase> reloaded =
      RuleDatabase::fromJSON(serialized, error);
  ASSERT_TRUE(reloaded) << error;
  EXPECT_EQ(reloaded->toJSON(), serialized);

  OnlineRuleMatcher matcher(*reloaded);
  OnlineRuleLookupResult hit = matcher.lookup(
      store.graphs[2].graphHash, "universe-v1", "semantics-v1",
      "costctx-v1");
  EXPECT_EQ(hit.status, OnlineRuleLookupStatus::Hit);
  ASSERT_NE(hit.entry, nullptr);
  EXPECT_EQ(hit.entry->classHash, "identity-class-i32");
  OnlineRuleLookupResult miss = matcher.lookup(
      "absent", "universe-v1", "semantics-v1", "costctx-v1");
  EXPECT_EQ(miss.status, OnlineRuleLookupStatus::Miss);
  OnlineRuleLookupResult mismatch = matcher.lookup(
      store.graphs[2].graphHash, "wrong-universe", "semantics-v1",
      "costctx-v1");
  EXPECT_EQ(mismatch.status, OnlineRuleLookupStatus::VersionMismatch);
}

TEST(RuleDatabaseTest,
     QuotientsOnlyUnreferencedFinalRootsAndKeepsSemanticLookup) {
  RuleDatabase database;
  database.eclassStoreHash = "store-v1";
  database.universeHash = "universe-v1";
  database.semanticsHash = "semantics-v1";
  database.costContextKey = "costctx-v1";
  database.costModelName = "test-model";
  database.costModelHash = "test-model-v1";

  PersistentGraphRecord input = makeGraph("ags.input", "i32", "{}", 0);
  PersistentGraphRecord rootA =
      makeGraph("arith.subi", "i32", "{variant = 0}", 1,
                {input.graphHash});
  PersistentGraphRecord rootB =
      makeGraph("arith.subi", "i32", "{variant = 1}", 1,
                {input.graphHash});
  database.graphs = {input, rootA, rootB};

  ParetoCandidateCost cost;
  RuleDatabaseEntry entryA;
  entryA.lhsGraphHash = rootA.graphHash;
  entryA.classHash = "class-a";
  entryA.resultType = "i32";
  entryA.sourceCost = cost;
  entryA.candidates.push_back({rootA.graphHash, cost});
  RuleDatabaseEntry entryB = entryA;
  entryB.lhsGraphHash = rootB.graphHash;
  entryB.classHash = "class-b";
  entryB.candidates.front().rootGraphHash = rootB.graphHash;
  database.entries = {entryA, entryB};

  const std::string semanticValueKey = "semantic-key";
  RuleDatabaseSemanticEntry semantic;
  semantic.semanticKey =
      getRuleDatabaseSemanticKeyDigest(semanticValueKey);
  semantic.classHash = "class-a";
  semantic.resultType = "i32";
  semantic.proofHashes = {"proof-v1"};
  semantic.candidates = {{rootA.graphHash, cost}, {rootB.graphHash, cost}};
  database.semanticEntries = {semantic};

  std::vector<CertifiedSemanticGraph> certificates = {
      {rootA.graphHash, semanticValueKey, "proof-v1"},
      {rootB.graphHash, semanticValueKey, "proof-v1"},
  };
  uint64_t prunedGraphs = 0;
  uint64_t prunedEntries = 0;
  std::string error;
  std::optional<RuleDatabase> quotiented =
      RuleDatabase::quotientFinalLayerBySemanticKey(
          database, certificates, prunedGraphs, prunedEntries, error);
  ASSERT_TRUE(quotiented) << error;
  EXPECT_EQ(prunedGraphs, 1u);
  EXPECT_EQ(quotiented->graphs.size(), 2u);
  EXPECT_EQ(quotiented->lookup(rootB.graphHash), nullptr);
  ASSERT_EQ(quotiented->semanticEntries.size(), 1u);
  ASSERT_EQ(quotiented->semanticEntries.front().candidates.size(), 1u);
  EXPECT_EQ(quotiented->semanticEntries.front().candidates.front().rootGraphHash,
            rootA.graphHash);
  std::vector<const RuleDatabaseSemanticEntry *> semanticMatches =
      quotiented->lookupSemantic(semanticValueKey, "i32");
  ASSERT_EQ(semanticMatches.size(), 1u);
  ASSERT_EQ(semanticMatches.front()->candidates.size(), 1u);
  EXPECT_EQ(semanticMatches.front()->candidates.front().rootGraphHash,
            rootA.graphHash);
  EXPECT_TRUE(RuleDatabase::fromJSON(quotiented->toJSON(), error)) << error;
}

TEST(RuleDatabaseTest, ExactEntriesUseOneBoundedClassProofReference) {
  PersistentEquivalenceClassStore store = makeStore();
  PersistentGraphRecord add{
      "", "arith.addi", "i32", "{}", 1,
      {store.graphs[0].graphHash, store.graphs[1].graphHash}};
  add.graphHash = getPersistentGraphHash(add);
  store.graphs.push_back(add);
  store.classes.front().memberGraphHashes.push_back(add.graphHash);
  llvm::sort(store.classes.front().memberGraphHashes);
  store.proofEdges.push_back({store.graphs[0].graphHash, add.graphHash,
                              "proof-v2", "certificate-v2"});

  ParetoExtractionResult pareto = makePareto(store);
  ParetoCandidateCost addCost;
  addCost.structural.operatorCount = 2;
  pareto.classes.front().allCandidates.push_back({add.graphHash, addCost});
  pareto.candidateCountBefore = 4;
  pareto.eclassStoreHash = store.getStableHash();

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  for (llvm::StringRef member : store.classes.front().memberGraphHashes) {
    const RuleDatabaseEntry *entry = database->lookup(member);
    if (member == store.graphs[0].graphHash) {
      EXPECT_EQ(entry, nullptr);
      continue;
    }
    ASSERT_NE(entry, nullptr);
    ASSERT_EQ(entry->proofHashes.size(), 1u);
    EXPECT_EQ(entry->proofHashes.front().rfind("bundle:", 0), 0u);
  }
}

TEST(RuleDatabaseTest, OmitsSingletonIdentitiesAndUnreferencedGraphs) {
  PersistentEquivalenceClassStore store = makeStore();
  PersistentGraphRecord one{
      "", "arith.constant", "i32", "{value = 1 : i32}", 0, {}};
  one.graphHash = getPersistentGraphHash(one);
  store.graphs.push_back(one);
  store.classes.push_back(
      {"one-class-i32", "i32", {one.graphHash}});

  ParetoExtractionResult pareto = makePareto(store);
  ParetoCandidateCost oneCost;
  oneCost.structural.operatorCount = 1;
  pareto.classes.push_back(
      {"one-class-i32", true, {{one.graphHash, oneCost}},
       {{one.graphHash, oneCost}}});
  pareto.candidateCountBefore = 4;
  pareto.candidateCountAfter = 3;
  pareto.structuralFallbackClassCount = 3;
  pareto.eclassStoreHash = store.getStableHash();

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  EXPECT_EQ(database->lookup(one.graphHash), nullptr);
  EXPECT_EQ(database->lookupGraph(one.graphHash), nullptr);
  EXPECT_EQ(database->entries.size(), 1u);
  EXPECT_EQ(database->graphs.size(), 3u);
}

TEST(RuleDatabaseTest, RejectsTamperedArtifact) {
  PersistentEquivalenceClassStore store = makeStore();
  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, makePareto(store), error);
  ASSERT_TRUE(database) << error;
  std::string serialized = database->toJSON();
  size_t position = serialized.find("test-model-v1");
  ASSERT_NE(position, std::string::npos);
  serialized[position] = 'X';
  EXPECT_FALSE(RuleDatabase::fromJSON(serialized, error));
  EXPECT_FALSE(error.empty());
}

TEST(RuleDatabaseTest, RejectsCandidateOutsideCertifiedClass) {
  PersistentEquivalenceClassStore store = makeStore();
  ParetoExtractionResult pareto = makePareto(store);
  pareto.classes.front().candidates.front().graphHash =
      store.graphs[1].graphHash;
  std::string error;
  EXPECT_FALSE(RuleDatabase::compile(store, pareto, error));
  EXPECT_FALSE(error.empty());
}

TEST(RuleDatabaseTest, RejectsEClassWithMixedConcreteResultTypes) {
  PersistentEquivalenceClassStore store = makeStore();
  store.classes.front().resultType = "i64";
  ParetoExtractionResult pareto = makePareto(store);
  std::string error;
  EXPECT_FALSE(RuleDatabase::compile(store, pareto, error));
  EXPECT_EQ(error, "e-class member result type does not match its class");
}

TEST(RuleDatabaseTest, RejectsIncompleteStructuralCosts) {
  PersistentEquivalenceClassStore store = makeStore();
  ParetoExtractionResult pareto = makePareto(store);
  pareto.invalidStructuralGraphCount = 1;
  std::string error;
  EXPECT_FALSE(RuleDatabase::compile(store, pareto, error));
  EXPECT_EQ(error,
            "Pareto report has graphs without a valid structural cost");
}

TEST(RuleDatabaseTest, RejectsIncompleteEquivalenceClassStore) {
  PersistentEquivalenceClassStore store = makeStore();
  ParetoExtractionResult pareto = makePareto(store);
  store.proofComplete = false;
  // The Pareto artifact still names the complete store hash. Compilation must
  // fail on store completeness before considering artifact hash consistency.
  std::string error;
  EXPECT_FALSE(RuleDatabase::compile(store, pareto, error));
  EXPECT_EQ(error,
            "cannot compile a rule database from an incomplete e-class store");
}

TEST(RuleDatabaseTest, MergesCompatibleFiniteUniverseShards) {
  PersistentEquivalenceClassStore i32Store =
      makeStore("i32", "universe-i32");
  PersistentEquivalenceClassStore i64Store =
      makeStore("i64", "universe-i64");
  std::string error;
  std::optional<RuleDatabase> i32Database =
      RuleDatabase::compile(i32Store, makePareto(i32Store), error);
  ASSERT_TRUE(i32Database) << error;
  std::optional<RuleDatabase> i64Database =
      RuleDatabase::compile(i64Store, makePareto(i64Store), error);
  ASSERT_TRUE(i64Database) << error;

  std::vector<RuleDatabase> shards{*i64Database, *i32Database};
  std::optional<RuleDatabase> merged = RuleDatabase::merge(shards, error);
  ASSERT_TRUE(merged) << error;
  EXPECT_EQ(merged->graphs.size(), 6u);
  EXPECT_EQ(merged->entries.size(), 2u);
  EXPECT_NE(merged->lookup(i32Store.graphs[2].graphHash), nullptr);
  EXPECT_NE(merged->lookup(i64Store.graphs[2].graphHash), nullptr);
  EXPECT_EQ(merged->semanticsHash, "semantics-v1");
  EXPECT_EQ(merged->costContextKey, "costctx-v1");
  EXPECT_EQ(merged->universeHash.rfind("bundle:", 0), 0u);

  std::string serialized = merged->toJSON();
  std::optional<RuleDatabase> reloaded =
      RuleDatabase::fromJSON(serialized, error);
  ASSERT_TRUE(reloaded) << error;
  EXPECT_EQ(reloaded->toJSON(), serialized);
}

TEST(RuleDatabaseTest, MergesSemanticCandidatesAcrossPartitions) {
  PersistentGraphRecord lhs =
      makeGraph("ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 0);
  ParetoCandidateCost lhsCost;
  lhsCost.structural.operatorCount = 1;
  ParetoCandidateCost rhsCost;

  auto makeShard = [](llvm::StringRef universe, llvm::StringRef eclasses,
                      const PersistentGraphRecord &graph,
                      const ParetoCandidateCost &cost,
                      llvm::StringRef proof) {
    RuleDatabase shard;
    shard.universeHash = universe.str();
    shard.eclassStoreHash = eclasses.str();
    shard.semanticsHash = "semantics-v1";
    shard.costContextKey = "costctx-v1";
    shard.costModelName = "test-model";
    shard.costModelHash = "test-model-v1";
    shard.graphs = {graph};
    shard.semanticEntries = {
        {getRuleDatabaseSemanticKeyDigest("same-normalized-denotation"),
         eclasses.str(), "i32",
         {proof.str()}, {{graph.graphHash, cost}}}};
    return shard;
  };

  std::vector<RuleDatabase> shards = {
      makeShard("partition-0", "eclass-0", lhs, lhsCost, "proof-0"),
      makeShard("partition-1", "eclass-1", rhs, rhsCost, "proof-1")};
  std::string error;
  std::optional<RuleDatabase> merged = RuleDatabase::merge(shards, error);
  ASSERT_TRUE(merged) << error;
  std::vector<const RuleDatabaseSemanticEntry *> matches =
      merged->lookupSemantic("same-normalized-denotation", "i32");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches.front()->candidates.size(), 2u);
  EXPECT_EQ(matches.front()->proofHashes,
            std::vector<std::string>({"proof-0", "proof-1"}));
  EXPECT_TRUE(RuleDatabase::fromJSON(merged->toJSON(), error)) << error;
}

TEST(RuleDatabaseTest, MergesEqualIdentityShardsDeterministically) {
  PersistentGraphRecord lhs =
      makeGraph("ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 0);
  ParetoCandidateCost lhsCost;
  lhsCost.structural.operatorCount = 1;
  ParetoCandidateCost rhsCost;

  auto makeShard = [&](const PersistentGraphRecord &graph,
                       const ParetoCandidateCost &cost,
                       llvm::StringRef classHash,
                       llvm::StringRef proof) {
    RuleDatabase shard;
    shard.universeHash = "same-universe";
    shard.eclassStoreHash = "same-eclasses";
    shard.semanticsHash = "semantics-v1";
    shard.costContextKey = "costctx-v1";
    shard.costModelName = "test-model";
    shard.costModelHash = "test-model-v1";
    shard.graphs = {graph};
    shard.semanticEntries = {
        {getRuleDatabaseSemanticKeyDigest("same-normalized-denotation"),
         classHash.str(), "i32",
         {proof.str()}, {{graph.graphHash, cost}}}};
    return shard;
  };

  RuleDatabase first = makeShard(lhs, lhsCost, "class-a", "proof-a");
  RuleDatabase second = makeShard(rhs, rhsCost, "class-b", "proof-b");
  std::string error;
  std::vector<RuleDatabase> forwardShards{first, second};
  std::optional<RuleDatabase> forward =
      RuleDatabase::merge(forwardShards, error);
  ASSERT_TRUE(forward) << error;
  std::vector<RuleDatabase> reverseShards{second, first};
  std::optional<RuleDatabase> reverse =
      RuleDatabase::merge(reverseShards, error);
  ASSERT_TRUE(reverse) << error;
  EXPECT_EQ(forward->toJSON(), reverse->toJSON());
  EXPECT_EQ(forward->getStableHash(), reverse->getStableHash());
}

TEST(RuleDatabaseTest, HierarchicalMergeEqualsFlatMerge) {
  std::vector<RuleDatabase> shards;
  std::string error;
  for (unsigned index = 0; index < 4; ++index) {
    PersistentEquivalenceClassStore store = makeStore(
        "i32", (llvm::Twine("hierarchical-universe-") +
                llvm::Twine(index)).str());
    std::optional<RuleDatabase> database =
        RuleDatabase::compile(store, makePareto(store), error);
    ASSERT_TRUE(database) << error;
    shards.push_back(std::move(*database));
  }

  std::optional<RuleDatabase> flat = RuleDatabase::merge(shards, error);
  ASSERT_TRUE(flat) << error;
  std::vector<RuleDatabase> firstHalf{shards[0], shards[1]};
  std::vector<RuleDatabase> secondHalf{shards[2], shards[3]};
  std::optional<RuleDatabase> first =
      RuleDatabase::merge(firstHalf, error);
  ASSERT_TRUE(first) << error;
  std::optional<RuleDatabase> second =
      RuleDatabase::merge(secondHalf, error);
  ASSERT_TRUE(second) << error;
  std::vector<RuleDatabase> groups{std::move(*first), std::move(*second)};
  std::optional<RuleDatabase> hierarchical =
      RuleDatabase::merge(groups, error);
  ASSERT_TRUE(hierarchical) << error;

  EXPECT_EQ(hierarchical->toJSON(), flat->toJSON());
  EXPECT_EQ(hierarchical->getStableHash(), flat->getStableHash());
}

TEST(RuleDatabaseTest, MergesTemplateCandidatesAcrossPartitions) {
  PersistentGraphRecord variable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 0);
  PersistentGraphRecord source = makeGraph(
      "arith.subi", "i32", "{}", 1,
      {variable.graphHash, zero.graphHash});
  PersistentGraphRecord alternate = makeGraph(
      "arith.addi", "i32", "{}", 1,
      {variable.graphHash, zero.graphHash});
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  ParetoCandidateCost identityCost;
  ParetoCandidateCost alternateCost;
  alternateCost.structural.operatorCount = 1;

  auto makeShard = [&](llvm::StringRef universe, llvm::StringRef eclasses,
                       const PersistentGraphRecord &candidate,
                       const ParetoCandidateCost &cost,
                       llvm::StringRef proof) {
    RuleDatabase shard;
    shard.universeHash = universe.str();
    shard.eclassStoreHash = eclasses.str();
    shard.semanticsHash = "semantics-v1";
    shard.costContextKey = "costctx-v1";
    shard.costModelName = "test-model";
    shard.costModelHash = "test-model-v1";
    shard.graphs = {variable, zero, source};
    if (llvm::none_of(shard.graphs, [&](const auto &graph) {
          return graph.graphHash == candidate.graphHash;
        }))
      shard.graphs.push_back(candidate);
    llvm::sort(shard.graphs, [](const auto &lhs, const auto &rhs) {
      return lhs.graphHash < rhs.graphHash;
    });
    shard.templates = {
        {source.graphHash,
         eclasses.str(),
         "i32",
         {proof.str()},
         sourceCost,
         {{candidate.graphHash, cost}}}};
    return shard;
  };

  std::vector<RuleDatabase> shards = {
      makeShard("partition-0", "template-eclass-0", variable,
                identityCost, "template-proof-0"),
      makeShard("partition-1", "template-eclass-1", alternate,
                alternateCost, "template-proof-1")};
  std::string error;
  std::optional<RuleDatabase> merged = RuleDatabase::merge(shards, error);
  ASSERT_TRUE(merged) << error;
  ASSERT_EQ(merged->templates.size(), 1u);
  EXPECT_EQ(merged->templates.front().candidates.size(), 2u);
  EXPECT_EQ(merged->templates.front().proofHashes,
            std::vector<std::string>({"template-proof-0",
                                      "template-proof-1"}));
  EXPECT_TRUE(RuleDatabase::fromJSON(merged->toJSON(), error)) << error;
}

TEST(RuleDatabaseTest, RoundTripsMatchesAndMergesQuantifiedTemplates) {
  PersistentGraphRecord memoryPattern = makeGraph(
      "ags.partitionable_memory", "none",
      "{axis = 0 : i32, ceil_div = 1 : i32}", 0);
  PersistentGraphRecord launch =
      makeGraph("ags.launch", "none", "{}", 0);
  PersistentGraphRecord sourcePattern = makeGraph(
      "ags.kernel", "none",
      "{ags.proof_only = true, ags.state_domain = \"memory\"}", 1,
      {memoryPattern.graphHash, launch.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "ags.kernel_partition", "none",
      "{ags.compiler_action = {axis = 0 : i32, ceil_div = 1 : i32, "
      "factor = 4 : i32, kind = \"launch_partition\"}, "
      "ags.proof_only = true}",
      1, {memoryPattern.graphHash, launch.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.weightedOperatorCost = 5;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.weightedOperatorCost = 2;
  RuleDatabase database;
  database.eclassStoreHash = "quantified-eclasses";
  database.universeHash = "quantified-universe-a";
  database.semanticsHash = "quantified-semantics";
  database.costContextKey = "quantified-cost-context";
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {memoryPattern, launch, sourcePattern, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {sourcePattern.graphHash,
       "quantified-launch-class",
       "none",
       {"quantified-proof"},
       sourceCost,
       {{candidate.graphHash, candidateCost}}}};

  std::string error;
  std::optional<RuleDatabase> reloaded =
      RuleDatabase::fromJSON(database.toJSON(), error);
  ASSERT_TRUE(reloaded) << error;
  ASSERT_EQ(reloaded->templates.size(), 1u);

  PersistentGraphRecord concreteMemory =
      makeGraph("tt.store", "none", "{concrete = true}", 7);
  PersistentGraphRecord concreteKernel = makeGraph(
      "ags.kernel", "none", sourcePattern.attributes, 8,
      {concreteMemory.graphHash, launch.graphHash});
  std::vector<PersistentGraphRecord> onlineGraphs = {
      concreteMemory, launch, concreteKernel};
  std::vector<RuleDatabaseTemplateMatch> matches =
      reloaded->matchTemplates(concreteKernel.graphHash, onlineGraphs);
  ASSERT_EQ(matches.size(), 1u);
  ASSERT_EQ(matches.front().bindings.size(), 1u);
  EXPECT_EQ(matches.front().bindings.front().first, memoryPattern.graphHash);
  EXPECT_EQ(matches.front().bindings.front().second, concreteMemory.graphHash);

  RuleDatabase shard = *reloaded;
  shard.universeHash = "quantified-universe-b";
  shard.eclassStoreHash = "quantified-eclasses-b";
  std::vector<RuleDatabase> shards{*reloaded, shard};
  std::optional<RuleDatabase> merged = RuleDatabase::merge(shards, error);
  ASSERT_TRUE(merged) << error;
  ASSERT_EQ(merged->templates.size(), 1u);
  EXPECT_EQ(merged->templates.front().proofHashes,
            std::vector<std::string>{"quantified-proof"});
  EXPECT_EQ(merged->templates.front().candidates.size(), 1u);
  EXPECT_TRUE(RuleDatabase::fromJSON(merged->toJSON(), error));
}

TEST(RuleDatabaseTest, RoundTripsAndLooksUpCertifiedSemanticIndex) {
  PersistentGraphRecord input =
      makeGraph("ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  ParetoCandidateCost cost;
  RuleDatabase database;
  database.eclassStoreHash = "semantic-eclasses";
  database.universeHash = "semantic-universe";
  database.semanticsHash = "semantic-version";
  database.costContextKey = "semantic-cost-context";
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {input};
  const std::string semanticKey =
      "shape[]layout[]indices[]value=(symbol 0 : i32);valid=(bool true : i1)";
  database.semanticEntries = {
      {getRuleDatabaseSemanticKeyDigest(semanticKey), "semantic-class", "i32",
       {"proof-hash"}, {{input.graphHash, cost}}}};

  std::string error;
  std::optional<RuleDatabase> reloaded =
      RuleDatabase::fromJSON(database.toJSON(), error);
  ASSERT_TRUE(reloaded) << error;
  std::vector<const RuleDatabaseSemanticEntry *> matches =
      reloaded->lookupSemantic(semanticKey, "i32");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches.front()->classHash, "semantic-class");
  EXPECT_TRUE(reloaded->lookupSemantic(semanticKey, "i64").empty());
}

TEST(RuleDatabaseTest, CompilesSemanticIndexOnlyFromReplayedV2Certificate) {
  MLIRContext context;
  Type i32 = IntegerType::get(&context, 32);
  TensorDenotation identity;
  identity.value = TensorSemanticExpr::symbol(&context, "input-0", i32);
  identity.defined = TensorSemanticExpr::boolean(&context, true);
  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  std::optional<SemanticEquivalenceProof> proof =
      oracle.prove(identity, identity);
  ASSERT_TRUE(proof);

  PersistentEquivalenceClassStore store = makeStore();
  store.proofEdges.front().proofHash = proof->getStableHash();
  store.proofEdges.front().proofCertificate = proof->toJSON();
  ParetoExtractionResult pareto = makePareto(store);
  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  ASSERT_EQ(database->semanticEntries.size(), 1u);
  ASSERT_EQ(database->templates.size(), 1u);
  EXPECT_EQ(database->templates.front().lhsPatternGraphHash,
            store.graphs[2].graphHash);
  ASSERT_EQ(database->templates.front().candidates.size(), 1u);
  EXPECT_EQ(database->templates.front().candidates.front().rootGraphHash,
            store.graphs.front().graphHash);
  EXPECT_EQ(database->semanticEntries.front().semanticKey,
            getRuleDatabaseSemanticKeyDigest(
                getStableTensorDenotationKey(identity)));
  EXPECT_EQ(database->lookupSemantic(getStableTensorDenotationKey(identity),
                                     "i32")
                .size(),
            1u);
  EXPECT_EQ(database->semanticEntries.front().proofHashes,
            std::vector<std::string>{proof->getStableHash()});
  EXPECT_EQ(database->semanticEntries.front().candidates.front().rootGraphHash,
            store.graphs.front().graphHash);
}

TEST(RuleDatabaseTest, RejectsTemplateWhoseRhsIntroducesANewVariable) {
  PersistentGraphRecord lhsVariable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhsVariable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 1);
  PersistentGraphRecord source = makeGraph(
      "arith.subi", "i32", "{}", 2,
      {lhsVariable.graphHash, zero.graphHash});
  RuleDatabase database;
  database.eclassStoreHash = "invalid-template-eclasses";
  database.universeHash = "invalid-template-universe";
  database.semanticsHash = "invalid-template-semantics";
  database.costContextKey = "invalid-template-context";
  database.costModelName = "structural";
  database.costModelHash = "structural-v1";
  database.graphs = {lhsVariable, rhsVariable, zero, source};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  RuleDatabaseTemplateEntry invalid;
  invalid.lhsPatternGraphHash = source.graphHash;
  invalid.classHash = "invalid-template-class";
  invalid.resultType = "i32";
  invalid.proofHashes = {"not-a-proof"};
  invalid.sourceCost = sourceCost;
  invalid.candidates = {{rhsVariable.graphHash, ParetoCandidateCost{}}};
  database.templates = {invalid};

  std::string error;
  EXPECT_FALSE(RuleDatabase::fromJSON(database.toJSON(), error));
  EXPECT_NE(error.find("closed symbolic instantiation"), std::string::npos);
}

TEST(RuleDatabaseTest, CompilesCertifiedSingletonSemanticIndex) {
  PersistentEquivalenceClassStore store = makeStore();
  CertifiedSemanticGraph certified{store.graphs[1].graphHash,
                                    "normalized-zero-i32",
                                    "reflexive-zero-proof"};
  std::string error;
  std::optional<RuleDatabase> database = RuleDatabase::compile(
      store, makePareto(store), error, {certified});
  ASSERT_TRUE(database) << error;
  ASSERT_EQ(database->semanticEntries.size(), 1u);
  const RuleDatabaseSemanticEntry &entry =
      database->semanticEntries.front();
  EXPECT_EQ(entry.semanticKey,
            getRuleDatabaseSemanticKeyDigest(certified.semanticKey));
  EXPECT_EQ(database->lookupSemantic(certified.semanticKey, "i32").size(),
            1u);
  EXPECT_EQ(entry.proofHashes,
            std::vector<std::string>{certified.proofHash});
  ASSERT_EQ(entry.candidates.size(), 1u);
  EXPECT_EQ(entry.candidates.front().rootGraphHash,
            store.graphs[1].graphHash);
}

TEST(RuleDatabaseTest,
     CoalescesCertifiedSemanticNormalFormAcrossDistinctEclasses) {
  PersistentEquivalenceClassStore store = makeStore();
  const std::string semanticKey = "shared-normalized-value-valid-form";
  std::vector<CertifiedSemanticGraph> certified = {
      {store.graphs[0].graphHash, semanticKey, "reflexive-input-proof"},
      {store.graphs[1].graphHash, semanticKey, "reflexive-zero-proof"},
  };
  std::string error;
  std::optional<RuleDatabase> database = RuleDatabase::compile(
      store, makePareto(store), error, certified);
  ASSERT_TRUE(database) << error;
  ASSERT_EQ(database->semanticEntries.size(), 1u);
  const RuleDatabaseSemanticEntry &entry =
      database->semanticEntries.front();
  EXPECT_EQ(entry.semanticKey,
            getRuleDatabaseSemanticKeyDigest(semanticKey));
  EXPECT_EQ(entry.classHash.rfind("semantic:", 0), 0u);
  EXPECT_EQ(entry.proofHashes,
            (std::vector<std::string>{"proof-v1",
                                      "reflexive-input-proof",
                                      "reflexive-zero-proof"}));
  ASSERT_EQ(entry.candidates.size(), 2u);
  std::vector<std::string> candidates;
  for (const RuleDatabaseCandidate &candidate : entry.candidates)
    candidates.push_back(candidate.rootGraphHash);
  std::vector<std::string> expectedCandidates = {
      store.graphs[0].graphHash, store.graphs[1].graphHash};
  llvm::sort(expectedCandidates);
  EXPECT_EQ(candidates, expectedCandidates);
  EXPECT_EQ(database->lookupSemantic(semanticKey, "i32").size(), 1u);
  EXPECT_TRUE(RuleDatabase::fromJSON(database->toJSON(), error)) << error;
}

TEST(RuleDatabaseTest, RejectsSemanticCertificationForUnknownGraph) {
  PersistentEquivalenceClassStore store = makeStore();
  CertifiedSemanticGraph certified{"missing-graph", "semantic-key",
                                    "proof-hash"};
  std::string error;
  EXPECT_FALSE(RuleDatabase::compile(store, makePareto(store), error,
                                     {certified}));
  EXPECT_NE(error.find("unknown graph"), std::string::npos);
}

TEST(RuleDatabaseTest, RejectsConflictingGraphContentAcrossShards) {
  PersistentEquivalenceClassStore store = makeStore();
  std::string error;
  std::optional<RuleDatabase> first =
      RuleDatabase::compile(store, makePareto(store), error);
  ASSERT_TRUE(first) << error;
  RuleDatabase conflicting = *first;
  conflicting.universeHash = "another-universe";
  conflicting.graphs.front().operationName = "ags.conflicting_input";

  std::vector<RuleDatabase> shards{*first, conflicting};
  EXPECT_FALSE(RuleDatabase::merge(shards, error));
  EXPECT_NE(error.find("conflicting graph hash"), std::string::npos);
}

TEST(RuleDatabaseTest, RejectsIncompatibleShardSemantics) {
  PersistentEquivalenceClassStore store = makeStore();
  std::string error;
  std::optional<RuleDatabase> first =
      RuleDatabase::compile(store, makePareto(store), error);
  ASSERT_TRUE(first) << error;
  RuleDatabase incompatible = *first;
  incompatible.universeHash = "another-universe";
  incompatible.semanticsHash = "semantics-v2";

  std::vector<RuleDatabase> shards{*first, incompatible};
  EXPECT_FALSE(RuleDatabase::merge(shards, error));
  EXPECT_NE(error.find("incompatible"), std::string::npos);
}

} // namespace

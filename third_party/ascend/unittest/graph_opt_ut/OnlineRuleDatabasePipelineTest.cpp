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

#include "TritonToGraph/GraphOptimization.h"
#include "TritonToGraph/LayoutMemoryOptimization.h"
#include "TritonToGraph/Passes.h"
#include "TritonToGraph/CanonicalGraphKey.h"
#include "TritonToGraph/RuleDatabase.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static PersistentGraphRecord makeGraph(
    std::string operation, std::string type, std::string attributes,
    unsigned operatorSize, std::vector<std::string> operands = {}) {
  PersistentGraphRecord graph;
  graph.operationName = std::move(operation);
  graph.resultType = std::move(type);
  graph.attributes = std::move(attributes);
  graph.operatorSize = operatorSize;
  graph.operandHashes = std::move(operands);
  graph.graphHash = getPersistentGraphHash(graph);
  return graph;
}

static std::string printAttributes(DictionaryAttr attributes) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << attributes;
  return storage;
}

TEST(OnlineRuleDatabasePipelineTest,
     MissingDatabaseDoesNotStartOnlineEnumeration) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @keep_without_database(%input: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %result = arith.subi %input, %zero : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  ASSERT_TRUE(options.autoGraphRuleDatabase.empty());
  ASSERT_FALSE(options.allowOnlineSemanticEnumeration);
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtracts = 0;
  module->walk([&](arith::SubIOp) { ++subtracts; });
  EXPECT_EQ(subtracts, 1u);
}

TEST(OnlineRuleDatabasePipelineTest, AppliesDatabaseRecipeWithoutEnumeration) {
  PersistentGraphRecord input = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 0);
  PersistentGraphRecord subtract =
      makeGraph("arith.subi", "i32", "{}", 1,
                {input.graphHash, zero.graphHash});

  PersistentEquivalenceClassStore store;
  store.universeHash = "online-rule-db-pipeline-test";
  store.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {input, zero, subtract};
  store.proofEdges = {
      {input.graphHash, subtract.graphHash, "proof-v1", "certificate-v1"}};
  store.classes = {
      {"identity-class", "i32", {input.graphHash, subtract.graphHash}},
      {"zero-class", "i32", {zero.graphHash}}};

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost inputCost;
  inputCost.stageCycles = 100;
  ParetoCandidateCost subtractCost;
  subtractCost.structural.operatorCount = 1;
  subtractCost.structural.weightedOperatorCost = 1;
  subtractCost.stageCycles = 100;
  ParetoCandidateCost zeroCost;
  ParetoExtractionResult pareto;
  pareto.eclassStoreHash = store.getStableHash();
  pareto.costContextKey = costContext.getStableKey();
  pareto.costModelName = "test-stage-model";
  pareto.costModelHash = "test-stage-model-v1";
  pareto.classes = {
      {"identity-class", false,
       {{input.graphHash, inputCost}, {subtract.graphHash, subtractCost}},
       {{input.graphHash, inputCost}}},
      {"zero-class", true, {{zero.graphHash, zeroCost}},
       {{zero.graphHash, zeroCost}}}};
  pareto.candidateCountBefore = 3;
  pareto.candidateCountAfter = 2;
  pareto.structuralFallbackClassCount = 1;

  // Equal StageModel estimates are not a reason to keep an extra operation;
  // the complete structural vector is the deterministic tie breaker.

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-pipeline", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database->toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %result = arith.subi %arg0, %zero : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     LexicographicStructuralChoiceIsNotRejectedByTelemetryScalar) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @layout_priority(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %first = arith.subi %arg0, %zero : i32
        %second = arith.subi %first, %zero : i32
        tt.return %second : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("layout_priority");
  ASSERT_TRUE(function);
  Value root =
      function->getRegion(0).front().getTerminator()->getOperand(0);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;

  auto candidate =
      llvm::find_if(canonical->graphs, [](const auto &graph) {
        return graph.operationName == "arith.subi" &&
               graph.operatorSize == 1;
      });
  ASSERT_NE(candidate, canonical->graphs.end());
  const std::string candidateHash = candidate->graphHash;

  ParetoCandidateCost sourceCost;
  sourceCost.structural.memoryStridePenalty = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  sourceCost.structural.operatorCount = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.memoryStridePenalty = 0;
  candidateCost.structural.weightedOperatorCost = 20000;
  candidateCost.structural.operatorCount = 1;

  // The candidate is lexicographically cheaper because it removes the stride
  // penalty, while the telemetry scalar is deliberately larger:
  //   source    ~= 1 * 2^24
  //   candidate ~= 20000 * 1024
  // The scalar must never veto the full structural-cost decision.
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "lexicographic-cost-eclasses";
  database.universeHash = "lexicographic-cost-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = canonical->graphs;
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {canonical->rootGraphHash,
       "layout-priority-class",
       "i32",
       {"layout-priority-proof"},
       sourceCost,
       {{candidateHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-lexicographic-cost", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesCertifiedSemanticEntryToUnseenSourceTopology) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @double_identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %first = arith.subi %arg0, %zero : i32
        %second = arith.subi %first, %zero : i32
        tt.return %second : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("double_identity");
  ASSERT_TRUE(function);
  Value root =
      function->getRegion(0).front().getTerminator()->getOperand(0);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;
  ASSERT_FALSE(canonical->normalizedSemanticKey.empty());

  PersistentGraphRecord input = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  ParetoCandidateCost inputCost;
  // An unseen online LHS has no offline StageModel score.  A cycle estimate
  // attached to its certified RHS must not be compared with the LHS node
  // count; semantic-normal-form lookup falls back to structural cost on both
  // sides and still removes the two identity subtractions.
  inputCost.stageCycles = 1000;
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "semantic-topology-eclasses";
  database.universeHash = "semantic-topology-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {input};
  database.semanticEntries = {{getRuleDatabaseSemanticKeyDigest(
                                   canonical->normalizedSemanticKey),
                               "identity-semantic-class", "i32",
                               {"replayed-proof"},
                               {{input.graphHash, inputCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-semantic-topology", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     SemanticUnionCanBeatAnExactShardLocalCandidate) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partitioned_identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %first = arith.subi %arg0, %zero : i32
        %second = arith.subi %first, %zero : i32
        tt.return %second : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("partitioned_identity");
  ASSERT_TRUE(function);
  Value root =
      function->getRegion(0).front().getTerminator()->getOperand(0);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;
  ASSERT_FALSE(canonical->normalizedSemanticKey.empty());

  auto input = llvm::find_if(canonical->graphs, [](const auto &graph) {
    return graph.operationName == "ags.input" && graph.resultType == "i32";
  });
  auto localCandidate =
      llvm::find_if(canonical->graphs, [](const auto &graph) {
        return graph.operationName == "arith.subi" &&
               graph.operatorSize == 1;
      });
  ASSERT_NE(input, canonical->graphs.end());
  ASSERT_NE(localCandidate, canonical->graphs.end());

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost localCost;
  localCost.structural.operatorCount = 1;
  localCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost globalCost;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "partitioned-semantic-eclasses";
  database.universeHash = "partitioned-semantic-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = canonical->graphs;
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {canonical->rootGraphHash,
       "source-shard-local-class",
       "i32",
       {"source-shard-proof"},
       sourceCost,
       {{localCandidate->graphHash, localCost}}}};
  database.semanticEntries = {
      {getRuleDatabaseSemanticKeyDigest(canonical->normalizedSemanticKey),
       "merged-cross-shard-class",
       "i32",
       {"cross-shard-semantic-proof"},
       {{input->graphHash, globalCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-partitioned-semantic-union", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     SemanticUnionUsesExactStageCostWhenSourceIsIndexed) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partitioned_stage_identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %first = arith.subi %arg0, %zero : i32
        %second = arith.subi %first, %zero : i32
        tt.return %second : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("partitioned_stage_identity");
  ASSERT_TRUE(function);
  Value root =
      function->getRegion(0).front().getTerminator()->getOperand(0);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;
  ASSERT_FALSE(canonical->normalizedSemanticKey.empty());

  auto input = llvm::find_if(canonical->graphs, [](const auto &graph) {
    return graph.operationName == "ags.input" && graph.resultType == "i32";
  });
  auto zero = llvm::find_if(canonical->graphs, [](const auto &graph) {
    return graph.operationName == "arith.constant";
  });
  auto localCandidate =
      llvm::find_if(canonical->graphs, [](const auto &graph) {
        return graph.operationName == "arith.subi" &&
               graph.operatorSize == 1;
      });
  ASSERT_NE(input, canonical->graphs.end());
  ASSERT_NE(zero, canonical->graphs.end());
  ASSERT_NE(localCandidate, canonical->graphs.end());

  PersistentGraphRecord add1 = makeGraph(
      "arith.addi", "i32", "{}", 1,
      {input->graphHash, zero->graphHash});
  PersistentGraphRecord add2 = makeGraph(
      "arith.addi", "i32", "{}", 2,
      {add1.graphHash, zero->graphHash});
  PersistentGraphRecord add3 = makeGraph(
      "arith.addi", "i32", "{}", 3,
      {add2.graphHash, zero->graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  sourceCost.stageCycles = 100;
  ParetoCandidateCost localCost;
  localCost.structural.operatorCount = 1;
  localCost.structural.weightedOperatorCost = 1;
  localCost.stageCycles = 50;
  ParetoCandidateCost globalCost;
  // The three-add identity is structurally worse than the two-sub source.
  // The exact source and merged candidate have comparable StageModel scores,
  // so the global candidate must still win by cycles.
  globalCost.structural.operatorCount = 3;
  globalCost.structural.weightedOperatorCost = 3;
  globalCost.stageCycles = 10;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "partitioned-stage-eclasses";
  database.universeHash = "partitioned-stage-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "test-stage-model";
  database.costModelHash = "test-stage-model-v1";
  database.graphs = canonical->graphs;
  database.graphs.insert(database.graphs.end(), {add1, add2, add3});
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {canonical->rootGraphHash,
       "source-stage-class",
       "i32",
       {"source-stage-proof"},
       sourceCost,
       {{localCandidate->graphHash, localCost}}}};
  database.semanticEntries = {
      {getRuleDatabaseSemanticKeyDigest(canonical->normalizedSemanticKey),
       "merged-stage-class",
       "i32",
       {"cross-shard-stage-proof"},
       {{add3.graphHash, globalCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-partitioned-stage-union", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  unsigned addCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  module->walk([&](arith::AddIOp) { ++addCount; });
  EXPECT_EQ(subtractCount, 0u);
  EXPECT_EQ(addCount, 3u);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesCertifiedSemanticEntryToUnseenMemoryStateTopology) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @redundant_store(%ptr: !tt.ptr<i32>, %value: i32) {
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("redundant_store");
  ASSERT_TRUE(function);
  Operation *root = nullptr;
  function.walk([&](triton::StoreOp store) { root = store; });
  ASSERT_NE(root, nullptr);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;
  ASSERT_FALSE(canonical->normalizedSemanticKey.empty());

  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord memory = makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord candidateStore = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "semantic-state-eclasses";
  database.universeHash = "semantic-state-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {pointer, value, memory, candidateStore};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.semanticEntries = {
      {getRuleDatabaseSemanticKeyDigest(canonical->normalizedSemanticKey),
       "single-store-semantic-class", "none", {"replayed-state-proof"},
       {{candidateStore.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-semantic-state", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned storeCount = 0;
  module->walk([&](triton::StoreOp) { ++storeCount; });
  EXPECT_EQ(storeCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest,
     SemanticMemoryUnionUsesExactStageCostWhenSourceIsIndexed) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partitioned_stage_overwrite(%ptr: !tt.ptr<i32>,
                                            %first: i32, %final: i32) {
        tt.store %ptr, %first : !tt.ptr<i32>
        tt.store %ptr, %final : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("partitioned_stage_overwrite");
  ASSERT_TRUE(function);
  Operation *root = nullptr;
  function.walk([&](triton::StoreOp store) { root = store; });
  ASSERT_NE(root, nullptr);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;
  ASSERT_FALSE(canonical->normalizedSemanticKey.empty());

  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord firstValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord finalValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 2 : i64}", 0);
  PersistentGraphRecord memory = makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord firstStore = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, firstValue.graphHash, memory.graphHash});
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 2,
      {pointer.graphHash, finalValue.graphHash, firstStore.graphHash});
  PersistentGraphRecord localCandidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, finalValue.graphHash, memory.graphHash});
  PersistentGraphRecord repeatedFirst = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 2,
      {pointer.graphHash, firstValue.graphHash, firstStore.graphHash});
  PersistentGraphRecord globalCandidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 3,
      {pointer.graphHash, finalValue.graphHash, repeatedFirst.graphHash});
  ASSERT_EQ(source.graphHash, canonical->rootGraphHash);

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  sourceCost.stageCycles = 100;
  ParetoCandidateCost localCost;
  localCost.structural.operatorCount = 1;
  localCost.structural.weightedOperatorCost = 1;
  localCost.stageCycles = 50;
  ParetoCandidateCost globalCost;
  // Deliberately make the merged state candidate structurally worse. Since
  // the exact source is indexed, all candidates share one StageModel context
  // and the cross-shard candidate must still win by cycles.
  globalCost.structural.operatorCount = 3;
  globalCost.structural.weightedOperatorCost = 3;
  globalCost.stageCycles = 10;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "partitioned-stage-memory-eclasses";
  database.universeHash = "partitioned-stage-memory-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "test-stage-model";
  database.costModelHash = "test-stage-model-v1";
  database.graphs = {pointer, firstValue, finalValue, memory, firstStore,
                     source, localCandidate, repeatedFirst, globalCandidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {source.graphHash,
       "source-stage-memory-class",
       "none",
       {"source-stage-memory-proof"},
       sourceCost,
       {{localCandidate.graphHash, localCost}}}};
  database.semanticEntries = {
      {getRuleDatabaseSemanticKeyDigest(canonical->normalizedSemanticKey),
       "merged-stage-memory-class",
       "none",
       {"cross-shard-stage-memory-proof"},
       {{globalCandidate.graphHash, globalCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-partitioned-stage-memory-union", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 3> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 3u);
  EXPECT_EQ(stores[0].getValue(), function.getArgument(1));
  EXPECT_EQ(stores[1].getValue(), function.getArgument(1));
  EXPECT_EQ(stores[2].getValue(), function.getArgument(2));
}

TEST(OnlineRuleDatabasePipelineTest,
     MemoryLexicographicChoiceIsNotRejectedByTelemetryScalar) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @layout_priority_store(%ptr: !tt.ptr<i32>,
                                     %first: i32, %final: i32) {
        tt.store %ptr, %first : !tt.ptr<i32>
        tt.store %ptr, %final : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::FuncOp function =
      module->lookupSymbol<triton::FuncOp>("layout_priority_store");
  ASSERT_TRUE(function);
  Operation *root = nullptr;
  function.walk([&](triton::StoreOp store) { root = store; });
  ASSERT_NE(root, nullptr);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(canonical) << error;

  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord firstValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord finalValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 2 : i64}", 0);
  PersistentGraphRecord memory = makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord firstStore = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, firstValue.graphHash, memory.graphHash});
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 2,
      {pointer.graphHash, finalValue.graphHash, firstStore.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, finalValue.graphHash, memory.graphHash});
  ASSERT_EQ(source.graphHash, canonical->rootGraphHash);

  ParetoCandidateCost sourceCost;
  sourceCost.structural.memoryStridePenalty = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  sourceCost.structural.operatorCount = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.memoryStridePenalty = 0;
  candidateCost.structural.weightedOperatorCost = 20000;
  candidateCost.structural.operatorCount = 1;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "lexicographic-memory-eclasses";
  database.universeHash = "lexicographic-memory-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {pointer, firstValue, finalValue, memory, firstStore,
                     source, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {source.graphHash,
       "layout-priority-memory-class",
       "none",
       {"layout-priority-memory-proof"},
       sourceCost,
       {{candidate.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-lexicographic-memory-cost", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 2> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1u);
  EXPECT_EQ(stores.front().getValue(), function.getArgument(2));
}

TEST(OnlineRuleDatabasePipelineTest, MaterializesSelectedMultiResultProjection) {
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord multiply = makeGraph(
      "arith.muli", "i32", "{}", 1,
      {lhs.graphHash, rhs.graphHash});
  PersistentGraphRecord extended = makeGraph(
      "arith.mului_extended", "i32",
      "{ags.operation_result_types = [i32, i32], "
      "ags.result_index = 0 : i64}",
      1, {lhs.graphHash, rhs.graphHash});

  PersistentEquivalenceClassStore store;
  store.universeHash = "online-multi-result-pipeline-test";
  store.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {lhs, rhs, multiply, extended};
  store.proofEdges = {{multiply.graphHash, extended.graphHash, "proof-v1",
                       "certificate-v1"}};
  store.classes = {
      {"lhs", "i32", {lhs.graphHash}},
      {"rhs", "i32", {rhs.graphHash}},
      {"multiply", "i32", {multiply.graphHash, extended.graphHash}}};

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost inputCost;
  ParetoCandidateCost originalCost;
  originalCost.structural.operatorCount = 1;
  originalCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost selectedCost;
  selectedCost.structural.operatorCount = 1;
  selectedCost.structural.weightedOperatorCost = 1;
  ParetoExtractionResult pareto;
  pareto.eclassStoreHash = store.getStableHash();
  pareto.costContextKey = costContext.getStableKey();
  pareto.costModelName = "structural-fallback-v1";
  pareto.costModelHash = "structural-fallback-v1";
  pareto.classes = {
      {"lhs", true, {{lhs.graphHash, inputCost}},
       {{lhs.graphHash, inputCost}}},
      {"rhs", true, {{rhs.graphHash, inputCost}},
       {{rhs.graphHash, inputCost}}},
      {"multiply", true,
       {{multiply.graphHash, originalCost},
        {extended.graphHash, selectedCost}},
       {{extended.graphHash, selectedCost}}}};
  pareto.candidateCountBefore = 4;
  pareto.candidateCountAfter = 3;
  pareto.structuralFallbackClassCount = 3;

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-multi-result", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database->toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @multiply(%lhs: i32, %rhs: i32) -> i32 {
        %result = arith.muli %lhs, %rhs : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned plainMultiplyCount = 0;
  unsigned extendedMultiplyCount = 0;
  module->walk([&](Operation *operation) {
    plainMultiplyCount += isa<arith::MulIOp>(operation);
    if (operation->getName().getStringRef() == "arith.mului_extended") {
      ++extendedMultiplyCount;
      EXPECT_EQ(operation->getNumResults(), 2u);
    }
  });
  EXPECT_EQ(plainMultiplyCount, 0u);
  EXPECT_EQ(extendedMultiplyCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest, CommitsDeclarativeModuleActionAfterRewrite) {
  PersistentGraphRecord launch =
      makeGraph("ags.launch", "none", "{}", 0);
  PersistentGraphRecord remap = makeGraph(
      "ags.launch_remap", "none",
      "{ags.module_attribute_patch = {hacc.coalesce_axis = 0 : i32, "
      "hacc.coalesce_factor = 1 : i32, "
      "hacc.coalesce_grid_ceil_div = 0 : i32}, "
      "ags.proof_only = true, ags.state_domain = \"launch\", axis = 0 : i32, "
      "ceil_div = 0 : i32, factor = 1 : i32}",
      1, {launch.graphHash});
  PersistentGraphRecord sourcePid = makeGraph(
      "tt.get_program_id", "i32", "{axis = 0 : i32}", 1,
      {launch.graphHash});
  PersistentGraphRecord remappedPid = makeGraph(
      "tt.get_program_id", "i32", "{axis = 0 : i32}", 2,
      {remap.graphHash});

  PersistentEquivalenceClassStore store;
  store.universeHash = "online-launch-action-pipeline-test";
  store.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {launch, remap, sourcePid, remappedPid};
  store.proofEdges = {
      {launch.graphHash, remap.graphHash, "proof-launch", "certificate-launch"},
      {sourcePid.graphHash, remappedPid.graphHash, "proof-pid",
       "certificate-pid"}};
  store.classes = {
      {"launch", "none", {launch.graphHash, remap.graphHash}},
      {"pid", "i32", {sourcePid.graphHash, remappedPid.graphHash}}};

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost launchCost;
  ParetoCandidateCost remapCost;
  remapCost.structural.operatorCount = 1;
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost selectedCost;
  selectedCost.structural.operatorCount = 2;
  selectedCost.structural.weightedOperatorCost = 1;
  ParetoExtractionResult pareto;
  pareto.eclassStoreHash = store.getStableHash();
  pareto.costContextKey = costContext.getStableKey();
  pareto.costModelName = "structural-fallback-v1";
  pareto.costModelHash = "structural-fallback-v1";
  pareto.classes = {
      {"launch", true,
       {{launch.graphHash, launchCost}, {remap.graphHash, remapCost}},
       {{launch.graphHash, launchCost}}},
      {"pid", true,
       {{sourcePid.graphHash, sourceCost},
        {remappedPid.graphHash, selectedCost}},
       {{remappedPid.graphHash, selectedCost}}}};
  pareto.candidateCountBefore = 4;
  pareto.candidateCountAfter = 2;
  pareto.structuralFallbackClassCount = 2;

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-launch-action", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database->toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @pid() -> i32 {
        %pid = tt.get_program_id x : i32
        tt.return %pid : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_axis"))
                .getInt(),
            0);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_factor"))
                .getInt(),
            1);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_grid_ceil_div"))
                .getInt(),
            0);
  bool sawPseudoOperation = false;
  module->walk([&](Operation *operation) {
    sawPseudoOperation |=
        operation->getName().getStringRef() == "ags.launch_remap";
  });
  EXPECT_FALSE(sawPseudoOperation);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesProofOnlyWholeKernelTransactionAndLaunchAction) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord falseMask =
      makeGraph("arith.constant", "i1", "{value = false}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord launch =
      makeGraph("ags.launch", "none", "{}", 0);
  PersistentGraphRecord loadGraph = makeGraph(
      "tt.load", "i32",
      "{operandSegmentSizes = array<i32: 1, 0, 0>}", 1,
      {pointer.graphHash, memory.graphHash});
  PersistentGraphRecord storeGraph = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 2,
      {pointer.graphHash, loadGraph.graphHash, falseMask.graphHash,
       memory.graphHash});
  PersistentGraphRecord remap = makeGraph(
      "ags.launch_remap", "none",
      "{ags.module_attribute_patch = {hacc.coalesce_axis = 0 : i32, "
      "hacc.coalesce_factor = 1 : i32, "
      "hacc.coalesce_grid_ceil_div = 0 : i32}, "
      "ags.proof_only = true, ags.state_domain = \"launch\", axis = 0 : i32, "
      "ceil_div = 0 : i32, factor = 1 : i32}",
      1, {launch.graphHash});
  const std::string kernelAttributes =
      "{ags.proof_only = true, ags.state_domain = \"memory\"}";
  PersistentGraphRecord sourceKernel = makeGraph(
      "ags.kernel", "none", kernelAttributes, 3,
      {storeGraph.graphHash, launch.graphHash});
  PersistentGraphRecord candidateKernel = makeGraph(
      "ags.kernel", "none", kernelAttributes, 2,
      {memory.graphHash, remap.graphHash});

  PersistentEquivalenceClassStore store;
  store.universeHash = "online-whole-kernel-transaction-test";
  store.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {pointer, falseMask, memory, launch, loadGraph, storeGraph,
                  remap, sourceKernel, candidateKernel};
  store.proofEdges = {
      {launch.graphHash, remap.graphHash, "proof-launch",
       "certificate-launch"},
      {sourceKernel.graphHash, candidateKernel.graphHash, "proof-kernel",
       "certificate-kernel"}};
  store.classes = {
      {"pointer", "!tt.ptr<i32>", {pointer.graphHash}},
      {"load", "i32", {loadGraph.graphHash}},
      {"false", "i1", {falseMask.graphHash}},
      {"memory", "none", {memory.graphHash}},
      {"launch", "none", {launch.graphHash, remap.graphHash}},
      {"store", "none", {storeGraph.graphHash}},
      {"kernel", "none",
       {sourceKernel.graphHash, candidateKernel.graphHash}}};

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost zeroCost;
  ParetoCandidateCost remapCost;
  remapCost.structural.operatorCount = 1;
  remapCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost loadCost;
  loadCost.structural.operatorCount = 1;
  loadCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost storeCost;
  storeCost.structural.operatorCount = 2;
  storeCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost sourceCost;
  sourceCost.structural.memoryStridePenalty = 1;
  sourceCost.structural.operatorCount = 3;
  sourceCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 2;
  candidateCost.structural.weightedOperatorCost = 20000;
  // The compiler-action kernel is preferred by the full lexicographic vector
  // because it removes the stride penalty. Its deliberately larger telemetry
  // scalar must not suppress the already-proved whole-kernel transaction.
  ParetoExtractionResult pareto;
  pareto.eclassStoreHash = store.getStableHash();
  pareto.costContextKey = costContext.getStableKey();
  pareto.costModelName = "structural-fallback-v1";
  pareto.costModelHash = "structural-fallback-v1";
  pareto.classes = {
      {"pointer", true, {{pointer.graphHash, zeroCost}},
       {{pointer.graphHash, zeroCost}}},
      {"load", true, {{loadGraph.graphHash, loadCost}},
       {{loadGraph.graphHash, loadCost}}},
      {"false", true, {{falseMask.graphHash, zeroCost}},
       {{falseMask.graphHash, zeroCost}}},
      {"memory", true, {{memory.graphHash, zeroCost}},
       {{memory.graphHash, zeroCost}}},
      {"launch", true,
       {{launch.graphHash, zeroCost}, {remap.graphHash, remapCost}},
       {{launch.graphHash, zeroCost}}},
      {"store", true, {{storeGraph.graphHash, storeCost}},
       {{storeGraph.graphHash, storeCost}}},
      {"kernel", true,
       {{sourceKernel.graphHash, sourceCost},
        {candidateKernel.graphHash, candidateCost}},
       {{candidateKernel.graphHash, candidateCost}}}};
  pareto.candidateCountBefore = 9;
  pareto.candidateCountAfter = 7;
  pareto.structuralFallbackClassCount = 7;

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-whole-kernel", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database->toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @kernel(%ptr: !tt.ptr<i32>) {
        %false = arith.constant false
        %value = tt.load %ptr : !tt.ptr<i32>
        tt.store %ptr, %value, %false : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned storeCount = 0;
  unsigned loadCount = 0;
  bool sawPseudoOperation = false;
  module->walk([&](Operation *operation) {
    storeCount += isa<triton::StoreOp>(operation);
    loadCount += isa<triton::LoadOp>(operation);
    sawPseudoOperation |= operation->getName().getStringRef() == "ags.kernel" ||
                          operation->getName().getStringRef() ==
                              "ags.launch_remap";
  });
  EXPECT_EQ(storeCount, 0u);
  EXPECT_EQ(loadCount, 0u);
  EXPECT_FALSE(sawPseudoOperation);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_factor"))
                .getInt(),
            1);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesAxisOneParameterizedLaunchPartitionCompilerAction) {
  PersistentGraphRecord memory =
      makeGraph("ags.partitionable_memory", "none",
                "{axis = 1 : i32, ceil_div = 1 : i32}", 0);
  PersistentGraphRecord bound =
      makeGraph("ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord launch =
      makeGraph("ags.launch", "none", "{}", 0);
  PersistentGraphRecord boundedLaunch = makeGraph(
      "ags.launch_bound", "none",
      "{ags.proof_only = true, ags.state_domain = \"launch\", "
      "axis = 1 : i32}",
      1, {bound.graphHash, launch.graphHash});
  const std::string sourceAttributes =
      "{ags.proof_only = true, ags.state_domain = \"memory\"}";
  PersistentGraphRecord sourceKernel =
      makeGraph("ags.kernel", "none", sourceAttributes, 1,
                {memory.graphHash, boundedLaunch.graphHash});
  const std::string partitionAttributes =
      "{ags.compiler_action = {axis = 1 : i32, ceil_div = 1 : i32, "
      "factor = 4 : i32, kind = \"launch_partition\"}, "
      "ags.module_attribute_patch = {hacc.coalesce_axis = 1 : i32, "
      "hacc.coalesce_factor = 4 : i32, "
      "hacc.coalesce_grid_ceil_div = 1 : i32}, "
      "ags.proof_only = true, ags.state_domain = \"memory\", "
      "axis = 1 : i32, ceil_div = 1 : i32, factor = 4 : i32}";
  PersistentGraphRecord partitionKernel = makeGraph(
      "ags.kernel_partition", "none", partitionAttributes, 1,
      {memory.graphHash, boundedLaunch.graphHash});

  PersistentEquivalenceClassStore store;
  store.universeHash = "online-parameterized-launch-partition-test";
  store.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {memory, bound, launch, boundedLaunch, sourceKernel,
                  partitionKernel};
  store.proofEdges = {
      {sourceKernel.graphHash, partitionKernel.graphHash,
       "proof-parameterized-partition",
       R"json({"steps":[{"reason":"quantified-guarded-tail-launch-partition"}]})json"}};
  store.classes = {
      {"memory", "none", {memory.graphHash}},
      {"bound", "i32", {bound.graphHash}},
      {"launch", "none", {launch.graphHash}},
      {"bounded-launch", "none", {boundedLaunch.graphHash}},
      {"kernel", "none", {sourceKernel.graphHash, partitionKernel.graphHash}}};

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost zeroCost;
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 5;
  ParetoCandidateCost partitionCost = sourceCost;
  partitionCost.structural.weightedOperatorCost = 2;
  ParetoExtractionResult pareto;
  pareto.eclassStoreHash = store.getStableHash();
  pareto.costContextKey = costContext.getStableKey();
  pareto.costModelName = "structural-fallback-v1";
  pareto.costModelHash = "structural-fallback-v1";
  pareto.classes = {
      {"memory", true, {{memory.graphHash, zeroCost}},
       {{memory.graphHash, zeroCost}}},
      {"bound", true, {{bound.graphHash, zeroCost}},
       {{bound.graphHash, zeroCost}}},
      {"launch", true, {{launch.graphHash, zeroCost}},
       {{launch.graphHash, zeroCost}}},
      {"bounded-launch", true, {{boundedLaunch.graphHash, zeroCost}},
       {{boundedLaunch.graphHash, zeroCost}}},
      {"kernel", true,
       {{sourceKernel.graphHash, sourceCost},
        {partitionKernel.graphHash, partitionCost}},
       {{partitionKernel.graphHash, partitionCost}}}};
  pareto.candidateCountBefore = 6;
  pareto.candidateCountAfter = 5;
  pareto.structuralFallbackClassCount = 5;

  std::string error;
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(store, pareto, error);
  ASSERT_TRUE(database) << error;
  ASSERT_EQ(database->templates.size(), 1u);
  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-parameterized-launch", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database->toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<cf::ControlFlowDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();

  OwningOpRef<ModuleOp> unsafeModule = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @unsafe(%ptr: !tt.ptr<i32>, %value: i32, %bound: i32)
          attributes {ags.launch_grid_bound_axes = array<i32: 1>,
                      ags.launch_grid_bound_arguments = array<i32: 2>} {
        %pid = tt.get_program_id y : i32
        %pointer = tt.addptr %ptr, %pid : !tt.ptr<i32>, i32
        tt.store %pointer, %value : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(unsafeModule);
  PassManager unsafeManager(&context);
  unsafeManager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(unsafeManager.run(*unsafeModule)));
  EXPECT_FALSE((*unsafeModule)->hasAttr("hacc.coalesce_factor"));

  OwningOpRef<ModuleOp> noPidModule = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @no_pid(%ptr: !tt.ptr<i32>, %value: i32, %bound: i32)
          attributes {ags.launch_grid_bound_axes = array<i32: 1>,
                      ags.launch_grid_bound_arguments = array<i32: 2>} {
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(noPidModule);
  PassManager noPidManager(&context);
  noPidManager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(noPidManager.run(*noPidModule)));
  EXPECT_FALSE((*noPidModule)->hasAttr("hacc.coalesce_factor"));

  OwningOpRef<ModuleOp> weakGuardModule = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @weak_guard(%ptr: !tt.ptr<i32>, %value: i32, %bound: i32)
          attributes {ags.launch_grid_bound_axes = array<i32: 1>,
                      ags.launch_grid_bound_arguments = array<i32: 2>} {
        %pid = tt.get_program_id y : i32
        %one = arith.constant 1 : i32
        %weaker_pid = arith.subi %pid, %one : i32
        %in_bounds = arith.cmpi slt, %weaker_pid, %bound : i32
        cf.cond_br %in_bounds, ^work, ^return
      ^work:
        %pointer = tt.addptr %ptr, %pid : !tt.ptr<i32>, i32
        tt.store %pointer, %value : !tt.ptr<i32>
        tt.return
      ^return:
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(weakGuardModule);
  PassManager weakGuardManager(&context);
  weakGuardManager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(weakGuardManager.run(*weakGuardModule)));
  EXPECT_FALSE((*weakGuardModule)->hasAttr("hacc.coalesce_factor"));

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @kernel(%ptr: !tt.ptr<i32>, %value: i32, %bound: i32)
          attributes {ags.launch_grid_bound_axes = array<i32: 1>,
                      ags.launch_grid_bound_arguments = array<i32: 2>} {
        %pid = tt.get_program_id y : i32
        %past_end = arith.cmpi sge, %pid, %bound : i32
        cf.cond_br %past_end, ^return, ^work
      ^return:
        tt.return
      ^work:
        %one = arith.constant 1 : i32
        %sum = arith.addi %value, %one : i32
        %pointer = tt.addptr %ptr, %pid : !tt.ptr<i32>, i32
        tt.store %pointer, %sum : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned storeCount = 0;
  unsigned programIdCount = 0;
  bool sawPseudoOperation = false;
  module->walk([&](Operation *operation) {
    storeCount += isa<triton::StoreOp>(operation);
    programIdCount += isa<triton::GetProgramIdOp>(operation);
    sawPseudoOperation |=
        operation->getName().getStringRef().starts_with("ags.");
  });
  EXPECT_EQ(storeCount, 4u);
  EXPECT_EQ(programIdCount, 1u);
  EXPECT_FALSE(sawPseudoOperation);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_factor"))
                .getInt(),
            4);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_axis"))
                .getInt(),
            1);
  EXPECT_EQ(cast<IntegerAttr>(
                (*module)->getAttr("hacc.coalesce_grid_ceil_div"))
                .getInt(),
            1);
}

TEST(OnlineRuleDatabasePipelineTest, AppliesStageBufferRecipe) {
  PersistentGraphRecord target = makeGraph(
      "ags.private_buffer", "memref<4xi32>",
      "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord source = makeGraph(
      "ags.input", "memref<4xi32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord buffer = makeGraph("ags.buffer", "none", "{}", 0);
  PersistentGraphRecord copy = makeGraph(
      "memref.copy", "none", "{ags.state_domain = \"buffer\"}", 1,
      {source.graphHash, target.graphHash, buffer.graphHash});
  PersistentGraphRecord readTarget = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 2,
      {target.graphHash, copy.graphHash});
  PersistentGraphRecord readSource = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 1,
      {source.graphHash, buffer.graphHash});

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  candidateCost.structural.weightedOperatorCost = 1;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-stage-buffer-test";
  database.universeHash = "online-stage-buffer-pipeline-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {target, source, buffer, copy, readTarget, readSource};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {readTarget.graphHash,
       "stage-buffer-copy-forwarding",
       "tensor<4xi32>",
       {"trusted-stage-buffer-certificate"},
       sourceCost,
       {{readSource.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-stage-buffer", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @copy_then_read(%source: memref<4xi32>) -> tensor<4xi32> {
        %target = memref.alloc() : memref<4xi32>
        memref.copy %source, %target : memref<4xi32> to memref<4xi32>
        %result = bufferization.to_tensor %target restrict writable
          : memref<4xi32> to tensor<4xi32>
        return %result : tensor<4xi32>
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  manager.addPass(createMergeConcatLoadBufferPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned copyCount = 0;
  Value tensorSource;
  module->walk([&](memref::CopyOp) { ++copyCount; });
  module->walk([&](bufferization::ToTensorOp operation) {
    tensorSource = operation.getBuffer();
  });
  EXPECT_EQ(copyCount, 0u);
  ASSERT_TRUE(tensorSource);
  auto function = module->lookupSymbol<func::FuncOp>("copy_then_read");
  ASSERT_TRUE(function);
  EXPECT_EQ(tensorSource, function.getArgument(0));
}

TEST(OnlineRuleDatabasePipelineTest,
     ReplacesOneCertifiedBufferStateBranchWithAnother) {
  PersistentGraphRecord targetA = makeGraph(
      "ags.private_buffer", "memref<4xi32>",
      "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord source = makeGraph(
      "ags.input", "memref<4xi32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord targetB = makeGraph(
      "ags.private_buffer", "memref<4xi32>",
      "{ags.symbol_id = 2 : i64}", 0);
  PersistentGraphRecord buffer = makeGraph("ags.buffer", "none", "{}", 0);
  PersistentGraphRecord copyA = makeGraph(
      "memref.copy", "none", "{ags.state_domain = \"buffer\"}", 1,
      {source.graphHash, targetA.graphHash, buffer.graphHash});
  PersistentGraphRecord copyBAfterA = makeGraph(
      "memref.copy", "none", "{ags.state_domain = \"buffer\"}", 2,
      {source.graphHash, targetB.graphHash, copyA.graphHash});
  PersistentGraphRecord readA = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 3,
      {targetA.graphHash, copyBAfterA.graphHash});
  PersistentGraphRecord copyB = makeGraph(
      "memref.copy", "none", "{ags.state_domain = \"buffer\"}", 1,
      {source.graphHash, targetB.graphHash, buffer.graphHash});
  PersistentGraphRecord readB = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 2,
      {targetB.graphHash, copyB.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 3;
  sourceCost.structural.weightedOperatorCost = 3;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 2;
  candidateCost.structural.weightedOperatorCost = 2;
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "trusted-buffer-branch-test";
  database.universeHash = "online-buffer-branch-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {targetA, source, targetB, buffer, copyA,
                     copyBAfterA, readA, copyB, readB};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {readA.graphHash,
       "buffer-state-branch-replacement",
       "tensor<4xi32>",
       {"trusted-buffer-state-branch-certificate"},
       sourceCost,
       {{readB.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-buffer-branch", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @replace_buffer_branch(%source: memref<4xi32>)
          -> tensor<4xi32> {
        %target_a = memref.alloc() : memref<4xi32>
        %target_b = memref.alloc() : memref<4xi32>
        memref.copy %source, %target_a : memref<4xi32> to memref<4xi32>
        memref.copy %source, %target_b : memref<4xi32> to memref<4xi32>
        %result = bufferization.to_tensor %target_a restrict writable
          : memref<4xi32> to tensor<4xi32>
        tt.return %result : tensor<4xi32>
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned copyCount = 0;
  unsigned allocCount = 0;
  Value tensorSource;
  module->walk([&](memref::CopyOp) { ++copyCount; });
  module->walk([&](memref::AllocOp) { ++allocCount; });
  module->walk([&](bufferization::ToTensorOp operation) {
    tensorSource = operation.getBuffer();
  });
  EXPECT_EQ(copyCount, 1u);
  EXPECT_EQ(allocCount, 1u);
  ASSERT_TRUE(tensorSource);
  EXPECT_TRUE(tensorSource.getDefiningOp<memref::AllocOp>());
}

TEST(OnlineRuleDatabasePipelineTest,
     RejectsStageBufferRecipeWhenWrittenBufferEscapes) {
  PersistentGraphRecord target = makeGraph(
      "ags.input", "memref<4xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord source = makeGraph(
      "ags.input", "memref<4xi32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord buffer = makeGraph("ags.buffer", "none", "{}", 0);
  PersistentGraphRecord copy = makeGraph(
      "memref.copy", "none", "{ags.state_domain = \"buffer\"}", 1,
      {source.graphHash, target.graphHash, buffer.graphHash});
  PersistentGraphRecord readTarget = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 2,
      {target.graphHash, copy.graphHash});
  PersistentGraphRecord readSource = makeGraph(
      "bufferization.to_tensor", "tensor<4xi32>", "{}", 1,
      {source.graphHash, buffer.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  candidateCost.structural.weightedOperatorCost = 1;
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "trusted-escaping-buffer-test";
  database.universeHash = "online-escaping-buffer-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {target, source, buffer, copy, readTarget, readSource};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {readTarget.graphHash,
       "escaping-stage-buffer-copy-forwarding",
       "tensor<4xi32>",
       {"trusted-escaping-stage-buffer-certificate"},
       sourceCost,
       {{readSource.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-escaping-buffer", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @copy_then_read(%source: memref<4xi32>,
                                %target: memref<4xi32>) -> tensor<4xi32> {
        memref.copy %source, %target : memref<4xi32> to memref<4xi32>
        %result = bufferization.to_tensor %target restrict writable
          : memref<4xi32> to tensor<4xi32>
        return %result : tensor<4xi32>
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  manager.addPass(createMergeConcatLoadBufferPass());
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned copyCount = 0;
  module->walk([&](memref::CopyOp) { ++copyCount; });
  EXPECT_EQ(copyCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest,
     RejectsDatabaseCandidateWithDifferentConcreteType) {
  PersistentGraphRecord input = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 0);
  PersistentGraphRecord subtract =
      makeGraph("arith.subi", "i32", "{}", 1,
                {input.graphHash, zero.graphHash});
  // The ordinal matches the real boundary input, but its concrete type does
  // not. This can arise when an offline universe enumerates several physical
  // layouts (or element types) for one symbolic input position.
  PersistentGraphRecord wrongTypeInput = makeGraph(
      "ags.input", "i64", "{ags.symbol_id = 0 : i64}", 0);

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost candidateCost;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-boundary-type-test";
  database.universeHash = "online-boundary-type-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {input, zero, subtract, wrongTypeInput};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {{subtract.graphHash,
                       "invalid-cross-type-class",
                       "i32",
                       {"trusted-cross-type-certificate"},
                       sourceCost,
                       {{wrongTypeInput.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-boundary-type", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %result = arith.subi %arg0, %zero : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  EXPECT_TRUE(failed(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest, AppliesZeroResultEffectRecipe) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord store = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost identityCost;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-effect-test";
  database.universeHash = "online-effect-pipeline-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {pointer, value, memory, store};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {{store.graphHash,
                       "effect-class",
                       "none",
                       {"trusted-certificate"},
                       sourceCost,
                       {{memory.graphHash, identityCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-effect", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @drop_store(%ptr: !tt.ptr<i32>, %value: i32) {
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned storeCount = 0;
  unsigned loadCount = 0;
  module->walk([&](triton::StoreOp) { ++storeCount; });
  module->walk([&](triton::LoadOp) { ++loadCount; });
  EXPECT_EQ(storeCount, 0u);
  EXPECT_EQ(loadCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesCertifiedPureTemplateToAnUnseenBoundSubgraph) {
  PersistentGraphRecord variable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 1);
  PersistentGraphRecord source = makeGraph(
      "arith.subi", "i32", "{}", 2,
      {variable.graphHash, zero.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost candidateCost;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "pure-template-eclasses";
  database.universeHash = "pure-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {variable, zero, source};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {source.graphHash,
       "pure-template-class",
       "i32",
       {"replayed-symbolic-proof"},
       sourceCost,
       {{variable.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-pure-template", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @bind_internal_value(%arg: i32) -> i32 {
        %one = arith.constant 1 : i32
        %bound = arith.addi %arg, %one : i32
        %zero = arith.constant 0 : i32
        %result = arith.subi %bound, %zero : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned addCount = 0;
  unsigned subtractCount = 0;
  module->walk([&](arith::AddIOp) { ++addCount; });
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(addCount, 1u);
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     BoundTemplateCanBeatAnExactShardLocalCandidate) {
  PersistentGraphRecord variable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 1);
  PersistentGraphRecord first = makeGraph(
      "arith.subi", "i32", "{}", 2,
      {variable.graphHash, zero.graphHash});
  PersistentGraphRecord source = makeGraph(
      "arith.subi", "i32", "{}", 3,
      {first.graphHash, zero.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost localCost;
  localCost.structural.operatorCount = 1;
  localCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost globalCost;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "partitioned-template-eclasses";
  database.universeHash = "partitioned-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {variable, zero, first, source};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {
      {source.graphHash,
       "source-shard-local-class",
       "i32",
       {"source-shard-proof"},
       sourceCost,
       {{first.graphHash, localCost}}}};
  database.templates = {
      {source.graphHash,
       "merged-template-class",
       "i32",
       {"cross-shard-template-proof"},
       sourceCost,
       {{variable.graphHash, globalCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-partitioned-template-union", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @partitioned_template_identity(%arg0: i32) -> i32 {
        %zero = arith.constant 0 : i32
        %first = arith.subi %arg0, %zero : i32
        %second = arith.subi %first, %zero : i32
        tt.return %second : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned subtractCount = 0;
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     StageCompatibilitySlotUsesTheSameGenericRuleDatabaseExecutor) {
  PersistentGraphRecord variable = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 1);
  PersistentGraphRecord source = makeGraph(
      "arith.subi", "i32", "{}", 2,
      {variable.graphHash, zero.graphHash});
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost candidateCost;
  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "stage-template-eclasses";
  database.universeHash = "stage-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {variable, zero, source};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {source.graphHash,
       "stage-template-class",
       "i32",
       {"replayed-symbolic-stage-proof"},
       sourceCost,
       {{variable.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-stage-rule-database", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @stage_bind(%arg: i32) -> i32 {
        %one = arith.constant 1 : i32
        %bound = arith.addi %arg, %one : i32
        %zero = arith.constant 0 : i32
        %result = arith.subi %bound, %zero : i32
        tt.return %result : i32
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  Builder builder(&context);
  (*module)->setAttr(
      kGraphOptimizationRuleMaskAttrName,
      builder.getI16IntegerAttr(
          getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver)));
  (*module)->setAttr(kAutoGraphRuleDatabaseAttrName,
                     builder.getStringAttr(databasePath));
  (*module)->setAttr(kAutoGraphOnlineBudgetAttrName,
                     builder.getI64IntegerAttr(10000));
  (*module)->setAttr(kAutoGraphRewriteBudgetAttrName,
                     builder.getI64IntegerAttr(64));
  (*module)->setAttr(kAutoGraphTarget91095AttrName,
                     builder.getBoolAttr(false));

  PassManager manager(&context);
  manager.addPass(createLayoutMemoryCompatibilityPass(
      LayoutMemoryCompatibilityPhase::BeforeDiagonal));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned addCount = 0;
  unsigned subtractCount = 0;
  module->walk([&](arith::AddIOp) { ++addCount; });
  module->walk([&](arith::SubIOp) { ++subtractCount; });
  EXPECT_EQ(addCount, 1u);
  EXPECT_EQ(subtractCount, 0u);
}

TEST(OnlineRuleDatabasePipelineTest,
     AppliesCertifiedMemoryTemplateWithTypedValueBindings) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord zero =
      makeGraph("arith.constant", "i32", "{value = 0 : i32}", 1);
  PersistentGraphRecord sum = makeGraph(
      "arith.addi", "i32", "{}", 2,
      {value.graphHash, zero.graphHash});
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 3,
      {pointer.graphHash, sum.graphHash, memory.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 3;
  sourceCost.structural.weightedOperatorCost = 3;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  candidateCost.structural.weightedOperatorCost = 1;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "memory-template-eclasses";
  database.universeHash = "memory-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {pointer, value, zero, sum, memory, source, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {source.graphHash,
       "memory-template-class",
       "none",
       {"replayed-symbolic-memory-proof"},
       sourceCost,
       {{candidate.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-memory-template", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @bind_store_value(%ptr: !tt.ptr<i32>, %value: i32) {
        %zero = arith.constant 0 : i32
        %sum = arith.addi %value, %zero : i32
        tt.store %ptr, %sum : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  unsigned addCount = 0;
  unsigned storeCount = 0;
  module->walk([&](arith::AddIOp) { ++addCount; });
  module->walk([&](triton::StoreOp) { ++storeCount; });
  EXPECT_EQ(addCount, 0u);
  EXPECT_EQ(storeCount, 1u);
}

TEST(OnlineRuleDatabasePipelineTest,
     ReCostsCertifiedRank2StoreTemplateAfterBindingAffinePointer) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "tensor<2x3x!tt.ptr<f32>>",
      "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "tensor<2x3xf32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});
  PersistentGraphRecord transposedPointer = makeGraph(
      "tt.trans", "tensor<3x2x!tt.ptr<f32>>",
      "{order = array<i32: 1, 0>}", 1, {pointer.graphHash});
  PersistentGraphRecord transposedValue = makeGraph(
      "tt.trans", "tensor<3x2xf32>",
      "{order = array<i32: 1, 0>}", 1, {value.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 3,
      {transposedPointer.graphHash, transposedValue.graphHash,
       memory.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 3;
  candidateCost.structural.weightedOperatorCost = 1;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "rank2-store-template-eclasses";
  database.universeHash = "rank2-store-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {pointer, value, memory, source, transposedPointer,
                     transposedValue, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {source.graphHash,
       "rank2-store-template-class",
       "none",
       {"replayed-symbolic-rank2-store-proof"},
       sourceCost,
       {{candidate.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-rank2-store-template", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @transpose_strided_store(
          %base: !tt.ptr<f32>, %value: tensor<2x3xf32>) {
        %r2 = tt.make_range {start = 0 : i32, end = 2 : i32}
          : tensor<2xi32>
        %r3 = tt.make_range {start = 0 : i32, end = 3 : i32}
          : tensor<3xi32>
        %row = tt.expand_dims %r2 {axis = 1 : i32}
          : tensor<2xi32> -> tensor<2x1xi32>
        %col = tt.expand_dims %r3 {axis = 0 : i32}
          : tensor<3xi32> -> tensor<1x3xi32>
        %row2d = tt.broadcast %row : tensor<2x1xi32> -> tensor<2x3xi32>
        %two = arith.constant 2 : i32
        %twos = tt.splat %two : i32 -> tensor<1x3xi32>
        %scaled = arith.muli %col, %twos : tensor<1x3xi32>
        %col2d = tt.broadcast %scaled
          : tensor<1x3xi32> -> tensor<2x3xi32>
        %offset = arith.addi %row2d, %col2d : tensor<2x3xi32>
        %bases = tt.splat %base
          : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
        %pointer = tt.addptr %bases, %offset
          : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
        tt.store %pointer, %value
          : tensor<2x3x!tt.ptr<f32>>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 1> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1u);
  auto pointerType = dyn_cast<RankedTensorType>(stores.front().getPtr().getType());
  auto valueType = dyn_cast<RankedTensorType>(stores.front().getValue().getType());
  ASSERT_TRUE(pointerType);
  ASSERT_TRUE(valueType);
  ASSERT_EQ(pointerType.getRank(), 2);
  ASSERT_EQ(valueType.getRank(), 2);
  EXPECT_EQ(pointerType.getDimSize(0), 3);
  EXPECT_EQ(pointerType.getDimSize(1), 2);
  EXPECT_EQ(valueType.getDimSize(0), 3);
  EXPECT_EQ(valueType.getDimSize(1), 2);
}

TEST(OnlineRuleDatabasePipelineTest,
     ReCostsLoopCarriedTemplateFromItsDynamicAffineInitPointer) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "tensor<2x3x!tt.ptr<f32>>",
      "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "tensor<2x3xf32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});
  PersistentGraphRecord transposedPointer = makeGraph(
      "tt.trans", "tensor<3x2x!tt.ptr<f32>>",
      "{order = array<i32: 1, 0>}", 1, {pointer.graphHash});
  PersistentGraphRecord transposedValue = makeGraph(
      "tt.trans", "tensor<3x2xf32>",
      "{order = array<i32: 1, 0>}", 1, {value.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 3,
      {transposedPointer.graphHash, transposedValue.graphHash,
       memory.graphHash});

  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 1;
  sourceCost.structural.weightedOperatorCost = 1;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 3;
  candidateCost.structural.weightedOperatorCost = 1;

  GraphCostContext costContext;
  costContext.target = "generic";
  RuleDatabase database;
  database.eclassStoreHash = "loop-affine-store-template-eclasses";
  database.universeHash = "loop-affine-store-template-universe";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-pareto-fallback-v2";
  database.costModelHash = "structural-pareto-fallback-v2";
  database.graphs = {pointer, value, memory, source, transposedPointer,
                     transposedValue, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.templates = {
      {source.graphHash,
       "loop-affine-store-template-class",
       "none",
       {"replayed-symbolic-loop-affine-store-proof"},
       sourceCost,
       {{candidate.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-loop-affine-store-template", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @transpose_loop_carried_dynamic_stride(
          %base: !tt.ptr<f32>, %value: tensor<2x3xf32>, %stride: i32) {
        %r2 = tt.make_range {start = 0 : i32, end = 2 : i32}
          : tensor<2xi32>
        %r3 = tt.make_range {start = 0 : i32, end = 3 : i32}
          : tensor<3xi32>
        %row = tt.expand_dims %r2 {axis = 1 : i32}
          : tensor<2xi32> -> tensor<2x1xi32>
        %col = tt.expand_dims %r3 {axis = 0 : i32}
          : tensor<3xi32> -> tensor<1x3xi32>
        %row2d = tt.broadcast %row : tensor<2x1xi32> -> tensor<2x3xi32>
        %strides = tt.splat %stride : i32 -> tensor<1x3xi32>
        %scaled = arith.muli %col, %strides : tensor<1x3xi32>
        %col2d = tt.broadcast %scaled
          : tensor<1x3xi32> -> tensor<2x3xi32>
        %offset = arith.addi %row2d, %col2d : tensor<2x3xi32>
        %bases = tt.splat %base
          : !tt.ptr<f32> -> tensor<2x3x!tt.ptr<f32>>
        %pointer = tt.addptr %bases, %offset
          : tensor<2x3x!tt.ptr<f32>>, tensor<2x3xi32>
        %lb = arith.constant 0 : index
        %ub = arith.constant 1 : index
        %step = arith.constant 1 : index
        %unused = scf.for %iv = %lb to %ub step %step
            iter_args(%iter = %pointer)
            -> (tensor<2x3x!tt.ptr<f32>>) {
          tt.store %iter, %value : tensor<2x3x!tt.ptr<f32>>
          scf.yield %iter : tensor<2x3x!tt.ptr<f32>>
        }
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 1> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1u);
  auto pointerType = dyn_cast<RankedTensorType>(stores.front().getPtr().getType());
  auto valueType = dyn_cast<RankedTensorType>(stores.front().getValue().getType());
  ASSERT_TRUE(pointerType);
  ASSERT_TRUE(valueType);
  EXPECT_EQ(pointerType.getDimSize(0), 3);
  EXPECT_EQ(pointerType.getDimSize(1), 2);
  EXPECT_EQ(valueType.getDimSize(0), 3);
  EXPECT_EQ(valueType.getDimSize(1), 2);
}

TEST(OnlineRuleDatabasePipelineTest, ReplacesCertifiedMemoryStateSuffix) {
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord firstValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord finalValue = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 2 : i64}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord firstStore = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, firstValue.graphHash, memory.graphHash});
  PersistentGraphRecord source = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 2,
      {pointer.graphHash, finalValue.graphHash, firstStore.graphHash});
  PersistentGraphRecord candidate = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, finalValue.graphHash, memory.graphHash});

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  candidateCost.structural.weightedOperatorCost = 1;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-effect-suffix-test";
  database.universeHash = "online-effect-suffix-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = {pointer, firstValue, finalValue, memory,
                     firstStore, source, candidate};
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.entries = {{source.graphHash,
                       "overwrite-class",
                       "none",
                       {"trusted-overwrite-certificate"},
                       sourceCost,
                       {{candidate.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-effect-suffix", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @overwrite(%ptr: !tt.ptr<i32>, %first: i32, %final: i32) {
        tt.store %ptr, %first : !tt.ptr<i32>
        tt.store %ptr, %final : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 2> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1u);
  auto function = *module->getOps<triton::FuncOp>().begin();
  EXPECT_EQ(stores.front().getValue(), function.getArgument(2));
}

TEST(OnlineRuleDatabasePipelineTest,
     ReplacesAdjacentTensorStoresWithOneWideStore) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @adjacent(%base: !tt.ptr<i32>, %high: tensor<4xi32>,
                        %low: tensor<4xi32>) {
        %base4 = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
        %high_range = tt.make_range {start = 4 : i32, end = 8 : i32}
          : tensor<4xi32>
        %high_ptr = tt.addptr %base4, %high_range
          : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        tt.store %high_ptr, %high : tensor<4x!tt.ptr<i32>>
        %low_range = tt.make_range {start = 0 : i32, end = 4 : i32}
          : tensor<4xi32>
        %low_ptr = tt.addptr %base4, %low_range
          : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        tt.store %low_ptr, %low : tensor<4x!tt.ptr<i32>>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  SmallVector<triton::StoreOp, 2> sourceStores;
  module->walk(
      [&](triton::StoreOp store) { sourceStores.push_back(store); });
  ASSERT_EQ(sourceStores.size(), 2u);

  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(sourceStores.back().getOperation(), error);
  ASSERT_TRUE(canonical) << error;

  auto inputHash = [&](int64_t symbol) -> std::string {
    std::string marker =
        (llvm::Twine("ags.symbol_id = ") + llvm::Twine(symbol) + " : i64")
            .str();
    auto graph = llvm::find_if(canonical->graphs, [&](const auto &candidate) {
      return candidate.operationName == "ags.input" &&
             candidate.attributes.find(marker) != std::string::npos;
    });
    return graph == canonical->graphs.end() ? std::string()
                                            : graph->graphHash;
  };
  auto graphHash = [&](StringRef operation, StringRef type) -> std::string {
    auto graph = llvm::find_if(canonical->graphs, [&](const auto &candidate) {
      return candidate.operationName == operation &&
             candidate.resultType == type;
    });
    return graph == canonical->graphs.end() ? std::string()
                                            : graph->graphHash;
  };
  std::string base = inputHash(0);
  std::string high = inputHash(1);
  std::string low = inputHash(2);
  std::string memory = graphHash("ags.memory", "none");
  ASSERT_FALSE(base.empty());
  ASSERT_FALSE(high.empty());
  ASSERT_FALSE(low.empty());
  ASSERT_FALSE(memory.empty());

  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  Type pointer8 = RankedTensorType::get({8}, pointer);
  Type value8 = RankedTensorType::get({8}, i32);
  std::string rangeAttributes = printAttributes(builder.getDictionaryAttr({
      builder.getNamedAttr("end", builder.getI32IntegerAttr(8)),
      builder.getNamedAttr("start", builder.getI32IntegerAttr(0))}));
  auto sliceAttributes = [&](int64_t offset) {
    return printAttributes(builder.getDictionaryAttr({
        builder.getNamedAttr("operandSegmentSizes",
                             builder.getDenseI32ArrayAttr({1, 1, 0, 0, 0})),
        builder.getNamedAttr("static_offsets",
                             builder.getDenseI64ArrayAttr({offset})),
        builder.getNamedAttr("static_sizes",
                             builder.getDenseI64ArrayAttr({4})),
        builder.getNamedAttr("static_strides",
                             builder.getDenseI64ArrayAttr({1}))}));
  };

  PersistentGraphRecord range = makeGraph(
      "tt.make_range", "tensor<8xi32>", rangeAttributes, 1);
  PersistentGraphRecord splat = makeGraph(
      "tt.splat", "tensor<8x!tt.ptr<i32>>", "{}", 1, {base});
  PersistentGraphRecord widePointer = makeGraph(
      "tt.addptr", "tensor<8x!tt.ptr<i32>>", "{}", 3,
      {splat.graphHash, range.graphHash});
  PersistentGraphRecord empty =
      makeGraph("tensor.empty", "tensor<8xi32>", "{}", 1);
  PersistentGraphRecord insertLow = makeGraph(
      "tensor.insert_slice", "tensor<8xi32>", sliceAttributes(0), 2,
      {low, empty.graphHash});
  PersistentGraphRecord insertHigh = makeGraph(
      "tensor.insert_slice", "tensor<8xi32>", sliceAttributes(4), 3,
      {high, insertLow.graphHash});
  PersistentGraphRecord wideStore = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 7,
      {widePointer.graphHash, insertHigh.graphHash, memory});

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 8;
  sourceCost.structural.weightedOperatorCost = 8;
  sourceCost.structural.memoryOperationCount = 2;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 7;
  candidateCost.structural.weightedOperatorCost = 7;
  candidateCost.structural.memoryOperationCount = 1;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-adjacent-store-test";
  database.universeHash = "online-adjacent-store-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = canonical->graphs;
  database.graphs.insert(database.graphs.end(),
                         {range, splat, widePointer, empty, insertLow,
                          insertHigh, wideStore});
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.graphs.erase(
      std::unique(database.graphs.begin(), database.graphs.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return lhs.graphHash == rhs.graphHash;
                  }),
      database.graphs.end());
  database.entries = {{canonical->rootGraphHash,
                       "adjacent-store-class",
                       "none",
                       {"trusted-adjacent-store-certificate"},
                       sourceCost,
                       {{wideStore.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-adjacent-store", "agsdb", fileDescriptor, databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::StoreOp, 2> stores;
  module->walk([&](triton::StoreOp store) { stores.push_back(store); });
  ASSERT_EQ(stores.size(), 1u);
  EXPECT_EQ(stores.front().getValue().getType(), value8);
  EXPECT_EQ(stores.front().getPtr().getType(), pointer8);
  unsigned inserts = 0;
  module->walk([&](tensor::InsertSliceOp) { ++inserts; });
  EXPECT_EQ(inserts, 2u);
}

TEST(OnlineRuleDatabasePipelineTest,
     MaterializesUnusedAtomicMemoryProjectionTransactionally) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @masked_atomic(%ptr: !tt.ptr<i32>, %value: i32) {
        %true = arith.constant true
        %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %value, %true
          : (!tt.ptr<i32>, i32, i1) -> i32
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  triton::AtomicRMWOp sourceAtomic;
  module->walk([&](triton::AtomicRMWOp atomic) { sourceAtomic = atomic; });
  ASSERT_TRUE(sourceAtomic);
  ASSERT_TRUE(sourceAtomic.getResult().use_empty());

  GraphOptimizationOptions graphOptions;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(graphOptions);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(sourceAtomic.getOperation(), error);
  ASSERT_TRUE(canonical) << error;

  auto atomicGraph = llvm::find_if(canonical->graphs, [](const auto &graph) {
    return graph.operationName == "tt.atomic_rmw";
  });
  ASSERT_NE(atomicGraph, canonical->graphs.end());
  ASSERT_EQ(atomicGraph->operandHashes.size(), 4u);
  PersistentGraphRecord unmaskedAtomic = makeGraph(
      "tt.atomic_rmw", atomicGraph->resultType, atomicGraph->attributes, 1,
      {atomicGraph->operandHashes[0], atomicGraph->operandHashes[1],
       atomicGraph->operandHashes[3]});
  PersistentGraphRecord projected = makeGraph(
      "ags.memory_after", "none",
      "{ags.state_domain = \"memory\"}", 2,
      {unmaskedAtomic.graphHash});

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoCandidateCost sourceCost;
  sourceCost.structural.operatorCount = 2;
  sourceCost.structural.weightedOperatorCost = 2;
  sourceCost.structural.memoryOperationCount = 1;
  ParetoCandidateCost candidateCost;
  candidateCost.structural.operatorCount = 1;
  candidateCost.structural.weightedOperatorCost = 1;
  candidateCost.structural.memoryOperationCount = 1;

  RuleDatabase database;
  database.eclassStoreHash = "trusted-atomic-projection-test";
  database.universeHash = "online-atomic-projection-test";
  database.semanticsHash = getBuiltinTTOpSemanticsStableHash();
  database.costContextKey = costContext.getStableKey();
  database.costModelName = "structural-fallback-v1";
  database.costModelHash = "structural-fallback-v1";
  database.graphs = canonical->graphs;
  database.graphs.push_back(unmaskedAtomic);
  database.graphs.push_back(projected);
  llvm::sort(database.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  database.graphs.erase(
      std::unique(database.graphs.begin(), database.graphs.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return lhs.graphHash == rhs.graphHash;
                  }),
      database.graphs.end());
  database.entries = {{canonical->rootGraphHash,
                       "atomic-projection-class",
                       "none",
                       {"trusted-atomic-projection-certificate"},
                       sourceCost,
                       {{projected.graphHash, candidateCost}}}};

  int fileDescriptor = -1;
  SmallString<128> databasePath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile(
      "ags-online-atomic-projection", "agsdb", fileDescriptor,
      databasePath));
  llvm::FileRemover removeDatabase(databasePath);
  {
    llvm::raw_fd_ostream output(fileDescriptor, /*shouldClose=*/true);
    output << database.toJSON();
  }

  GraphOptimizationOptions options;
  options.enabledRuleMask =
      getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);
  options.enabledNarrowUnsignedRuleMask = 0;
  options.autoGraphRuleDatabase = databasePath.str().str();
  PassManager manager(&context);
  manager.addPass(createGraphOptimizePass(options));
  ASSERT_TRUE(succeeded(manager.run(*module)));

  SmallVector<triton::AtomicRMWOp, 2> atomics;
  module->walk(
      [&](triton::AtomicRMWOp atomic) { atomics.push_back(atomic); });
  ASSERT_EQ(atomics.size(), 1u);
  EXPECT_FALSE(atomics.front().getMask());
}

} // namespace

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

#include "TritonToGraph/ParetoExtractor.h"

#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir::triton::cfg;

namespace {

static PersistentEquivalenceClassStore makeStore() {
  PersistentEquivalenceClassStore store;
  store.universeHash = "universe";
  store.semanticsHash = "semantics";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {
      {"input", "ags.input", "i32", "", 0, {}},
      {"zero", "arith.constant", "i32", "value=0", 1, {}},
      {"sub", "arith.subi", "i32", "", 2, {"input", "zero"}},
      {"cast", "arith.extui", "i64", "", 1, {"input"}},
  };
  store.classes = {
      {"identity-class", "i32", {"input", "sub"}},
      {"zero-class", "i32", {"zero"}},
      {"cast-class", "i64", {"cast"}},
  };
  return store;
}

static GraphCostContext makeContext() {
  GraphCostContext context;
  context.target = "ascend950dt";
  context.shapeClass = "scalar";
  return context;
}

TEST(ParetoExtractorTest, StructuralFallbackPrunesDominatedGraph) {
  PersistentEquivalenceClassStore store = makeStore();
  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());

  EXPECT_EQ(result.candidateCountBefore, 4u);
  EXPECT_EQ(result.candidateCountAfter, 3u);
  EXPECT_EQ(result.structuralFallbackClassCount, 3u);
  EXPECT_EQ(result.invalidStructuralGraphCount, 0u);
  EXPECT_TRUE(result.hasStructuralCostForEveryGraph());
  ASSERT_EQ(result.classes.size(), 3u);
  auto identity = llvm::find_if(result.classes, [](const auto &eclass) {
    return eclass.classHash == "identity-class";
  });
  ASSERT_NE(identity, result.classes.end());
  ASSERT_EQ(identity->candidates.size(), 1u);
  EXPECT_EQ(identity->candidates.front().graphHash, "input");
  EXPECT_FALSE(identity->candidates.front().cost.stageCycles);

  ParetoExtractionResult repeated = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());
  EXPECT_EQ(result.toJSON(), repeated.toJSON());
}

TEST(ParetoExtractorTest, EqualCostGraphsKeepOneStableRepresentative) {
  PersistentEquivalenceClassStore store;
  store.universeHash = "equal-cost-universe";
  store.semanticsHash = "semantics";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {
      {"a", "ags.input", "i32", "{ags.symbol_id = 0}", 0, {}},
      {"b", "ags.input", "i32", "{ags.symbol_id = 1}", 0, {}},
      {"c", "ags.input", "i32", "{ags.symbol_id = 2}", 0, {}},
  };
  store.classes = {{"equal-cost-class", "i32", {"a", "b", "c"}}};

  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());
  ASSERT_EQ(result.classes.size(), 1u);
  EXPECT_EQ(result.classes.front().allCandidates.size(), 3u);
  ASSERT_EQ(result.classes.front().candidates.size(), 1u);
  EXPECT_EQ(result.classes.front().candidates.front().graphHash, "a");
  EXPECT_EQ(result.candidateCountBefore, 3u);
  EXPECT_EQ(result.candidateCountAfter, 1u);
}

TEST(ParetoExtractorTest, StructuralFallbackPrefersUnitStrideMemoryLayout) {
  PersistentEquivalenceClassStore store;
  store.universeHash = "layout-universe";
  store.semanticsHash = "semantics";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {
      {"base", "ags.input", "!tt.ptr<f32>", "{}", 0, {}},
      {"value23", "ags.input", "tensor<2x3xf32>", "{}", 0, {}},
      {"value32", "ags.input", "tensor<3x2xf32>", "{}", 0, {}},
      {"memory", "ags.memory", "none", "{}", 0, {}},
      {"two", "arith.constant", "i32", "{value = 2 : i32}", 1, {}},
      {"range2", "tt.make_range", "tensor<2xi32>",
       "{end = 2 : i32, start = 0 : i32}", 1, {}},
      {"range3", "tt.make_range", "tensor<3xi32>",
       "{end = 3 : i32, start = 0 : i32}", 1, {}},
      {"r2e1", "tt.expand_dims", "tensor<2x1xi32>",
       "{axis = 1 : i32}", 2, {"range2"}},
      {"r3e0", "tt.expand_dims", "tensor<1x3xi32>",
       "{axis = 0 : i32}", 2, {"range3"}},
      {"scale13", "tt.splat", "tensor<1x3xi32>", "{}", 2, {"two"}},
      {"scaled13", "arith.muli", "tensor<1x3xi32>", "{}", 4,
       {"r3e0", "scale13"}},
      {"r2b23", "tt.broadcast", "tensor<2x3xi32>", "{}", 3,
       {"r2e1"}},
      {"r3b23", "tt.broadcast", "tensor<2x3xi32>", "{}", 5,
       {"scaled13"}},
      {"offset23", "arith.addi", "tensor<2x3xi32>", "{}", 9,
       {"r2b23", "r3b23"}},
      {"base23", "tt.splat", "tensor<2x3x!tt.ptr<f32>>", "{}", 1,
       {"base"}},
      {"ptr23", "tt.addptr", "tensor<2x3x!tt.ptr<f32>>", "{}", 11,
       {"base23", "offset23"}},
      {"store23", "tt.store", "none", "{}", 12,
       {"ptr23", "value23", "memory"}},
      {"r3e1", "tt.expand_dims", "tensor<3x1xi32>",
       "{axis = 1 : i32}", 2, {"range3"}},
      {"r2e0", "tt.expand_dims", "tensor<1x2xi32>",
       "{axis = 0 : i32}", 2, {"range2"}},
      {"scale31", "tt.splat", "tensor<3x1xi32>", "{}", 2, {"two"}},
      {"scaled31", "arith.muli", "tensor<3x1xi32>", "{}", 4,
       {"r3e1", "scale31"}},
      {"r3b32", "tt.broadcast", "tensor<3x2xi32>", "{}", 5,
       {"scaled31"}},
      {"r2b32", "tt.broadcast", "tensor<3x2xi32>", "{}", 3,
       {"r2e0"}},
      {"offset32", "arith.addi", "tensor<3x2xi32>", "{}", 9,
       {"r3b32", "r2b32"}},
      {"base32", "tt.splat", "tensor<3x2x!tt.ptr<f32>>", "{}", 1,
       {"base"}},
      {"ptr32", "tt.addptr", "tensor<3x2x!tt.ptr<f32>>", "{}", 11,
       {"base32", "offset32"}},
      {"store32", "tt.store", "none", "{}", 12,
       {"ptr32", "value32", "memory"}},
  };
  store.classes = {{"layout-class", "none", {"store23", "store32"}}};

  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());
  ASSERT_EQ(result.classes.size(), 1u);
  ASSERT_EQ(result.classes.front().allCandidates.size(), 2u);
  auto strided = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "store23";
      });
  auto contiguous = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "store32";
      });
  ASSERT_NE(strided, result.classes.front().allCandidates.end());
  ASSERT_NE(contiguous, result.classes.front().allCandidates.end());
  EXPECT_EQ(strided->cost.structural.memoryStridePenalty, 1u);
  EXPECT_EQ(contiguous->cost.structural.memoryStridePenalty, 0u);
  ASSERT_EQ(result.classes.front().candidates.size(), 1u);
  EXPECT_EQ(result.classes.front().candidates.front().graphHash, "store32");
}

TEST(ParetoExtractorTest, CompleteStageModelIsUsedForWholeClass) {
  PersistentEquivalenceClassStore store = makeStore();
  unsigned evaluationCount = 0;
  StageModelCostAdapter stageModel(
      [&evaluationCount](const PersistentEquivalenceClassStore &,
                         const PersistentGraphRecord &graph,
                         const GraphCostContext &,
                         std::string &) -> std::optional<uint64_t> {
        ++evaluationCount;
        return graph.operationName == "ags.input" ? 1 : 10 + graph.operatorSize;
      },
      "fake-stage-model-v1");
  ParetoExtractionResult result =
      ParetoExtractor().extract(store, stageModel, makeContext());

  EXPECT_TRUE(result.usedStageModelForEveryClass());
  EXPECT_TRUE(result.hasStructuralCostForEveryGraph());
  EXPECT_EQ(result.structuralFallbackClassCount, 0u);
  EXPECT_EQ(evaluationCount, 2u);
  for (const ParetoEquivalenceClass &eclass : result.classes) {
    EXPECT_FALSE(eclass.usedStructuralFallback);
    for (const ParetoCandidateRecord &candidate : eclass.candidates) {
      if (eclass.allCandidates.size() > 1)
        EXPECT_TRUE(candidate.cost.stageCycles.has_value());
      else
        EXPECT_FALSE(candidate.cost.stageCycles.has_value());
    }
  }
  EXPECT_NE(result.toJSON().find("singleton-no-choice"), std::string::npos);
}

TEST(ParetoExtractorTest, OneUnknownCandidateFallsBackForEntireClass) {
  PersistentEquivalenceClassStore store = makeStore();
  store.classes.front().memberGraphHashes = {"sub", "input"};
  unsigned evaluationCount = 0;
  StageModelCostAdapter partialStageModel(
      [&](const PersistentEquivalenceClassStore &,
          const PersistentGraphRecord &graph, const GraphCostContext &,
          std::string &reason) -> std::optional<uint64_t> {
        ++evaluationCount;
        if (graph.operationName == "arith.subi") {
          reason = "subi materialization unsupported";
          return std::nullopt;
        }
        return 7;
      },
      "partial-stage-model-v1");
  ParetoExtractionResult result =
      ParetoExtractor().extract(store, partialStageModel, makeContext());

  auto identity = llvm::find_if(result.classes, [](const auto &eclass) {
    return eclass.classHash == "identity-class";
  });
  ASSERT_NE(identity, result.classes.end());
  EXPECT_TRUE(identity->usedStructuralFallback);
  for (const ParetoCandidateRecord &candidate : identity->candidates)
    EXPECT_FALSE(candidate.cost.stageCycles.has_value());
  EXPECT_EQ(result.structuralFallbackClassCount, 1u);
  EXPECT_EQ(evaluationCount, 1u);
  EXPECT_TRUE(llvm::is_contained(result.stageModelUnknownReasons,
                                 "subi materialization unsupported"));
}

TEST(ParetoExtractorTest, InvalidPersistentGraphFailsStructuralCompleteness) {
  PersistentEquivalenceClassStore store = makeStore();
  store.graphs.push_back(
      {"broken", "arith.addi", "i32", "", 2, {"input", "missing"}});
  store.classes.push_back({"broken-class", "i32", {"broken"}});

  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());

  EXPECT_EQ(result.invalidStructuralGraphCount, 1u);
  EXPECT_FALSE(result.hasStructuralCostForEveryGraph());
  EXPECT_TRUE(llvm::is_contained(result.stageModelUnknownReasons,
                                 "invalid or cyclic persistent graph: broken"));
  EXPECT_NE(result.toJSON().find("\"structural_cost_complete\": false"),
            std::string::npos);
}

TEST(ParetoExtractorTest, LaunchRemapAmortizesPhysicalProgramWork) {
  PersistentEquivalenceClassStore store;
  store.universeHash = "launch-universe";
  store.semanticsHash = "semantics";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {
      {"pointer", "ags.input", "!tt.ptr<i32>", "", 0, {}},
      {"value", "ags.input", "i32", "", 0, {}},
      {"memory", "ags.memory", "none", "{}", 0, {}},
      {"launch", "ags.launch", "none", "{}", 0, {}},
      {"write0", "tt.store", "none", "{}", 1,
       {"pointer", "value", "memory"}},
      {"write1", "tt.store", "none", "{}", 2,
       {"pointer", "value", "write0"}},
      {"remap", "ags.launch_remap", "none",
       "{ags.proof_only = true, axis = 0 : i32, ceil_div = 0 : i32, "
       "factor = 2 : i32}",
       1, {"launch"}},
      {"source-kernel", "ags.kernel", "none",
       "{ags.proof_only = true}", 2, {"write0", "launch"}},
      {"grouped-kernel", "ags.kernel", "none",
       "{ags.proof_only = true}", 3, {"write1", "remap"}},
  };
  store.classes = {
      {"kernel-class", "none", {"source-kernel", "grouped-kernel"}}};

  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());
  ASSERT_EQ(result.classes.size(), 1u);
  ASSERT_EQ(result.classes.front().allCandidates.size(), 2u);
  auto source = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "source-kernel";
      });
  auto grouped = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "grouped-kernel";
      });
  ASSERT_NE(source, result.classes.front().allCandidates.end());
  ASSERT_NE(grouped, result.classes.front().allCandidates.end());
  EXPECT_EQ(source->cost.structural.weightedOperatorCost, 5u);
  EXPECT_EQ(grouped->cost.structural.weightedOperatorCost, 3u);
  EXPECT_LT(grouped->cost.structural.weightedOperatorCost,
            source->cost.structural.weightedOperatorCost);
}

TEST(ParetoExtractorTest, ParameterizedPartitionAmortizesWithoutLaneCopies) {
  PersistentEquivalenceClassStore store;
  store.universeHash = "parameterized-launch-universe";
  store.semanticsHash = "semantics";
  store.enumerationComplete = true;
  store.fingerprintComplete = true;
  store.proofComplete = true;
  store.graphs = {
      {"pointer", "ags.input", "!tt.ptr<i32>", "", 0, {}},
      {"value", "ags.input", "i32", "", 0, {}},
      {"memory", "ags.memory", "none", "{}", 0, {}},
      {"launch", "ags.launch", "none", "{}", 0, {}},
      {"write", "tt.store", "none", "{}", 1,
       {"pointer", "value", "memory"}},
      {"source", "ags.kernel", "none", "{ags.proof_only = true}", 2,
       {"write", "launch"}},
      {"partition", "ags.kernel_partition", "none",
       "{ags.proof_only = true, axis = 0 : i32, ceil_div = 0 : i32, "
       "factor = 16 : i32}",
       2, {"write", "launch"}},
  };
  store.classes = {{"kernel-class", "none", {"source", "partition"}}};

  ParetoExtractionResult result = ParetoExtractor().extract(
      store, StructuralFallbackCostModel(), makeContext());
  ASSERT_EQ(result.classes.size(), 1u);
  auto source = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "source";
      });
  auto partition = llvm::find_if(
      result.classes.front().allCandidates,
      [](const ParetoCandidateRecord &candidate) {
        return candidate.graphHash == "partition";
      });
  ASSERT_NE(source, result.classes.front().allCandidates.end());
  ASSERT_NE(partition, result.classes.front().allCandidates.end());
  EXPECT_EQ(source->cost.structural.weightedOperatorCost, 5u);
  EXPECT_EQ(partition->cost.structural.weightedOperatorCost, 1u);
  EXPECT_EQ(partition->cost.structural.operatorCount,
            source->cost.structural.operatorCount);
}

} // namespace

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

#ifndef TRITON_TO_GRAPH_PARETO_EXTRACTOR_H
#define TRITON_TO_GRAPH_PARETO_EXTRACTOR_H

#include "TritonToGraph/EquivalenceClassStore.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir::triton::cfg {

// Cost context is intentionally separate from equivalence. A proof class is
// target independent; Pareto winners may differ by concrete target, shape and
// layout interval.
struct GraphCostContext {
  std::string target;
  std::string shapeClass = "universe";
  std::string layoutClass = "unknown";
  std::string dtypeClass = "from-result-type";

  std::string getStableKey() const;
};

// Hardware-independent features are always available. They are the fallback
// when a complete candidate class cannot be materialized and scored by the
// StageModel, and remain useful Pareto dimensions/tie breakers otherwise.
struct StructuralCostFeatures {
  uint64_t operatorCount = 0;
  uint64_t weightedOperatorCost = 0;
  uint64_t maxDepth = 0;
  uint64_t conversionCount = 0;
  uint64_t memoryOperationCount = 0;
  // Sum of the excess absolute stride of statically understood innermost
  // memory lanes. Unit-stride (including reverse unit-stride) contributes
  // zero; stride S contributes |S|-1. Unknown/non-affine accesses do not
  // invent a score and therefore cannot by themselves select a rewrite.
  // This is a generic layout-cost dimension, not a workload matcher.
  uint64_t memoryStridePenalty = 0;

  bool dominates(const StructuralCostFeatures &other) const;
  bool equals(const StructuralCostFeatures &other) const;
};

// Compute the same target-independent structural features for either an
// offline store or an online canonical graph overlay. `lookup` must return a
// stable record for every graph reachable from `rootGraphHash`.
std::optional<StructuralCostFeatures> estimateStructuralCost(
    llvm::StringRef rootGraphHash,
    llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)> lookup);

struct ParetoCandidateCost {
  StructuralCostFeatures structural;
  std::optional<uint64_t> stageCycles;
};

struct ParetoCandidateRecord {
  std::string graphHash;
  ParetoCandidateCost cost;
};

struct ParetoEquivalenceClass {
  std::string classHash;
  bool usedStructuralFallback = true;
  // Full evaluated class is retained for online source-cost comparison. The
  // smaller `candidates` vector is the Pareto frontier persisted as RHS forms.
  std::vector<ParetoCandidateRecord> allCandidates;
  std::vector<ParetoCandidateRecord> candidates;
};

struct ParetoExtractionResult {
  std::string eclassStoreHash;
  std::string costContextKey;
  std::string costModelName;
  std::string costModelHash;
  std::vector<ParetoEquivalenceClass> classes;
  uint64_t candidateCountBefore = 0;
  uint64_t candidateCountAfter = 0;
  uint64_t structuralFallbackClassCount = 0;
  uint64_t invalidStructuralGraphCount = 0;
  std::vector<std::string> stageModelUnknownReasons;

  bool usedStageModelForEveryClass() const {
    return structuralFallbackClassCount == 0;
  }
  bool hasStructuralCostForEveryGraph() const {
    return invalidStructuralGraphCount == 0;
  }
  void writeJSON(llvm::raw_ostream &output) const;
  std::string toJSON() const;
};

// A whole-graph interface: the existing Ascend StageModel runs after candidate
// materialization/lowering and estimates an entire module/pipeline. It is not
// a per-node weight table.
class OfflineGraphCostModel {
public:
  virtual ~OfflineGraphCostModel() = default;
  virtual llvm::StringRef getName() const = 0;
  virtual std::string getVersionHash() const = 0;
  // True only for a target model that can produce hardware cycle estimates.
  // The extractor uses this to avoid materializing singleton e-classes: a
  // class with no alternative has no cost decision to make.
  virtual bool providesStageCycles() const { return false; }
  virtual std::optional<uint64_t>
  estimateStageCycles(const PersistentEquivalenceClassStore &store,
                      const PersistentGraphRecord &root,
                      const GraphCostContext &context,
                      std::string &unknownReason) const = 0;
};

// Explicit fallback for offline universes that cannot yet materialize a full
// target module. It never pretends that structural units are hardware cycles.
class StructuralFallbackCostModel final : public OfflineGraphCostModel {
public:
  llvm::StringRef getName() const override {
    return "structural-pareto-fallback-v2";
  }
  std::string getVersionHash() const override;
  std::optional<uint64_t>
  estimateStageCycles(const PersistentEquivalenceClassStore &store,
                      const PersistentGraphRecord &root,
                      const GraphCostContext &context,
                      std::string &unknownReason) const override;
};

// Adapter boundary for the existing module-level StageModel. The evaluator is
// supplied by the stage that can materialize a candidate DAG into complete IR,
// run -ascend-perf-model and return ascend.scheduled_cycles. This keeps the
// generic Pareto engine independent of lowering implementation details.
class StageModelCostAdapter final : public OfflineGraphCostModel {
public:
  using Evaluator = std::function<std::optional<uint64_t>(
      const PersistentEquivalenceClassStore &,
      const PersistentGraphRecord &, const GraphCostContext &, std::string &)>;

  StageModelCostAdapter(Evaluator evaluator, std::string evaluatorVersion);

  llvm::StringRef getName() const override {
    return "ascend-stage-model-adapter-v1";
  }
  bool providesStageCycles() const override { return true; }
  std::string getVersionHash() const override;
  std::optional<uint64_t>
  estimateStageCycles(const PersistentEquivalenceClassStore &store,
                      const PersistentGraphRecord &root,
                      const GraphCostContext &context,
                      std::string &unknownReason) const override;

private:
  Evaluator evaluator;
  std::string evaluatorVersion;
};

class ParetoExtractor {
public:
  ParetoExtractionResult
  extract(const PersistentEquivalenceClassStore &store,
          const OfflineGraphCostModel &costModel,
          const GraphCostContext &context,
          llvm::StringRef precomputedStoreHash = {}) const;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_PARETO_EXTRACTOR_H

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

#ifndef TRITON_TO_GRAPH_STAGE_MODEL_GRAPH_COST_MODEL_H
#define TRITON_TO_GRAPH_STAGE_MODEL_GRAPH_COST_MODEL_H

#include "TritonToGraph/ParetoExtractor.h"

#include "llvm/ADT/StringMap.h"

#include <memory>
#include <string>

namespace mlir::triton::cfg {

// Offline-only bridge from a persistent AGS candidate DAG to the existing
// module-level Ascend StageCostModel. It materializes a standalone typed
// function, runs the production cost-model pipeline, and accepts a score only
// when every non-zero-cost DAG operation was consumed by that pipeline.
class AscendStageModelGraphCostModel final : public OfflineGraphCostModel {
public:
  // An empty path selects the StageModel's built-in Ascend 910B configuration.
  // Other targets must provide an explicit hardware configuration file.
  explicit AscendStageModelGraphCostModel(
      std::string hardwareConfigPath = "");
  ~AscendStageModelGraphCostModel() override;

  llvm::StringRef getName() const override {
    return "ascend-stage-model-whole-graph-v2";
  }
  bool providesStageCycles() const override { return true; }
  std::string getVersionHash() const override;
  std::optional<uint64_t>
  estimateStageCycles(const PersistentEquivalenceClassStore &store,
                      const PersistentGraphRecord &root,
                      const GraphCostContext &context,
                      std::string &unknownReason) const override;

private:
  std::string hardwareConfigPath;
  std::string hardwareConfigHash;
  std::string configurationError;
  // Pareto extraction queries many roots from one immutable persistent store.
  // Keep its graph index once per model instance instead of rebuilding the
  // complete catalog for every candidate.
  mutable const PersistentEquivalenceClassStore *indexedStore = nullptr;
  mutable llvm::StringMap<const PersistentGraphRecord *> indexedGraphs;
  mutable std::unique_ptr<MLIRContext> materializationContext;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_STAGE_MODEL_GRAPH_COST_MODEL_H

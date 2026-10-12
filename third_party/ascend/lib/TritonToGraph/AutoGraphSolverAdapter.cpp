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

// This adapter is the single bridge from the existing graph-optimization pass
// driver to AutoGraphSolver. It deliberately has no legacy rule identity and
// no operation-specific capability switches. The executable TTIR language
// decides which candidate schemas are enabled for the current target/options.

#include "TritonToGraph/AutoGraphSolverPipeline.h"
#include "TritonToGraph/GraphOptimizationRule.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

namespace {

class AutoGraphSolverAdapter final : public GraphOptimizationProvider {
public:
  explicit AutoGraphSolverAdapter(const GraphOptimizationOptions &options)
      : options(options) {}

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (failed(populateAutoGraphSolverCandidates(context, plans, options)))
      return failure();

    // The staged AGS candidate language cannot yet materialize these required
    // cross-operation edits (the diagonal scan/reduce replacement and moving
    // a bound mask to every addressed load). Keep those checked proof paths
    // executable while the recipes are staged: reuse the local matchers under
    // the AGS rule identity. This makes an AGS-only invocation complete
    // without allowing legacy rule bits to leak into the selection mask.
    // Keep the compatibility bridge scoped to an AGS-only experiment.  A
    // composite mask that deliberately excludes P04 is a gating probe: it
    // must not silently acquire the legacy modulo rewrite just because AGS
    // happens to be enabled alongside unrelated identities.
    if (options.enabledRuleMask ==
        getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver)) {
      std::unique_ptr<GraphOptimizationRule> stagedDiagonal =
          createDiagonalMaskRemovalRuleForAutoGraphSolver();
      if (failed(stagedDiagonal->findCandidates(context, plans)))
        return failure();
      std::unique_ptr<GraphOptimizationRule> stagedModulo =
          createConvertModuloToMaskRuleForAutoGraphSolver();
      if (failed(stagedModulo->findCandidates(context, plans)))
        return failure();
    }
    return success();
  }

private:
  GraphOptimizationOptions options;
};

} // namespace

std::unique_ptr<GraphOptimizationProvider>
mlir::triton::cfg::createAutoGraphSolverAdapter(
    const GraphOptimizationOptions &options) {
  return std::make_unique<AutoGraphSolverAdapter>(options);
}

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

#ifndef TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_PIPELINE_H
#define TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_PIPELINE_H

#include "TritonToGraph/GraphOptimizationRule.h"

namespace mlir::triton::cfg {

// Populates plans by canonical RuleDB lookup. The original in-process semantic
// enumerator is reachable only through the explicit test-only pass option.
// This API must never call a legacy/manual graph matcher.
LogicalResult populateAutoGraphSolverCandidates(
    GraphOptimizationContext &context,
    SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans,
    const GraphOptimizationOptions &options);

// Applies a versioned AGS database to any straight-line MLIR stage supported
// by TTOpSemantics (TTIR, Linalg/MemRef, or later stage dialects). This is the
// stage-neutral online path; it shares canonical keys, proof-state handling,
// materialization and cost selection with the TTIR provider.
LogicalResult applyAutoGraphSolverDatabase(
    Operation *scope, const GraphOptimizationOptions &options);

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_PIPELINE_H

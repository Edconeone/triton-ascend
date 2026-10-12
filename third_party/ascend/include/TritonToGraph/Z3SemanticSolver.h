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

#ifndef TRITON_TO_GRAPH_Z3_SEMANTIC_SOLVER_H
#define TRITON_TO_GRAPH_Z3_SEMANTIC_SOLVER_H

#include "TritonToGraph/AutoGraphSolver.h"

#include <cstdint>
#include <memory>

namespace mlir::triton::cfg {

struct Z3SemanticSolverStatistics {
  uint64_t queries = 0;
  uint64_t proved = 0;
  uint64_t refuted = 0;
  uint64_t unknown = 0;
  uint64_t unsupported = 0;
  uint64_t timedOut = 0;
  uint64_t otherUnknown = 0;
  uint64_t simplifierProved = 0;
  uint64_t simplifierRefuted = 0;
  uint64_t retryChecks = 0;
  uint64_t retryUnsatChecks = 0;
  uint64_t retrySatChecks = 0;
  uint64_t retryUnknown = 0;
  uint64_t bitBlastChecks = 0;
  uint64_t bitBlastUnsatChecks = 0;
  uint64_t bitBlastSatChecks = 0;
  uint64_t bitBlastUnknown = 0;
  uint64_t finiteShiftDomainChecks = 0;
  uint64_t finiteShiftDomainUnsatChecks = 0;
  uint64_t finiteShiftDomainSatChecks = 0;
  uint64_t finiteShiftDomainUnknownChecks = 0;
  uint64_t workerContexts = 0;
  uint64_t workerContextCreations = 0;
  uint64_t workerContextRecycles = 0;
};

// Offline exact fallback for the integer/boolean subset of
// TensorIndexSemantics.  It asks Z3 for a model of semantic inequality:
//
//   valid(lhs) != valid(rhs) ||
//     (valid(lhs) && valid(rhs) && value(lhs) != value(rhs))
//
// UNSAT is a proof of equivalence, SAT is a concrete refutation, and timeout
// or an unsupported semantic construct remains Unknown.  This class is built
// into the rule-generation tool only; it is not linked into the online pass.
class Z3SemanticSolver final : public SemanticEquivalenceBackend {
public:
  explicit Z3SemanticSolver(unsigned timeoutMilliseconds = 20,
                            unsigned maxQueriesPerWorkerContext = 4096,
                            unsigned retryTimeoutMilliseconds = 0,
                            unsigned bitBlastTimeoutMilliseconds = 0,
                            unsigned maxBitBlastWorkers = 8);
  ~Z3SemanticSolver() override;

  StringRef getProofBackendName() const override;
  SemanticProofResult
  compareNormalized(const TensorDenotation &lhs,
                    const TensorDenotation &rhs) const override;
  SemanticProofResult
  replay(const SemanticEquivalenceProof &proof) const override;

  Z3SemanticSolverStatistics getStatistics() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_Z3_SEMANTIC_SOLVER_H

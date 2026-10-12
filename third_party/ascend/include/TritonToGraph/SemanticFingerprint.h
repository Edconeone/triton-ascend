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

#ifndef TRITON_TO_GRAPH_SEMANTIC_FINGERPRINT_H
#define TRITON_TO_GRAPH_SEMANTIC_FINGERPRINT_H

#include "TritonToGraph/AutoGraphSolver.h"
#include "TritonToGraph/OfflineGraphEnumerator.h"

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir::triton::cfg {

struct OfflineSemanticFingerprintBucket {
  std::string fingerprint;
  std::vector<unsigned> graphIds;
};

struct OfflineSemanticFingerprintGap {
  unsigned graphId = 0;
  std::string graphKey;
  std::string reason;
  std::string semanticKey;
};

// A fingerprint is only a sound pre-filter. Graphs in different concrete
// sample buckets cannot be equivalent; graphs in the same bucket still need
// the EquivalenceProver. Graphs outside the executable fingerprint fragment
// are wildcards and remain comparable with every graph of the same type.
struct OfflineSemanticFingerprintResult {
  static constexpr unsigned MaxWildcardGraphDetails = 256;

  std::vector<OfflineSemanticFingerprintBucket> buckets;
  std::vector<unsigned> wildcardGraphIds;
  // Keep every id for completeness/proof accounting, but retain only a
  // bounded diagnostic sample. A broad unsupported semantic family can
  // otherwise duplicate hundreds of thousands of full denotation strings in
  // the JSON report and consume more memory than the search itself.
  std::vector<OfflineSemanticFingerprintGap> wildcardGraphs;
  std::vector<std::string> unsupportedReasons;
  // Indexed by OfflineTypedGraph::id. This is intentionally not serialized in
  // the fingerprint report; it is the in-memory handoff to the proof stage.
  std::vector<std::optional<TensorDenotation>> denotations;
  // True only when every available denotation above is normalizer output.
  // Programmatic/external results default to false and keep the ordinary
  // oracle path; production fingerprinting can skip redundant normalization.
  bool denotationsNormalized = false;
  // Stable exact-denotation group id for each graph. The fingerprinter already
  // computes this quotient before concrete-model evaluation, so handing it to
  // the prover avoids printing and grouping every normalized denotation a
  // second time. The maximum unsigned value denotes an unavailable/wildcard
  // denotation.
  std::vector<unsigned> exactDenotationGroupIds;
  unsigned modelCount = 0;
  uint64_t comparablePairsBefore = 0;
  uint64_t proofCandidatePairsAfter = 0;
  // Multiple typed DAGs often normalize to the same denotation. Fingerprint
  // that semantic normal form once and reuse the result; this is a semantic
  // quotient and contains no operation-specific knowledge.
  uint64_t uniqueDenotationsFingerprinted = 0;
  uint64_t denotationFingerprintCacheHits = 0;
  // Concrete-model evaluation is adaptive: model zero validates every unique
  // denotation, then later models run only for buckets that still contain
  // distinct semantic normal forms. This preserves the eager model partition
  // while avoiding work that cannot change it.
  uint64_t denotationModelEvaluations = 0;
  uint64_t denotationModelEvaluationsAvoided = 0;
  unsigned parallelJobs = 1;

  uint64_t getFingerprintedGraphCount() const;
  bool isComplete() const { return wildcardGraphIds.empty(); }
  void writeJSON(llvm::raw_ostream &output, llvm::StringRef universeHash,
                 bool enumerationComplete) const;
  std::string toJSON(llvm::StringRef universeHash,
                     bool enumerationComplete) const;
};

// Computes deterministic concrete-model signatures from the common
// TensorDenotation language, including tensor element/index samples and
// finite Sum/ForAll binders. The implementation contains no TT operation
// names. Unsupported semantic kinds yield wildcard graphs instead of an
// unsound negative filter.
class OfflineSemanticFingerprinter {
public:
  explicit OfflineSemanticFingerprinter(unsigned modelCount = 4,
                                         unsigned parallelJobs = 1)
      : modelCount(modelCount), parallelJobs(std::max(1u, parallelJobs)) {}

  OfflineSemanticFingerprintResult
  fingerprint(llvm::ArrayRef<OfflineTypedGraph> graphs,
              const TTOpSemanticsRegistry &semantics,
              const TensorSemanticNormalizer &normalizer) const;

  // Searches an additional deterministic model range for a concrete semantic
  // counterexample. A true result soundly refutes equivalence; false means the
  // finite range did not distinguish the denotations and is never treated as
  // a proof of equality. This remains generic over TensorDenotation and has no
  // TT operation names or workload matchers.
  static bool hasConcreteCounterexample(const TensorDenotation &lhs,
                                        const TensorDenotation &rhs,
                                        unsigned firstModel,
                                        unsigned additionalModelCount,
                                        std::string &reason);

  // Returns the deterministic signature for one denotation over an explicit
  // model range. The proof stage uses this to cache adaptive counterexample
  // signatures per graph instead of re-evaluating the same denotation for
  // every candidate pair. Failure means the denotation is outside the
  // executable fingerprint fragment and must remain conservatively unknown.
  static std::optional<std::string>
  fingerprintDenotationRange(const TensorDenotation &denotation,
                             unsigned firstModel, unsigned modelCount,
                             std::string &reason);

private:
  unsigned modelCount;
  unsigned parallelJobs;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_SEMANTIC_FINGERPRINT_H

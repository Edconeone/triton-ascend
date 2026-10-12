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

#ifndef TRITON_TO_GRAPH_OFFLINE_EQUIVALENCE_PROVER_H
#define TRITON_TO_GRAPH_OFFLINE_EQUIVALENCE_PROVER_H

#include "TritonToGraph/SemanticFingerprint.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir::triton::cfg {

struct OfflineSemanticProofPair {
  unsigned lhsGraphId = 0;
  unsigned rhsGraphId = 0;
  std::string lhsGraphKey;
  std::string rhsGraphKey;
  SemanticProofStatus status = SemanticProofStatus::Unknown;
  std::string proofHash;
  std::string proofCertificate;
  // Large exact-denotation groups reuse one replayable certificate. Keep the
  // public inline string for ordinary/small proofs while allowing production
  // forest edges to share immutable certificate storage.
  std::shared_ptr<const std::string> sharedProofCertificate;
  std::string reason;
  std::string lhsSemanticKey;
  std::string rhsSemanticKey;
  // Unknown relations often reuse one normalized denotation in many pairs.
  // Share those large diagnostic keys in production instead of copying the
  // same printed expression hundreds of thousands of times.
  std::shared_ptr<const std::string> sharedLhsSemanticKey;
  std::shared_ptr<const std::string> sharedRhsSemanticKey;

  llvm::StringRef getProofCertificate() const {
    return sharedProofCertificate ? llvm::StringRef(*sharedProofCertificate)
                                  : llvm::StringRef(proofCertificate);
  }
  llvm::StringRef getLhsSemanticKey() const {
    return sharedLhsSemanticKey ? llvm::StringRef(*sharedLhsSemanticKey)
                                : llvm::StringRef(lhsSemanticKey);
  }
  llvm::StringRef getRhsSemanticKey() const {
    return sharedRhsSemanticKey ? llvm::StringRef(*sharedRhsSemanticKey)
                                : llvm::StringRef(rhsSemanticKey);
  }
};

// A pending pair is scheduling state, never proof evidence. It records an
// exact query that returned Unknown so a later run can retry that frontier
// before scanning new pairs with a larger solver budget.
struct OfflinePendingProofPair {
  unsigned lhsGraphId = 0;
  unsigned rhsGraphId = 0;
  std::string lhsGraphKey;
  std::string rhsGraphKey;
  uint64_t attemptCount = 0;
  std::string lastReason;
  std::string lhsSemanticKey;
  std::string rhsSemanticKey;
};

// Restartable proof-progress cache. Only Proved and Refuted entries in `pairs`
// are terminal proof results. `pendingPairs` contains Unknown scheduling hints;
// loading one never merges graphs or classifies a pair. Graph keys bind numeric
// ids to one deterministic enumeration, while the universe hash binds the
// cache to the finite search space that produced it.
struct OfflineResolvedProofCache {
  std::string universeHash;
  uint64_t graphCount = 0;
  std::vector<OfflineSemanticProofPair> pairs;
  std::vector<OfflinePendingProofPair> pendingPairs;

  void writeJSON(llvm::raw_ostream &output) const;
  std::string toJSON() const;
  static std::optional<OfflineResolvedProofCache>
  fromJSON(llvm::StringRef json, std::string &error);
};

struct OfflineEquivalenceProofOptions {
  // Called cooperatively while candidate pairs are formed and proved. A true
  // result stops the stage without allowing a partial proof set to masquerade
  // as a complete equivalence relation.
  std::function<bool()> shouldStop;

  // Refuted pairs carry no evidence consumed by the e-class store. Keeping
  // them is useful in small unit tests, but production rule generation can
  // omit their verbose graph keys and reasons while retaining exact counts.
  bool retainRefutedPairs = true;

  // Graph ids are sufficient for the in-memory e-class builder, which already
  // owns the enumeration table. Production reports may omit duplicated
  // canonical graph-key strings; diagnostic/unit-test mode keeps them.
  bool retainGraphKeys = true;

  // A proved equality edge makes every pair already connected by proved
  // edges equivalent by transitivity. Production builds keep only this
  // replayable spanning forest instead of materializing O(N^2) redundant
  // certificates inside a large fingerprint bucket.
  bool pruneTransitivelyImpliedPairs = false;

  // Continue primary fingerprint collisions for at most this many additional
  // deterministic models. Production quotient mode refines only sub-buckets
  // that still contain distinct exact denotations, before invoking the
  // symbolic oracle. A differing concrete Value/Definedness result is a sound
  // refutation; exhausting the finite range never proves equality.
  unsigned additionalCounterexampleModels = 0;

  // Independent denotations can be evaluated over the additional model
  // range concurrently. This changes only throughput, never the generated
  // signatures or proof relation.
  unsigned counterexampleJobs = 1;

  // Optional restart state from an earlier interrupted offline proof run.
  // Replayable Proved and deterministic Refuted entries are reused. Pending
  // Unknown entries are only a priority frontier and are retried with the
  // current solver budget.
  const OfflineResolvedProofCache *resolvedProofCache = nullptr;
};

// Auditable output of the proof stage. Pairs rejected by different complete
// fingerprints have concrete counterexamples and do not enter the expensive
// normal-form prover. In exhaustive mode every remaining pair has a tri-state
// record; production mode may instead account for it as transitively proved by
// the retained, replayable certificate forest.
struct OfflineEquivalenceProofResult {
  std::vector<OfflineSemanticProofPair> pairs;
  uint64_t prefilterRefutedPairs = 0;
  uint64_t candidatePairs = 0;
  uint64_t processedPairs = 0;
  uint64_t provedPairs = 0;
  uint64_t transitivelyProvedPairs = 0;
  uint64_t refutedPairs = 0;
  uint64_t counterexampleRefutedPairs = 0;
  // Number of denotation/model evaluations performed by the additional
  // counterexample filter. Production quotient mode evaluates one model at a
  // time only for exact semantic groups that still collide.
  uint64_t counterexampleModelEvaluations = 0;
  uint64_t counterexampleModelEvaluationsAvoided = 0;
  // A bounded primary-normalizer probe runs before the additional concrete
  // models. It coalesces only relations proved by the cheap in-process
  // procedure and never invokes an exact fallback.
  uint64_t symbolicPremergeComparisons = 0;
  uint64_t symbolicPremergeMergedGroups = 0;
  // After concrete refinement, remaining same-signature relations are probed
  // with the complete oracle in parallel. Proved certificates are retained for
  // final proof-forest construction; negative/unknown answers are also cached.
  uint64_t exactPostRefinementComparisons = 0;
  uint64_t exactPostRefinementProved = 0;
  uint64_t resolvedProofCacheHits = 0;
  uint64_t resolvedProofCacheRejected = 0;
  uint64_t pendingProofCacheAttempts = 0;
  uint64_t pendingProofCacheResolved = 0;
  uint64_t pendingProofCacheRemaining = 0;
  uint64_t pendingProofCacheRejected = 0;
  uint64_t unknownPairs = 0;
  uint64_t oracleComparisons = 0;
  uint64_t semanticQuotientGroups = 0;
  uint64_t semanticQuotientClasses = 0;
  unsigned counterexampleJobs = 1;
  bool usedSemanticQuotient = false;
  bool interrupted = false;
  OfflineResolvedProofCache resolvedProofCache;

  bool isComplete() const { return !interrupted && unknownPairs == 0; }
  void writeJSON(llvm::raw_ostream &output, llvm::StringRef universeHash,
                 bool enumerationComplete,
                 bool fingerprintComplete) const;
  std::string toJSON(llvm::StringRef universeHash,
                     bool enumerationComplete,
                     bool fingerprintComplete) const;
};

// Generic bridge from fingerprint buckets to the equivalence oracle. It has
// no operation names or workload matchers: adding an operation changes only
// its schema/semantics and the finite universe.
class OfflineEquivalenceProver {
public:
  OfflineEquivalenceProofResult
  prove(llvm::ArrayRef<OfflineTypedGraph> graphs,
        const OfflineSemanticFingerprintResult &fingerprints,
        const SemanticEquivalenceOracle &oracle,
        const OfflineEquivalenceProofOptions &options = {}) const;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_OFFLINE_EQUIVALENCE_PROVER_H

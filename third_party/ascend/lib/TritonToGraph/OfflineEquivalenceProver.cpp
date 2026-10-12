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

#include "TritonToGraph/OfflineEquivalenceProver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static llvm::StringRef stringifyStatus(SemanticProofStatus status) {
  switch (status) {
  case SemanticProofStatus::Proved:
    return "proved";
  case SemanticProofStatus::Refuted:
    return "refuted";
  case SemanticProofStatus::Unknown:
    return "unknown";
  }
  llvm_unreachable("unknown semantic proof status");
}

static std::optional<SemanticProofStatus> parseResolvedStatus(StringRef text) {
  if (text == "proved")
    return SemanticProofStatus::Proved;
  if (text == "refuted")
    return SemanticProofStatus::Refuted;
  return std::nullopt;
}

static uint64_t pairKey(unsigned lhs, unsigned rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  return (static_cast<uint64_t>(lhs) << 32) | rhs;
}

static uint64_t chooseTwo(uint64_t count) {
  return count < 2 ? 0 : count * (count - 1) / 2;
}

template <typename Function>
static void parallelForWork(unsigned count, unsigned requestedJobs,
                            Function &&function) {
  const unsigned jobs = std::min(count, std::max(1u, requestedJobs));
  if (jobs <= 1 || count <= 1) {
    for (unsigned index = 0; index < count; ++index)
      function(index);
    return;
  }

  std::atomic<unsigned> next{0};
  std::vector<std::thread> workers;
  workers.reserve(jobs);
  for (unsigned worker = 0; worker < jobs; ++worker) {
    workers.emplace_back([&] {
      while (true) {
        unsigned index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count)
          return;
        function(index);
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
}

} // namespace

void OfflineResolvedProofCache::writeJSON(llvm::raw_ostream &output) const {
  llvm::DenseSet<uint64_t> resolvedKeys;
  for (const OfflineSemanticProofPair &pair : pairs)
    if (pair.status != SemanticProofStatus::Unknown)
      resolvedKeys.insert(pairKey(pair.lhsGraphId, pair.rhsGraphId));
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("schema_version", 2);
    json.attribute("universe_hash", universeHash);
    json.attribute("graph_count", static_cast<int64_t>(graphCount));
    json.attributeArray("pairs", [&] {
      for (const OfflineSemanticProofPair &pair : pairs) {
        if (pair.status == SemanticProofStatus::Unknown)
          continue;
        json.object([&] {
          json.attribute("lhs_graph_id", static_cast<int64_t>(pair.lhsGraphId));
          json.attribute("rhs_graph_id", static_cast<int64_t>(pair.rhsGraphId));
          json.attribute("lhs_graph_key", pair.lhsGraphKey);
          json.attribute("rhs_graph_key", pair.rhsGraphKey);
          json.attribute("status", stringifyStatus(pair.status));
          if (!pair.proofHash.empty())
            json.attribute("proof_hash", pair.proofHash);
          if (!pair.getProofCertificate().empty())
            json.attribute("proof_certificate", pair.getProofCertificate());
        });
      }
    });
    json.attributeArray("pending_pairs", [&] {
      llvm::DenseSet<uint64_t> emitted;
      for (const OfflinePendingProofPair &pair : pendingPairs) {
        const uint64_t key = pairKey(pair.lhsGraphId, pair.rhsGraphId);
        if (resolvedKeys.contains(key) || !emitted.insert(key).second)
          continue;
        json.object([&] {
          json.attribute("lhs_graph_id", static_cast<int64_t>(pair.lhsGraphId));
          json.attribute("rhs_graph_id", static_cast<int64_t>(pair.rhsGraphId));
          json.attribute("lhs_graph_key", pair.lhsGraphKey);
          json.attribute("rhs_graph_key", pair.rhsGraphKey);
          json.attribute("attempt_count",
                         static_cast<int64_t>(pair.attemptCount));
          if (!pair.lastReason.empty())
            json.attribute("last_reason", pair.lastReason);
          if (!pair.lhsSemanticKey.empty())
            json.attribute("lhs_semantic_key", pair.lhsSemanticKey);
          if (!pair.rhsSemanticKey.empty())
            json.attribute("rhs_semantic_key", pair.rhsSemanticKey);
        });
      }
    });
  });
  output << '\n';
}

std::string OfflineResolvedProofCache::toJSON() const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream);
  return storage;
}

std::optional<OfflineResolvedProofCache>
OfflineResolvedProofCache::fromJSON(StringRef contents, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(contents);
  if (!parsed) {
    error = "resolved proof cache is not valid JSON: " +
            llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  llvm::json::Object *object = parsed->getAsObject();
  if (!object) {
    error = "resolved proof cache root must be an object";
    return std::nullopt;
  }
  std::optional<int64_t> schemaVersion = object->getInteger("schema_version");
  std::optional<StringRef> universeHash = object->getString("universe_hash");
  std::optional<int64_t> graphCount = object->getInteger("graph_count");
  llvm::json::Array *pairs = object->getArray("pairs");
  if (!schemaVersion || (*schemaVersion != 1 && *schemaVersion != 2) ||
      !universeHash || !graphCount || *graphCount < 0 || !pairs) {
    error = "resolved proof cache header is incomplete or unsupported";
    return std::nullopt;
  }

  OfflineResolvedProofCache cache;
  cache.universeHash = universeHash->str();
  cache.graphCount = static_cast<uint64_t>(*graphCount);
  cache.pairs.reserve(pairs->size());
  for (const llvm::json::Value &value : *pairs) {
    const llvm::json::Object *pairObject = value.getAsObject();
    if (!pairObject) {
      error = "resolved proof cache pair must be an object";
      return std::nullopt;
    }
    std::optional<int64_t> lhs = pairObject->getInteger("lhs_graph_id");
    std::optional<int64_t> rhs = pairObject->getInteger("rhs_graph_id");
    std::optional<StringRef> lhsKey = pairObject->getString("lhs_graph_key");
    std::optional<StringRef> rhsKey = pairObject->getString("rhs_graph_key");
    std::optional<StringRef> statusText = pairObject->getString("status");
    std::optional<SemanticProofStatus> status =
        statusText ? parseResolvedStatus(*statusText) : std::nullopt;
    if (!lhs || !rhs || *lhs < 0 || *rhs < 0 ||
        static_cast<uint64_t>(*lhs) > std::numeric_limits<unsigned>::max() ||
        static_cast<uint64_t>(*rhs) > std::numeric_limits<unsigned>::max() ||
        !lhsKey || !rhsKey || !status) {
      error = "resolved proof cache pair is incomplete or invalid";
      return std::nullopt;
    }
    OfflineSemanticProofPair pair;
    pair.lhsGraphId = static_cast<unsigned>(*lhs);
    pair.rhsGraphId = static_cast<unsigned>(*rhs);
    pair.lhsGraphKey = lhsKey->str();
    pair.rhsGraphKey = rhsKey->str();
    pair.status = *status;
    if (std::optional<StringRef> proofHash =
            pairObject->getString("proof_hash"))
      pair.proofHash = proofHash->str();
    if (std::optional<StringRef> certificate =
            pairObject->getString("proof_certificate"))
      pair.proofCertificate = certificate->str();
    if (pair.status == SemanticProofStatus::Proved &&
        (pair.proofHash.empty() || pair.proofCertificate.empty())) {
      error = "proved cache pair has no replayable certificate";
      return std::nullopt;
    }
    cache.pairs.push_back(std::move(pair));
  }
  if (*schemaVersion == 2) {
    llvm::json::Array *pendingPairs = object->getArray("pending_pairs");
    if (!pendingPairs) {
      error = "proof-progress cache has no pending pair array";
      return std::nullopt;
    }
    cache.pendingPairs.reserve(pendingPairs->size());
    for (const llvm::json::Value &value : *pendingPairs) {
      const llvm::json::Object *pairObject = value.getAsObject();
      if (!pairObject) {
        error = "pending proof cache pair must be an object";
        return std::nullopt;
      }
      std::optional<int64_t> lhs = pairObject->getInteger("lhs_graph_id");
      std::optional<int64_t> rhs = pairObject->getInteger("rhs_graph_id");
      std::optional<StringRef> lhsKey = pairObject->getString("lhs_graph_key");
      std::optional<StringRef> rhsKey = pairObject->getString("rhs_graph_key");
      std::optional<int64_t> attempts = pairObject->getInteger("attempt_count");
      if (!lhs || !rhs || *lhs < 0 || *rhs < 0 ||
          static_cast<uint64_t>(*lhs) > std::numeric_limits<unsigned>::max() ||
          static_cast<uint64_t>(*rhs) > std::numeric_limits<unsigned>::max() ||
          !lhsKey || !rhsKey || !attempts || *attempts < 0) {
        error = "pending proof cache pair is incomplete or invalid";
        return std::nullopt;
      }
      OfflinePendingProofPair pair;
      pair.lhsGraphId = static_cast<unsigned>(*lhs);
      pair.rhsGraphId = static_cast<unsigned>(*rhs);
      pair.lhsGraphKey = lhsKey->str();
      pair.rhsGraphKey = rhsKey->str();
      pair.attemptCount = static_cast<uint64_t>(*attempts);
      if (std::optional<StringRef> reason =
              pairObject->getString("last_reason"))
        pair.lastReason = reason->str();
      if (std::optional<StringRef> semanticKey =
              pairObject->getString("lhs_semantic_key"))
        pair.lhsSemanticKey = semanticKey->str();
      if (std::optional<StringRef> semanticKey =
              pairObject->getString("rhs_semantic_key"))
        pair.rhsSemanticKey = semanticKey->str();
      cache.pendingPairs.push_back(std::move(pair));
    }
  }
  return cache;
}

void OfflineEquivalenceProofResult::writeJSON(
    llvm::raw_ostream &output, llvm::StringRef universeHash,
    bool enumerationComplete, bool fingerprintComplete) const {
  // Production proof forests may retain hundreds of thousands of replayable
  // edges. Stream entries directly so report emission does not duplicate the
  // complete forest in a json::Array and a second serialized string.
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("universe_hash", universeHash);
    json.attribute("enumeration_complete", enumerationComplete);
    json.attribute("fingerprint_complete", fingerprintComplete);
    json.attribute("proof_complete", isComplete());
    json.attribute("interrupted", interrupted);
    json.attribute("prefilter_refuted_pair_count",
                   static_cast<int64_t>(prefilterRefutedPairs));
    json.attribute("proof_candidate_pair_count",
                   static_cast<int64_t>(candidatePairs));
    json.attribute("processed_pair_count",
                   static_cast<int64_t>(processedPairs));
    json.attribute("retained_pair_count", static_cast<int64_t>(pairs.size()));
    json.attribute("proved_pair_count", static_cast<int64_t>(provedPairs));
    json.attribute("transitively_proved_pair_count",
                   static_cast<int64_t>(transitivelyProvedPairs));
    json.attribute("refuted_pair_count", static_cast<int64_t>(refutedPairs));
    json.attribute("counterexample_refuted_pair_count",
                   static_cast<int64_t>(counterexampleRefutedPairs));
    json.attribute("counterexample_model_evaluations",
                   static_cast<int64_t>(counterexampleModelEvaluations));
    json.attribute(
        "counterexample_model_evaluations_avoided",
        static_cast<int64_t>(counterexampleModelEvaluationsAvoided));
    json.attribute("symbolic_premerge_comparisons",
                   static_cast<int64_t>(symbolicPremergeComparisons));
    json.attribute("symbolic_premerge_merged_groups",
                   static_cast<int64_t>(symbolicPremergeMergedGroups));
    json.attribute("exact_post_refinement_comparisons",
                   static_cast<int64_t>(exactPostRefinementComparisons));
    json.attribute("exact_post_refinement_proved",
                   static_cast<int64_t>(exactPostRefinementProved));
    json.attribute("resolved_proof_cache_hits",
                   static_cast<int64_t>(resolvedProofCacheHits));
    json.attribute("resolved_proof_cache_rejected",
                   static_cast<int64_t>(resolvedProofCacheRejected));
    json.attribute("pending_proof_cache_attempts",
                   static_cast<int64_t>(pendingProofCacheAttempts));
    json.attribute("pending_proof_cache_resolved",
                   static_cast<int64_t>(pendingProofCacheResolved));
    json.attribute("pending_proof_cache_remaining",
                   static_cast<int64_t>(pendingProofCacheRemaining));
    json.attribute("pending_proof_cache_rejected",
                   static_cast<int64_t>(pendingProofCacheRejected));
    json.attribute("unknown_pair_count", static_cast<int64_t>(unknownPairs));
    json.attribute("oracle_comparison_count",
                   static_cast<int64_t>(oracleComparisons));
    json.attribute("used_semantic_quotient", usedSemanticQuotient);
    json.attribute("semantic_quotient_group_count",
                   static_cast<int64_t>(semanticQuotientGroups));
    json.attribute("semantic_quotient_class_count",
                   static_cast<int64_t>(semanticQuotientClasses));
    json.attribute("counterexample_jobs",
                   static_cast<int64_t>(counterexampleJobs));
    json.attributeArray("pairs", [&] {
      for (const OfflineSemanticProofPair &pair : pairs) {
        json.object([&] {
          json.attribute("lhs_graph_id", static_cast<int64_t>(pair.lhsGraphId));
          json.attribute("rhs_graph_id", static_cast<int64_t>(pair.rhsGraphId));
          json.attribute("lhs_graph_key", pair.lhsGraphKey);
          json.attribute("rhs_graph_key", pair.rhsGraphKey);
          json.attribute("status", stringifyStatus(pair.status));
          json.attribute("reason", pair.reason);
          if (!pair.getLhsSemanticKey().empty())
            json.attribute("lhs_semantic_key", pair.getLhsSemanticKey());
          if (!pair.getRhsSemanticKey().empty())
            json.attribute("rhs_semantic_key", pair.getRhsSemanticKey());
          if (!pair.proofHash.empty())
            json.attribute("proof_hash", pair.proofHash);
          if (!pair.getProofCertificate().empty())
            json.attribute("proof_certificate", pair.getProofCertificate());
        });
      }
    });
  });
  output << '\n';
}

std::string OfflineEquivalenceProofResult::toJSON(
    llvm::StringRef universeHash, bool enumerationComplete,
    bool fingerprintComplete) const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream, universeHash, enumerationComplete, fingerprintComplete);
  return storage;
}

OfflineEquivalenceProofResult OfflineEquivalenceProver::prove(
    llvm::ArrayRef<OfflineTypedGraph> graphs,
    const OfflineSemanticFingerprintResult &fingerprints,
    const SemanticEquivalenceOracle &oracle,
    const OfflineEquivalenceProofOptions &options) const {
  OfflineEquivalenceProofResult result;
  result.counterexampleJobs = std::max(1u, options.counterexampleJobs);
  result.prefilterRefutedPairs = fingerprints.comparablePairsBefore -
                                 fingerprints.proofCandidatePairsAfter;
  auto compareDenotations = [&](const TensorDenotation &lhs,
                                const TensorDenotation &rhs) {
    return fingerprints.denotationsNormalized
               ? oracle.compareNormalized(lhs, rhs)
               : oracle.compare(lhs, rhs);
  };
  auto comparePrimaryDenotations = [&](const TensorDenotation &lhs,
                                       const TensorDenotation &rhs) {
    return fingerprints.denotationsNormalized
               ? oracle.comparePrimaryNormalized(lhs, rhs)
               : oracle.comparePrimary(lhs, rhs);
  };

  uint64_t workItems = 0;
  auto shouldStop = [&]() {
    if (!options.shouldStop || (workItems++ & 1023) != 0)
      return false;
    if (!options.shouldStop())
      return false;
    result.interrupted = true;
    return true;
  };

  auto retainGraphKeys = [&](OfflineSemanticProofPair &pair) {
    if (!options.retainGraphKeys)
      return;
    if (pair.lhsGraphId < graphs.size())
      pair.lhsGraphKey = graphs[pair.lhsGraphId].canonicalKey;
    if (pair.rhsGraphId < graphs.size())
      pair.rhsGraphKey = graphs[pair.rhsGraphId].canonicalKey;
  };
  struct PairEvaluation {
    OfflineSemanticProofPair pair;
    bool counterexampleRefuted = false;
  };
  struct CachedCounterexampleFingerprint {
    bool computed = false;
    std::optional<std::string> signature;
    std::string reason;
  };
  std::vector<CachedCounterexampleFingerprint> counterexampleFingerprints(
      fingerprints.denotations.size());
  std::vector<std::shared_ptr<const std::string>> stableDenotationKeys(
      fingerprints.denotations.size());
  auto getStableDenotationKey =
      [&](unsigned graphId) -> const std::shared_ptr<const std::string> & {
    std::shared_ptr<const std::string> &cached = stableDenotationKeys[graphId];
    if (!cached)
      cached = std::make_shared<const std::string>(
          getStableTensorDenotationKey(*fingerprints.denotations[graphId]));
    return cached;
  };
  // Symbolic premerge is an accelerator, not a second authority. Remember its
  // primary refutations, but let its Unknown answers reach the complete oracle
  // after concrete refinement. Final negative/unknown answers are cached so
  // multiplicity accounting does not repeat a deterministic query. Proved
  // answers are intentionally replayed below to construct the persistent
  // certificate.
  llvm::DenseMap<uint64_t, SemanticProofStatus> symbolicProbeCache;
  llvm::DenseMap<uint64_t, SemanticProofResult> exactProbeCache;
  llvm::DenseSet<uint64_t> resolvedProofCacheKeys;
  llvm::DenseSet<uint64_t> loadedResolvedProofCacheKeys;
  llvm::DenseMap<uint64_t, unsigned> pendingProofCacheIndices;
  result.resolvedProofCache.graphCount = graphs.size();

  auto retirePendingProof = [&](uint64_t key) {
    if (!pendingProofCacheIndices.erase(key))
      return;
    ++result.pendingProofCacheResolved;
    --result.pendingProofCacheRemaining;
  };

  auto rememberPendingProof = [&](unsigned lhsGraphId, unsigned rhsGraphId,
                                  StringRef reason) {
    const uint64_t key = pairKey(lhsGraphId, rhsGraphId);
    if (resolvedProofCacheKeys.contains(key))
      return;
    auto found = pendingProofCacheIndices.find(key);
    if (found != pendingProofCacheIndices.end()) {
      OfflinePendingProofPair &pending =
          result.resolvedProofCache.pendingPairs[found->second];
      ++pending.attemptCount;
      pending.lastReason = reason.str();
      return;
    }
    OfflinePendingProofPair pending;
    pending.lhsGraphId = lhsGraphId;
    pending.rhsGraphId = rhsGraphId;
    pending.lhsGraphKey = graphs[lhsGraphId].canonicalKey;
    pending.rhsGraphKey = graphs[rhsGraphId].canonicalKey;
    pending.attemptCount = 1;
    pending.lastReason = reason.str();
    pending.lhsSemanticKey = *getStableDenotationKey(lhsGraphId);
    pending.rhsSemanticKey = *getStableDenotationKey(rhsGraphId);
    pendingProofCacheIndices.try_emplace(
        key, result.resolvedProofCache.pendingPairs.size());
    result.resolvedProofCache.pendingPairs.push_back(std::move(pending));
    ++result.pendingProofCacheRemaining;
  };

  auto rememberResolvedProof = [&](unsigned lhsGraphId, unsigned rhsGraphId,
                                   const SemanticProofResult &proof) {
    if (proof.status == SemanticProofStatus::Unknown)
      return;
    const uint64_t key = pairKey(lhsGraphId, rhsGraphId);
    if (!resolvedProofCacheKeys.insert(key).second)
      return;
    OfflineSemanticProofPair pair;
    pair.lhsGraphId = lhsGraphId;
    pair.rhsGraphId = rhsGraphId;
    pair.lhsGraphKey = graphs[lhsGraphId].canonicalKey;
    pair.rhsGraphKey = graphs[rhsGraphId].canonicalKey;
    pair.status = proof.status;
    if (proof.status == SemanticProofStatus::Proved) {
      if (!proof.proof) {
        resolvedProofCacheKeys.erase(key);
        return;
      }
      pair.proofHash = proof.proof->getStableHash();
      pair.proofCertificate = proof.proof->toJSON();
    }
    retirePendingProof(key);
    result.resolvedProofCache.pairs.push_back(std::move(pair));
  };

  if (options.resolvedProofCache) {
    for (const OfflineSemanticProofPair &cached :
         options.resolvedProofCache->pairs) {
      if (cached.status == SemanticProofStatus::Unknown ||
          cached.lhsGraphId >= graphs.size() ||
          cached.rhsGraphId >= graphs.size() ||
          graphs[cached.lhsGraphId].canonicalKey != cached.lhsGraphKey ||
          graphs[cached.rhsGraphId].canonicalKey != cached.rhsGraphKey) {
        ++result.resolvedProofCacheRejected;
        continue;
      }
      const uint64_t key = pairKey(cached.lhsGraphId, cached.rhsGraphId);
      if (!resolvedProofCacheKeys.insert(key).second)
        continue;
      if (cached.status == SemanticProofStatus::Refuted) {
        loadedResolvedProofCacheKeys.insert(key);
        symbolicProbeCache.try_emplace(key, SemanticProofStatus::Refuted);
        result.resolvedProofCache.pairs.push_back(cached);
        continue;
      }

      std::string decodeError;
      MLIRContext &context =
          *graphs[cached.lhsGraphId].resultType.getContext();
      std::optional<SemanticEquivalenceProof> decoded =
          SemanticEquivalenceProof::fromJSON(cached.getProofCertificate(),
                                             context, decodeError);
      if (!decoded || decoded->getStableHash() != cached.proofHash) {
        resolvedProofCacheKeys.erase(key);
        ++result.resolvedProofCacheRejected;
        continue;
      }
      SemanticProofResult proof;
      proof.status = SemanticProofStatus::Proved;
      proof.reason = "reused resolved proof cache certificate";
      proof.proof = std::move(*decoded);
      loadedResolvedProofCacheKeys.insert(key);
      exactProbeCache.try_emplace(key, std::move(proof));
      result.resolvedProofCache.pairs.push_back(cached);
    }
    for (const OfflinePendingProofPair &cached :
         options.resolvedProofCache->pendingPairs) {
      const uint64_t key = pairKey(cached.lhsGraphId, cached.rhsGraphId);
      if (cached.lhsGraphId >= graphs.size() ||
          cached.rhsGraphId >= graphs.size() ||
          cached.lhsGraphId >= fingerprints.denotations.size() ||
          cached.rhsGraphId >= fingerprints.denotations.size() ||
          !fingerprints.denotations[cached.lhsGraphId] ||
          !fingerprints.denotations[cached.rhsGraphId] ||
          graphs[cached.lhsGraphId].canonicalKey != cached.lhsGraphKey ||
          graphs[cached.rhsGraphId].canonicalKey != cached.rhsGraphKey ||
          resolvedProofCacheKeys.contains(key) ||
          pendingProofCacheIndices.contains(key)) {
        ++result.pendingProofCacheRejected;
        continue;
      }
      pendingProofCacheIndices.try_emplace(
          key, result.resolvedProofCache.pendingPairs.size());
      result.resolvedProofCache.pendingPairs.push_back(cached);
      OfflinePendingProofPair &pending =
          result.resolvedProofCache.pendingPairs.back();
      if (pending.lhsSemanticKey.empty())
        pending.lhsSemanticKey = *getStableDenotationKey(pending.lhsGraphId);
      if (pending.rhsSemanticKey.empty())
        pending.rhsSemanticKey = *getStableDenotationKey(pending.rhsGraphId);
      ++result.pendingProofCacheRemaining;
    }
  }

  // Retry the persisted Unknown frontier before searching for new pairs. The
  // entries are only scheduling hints: every one is submitted to the current
  // exact oracle again, and only a fresh Proved/Refuted result becomes cached
  // evidence. Batching preserves cooperative wall/RSS cancellation.
  if (!pendingProofCacheIndices.empty()) {
    SmallVector<std::pair<uint64_t, unsigned>, 0> frontier;
    frontier.reserve(pendingProofCacheIndices.size());
    for (const auto &entry : pendingProofCacheIndices)
      frontier.push_back({entry.first, entry.second});
    llvm::sort(frontier, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    constexpr unsigned kPendingProofBatchSize = 8192;
    for (unsigned begin = 0; begin < frontier.size();
         begin += kPendingProofBatchSize) {
      const unsigned count = std::min<unsigned>(
          kPendingProofBatchSize, frontier.size() - begin);
      std::vector<SemanticProofResult> outcomes(count);
      parallelForWork(count, result.counterexampleJobs,
                      [&](unsigned localIndex) {
                        const OfflinePendingProofPair &pending =
                            result.resolvedProofCache.pendingPairs[
                                frontier[begin + localIndex].second];
                        outcomes[localIndex] = compareDenotations(
                            *fingerprints.denotations[pending.lhsGraphId],
                            *fingerprints.denotations[pending.rhsGraphId]);
                      });
      result.oracleComparisons += count;
      result.pendingProofCacheAttempts += count;
      for (unsigned localIndex = 0; localIndex < count; ++localIndex) {
        const uint64_t key = frontier[begin + localIndex].first;
        const unsigned pendingIndex = frontier[begin + localIndex].second;
        OfflinePendingProofPair &pending =
            result.resolvedProofCache.pendingPairs[pendingIndex];
        SemanticProofResult &outcome = outcomes[localIndex];
        if (outcome.status == SemanticProofStatus::Unknown) {
          ++pending.attemptCount;
          pending.lastReason = outcome.reason;
          continue;
        }
        const unsigned lhsGraphId = pending.lhsGraphId;
        const unsigned rhsGraphId = pending.rhsGraphId;
        rememberResolvedProof(lhsGraphId, rhsGraphId, outcome);
        if (outcome.status == SemanticProofStatus::Proved && outcome.proof)
          exactProbeCache.try_emplace(key, std::move(outcome));
        else if (outcome.status == SemanticProofStatus::Refuted)
          symbolicProbeCache.try_emplace(key, SemanticProofStatus::Refuted);
      }
      if (options.shouldStop && options.shouldStop()) {
        result.interrupted = true;
        return result;
      }
    }
  }
  auto getCounterexampleFingerprint =
      [&](unsigned graphId) -> const CachedCounterexampleFingerprint & {
    CachedCounterexampleFingerprint &cached =
        counterexampleFingerprints[graphId];
    if (!cached.computed) {
      cached.computed = true;
      cached.signature =
          OfflineSemanticFingerprinter::fingerprintDenotationRange(
              *fingerprints.denotations[graphId], fingerprints.modelCount,
              options.additionalCounterexampleModels, cached.reason);
      result.counterexampleModelEvaluations +=
          options.additionalCounterexampleModels;
    }
    return cached;
  };
  auto evaluatePair = [&](unsigned lhsGraphId, unsigned rhsGraphId,
                          bool replaySerializedCertificate = true)
      -> PairEvaluation {
    PairEvaluation evaluation;
    OfflineSemanticProofPair &pair = evaluation.pair;
    pair.lhsGraphId = lhsGraphId;
    pair.rhsGraphId = rhsGraphId;
    if (pair.lhsGraphId >= fingerprints.denotations.size() ||
        pair.rhsGraphId >= fingerprints.denotations.size() ||
        !fingerprints.denotations[pair.lhsGraphId] ||
        !fingerprints.denotations[pair.rhsGraphId]) {
      pair.reason = "denotation unavailable after fingerprinting";
      pair.status = SemanticProofStatus::Unknown;
      return evaluation;
    }

    bool sameExactDenotation =
        pair.lhsGraphId < fingerprints.exactDenotationGroupIds.size() &&
        pair.rhsGraphId < fingerprints.exactDenotationGroupIds.size() &&
        fingerprints.exactDenotationGroupIds[pair.lhsGraphId] ==
            fingerprints.exactDenotationGroupIds[pair.rhsGraphId];
    if (!sameExactDenotation &&
        options.additionalCounterexampleModels != 0) {
      const CachedCounterexampleFingerprint &lhsFingerprint =
          getCounterexampleFingerprint(pair.lhsGraphId);
      const CachedCounterexampleFingerprint &rhsFingerprint =
          getCounterexampleFingerprint(pair.rhsGraphId);
      if (lhsFingerprint.signature && rhsFingerprint.signature &&
          *lhsFingerprint.signature != *rhsFingerprint.signature) {
        pair.status = SemanticProofStatus::Refuted;
        pair.reason =
            (llvm::Twine("concrete semantic counterexample in model range [") +
             llvm::Twine(fingerprints.modelCount) + ", " +
             llvm::Twine(fingerprints.modelCount +
                         options.additionalCounterexampleModels) +
             ")")
                .str();
        evaluation.counterexampleRefuted = true;
        return evaluation;
      }
    }

    const uint64_t key = pairKey(pair.lhsGraphId, pair.rhsGraphId);
    auto cachedExact = exactProbeCache.find(key);
    SemanticProofResult proof;
    if (cachedExact != exactProbeCache.end())
      proof = cachedExact->second;

    auto cachedProbe = symbolicProbeCache.find(key);
    if (cachedExact == exactProbeCache.end() &&
        cachedProbe != symbolicProbeCache.end() &&
        cachedProbe->second != SemanticProofStatus::Proved) {
      pair.status = cachedProbe->second;
      pair.reason = pair.status == SemanticProofStatus::Refuted
                        ? "reused symbolic premerge refutation"
                        : "reused symbolic premerge unknown result";
      if (pair.status == SemanticProofStatus::Unknown) {
        pair.sharedLhsSemanticKey = getStableDenotationKey(pair.lhsGraphId);
        pair.sharedRhsSemanticKey = getStableDenotationKey(pair.rhsGraphId);
      }
      return evaluation;
    }

    if (cachedExact == exactProbeCache.end()) {
      ++result.oracleComparisons;
      proof = compareDenotations(
          *fingerprints.denotations[pair.lhsGraphId],
          *fingerprints.denotations[pair.rhsGraphId]);
    }
    pair.status = proof.status;
    pair.reason = proof.reason;
    // Remember deterministic negative/unknown answers so multiplicity
    // accounting can reuse them. A proved exact-probe result already carries
    // its certificate and reaches the serialization/replay path below without
    // invoking the backend a second time.
    if (pair.status != SemanticProofStatus::Proved)
      symbolicProbeCache.try_emplace(
          pairKey(pair.lhsGraphId, pair.rhsGraphId), pair.status);
    if (pair.status == SemanticProofStatus::Proved && proof.proof) {
      pair.proofHash = proof.proof->getStableHash();
      pair.proofCertificate = proof.proof->toJSON();
      if (replaySerializedCertificate) {
        std::string decodeError;
        MLIRContext &context = *graphs[pair.lhsGraphId].resultType.getContext();
        std::optional<SemanticEquivalenceProof> decoded =
            SemanticEquivalenceProof::fromJSON(pair.proofCertificate, context,
                                               decodeError);
        SemanticProofResult replay =
            decoded ? oracle.replay(*decoded) : SemanticProofResult{};
        if (!replay.isProved() || !replay.proof ||
            replay.proof->getStableHash() != pair.proofHash) {
          pair.status = SemanticProofStatus::Unknown;
          pair.proofHash.clear();
          pair.proofCertificate.clear();
          pair.reason =
              decodeError.empty()
                  ? "deserialized proof replay failed"
                  : "proof certificate decode failed: " + decodeError;
        }
      }
    }
    if (pair.status == SemanticProofStatus::Unknown) {
      pair.sharedLhsSemanticKey = getStableDenotationKey(pair.lhsGraphId);
      pair.sharedRhsSemanticKey = getStableDenotationKey(pair.rhsGraphId);
      if (cachedExact == exactProbeCache.end())
        rememberPendingProof(pair.lhsGraphId, pair.rhsGraphId, pair.reason);
    } else if (cachedExact == exactProbeCache.end()) {
      rememberResolvedProof(pair.lhsGraphId, pair.rhsGraphId, proof);
    }
    return evaluation;
  };

  // Production output keeps a replayable proof forest and omits individual
  // refuted pairs. In that mode, quotient each fingerprint bucket by the exact
  // normalized denotation before invoking the oracle. This is generic over the
  // semantic language: equal stable denotation keys are still connected by
  // replayed proof certificates, while different keys are compared through
  // one representative of each discovered equivalence class. Logical pair
  // counts remain exact without materializing an O(N^2) pair set.
  bool canUseSemanticQuotient =
      options.pruneTransitivelyImpliedPairs && !options.retainRefutedPairs &&
      fingerprints.wildcardGraphIds.empty();
  if (canUseSemanticQuotient)
    for (const OfflineSemanticFingerprintBucket &bucket : fingerprints.buckets)
      for (unsigned graphId : bucket.graphIds)
        if (graphId >= fingerprints.denotations.size() ||
            !fingerprints.denotations[graphId]) {
          canUseSemanticQuotient = false;
          break;
        }

  if (canUseSemanticQuotient &&
      options.additionalCounterexampleModels != 0) {
    bool hasGlobalExactGroupIds =
        fingerprints.exactDenotationGroupIds.size() ==
        fingerprints.denotations.size();
    if (hasGlobalExactGroupIds)
      for (const OfflineSemanticFingerprintBucket &bucket :
           fingerprints.buckets) {
        if (bucket.graphIds.size() < 2)
          continue;
        for (unsigned graphId : bucket.graphIds)
          if (fingerprints.exactDenotationGroupIds[graphId] ==
              std::numeric_limits<unsigned>::max()) {
            hasGlobalExactGroupIds = false;
            break;
          }
        if (!hasGlobalExactGroupIds)
          break;
      }

    struct AdaptiveCounterexampleBucket {
      std::string familyFingerprint;
      std::string signature;
      SmallVector<unsigned, 8> representatives;
    };
    SmallVector<AdaptiveCounterexampleBucket, 64> active;
    llvm::DenseMap<unsigned, SmallVector<unsigned, 4>> premergedMembers;
    struct PremergedClass {
      unsigned representative = 0;
      SmallVector<unsigned, 4> members;
    };
    struct BucketPremergeWork {
      std::string fingerprint;
      SmallVector<unsigned, 0> remaining;
      SmallVector<PremergedClass, 8> classes;
      bool hasUnknownRelation = false;
    };
    SmallVector<BucketPremergeWork, 64> premergeWork;
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets) {
      if (bucket.graphIds.size() < 2)
        continue;

      llvm::DenseSet<unsigned> seen;
      SmallVector<unsigned, 0> exactRepresentatives;
      for (unsigned graphId : bucket.graphIds) {
        unsigned identity = hasGlobalExactGroupIds
                                ? fingerprints.exactDenotationGroupIds[graphId]
                                : graphId;
        if (seen.insert(identity).second)
          exactRepresentatives.push_back(graphId);
      }

      // One exact denotation group needs no negative filter: every member is
      // connected by an exact-normal-form certificate below. For a real
      // collision, refine only the sub-buckets that still contain two or more
      // distinct exact groups. This is the same generic adaptive partition as
      // the primary fingerprinter, continued over the extra model range.
      if (exactRepresentatives.size() < 2) {
        if (!exactRepresentatives.empty()) {
          CachedCounterexampleFingerprint &cached =
              counterexampleFingerprints[exactRepresentatives.front()];
          cached.computed = true;
          cached.signature = bucket.fingerprint + ":only";
        }
        continue;
      }

      premergeWork.push_back(
          {bucket.fingerprint, std::move(exactRepresentatives), {}, false});
    }

    // Concrete models are poor at separating expressions that the cheap
    // normalizer can already prove equal. Before spending up to 96 more
    // evaluations on each exact normal form, probe at most eight class pivots
    // per bucket with that primary procedure only. Expensive exact backends
    // (for example Z3) are deliberately deferred until after concrete
    // fingerprint refinement; invoking them here makes millions of candidates
    // consume their timeout before the cheap counterexample filter can split
    // the bucket. Final e-class construction below still proves every retained
    // relation with the complete oracle and replays its serialized certificate.
    constexpr unsigned kPremergeClassProbeLimit = 8;
    struct SymbolicProbe {
      unsigned bucketIndex = 0;
      unsigned graphId = 0;
      unsigned pivotGraphId = 0;
    };
    for (unsigned round = 0; round < kPremergeClassProbeLimit; ++round) {
      SmallVector<SymbolicProbe, 0> probes;
      std::vector<unsigned> newClassIndices(
          premergeWork.size(), std::numeric_limits<unsigned>::max());
      for (auto [bucketIndex, work] : llvm::enumerate(premergeWork)) {
        if (work.remaining.empty())
          continue;
        const unsigned pivot = work.remaining.front();
        newClassIndices[bucketIndex] = work.classes.size();
        work.classes.push_back({pivot, {pivot}});
        for (unsigned index = 1; index < work.remaining.size(); ++index)
          probes.push_back({static_cast<unsigned>(bucketIndex),
                            work.remaining[index], pivot});
      }
      if (probes.empty()) {
        for (BucketPremergeWork &work : premergeWork)
          work.remaining.clear();
        break;
      }

      std::vector<SemanticProofStatus> statuses(
          probes.size(), SemanticProofStatus::Unknown);
      parallelForWork(probes.size(), result.counterexampleJobs,
                      [&](unsigned index) {
                        const SymbolicProbe &probe = probes[index];
                        const TensorDenotation &lhs =
                            *fingerprints.denotations[probe.graphId];
                        const TensorDenotation &rhs =
                            *fingerprints.denotations[probe.pivotGraphId];
                        statuses[index] =
                            comparePrimaryDenotations(lhs, rhs).status;
                      });
      result.symbolicPremergeComparisons += probes.size();

      std::vector<SmallVector<unsigned, 0>> survivors(premergeWork.size());
      for (auto [index, probe] : llvm::enumerate(probes)) {
        if (statuses[index] == SemanticProofStatus::Proved) {
          premergeWork[probe.bucketIndex]
              .classes[newClassIndices[probe.bucketIndex]]
              .members.push_back(probe.graphId);
          ++result.symbolicPremergeMergedGroups;
        } else {
          // A primary refutation is final. A primary Unknown is not cached:
          // the complete oracle must get a chance to ask its exact fallback
          // after concrete refinement has reduced the search space.
          if (statuses[index] == SemanticProofStatus::Refuted)
            symbolicProbeCache.try_emplace(
                pairKey(probe.graphId, probe.pivotGraphId), statuses[index]);
          survivors[probe.bucketIndex].push_back(probe.graphId);
          if (statuses[index] == SemanticProofStatus::Unknown)
            premergeWork[probe.bucketIndex].hasUnknownRelation = true;
        }
      }
      for (auto [bucketIndex, work] : llvm::enumerate(premergeWork))
        work.remaining = std::move(survivors[bucketIndex]);
      if (options.shouldStop && options.shouldStop()) {
        result.interrupted = true;
        return result;
      }
    }

    // More than eight pairwise-distinct classes are left as independent
    // representatives. The concrete partition can still refute them; no
    // equality is inferred from reaching the symbolic search budget.
    uint64_t eagerModelEvaluations = 0;
    for (BucketPremergeWork &work : premergeWork) {
      if (work.remaining.size() > 1)
        work.hasUnknownRelation = true;
      for (unsigned graphId : work.remaining)
        work.classes.push_back({graphId, {graphId}});

      SmallVector<unsigned, 0> representatives;
      representatives.reserve(work.classes.size());
      for (PremergedClass &premerged : work.classes) {
        representatives.push_back(premerged.representative);
        premergedMembers.try_emplace(premerged.representative,
                                     std::move(premerged.members));
      }
      if (representatives.size() < 2) {
        for (unsigned graphId :
             premergedMembers.find(representatives.front())->second) {
          CachedCounterexampleFingerprint &cached =
              counterexampleFingerprints[graphId];
          cached.computed = true;
          cached.signature = work.fingerprint + ":symbolic-only";
        }
        continue;
      }
      if (!work.hasUnknownRelation) {
        // Every cross-class relation in this bucket was symbolically refuted.
        // Mark the cache as computed-without-a-concrete-signature so the final
        // classification repeats the exact oracle comparisons instead of
        // spending concrete models or misreporting a symbolic refutation as a
        // sampled counterexample.
        for (unsigned representative : representatives)
          for (unsigned graphId :
               premergedMembers.find(representative)->second)
            counterexampleFingerprints[graphId].computed = true;
        continue;
      }
      eagerModelEvaluations +=
          static_cast<uint64_t>(representatives.size()) *
          options.additionalCounterexampleModels;
      active.push_back({work.fingerprint, work.fingerprint + ":0",
                        std::move(representatives)});
    }

    auto completeBucket = [&](AdaptiveCounterexampleBucket &completed) {
      for (unsigned graphId : completed.representatives) {
        auto members = premergedMembers.find(graphId);
        if (members == premergedMembers.end()) {
          CachedCounterexampleFingerprint &cached =
              counterexampleFingerprints[graphId];
          cached.computed = true;
          cached.signature = completed.signature;
          continue;
        }
        for (unsigned member : members->second) {
          CachedCounterexampleFingerprint &cached =
              counterexampleFingerprints[member];
          cached.computed = true;
          cached.signature = completed.signature;
        }
      }
    };

    // Continue every primary collision bucket together. One flattened batch
    // per model keeps all requested workers busy even after refinement has
    // produced many two- or three-member buckets, and it avoids creating a
    // fresh thread team for every small bucket.
    uint64_t nextPartitionId = 0;
    for (unsigned modelOffset = 0;
         modelOffset < options.additionalCounterexampleModels &&
         !active.empty();
         ++modelOffset) {
      SmallVector<AdaptiveCounterexampleBucket, 64> next;
      SmallVector<unsigned, 0> evaluationGraphIds;
      SmallVector<unsigned, 0> evaluationOffsets;
      evaluationOffsets.reserve(active.size() + 1);
      evaluationOffsets.push_back(0);
      for (AdaptiveCounterexampleBucket &collision : active) {
        if (collision.representatives.size() == 1) {
          completeBucket(collision);
          evaluationOffsets.push_back(evaluationGraphIds.size());
          continue;
        }
        evaluationGraphIds.append(collision.representatives.begin(),
                                  collision.representatives.end());
        evaluationOffsets.push_back(evaluationGraphIds.size());
      }

      struct ModelEvaluation {
        std::optional<std::string> fingerprint;
        std::string reason;
      };
      std::vector<ModelEvaluation> evaluations(evaluationGraphIds.size());
      parallelForWork(
          evaluationGraphIds.size(), result.counterexampleJobs,
          [&](unsigned index) {
            const unsigned graphId = evaluationGraphIds[index];
            evaluations[index].fingerprint =
                OfflineSemanticFingerprinter::fingerprintDenotationRange(
                    *fingerprints.denotations[graphId],
                    fingerprints.modelCount + modelOffset,
                    /*modelCount=*/1, evaluations[index].reason);
          });
      result.counterexampleModelEvaluations += evaluationGraphIds.size();

      for (auto [collisionIndex, collision] : llvm::enumerate(active)) {
        const unsigned begin = evaluationOffsets[collisionIndex];
        const unsigned end = evaluationOffsets[collisionIndex + 1];
        if (begin == end)
          continue;
        llvm::StringMap<SmallVector<unsigned, 8>> partitions;
        for (unsigned index = begin; index < end; ++index) {
          const unsigned graphId = evaluationGraphIds[index];
          ModelEvaluation &evaluation = evaluations[index];
          if (!evaluation.fingerprint) {
            auto members = premergedMembers.find(graphId);
            if (members == premergedMembers.end()) {
              CachedCounterexampleFingerprint &cached =
                  counterexampleFingerprints[graphId];
              cached.computed = true;
              cached.reason = evaluation.reason;
            } else {
              for (unsigned member : members->second) {
                CachedCounterexampleFingerprint &cached =
                    counterexampleFingerprints[member];
                cached.computed = true;
                cached.reason = evaluation.reason;
              }
            }
            continue;
          }
          partitions[*evaluation.fingerprint].push_back(graphId);
        }
        for (auto &partition : partitions) {
          AdaptiveCounterexampleBucket child;
          child.familyFingerprint = collision.familyFingerprint;
          // The label is an opaque partition identity, not a proof hash. A
          // bounded stable label avoids retaining a model-history string whose
          // length grows with every refinement round.
          child.signature =
              (llvm::Twine(child.familyFingerprint) + ":" +
               llvm::Twine(++nextPartitionId))
                  .str();
          child.representatives = std::move(partition.second);
          next.push_back(std::move(child));
        }
      }
      active = std::move(next);
      if (options.shouldStop && options.shouldStop()) {
        result.interrupted = true;
        return result;
      }
    }
    // Concrete signatures have now removed cheap counterexamples. Probe the
    // remaining relations with an optional exact backend in parallel, cache
    // every answer, and let final proof-forest construction consume those
    // answers without serializing millions of SMT calls. Bounded batches keep
    // the wall/RSS gate responsive even when one signature bucket is large.
    if (oracle.hasFallbackBackend() && !active.empty()) {
      struct ExactProbe {
        unsigned bucketIndex = 0;
        unsigned graphId = 0;
        unsigned pivotGraphId = 0;
      };
      std::vector<SmallVector<unsigned, 0>> remaining(active.size());
      for (auto [bucketIndex, collision] : llvm::enumerate(active))
        remaining[bucketIndex].append(collision.representatives.begin(),
                                      collision.representatives.end());

      constexpr unsigned kExactProbeClassLimit = 8;
      constexpr unsigned kExactProbeBatchSize = 8192;
      for (unsigned round = 0; round < kExactProbeClassLimit; ++round) {
        SmallVector<ExactProbe, 0> probes;
        for (auto [bucketIndex, representatives] :
             llvm::enumerate(remaining)) {
          if (representatives.empty())
            continue;
          const unsigned pivot = representatives.front();
          for (unsigned index = 1; index < representatives.size(); ++index)
            probes.push_back({static_cast<unsigned>(bucketIndex),
                              representatives[index], pivot});
        }
        if (probes.empty())
          break;

        std::vector<SmallVector<unsigned, 0>> survivors(active.size());
        for (unsigned begin = 0; begin < probes.size();
             begin += kExactProbeBatchSize) {
          const unsigned count =
              std::min<unsigned>(kExactProbeBatchSize, probes.size() - begin);
          std::vector<SemanticProofResult> outcomes(count);
          std::vector<uint8_t> cacheHits(count, 0);
          parallelForWork(count, result.counterexampleJobs,
                          [&](unsigned localIndex) {
                            const ExactProbe &probe = probes[begin + localIndex];
                            const uint64_t key =
                                pairKey(probe.graphId, probe.pivotGraphId);
                            auto cachedExact = exactProbeCache.find(key);
                            if (cachedExact != exactProbeCache.end()) {
                              outcomes[localIndex] = cachedExact->second;
                              cacheHits[localIndex] =
                                  loadedResolvedProofCacheKeys.contains(key)
                                      ? 1
                                      : 2;
                              return;
                            }
                            auto cachedStatus = symbolicProbeCache.find(key);
                            if (cachedStatus != symbolicProbeCache.end() &&
                                cachedStatus->second ==
                                    SemanticProofStatus::Refuted) {
                              outcomes[localIndex].status =
                                  SemanticProofStatus::Refuted;
                              outcomes[localIndex].reason =
                                  "reused cached exact refutation";
                              cacheHits[localIndex] =
                                  loadedResolvedProofCacheKeys.contains(key)
                                      ? 1
                                      : 2;
                              return;
                            }
                            outcomes[localIndex] = compareDenotations(
                                *fingerprints.denotations[probe.graphId],
                                *fingerprints.denotations[probe.pivotGraphId]);
                          });

          result.oracleComparisons += std::count(
              cacheHits.begin(), cacheHits.end(), static_cast<uint8_t>(0));
          result.exactPostRefinementComparisons += count;
          for (unsigned localIndex = 0; localIndex < count; ++localIndex) {
            const ExactProbe &probe = probes[begin + localIndex];
            SemanticProofResult &outcome = outcomes[localIndex];
            const uint64_t key = pairKey(probe.graphId, probe.pivotGraphId);
            if (cacheHits[localIndex] == 1)
              ++result.resolvedProofCacheHits;
            if (cacheHits[localIndex] == 0)
              rememberResolvedProof(probe.graphId, probe.pivotGraphId,
                                    outcome);
            if (cacheHits[localIndex] == 0 &&
                outcome.status == SemanticProofStatus::Unknown)
              rememberPendingProof(probe.graphId, probe.pivotGraphId,
                                   outcome.reason);
            if (outcome.status == SemanticProofStatus::Proved &&
                outcome.proof) {
              if (cacheHits[localIndex] == 0)
                exactProbeCache.try_emplace(key, std::move(outcome));
              ++result.exactPostRefinementProved;
              continue;
            }
            if (cacheHits[localIndex] == 0 &&
                outcome.status != SemanticProofStatus::Proved)
              symbolicProbeCache.try_emplace(key, outcome.status);
            survivors[probe.bucketIndex].push_back(probe.graphId);
          }

          if (options.shouldStop && options.shouldStop()) {
            result.interrupted = true;
            return result;
          }
        }
        remaining = std::move(survivors);
      }
    }

    for (AdaptiveCounterexampleBucket &collision : active)
      completeBucket(collision);
    result.counterexampleModelEvaluationsAvoided =
        eagerModelEvaluations - result.counterexampleModelEvaluations;
  }

  if (canUseSemanticQuotient) {
    result.usedSemanticQuotient = true;
    result.candidatePairs = fingerprints.proofCandidatePairsAfter;
    for (const OfflineSemanticFingerprintBucket &bucket :
         fingerprints.buckets) {
      if (bucket.graphIds.size() < 2)
        continue;

      struct SemanticGroup {
        unsigned stableGroupId = std::numeric_limits<unsigned>::max();
        std::string key;
        std::vector<unsigned> graphIds;
      };
      std::vector<SemanticGroup> groups;
      bool hasExactGroupIds =
          fingerprints.exactDenotationGroupIds.size() ==
          fingerprints.denotations.size();
      if (hasExactGroupIds)
        for (unsigned graphId : bucket.graphIds)
          if (graphId >= fingerprints.exactDenotationGroupIds.size() ||
              fingerprints.exactDenotationGroupIds[graphId] ==
                  std::numeric_limits<unsigned>::max()) {
            hasExactGroupIds = false;
            break;
          }
      if (hasExactGroupIds) {
        llvm::DenseMap<unsigned, unsigned> localGroupIndices;
        groups.reserve(bucket.graphIds.size());
        for (unsigned graphId : bucket.graphIds) {
          const unsigned stableGroupId =
              fingerprints.exactDenotationGroupIds[graphId];
          auto [entry, inserted] =
              localGroupIndices.try_emplace(stableGroupId, groups.size());
          if (inserted)
            groups.push_back({stableGroupId, {}, {}});
          groups[entry->second].graphIds.push_back(graphId);
        }
      } else {
        // Compatibility path for focused tests and external callers that
        // construct fingerprint results directly instead of using the
        // production fingerprinter.
        llvm::StringMap<std::vector<unsigned>> graphsByDenotation;
        for (unsigned graphId : bucket.graphIds)
          graphsByDenotation[*getStableDenotationKey(graphId)]
              .push_back(graphId);
        groups.reserve(graphsByDenotation.size());
        for (auto &entry : graphsByDenotation)
          groups.push_back(
              {std::numeric_limits<unsigned>::max(), entry.getKey().str(),
               std::move(entry.getValue())});
      }
      llvm::sort(groups, [](const SemanticGroup &lhs,
                            const SemanticGroup &rhs) {
        if (lhs.stableGroupId != std::numeric_limits<unsigned>::max() &&
            rhs.stableGroupId != std::numeric_limits<unsigned>::max())
          return lhs.stableGroupId < rhs.stableGroupId;
        return lhs.key < rhs.key;
      });
      result.semanticQuotientGroups += groups.size();

      std::vector<unsigned> parent(groups.size());
      for (unsigned index = 0; index < parent.size(); ++index)
        parent[index] = index;
      auto findRoot = [&](unsigned id) {
        unsigned root = id;
        while (parent[root] != root)
          root = parent[root];
        while (parent[id] != id) {
          unsigned next = parent[id];
          parent[id] = root;
          id = next;
        }
        return root;
      };

      // A concrete signature mismatch is already a replayable counterexample
      // for every graph represented by the two semantic groups.  Partition
      // the quadratic symbolic search by that signature instead of merely
      // checking the signature after visiting every pair.  A missing
      // signature remains conservative and is compared with every earlier
      // group; equal signatures still require the symbolic oracle below.
      std::vector<std::optional<std::string>> groupCounterexampleSignatures(
          groups.size());
      if (options.additionalCounterexampleModels != 0)
        for (unsigned groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
          const CachedCounterexampleFingerprint &fingerprint =
              getCounterexampleFingerprint(groups[groupIndex].graphIds.front());
          if (fingerprint.signature)
            groupCounterexampleSignatures[groupIndex] = *fingerprint.signature;
        }
      llvm::StringMap<SmallVector<unsigned, 8>> priorGroupsBySignature;
      SmallVector<unsigned, 8> priorGroupsWithoutSignature;

      const uint64_t provedBeforeBucket = result.provedPairs;
      for (unsigned groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        SemanticGroup &group = groups[groupIndex];
        const unsigned representative = group.graphIds.front();

        // The stable key is an exact normalized denotation key. Retain a
        // certificate edge from its representative to every member so the
        // persistent e-class store can replay the complete spanning forest.
        if (group.graphIds.size() > 1) {
          if (shouldStop())
            return result;
          // Fingerprinting formed this group only after an exact structural
          // comparison of normalized denotations. The oracle still constructs
          // the replayable certificate, but parsing and replaying that same
          // freshly serialized certificate here adds no proof power. Generic
          // cross-group equalities retain the full serialization replay below.
          PairEvaluation evidence = evaluatePair(
              representative, group.graphIds[1],
              /*replaySerializedCertificate=*/false);
          if (evidence.pair.status != SemanticProofStatus::Proved) {
            // Equal normalized keys must be reflexively provable. Fail closed
            // if proof construction or replay ever violates that invariant.
            evidence.pair.status = SemanticProofStatus::Unknown;
            if (evidence.pair.reason.empty())
              evidence.pair.reason =
                  "equal normalized denotations were not provably equal";
            retainGraphKeys(evidence.pair);
            result.pairs.push_back(std::move(evidence.pair));
            ++result.unknownPairs;
            return result;
          }
          auto sharedCertificate = std::make_shared<const std::string>(
              std::move(evidence.pair.proofCertificate));
          for (unsigned memberIndex = 1; memberIndex < group.graphIds.size();
               ++memberIndex) {
            if (shouldStop())
              return result;
            OfflineSemanticProofPair pair;
            pair.lhsGraphId = representative;
            pair.rhsGraphId = group.graphIds[memberIndex];
            pair.status = SemanticProofStatus::Proved;
            pair.proofHash = evidence.pair.proofHash;
            pair.sharedProofCertificate = sharedCertificate;
            pair.reason = evidence.pair.reason;
            retainGraphKeys(pair);
            result.pairs.push_back(std::move(pair));
            ++result.provedPairs;
          }
        }

        // A new semantic group may bridge several previously distinct proof
        // classes. Compare it with every current representative and keep one
        // replayable edge for each class that it joins.
        SmallVector<unsigned, 16> previousCandidates;
        if (groupCounterexampleSignatures[groupIndex]) {
          auto sameSignature = priorGroupsBySignature.find(
              *groupCounterexampleSignatures[groupIndex]);
          if (sameSignature != priorGroupsBySignature.end())
            previousCandidates.append(sameSignature->second.begin(),
                                      sameSignature->second.end());
          previousCandidates.append(priorGroupsWithoutSignature.begin(),
                                    priorGroupsWithoutSignature.end());
        } else {
          previousCandidates.reserve(groupIndex);
          for (unsigned previous = 0; previous < groupIndex; ++previous)
            previousCandidates.push_back(previous);
        }
        llvm::DenseSet<unsigned> comparedRoots;
        for (unsigned previous : previousCandidates) {
          unsigned previousRoot = findRoot(previous);
          unsigned currentRoot = findRoot(groupIndex);
          if (previousRoot == currentRoot ||
              !comparedRoots.insert(previousRoot).second)
            continue;
          if (shouldStop())
            return result;
          PairEvaluation evaluation = evaluatePair(
              representative, groups[previousRoot].graphIds.front());
          if (evaluation.pair.status != SemanticProofStatus::Proved)
            continue;
          retainGraphKeys(evaluation.pair);
          result.pairs.push_back(std::move(evaluation.pair));
          ++result.provedPairs;
          currentRoot = findRoot(groupIndex);
          previousRoot = findRoot(previousRoot);
          if (currentRoot != previousRoot)
            parent[previousRoot] = currentRoot;
        }
        if (groupCounterexampleSignatures[groupIndex])
          priorGroupsBySignature[*groupCounterexampleSignatures[groupIndex]]
              .push_back(groupIndex);
        else
          priorGroupsWithoutSignature.push_back(groupIndex);
      }

      // The incremental pass above compares a new group with one current
      // representative per reachable proof class.  A proof discovered late
      // in the bucket can change those representatives and expose another
      // equality that was not in an earlier group's candidate frontier.  Run
      // a generic fixed-point closure before multiplicity accounting so the
      // final class-class classifier can never discover a missing equality.
      // Concrete signatures remain a sound negative partition; pairs with a
      // shared (or unavailable) signature still go through the exact oracle.
      constexpr unsigned kMaximumTransientClosureRetries = 3;
      unsigned transientClosureRetries = 0;
      bool continueProofClassClosure = true;
      while (continueProofClassClosure) {
        bool mergedProofClasses = false;
        bool sawUnknownProofClass = false;
        SmallVector<unsigned, 16> closureRoots;
        std::vector<unsigned> closureRepresentatives(groups.size(), 0);
        std::vector<bool> hasClosureRepresentative(groups.size(), false);
        for (unsigned groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
          unsigned root = findRoot(groupIndex);
          if (!hasClosureRepresentative[root]) {
            closureRepresentatives[root] =
                groups[groupIndex].graphIds.front();
            hasClosureRepresentative[root] = true;
          }
        }
        for (unsigned root = 0; root < groups.size(); ++root)
          if (hasClosureRepresentative[root])
            closureRoots.push_back(root);

        for (size_t lhsIndex = 0; lhsIndex < closureRoots.size(); ++lhsIndex)
          for (size_t rhsIndex = lhsIndex + 1;
               rhsIndex < closureRoots.size(); ++rhsIndex) {
            unsigned lhsRoot = findRoot(closureRoots[lhsIndex]);
            unsigned rhsRoot = findRoot(closureRoots[rhsIndex]);
            if (lhsRoot == rhsRoot)
              continue;
            if (shouldStop())
              return result;

            const unsigned lhsRepresentative =
                closureRepresentatives[lhsRoot];
            const unsigned rhsRepresentative =
                closureRepresentatives[rhsRoot];
            if (options.additionalCounterexampleModels != 0) {
              const CachedCounterexampleFingerprint &lhsFingerprint =
                  getCounterexampleFingerprint(lhsRepresentative);
              const CachedCounterexampleFingerprint &rhsFingerprint =
                  getCounterexampleFingerprint(rhsRepresentative);
              if (lhsFingerprint.signature && rhsFingerprint.signature &&
                  *lhsFingerprint.signature != *rhsFingerprint.signature)
                continue;
            }

            PairEvaluation closure =
                evaluatePair(lhsRepresentative, rhsRepresentative);
            if (closure.pair.status == SemanticProofStatus::Unknown) {
              sawUnknownProofClass = true;
              continue;
            }
            if (closure.pair.status != SemanticProofStatus::Proved)
              continue;
            retainGraphKeys(closure.pair);
            result.pairs.push_back(std::move(closure.pair));
            ++result.provedPairs;
            lhsRoot = findRoot(lhsRoot);
            rhsRoot = findRoot(rhsRoot);
            if (lhsRoot != rhsRoot) {
              parent[rhsRoot] = lhsRoot;
              mergedProofClasses = true;
            }
          }
        if (mergedProofClasses) {
          transientClosureRetries = 0;
          continue;
        }
        continueProofClassClosure =
            sawUnknownProofClass &&
            transientClosureRetries++ < kMaximumTransientClosureRetries;
      }

      std::vector<uint64_t> classGraphCounts(groups.size(), 0);
      std::vector<unsigned> classRepresentatives(groups.size(), 0);
      std::vector<bool> hasRepresentative(groups.size(), false);
      for (unsigned groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        unsigned root = findRoot(groupIndex);
        classGraphCounts[root] += groups[groupIndex].graphIds.size();
        if (!hasRepresentative[root]) {
          classRepresentatives[root] = groups[groupIndex].graphIds.front();
          hasRepresentative[root] = true;
        }
      }
      SmallVector<unsigned, 16> classRoots;
      uint64_t logicallyProved = 0;
      for (unsigned root = 0; root < groups.size(); ++root)
        if (hasRepresentative[root]) {
          classRoots.push_back(root);
          logicallyProved += chooseTwo(classGraphCounts[root]);
        }
      result.semanticQuotientClasses += classRoots.size();
      const uint64_t directProofs = result.provedPairs - provedBeforeBucket;
      if (logicallyProved < directProofs) {
        result.interrupted = true;
        return result;
      }
      result.transitivelyProvedPairs += logicallyProved - directProofs;

      // Representatives of the final proof classes cannot be equal.  Compute
      // different-signature Cartesian products arithmetically, then invoke the
      // symbolic oracle only within one signature bucket or for a class whose
      // concrete signature could not be evaluated.  Equality of every member
      // with its representative makes each multiplied tri-state count sound.
      struct SignedClassBucket {
        std::string signature;
        SmallVector<unsigned, 8> roots;
        uint64_t graphCount = 0;
      };
      SmallVector<SignedClassBucket, 16> signedClassBuckets;
      llvm::StringMap<unsigned> signedClassBucketIndices;
      std::vector<std::optional<std::string>> classSignatures(classRoots.size());
      for (size_t classIndex = 0; classIndex < classRoots.size(); ++classIndex) {
        unsigned root = classRoots[classIndex];
        if (options.additionalCounterexampleModels != 0) {
          const CachedCounterexampleFingerprint &fingerprint =
              getCounterexampleFingerprint(classRepresentatives[root]);
          if (fingerprint.signature)
            classSignatures[classIndex] = *fingerprint.signature;
        }
        if (!classSignatures[classIndex])
          continue;
        auto [entry, inserted] = signedClassBucketIndices.try_emplace(
            *classSignatures[classIndex], signedClassBuckets.size());
        if (inserted)
          signedClassBuckets.push_back(
              {*classSignatures[classIndex], {}, 0});
        SignedClassBucket &bucket = signedClassBuckets[entry->second];
        bucket.roots.push_back(root);
        bucket.graphCount += classGraphCounts[root];
      }

      uint64_t previousSignedGraphCount = 0;
      for (const SignedClassBucket &bucket : signedClassBuckets) {
        uint64_t counterexamplePairs =
            previousSignedGraphCount * bucket.graphCount;
        result.refutedPairs += counterexamplePairs;
        result.counterexampleRefutedPairs += counterexamplePairs;
        previousSignedGraphCount += bucket.graphCount;
      }

      auto classifyClassPair = [&](unsigned lhsRoot, unsigned rhsRoot) {
        PairEvaluation evaluation = evaluatePair(
            classRepresentatives[lhsRoot], classRepresentatives[rhsRoot]);
        uint64_t multiplicity =
            classGraphCounts[lhsRoot] * classGraphCounts[rhsRoot];
        switch (evaluation.pair.status) {
        case SemanticProofStatus::Proved:
          // The class-building pass compared every pair that survived the
          // concrete partition. A proved result here means it missed a merge;
          // never publish such a partial relation as complete.
          evaluation.pair.status = SemanticProofStatus::Unknown;
          evaluation.pair.reason =
              "semantic quotient missed a provable class merge";
          [[fallthrough]];
        case SemanticProofStatus::Unknown:
          result.unknownPairs += multiplicity;
          retainGraphKeys(evaluation.pair);
          result.pairs.push_back(std::move(evaluation.pair));
          break;
        case SemanticProofStatus::Refuted:
          result.refutedPairs += multiplicity;
          if (evaluation.counterexampleRefuted)
            result.counterexampleRefutedPairs += multiplicity;
          break;
        }
      };

      for (const SignedClassBucket &bucket : signedClassBuckets)
        for (size_t lhs = 0; lhs < bucket.roots.size(); ++lhs)
          for (size_t rhs = lhs + 1; rhs < bucket.roots.size(); ++rhs) {
            if (shouldStop())
              return result;
            classifyClassPair(bucket.roots[lhs], bucket.roots[rhs]);
          }

      // A class without a concrete signature may equal any other class.  Keep
      // the conservative all-pairs behavior for it, while visiting two
      // unsigned classes only once.
      for (size_t lhs = 0; lhs < classRoots.size(); ++lhs) {
        if (classSignatures[lhs])
          continue;
        for (size_t rhs = 0; rhs < classRoots.size(); ++rhs) {
          if (rhs == lhs || (!classSignatures[rhs] && rhs < lhs))
            continue;
          if (shouldStop())
            return result;
          classifyClassPair(classRoots[lhs], classRoots[rhs]);
        }
      }
    }
    result.processedPairs = result.candidatePairs;
    if (result.provedPairs + result.transitivelyProvedPairs +
            result.refutedPairs + result.unknownPairs !=
        result.candidatePairs)
      result.interrupted = true;
    return result;
  }

  llvm::DenseSet<uint64_t> candidates;
  for (const OfflineSemanticFingerprintBucket &bucket : fingerprints.buckets)
    for (size_t lhs = 0; lhs < bucket.graphIds.size(); ++lhs)
      for (size_t rhs = lhs + 1; rhs < bucket.graphIds.size(); ++rhs) {
        if (shouldStop()) {
          result.candidatePairs = fingerprints.proofCandidatePairsAfter;
          return result;
        }
        candidates.insert(pairKey(bucket.graphIds[lhs], bucket.graphIds[rhs]));
      }

  // Wildcards cannot be rejected by fingerprint. Compare them with every
  // graph of the same result type, de-duplicating wildcard/wildcard pairs.
  for (unsigned wildcard : fingerprints.wildcardGraphIds) {
    if (wildcard >= graphs.size())
      continue;
    for (const OfflineTypedGraph &graph : graphs) {
      if (shouldStop()) {
        result.candidatePairs = fingerprints.proofCandidatePairsAfter;
        return result;
      }
      if (graph.id == wildcard || graph.resultType != graphs[wildcard].resultType)
        continue;
      candidates.insert(pairKey(wildcard, graph.id));
    }
  }

  std::vector<uint64_t> orderedCandidates(candidates.begin(), candidates.end());
  result.candidatePairs = orderedCandidates.size();
  llvm::sort(orderedCandidates);
  std::vector<unsigned> provedParent(graphs.size());
  for (unsigned id = 0; id < provedParent.size(); ++id)
    provedParent[id] = id;
  auto findProvedRoot = [&](unsigned id) {
    unsigned root = id;
    while (provedParent[root] != root)
      root = provedParent[root];
    while (provedParent[id] != id) {
      unsigned parent = provedParent[id];
      provedParent[id] = root;
      id = parent;
    }
    return root;
  };
  for (uint64_t key : orderedCandidates) {
    if (shouldStop())
      break;
    ++result.processedPairs;
    const unsigned lhsGraphId = static_cast<unsigned>(key >> 32);
    const unsigned rhsGraphId = static_cast<unsigned>(key);
    if (options.pruneTransitivelyImpliedPairs &&
        lhsGraphId < provedParent.size() &&
        rhsGraphId < provedParent.size() &&
        findProvedRoot(lhsGraphId) == findProvedRoot(rhsGraphId)) {
      ++result.transitivelyProvedPairs;
      continue;
    }
    PairEvaluation evaluation = evaluatePair(lhsGraphId, rhsGraphId);
    OfflineSemanticProofPair &pair = evaluation.pair;
    if (evaluation.counterexampleRefuted)
      ++result.counterexampleRefutedPairs;

    switch (pair.status) {
    case SemanticProofStatus::Proved:
      ++result.provedPairs;
      if (options.pruneTransitivelyImpliedPairs &&
          pair.lhsGraphId < provedParent.size() &&
          pair.rhsGraphId < provedParent.size()) {
        unsigned lhsRoot = findProvedRoot(pair.lhsGraphId);
        unsigned rhsRoot = findProvedRoot(pair.rhsGraphId);
        if (lhsRoot != rhsRoot)
          provedParent[rhsRoot] = lhsRoot;
      }
      break;
    case SemanticProofStatus::Refuted:
      ++result.refutedPairs;
      break;
    case SemanticProofStatus::Unknown:
      ++result.unknownPairs;
      break;
    }
    if (pair.status != SemanticProofStatus::Refuted ||
        options.retainRefutedPairs) {
      retainGraphKeys(pair);
      result.pairs.push_back(std::move(pair));
    }
  }
  return result;
}

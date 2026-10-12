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

#ifndef TRITON_TO_GRAPH_OFFLINE_GRAPH_ENUMERATOR_H
#define TRITON_TO_GRAPH_OFFLINE_GRAPH_ENUMERATOR_H

#include "TritonToGraph/GraphUniverseSpec.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir::triton::cfg {

struct OfflineTypeDomain {
  SmallVector<Type, 16> types;
  std::vector<std::string> unsupportedUniverseFeatures;

  bool isComplete() const { return unsupportedUniverseFeatures.empty(); }
};

// Materializes the finite scalar + ranked-tensor type domain declared by the
// universe. This is shared by the command-line builder and tests so the
// completeness boundary has one implementation.
OfflineTypeDomain buildOfflineTypeDomain(MLIRContext &context,
                                         const GraphUniverseSpec &spec);

// One canonical typed DAG in the offline search universe. `operatorSize`
// counts distinct reachable non-terminal operations; a shared operand DAG is
// counted once. Boundary inputs and constants are layer 0. The reachable ID
// set is an enumeration-time certificate for that size and is intentionally
// not part of the persistent RuleDB format.
struct OfflineTypedGraph {
  unsigned id = 0;
  unsigned operatorSize = 0;
  std::string operationName;
  TTIRCandidateEffectKind effectKind = TTIRCandidateEffectKind::Pure;
  Type resultType;
  DictionaryAttr attributes;
  SmallVector<unsigned, 3> operands;
  SmallVector<unsigned, 8> reachableOperatorIds;
  std::string canonicalKey;
};

// A cheap, exact structural summary of a denotation. It is intentionally
// weaker than semantic equivalence: equal signatures are only a candidate
// bucket and still require structural equality or the semantic oracle before
// a graph can be discarded. The fields are split by semantic domain so the
// same object can later become the generic MayExtend key for Shape/Index/
// Effect/Launch/Buffer constraints.
enum class AbstractSemanticSignaturePrecision : uint8_t {
  Exact,
  Conservative,
  Sampled,
};

struct AbstractSemanticSignature {
  std::string resultType;
  SmallVector<int64_t, 4> shape;
  SmallVector<std::string, 4> shapeSymbols;
  std::string layoutIdentity;
  uint64_t indexHash = 0;
  uint64_t valueHash = 0;
  uint64_t definednessHash = 0;
  uint64_t effectHash = 0;
  uint64_t memoryHash = 0;
  uint64_t launchHash = 0;
  uint64_t bufferHash = 0;
  uint8_t stateFlags = 0;
  AbstractSemanticSignaturePrecision precision =
      AbstractSemanticSignaturePrecision::Exact;

  bool isExact() const {
    return precision == AbstractSemanticSignaturePrecision::Exact;
  }
  uint64_t getBucketHash() const;
  bool structurallyEquals(const AbstractSemanticSignature &other) const;
};

AbstractSemanticSignature
buildAbstractSemanticSignature(Type resultType,
                               const TensorDenotation &denotation);

struct OfflineEnumerationResult;

// A reusable snapshot of every graph below the final operator-size layer.
// Final-layer partitions share this exact prefix. Keeping it once avoids
// rebuilding the same typed Cartesian product in every shard.  The cache also
// records which graphs were proof-certified operand representatives while the
// prefix was built.  A consumer re-denotes those representatives under the
// hash-bound semantics before extending them; no denotation, e-class, or
// rewrite proof is serialized.
struct OfflineEnumerationPrefixCache {
  static constexpr unsigned kSchemaVersion = 2;

  unsigned schemaVersion = kSchemaVersion;
  std::string partitionFamilyHash;
  std::string semanticsHash;
  unsigned maximumOperatorSize = 0;
  bool quotientEquivalentOperands = false;
  std::vector<OfflineTypedGraph> graphs;
  // One byte per graph.  A set bit means the prefix build proved that this
  // graph is the deterministic representative retained for later expansion.
  std::vector<uint8_t> operandFrontier;
  std::vector<std::string> realizedCandidateSchemas;
  // Populated only by fromJSON after a full content-hash verification.  It is
  // an in-memory validation result, not serialized state, and avoids hashing
  // a multi-gigabyte cache a second time immediately before replay.
  std::string validatedStableHash;

  std::string getStableHash() const;
  // The current representation is a versioned JSON-Lines record stream so a
  // shard never constructs one enormous JSON DOM. fromJSON also accepts the
  // legacy single-object representation for compatibility.
  void writeJSON(llvm::raw_ostream &output) const;
  std::string toJSON() const;

  // Stream a cache directly from a complete enumeration without duplicating
  // the potentially multi-gigabyte graph prefix in memory.
  static bool writeJSONFromEnumeration(
      llvm::raw_ostream &output,
      const OfflineEnumerationResult &enumeration,
      llvm::StringRef semanticsHash, unsigned maximumOperatorSize,
      std::string &error);

  static std::optional<OfflineEnumerationPrefixCache>
  fromEnumeration(const OfflineEnumerationResult &enumeration,
                  llvm::StringRef semanticsHash,
                  unsigned maximumOperatorSize, std::string &error);
  static std::optional<OfflineEnumerationPrefixCache>
  fromJSON(llvm::StringRef serialized, MLIRContext &context,
           std::string &error,
           size_t maxGraphs = std::numeric_limits<size_t>::max());
};

struct OfflineEnumerationOptions {
  // A safety limit is not a completeness boundary. Hitting it makes the run
  // incomplete and must be visible in the report and process exit status.
  unsigned maxGraphs = 1000000;
  // Retain every generated graph as a possible rewrite LHS, but expand larger
  // graphs from only one proof-certified representative per semantic e-class.
  // For a depth-1 universe, also retain one typed alpha-normal form for each
  // boundary-input equality partition; input names themselves are immaterial.
  const TTOpSemanticsRegistry *semantics = nullptr;
  const TensorSemanticNormalizer *normalizer = nullptr;
  bool quotientEquivalentOperands = false;
  // A standalone depth-1 universe may quotient boundary input names while
  // preserving their equality partition. A depth-1 build used as the prefix
  // of a deeper family must retain distinct names so sibling subgraphs can
  // still express same-input versus different-input relationships.
  bool alphaNormalizeFinalLayerInputs = true;
  // Final-layer graphs cannot feed another candidate. Their denotability can
  // therefore be checked concurrently and compacted back into deterministic
  // generation order without changing the explored DAG universe.
  unsigned semanticValidationJobs = 1;
  // Final-layer graphs cannot feed another candidate. When enabled, denote
  // them eagerly and retain one proof-certified representative per exact
  // semantic class instead of retaining every syntactic spelling. This is a
  // proof-safe semantic catalog mode: it preserves one executable candidate
  // for every proved root meaning, but intentionally does not claim that the
  // syntactic graph catalog is exhaustive.
  bool finalLayerSemanticQuotient = false;
  // Proof-safe recursive catalog mode.  Unlike the default complete mode, it
  // does not retain a syntactic duplicate once an earlier expansion frontier
  // already contains a proved semantic representative.  This is explicitly
  // a semantic-coverage catalog, not a complete syntactic graph catalog.
  bool semanticFrontierCatalogOnly = false;
  // A standalone enumeration that will be serialized as a deeper family's
  // prefix must classify its own last layer as an operand frontier.  Ordinary
  // final layers only need denotability and keep the cheaper validation path.
  bool recordFinalLayerOperandFrontier = false;
  // Schema reachability is a fast forward-chaining audit, not an exhaustive
  // universe build. It retains only the first well-typed realization of each
  // declarative semantics clause/arity/result projection. The result is never
  // allowed to claim exhaustive enumeration or emit a RuleDB.
  bool schemaReachabilityOnly = false;
  // Optional complete lower-layer snapshot. Its family/semantics identity is
  // validated by the caller; the enumerator validates graph topology and
  // re-denotes every cached operand-frontier representative before continuing.
  OfflineEnumerationPrefixCache *prefixCache = nullptr;
  // A command-line shard owns its deserialized cache exclusively.  Allow it
  // to transfer the graph vector into the result instead of retaining two
  // multi-gigabyte copies.  Tests and embedding callers keep copy semantics
  // unless they opt in explicitly.
  bool consumePrefixCache = false;
  // Optional cooperative cancellation hook for resource budgets.  The
  // enumerator polls it inside the Cartesian-product walk so a single large
  // enumeration stage cannot run past a caller's wall/RSS limit unchecked.
  std::function<bool()> shouldStop;
};

struct OfflineEnumerationResult {
  std::vector<OfflineTypedGraph> graphs;
  // Populated for complete quotient enumerations and consumed only by prefix
  // cache serialization.  It is not part of the public JSON report.
  std::vector<uint8_t> operandFrontier;
  // Number of exact semantic quotient representatives retained at each
  // operator-size layer.  This is reporting state only; it does not alter the
  // graph catalog or the expansion frontier.
  std::vector<uint64_t> exactQuotientClassesBySize;
  std::vector<unsigned> graphCountBySize;
  std::vector<std::string> resultTypes;
  std::vector<std::string> missingCandidateSchemaOperations;
  std::vector<std::string> missingAttributeDomainOperations;
  std::vector<std::string> unrealizedCandidateSchemaOperations;
  std::vector<std::string> unrealizedCandidateSchemas;
  // Variadic schemas may have concrete arities that are legal in the
  // language but impossible in this universe's finite type/rank domain.
  // They are reported for auditability and are not reachability failures.
  std::vector<std::string> inapplicableCandidateSchemas;
  std::vector<std::string> nonSearchableSemanticSchemas;
  std::vector<std::pair<std::string, uint64_t>> graphCountByOperation;
  std::vector<std::pair<std::string, uint64_t>> graphCountBySchema;
  std::vector<std::string> unsupportedUniverseFeatures;
  uint64_t attemptedCandidates = 0;
  uint64_t rejectedByType = 0;
  // Type-correct operand tuples can still be outside an OP's denotable
  // semantic domain (for example, an invalid effect/state combination). Such
  // tuples are rejected before they enter the finite graph universe.
  uint64_t rejectedBySemantics = 0;
  uint64_t semanticValidationGraphCount = 0;
  unsigned semanticValidationJobs = 1;
  // Number of partial type assignments evaluated while proving that each
  // selected declarative schema is applicable to the finite universe.
  uint64_t schemaTypingAssignmentsTested = 0;
  uint64_t exactDuplicates = 0;
  uint64_t operandQuotientPruned = 0;
  // When the current operands already consume the complete DAG-node budget,
  // another operand can only be a terminal or a node already present in that
  // reachable DAG. The enumerator uses this exact domain instead of scanning
  // the complete typed operand frontier again.
  uint64_t saturatedReachabilityDomains = 0;
  uint64_t saturatedReachabilityCandidatesSkipped = 0;
  // Depth-1 universes can quotient boundary-input names by typed alpha
  // renaming while retaining every operand-equality partition.
  uint64_t inputRenamingQuotientPruned = 0;
  // Number of final-layer syntactic graphs omitted after an exact semantic
  // proof against an already retained final-layer representative.
  uint64_t finalLayerSemanticQuotientPruned = 0;
  uint64_t finalLayerAbstractSignaturePruned = 0;
  // Final-layer normalized denotations that are structurally identical to an
  // existing representative.  This is a cheap exact check before invoking
  // the semantic oracle; it is kept separate from the abstract-signature
  // counter because normalization can expose equality that the raw
  // denotation signature does not.
  uint64_t finalLayerNormalizedStructuralPruned = 0;
  // Intermediate graphs omitted by semanticFrontierCatalogOnly after a
  // structural/semantic proof against an existing expansion representative.
  uint64_t semanticFrontierCatalogPruned = 0;
  // Number of short operand prefixes skipped by exact final-layer sharding.
  // Unary/binary schemas filter at operand zero; wider schemas first select a
  // bounded shard group with operand zero and then one owner with operand one.
  // A low-cardinality first operand therefore cannot monopolize a shard, while
  // unrelated shards avoid repeating its second-operand walk. The exact set
  // union is preserved across all partition indices.
  uint64_t finalLayerPartitionPrefixesPruned = 0;
  unsigned finalLayerPartitionCount = 1;
  unsigned finalLayerPartitionIndex = 0;
  std::string finalLayerPartitionScheme;
  std::string finalLayerPartitionFamilyHash;
  uint64_t completePrefixGraphCount = 0;
  std::string completePrefixHash;
  std::vector<std::string> realizedPrefixCandidateSchemas;
  uint64_t reusedPrefixGraphCount = 0;
  uint64_t reusedPrefixOperandFrontierGraphCount = 0;
  std::string reusedPrefixCacheHash;
  bool quotientEquivalentOperands = false;
  bool finalLayerSemanticQuotient = false;
  bool semanticFrontierCatalogOnly = false;
  bool semanticCatalogCompressed = false;
  bool schemaReachabilityOnly = false;
  bool hitGraphLimit = false;
  bool interrupted = false;

  bool isComplete() const {
    return !schemaReachabilityOnly && !semanticCatalogCompressed &&
           !hitGraphLimit && !interrupted &&
           missingCandidateSchemaOperations.empty() &&
           missingAttributeDomainOperations.empty() &&
           (finalLayerPartitionCount > 1 ||
            (unrealizedCandidateSchemaOperations.empty() &&
             unrealizedCandidateSchemas.empty())) &&
           unsupportedUniverseFeatures.empty();
  }

  // Completeness of the proof-safe semantic catalog. This deliberately
  // differs from isComplete(): the latter means every syntactic graph was
  // retained, while this predicate also accepts an exact final-layer
  // semantic quotient.
  bool isSemanticCoverageComplete() const {
    return !schemaReachabilityOnly && !hitGraphLimit && !interrupted &&
           missingCandidateSchemaOperations.empty() &&
           missingAttributeDomainOperations.empty() &&
           (finalLayerPartitionCount > 1 ||
            (unrealizedCandidateSchemaOperations.empty() &&
             unrealizedCandidateSchemas.empty())) &&
           unsupportedUniverseFeatures.empty();
  }

  bool isSchemaReachabilityComplete() const {
    return !hitGraphLimit && !interrupted &&
           missingCandidateSchemaOperations.empty() &&
           missingAttributeDomainOperations.empty() &&
           unrealizedCandidateSchemaOperations.empty() &&
           unrealizedCandidateSchemas.empty() &&
           unsupportedUniverseFeatures.empty();
  }

  std::string toJSON(StringRef universeHash) const;
};

// Exhaustively enumerates the Cartesian product implied by operation arity,
// typed operand DAGs and `maxNodes`. It contains no operation-name or
// workload-specific matching logic: all construction legality comes from the
// schemas generated from TTOpSemantics.
class OfflineTypedGraphEnumerator {
public:
  explicit OfflineTypedGraphEnumerator(OfflineEnumerationOptions options = {})
      : options(options) {}

  OfflineEnumerationResult
  enumerate(const GraphUniverseSpec &spec,
            ArrayRef<TTIRCandidateOpSchema> schemas,
            ArrayRef<Type> resultTypes) const;

private:
  OfflineEnumerationOptions options;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_OFFLINE_GRAPH_ENUMERATOR_H

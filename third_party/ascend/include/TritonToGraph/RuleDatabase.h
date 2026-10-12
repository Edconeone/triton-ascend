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

#ifndef TRITON_TO_GRAPH_RULE_DATABASE_H
#define TRITON_TO_GRAPH_RULE_DATABASE_H

#include "TritonToGraph/ParetoExtractor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir::triton::cfg {

struct RuleDatabaseCandidate {
  std::string rootGraphHash;
  ParetoCandidateCost cost;
};

// One canonical LHS lookup record. Candidates point into the shared graph
// table, so a class with N source forms and P Pareto winners costs O(N*P), not
// the O(N^2) expansion of every pairwise rewrite.
struct RuleDatabaseEntry {
  std::string lhsGraphHash;
  std::string classHash;
  std::string resultType;
  std::vector<std::string> proofHashes;
  ParetoCandidateCost sourceCost;
  std::vector<RuleDatabaseCandidate> candidates;
};

// A replay-validated, universally quantified rule.  `lhsPatternGraphHash`
// points at a graph in the shared graph table.  During online matching,
// terminal ags.input/ags.memory/ags.buffer nodes in that graph are typed
// variables; every other node is matched structurally. A pure graph template
// is emitted only from a direct replay-validated
// symbolic proof whose RHS introduces no new variables. Compiler-action
// templates additionally require an explicit quantified theorem step. This is
// an indexed instantiation of an offline proof rather than online equivalence
// search.
struct RuleDatabaseTemplateEntry {
  std::string lhsPatternGraphHash;
  std::string classHash;
  std::string resultType;
  std::vector<std::string> proofHashes;
  ParetoCandidateCost sourceCost;
  std::vector<RuleDatabaseCandidate> candidates;
};

struct RuleDatabaseTemplateMatch {
  const RuleDatabaseTemplateEntry *entry = nullptr;
  // Pattern terminal graph hash -> online canonical graph hash.  Keeping the
  // bindings makes later generic recipe substitution possible; compiler-action
  // templates currently need only the certified entry.
  std::vector<std::pair<std::string, std::string>> bindings;
};

// One proof-backed semantic-normal-form index entry.  Unlike an exact graph
// entry, this may match a source topology that was not enumerated offline, but
// only when interpreting that topology through the same versioned .ttsem
// language produces the same versioned normalized-denotation digest. The
// online path performs no candidate enumeration and no pairwise proof search.
struct RuleDatabaseSemanticEntry {
  // Versioned SHA-256 digest of the normalized denotation key.
  std::string semanticKey;
  std::string classHash;
  std::string resultType;
  std::vector<std::string> proofHashes;
  std::vector<RuleDatabaseCandidate> candidates;
};

// Offline certification that one persistent graph denotes a normalized
// semantic form. The proof hash is normally a reflexive Value+Valid proof
// generated from the fingerprinter's exact denotation. This lets a singleton
// optimal e-class seed semantic lookup without adding an online search path.
struct CertifiedSemanticGraph {
  std::string graphHash;
  std::string semanticKey;
  std::string proofHash;
};

// RuleDB artifacts index normalized denotations by a versioned SHA-256 digest
// instead of embedding the often-kilobyte canonical expression in every
// entry. Online callers pass the canonical expression to lookupSemantic(); it
// is reduced to the same digest before the immutable index is searched.
std::string getRuleDatabaseSemanticKeyDigest(llvm::StringRef semanticKey);

struct RuleDatabase {
  static constexpr unsigned kSchemaVersion = 7;

  unsigned schemaVersion = kSchemaVersion;
  std::string eclassStoreHash;
  std::string universeHash;
  std::string semanticsHash;
  std::string costContextKey;
  std::string costModelName;
  std::string costModelHash;
  std::vector<PersistentGraphRecord> graphs;
  std::vector<RuleDatabaseEntry> entries;
  std::vector<RuleDatabaseTemplateEntry> templates;
  std::vector<RuleDatabaseSemanticEntry> semanticEntries;

  std::string getStableHash() const;
  void writeJSON(llvm::raw_ostream &output) const;
  std::string toJSON() const;
  const RuleDatabaseEntry *lookup(llvm::StringRef lhsGraphHash) const;
  const PersistentGraphRecord *lookupGraph(llvm::StringRef graphHash) const;
  std::vector<RuleDatabaseTemplateMatch> matchTemplates(
      llvm::StringRef onlineRootGraphHash,
      llvm::ArrayRef<PersistentGraphRecord> onlineGraphs) const;
  std::vector<const RuleDatabaseSemanticEntry *>
  lookupSemantic(llvm::StringRef semanticKey,
                 llvm::StringRef resultType) const;

  // Remove only duplicate maximal-layer roots that have replay-certified
  // normalized semantics. Operand graphs are retained, compiler-action roots
  // are retained, and the semantic index is filtered to the remaining
  // candidates. This is a generic catalog compression step; it does not
  // invent an equality or perform an online search.
  static std::optional<RuleDatabase>
  quotientFinalLayerBySemanticKey(
      const RuleDatabase &database,
      llvm::ArrayRef<CertifiedSemanticGraph> certifiedSemanticGraphs,
      uint64_t &prunedGraphCount, uint64_t &prunedEntryCount,
      std::string &error);

  static std::optional<RuleDatabase>
  compile(const PersistentEquivalenceClassStore &store,
          const ParetoExtractionResult &pareto, std::string &error,
          llvm::ArrayRef<CertifiedSemanticGraph> certifiedSemanticGraphs = {},
          llvm::StringRef prevalidatedStoreHash = {});
  // Merge independently generated finite-universe shards. Proof semantics and
  // cost context must match exactly; universe hashes may differ. Shared graph
  // hashes are deduplicated by content and conflicts fail closed.
  static std::optional<RuleDatabase>
  merge(llvm::ArrayRef<RuleDatabase> shards, std::string &error);
  static std::optional<RuleDatabase> fromJSON(llvm::StringRef serialized,
                                               std::string &error);
};

enum class OnlineRuleLookupStatus : uint8_t {
  Hit,
  Miss,
  VersionMismatch,
};

struct OnlineRuleLookupResult {
  OnlineRuleLookupStatus status = OnlineRuleLookupStatus::Miss;
  const RuleDatabaseEntry *entry = nullptr;
  std::string reason;
};

// Online lookup deliberately does no candidate enumeration or proof search.
// Canonical graph construction and guard extraction are separate front-end
// steps; this component only validates artifact versions and performs O(log N)
// indexed lookup in the immutable database.
class OnlineRuleMatcher {
public:
  explicit OnlineRuleMatcher(const RuleDatabase &database)
      : database(database) {}

  OnlineRuleLookupResult lookup(llvm::StringRef canonicalGraphHash,
                                llvm::StringRef universeHash,
                                llvm::StringRef semanticsHash,
                                llvm::StringRef costContextKey) const;

private:
  const RuleDatabase &database;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_RULE_DATABASE_H

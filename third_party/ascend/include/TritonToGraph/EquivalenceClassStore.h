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

#ifndef TRITON_TO_GRAPH_EQUIVALENCE_CLASS_STORE_H
#define TRITON_TO_GRAPH_EQUIVALENCE_CLASS_STORE_H

#include "TritonToGraph/OfflineEquivalenceProver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace mlir {
class MLIRContext;
}

namespace mlir::triton::cfg {

// Stable, graph-id-independent representation of one enumerated DAG. Operand
// references use the recursively computed graph hashes rather than temporary
// enumeration IDs, so the same graph has the same identity after reload or an
// incremental build.
struct PersistentGraphRecord {
  std::string graphHash;
  std::string operationName;
  std::string resultType;
  std::string attributes;
  unsigned operatorSize = 0;
  std::vector<std::string> operandHashes;
  TTIRCandidateEffectKind effectKind = TTIRCandidateEffectKind::Pure;
};

// The single canonical graph identity implementation shared by offline store
// construction and online TTIR import. Callers must fill operation, type,
// semantic attributes, operator size and already-canonical operand hashes.
std::string getPersistentGraphHash(const PersistentGraphRecord &graph);

// Convert temporary enumeration IDs into the stable recursive graph identity
// used by the e-class store and RuleDB. The vector remains indexed by
// OfflineTypedGraph::id so later offline stages can attach derived evidence
// without rebuilding graph identity independently.
std::optional<std::vector<PersistentGraphRecord>>
buildPersistentGraphRecords(llvm::ArrayRef<OfflineTypedGraph> graphs,
                            std::string &error);

// Reconstruct a topological OfflineTypedGraph table from persistent graph
// records. This is used by cross-shard proof replay: RuleDB merge has already
// deduplicated graph identities, so the merged catalog can be re-denoted and
// re-proved without retaining the original enumerator process.
std::optional<std::vector<OfflineTypedGraph>>
reconstructOfflineGraphs(llvm::ArrayRef<PersistentGraphRecord> records,
                         MLIRContext &context, std::string &error);

// One certified edge in the equivalence relation. Only Proved pairs enter the
// store; Refuted and Unknown pairs can never merge classes.
struct PersistentProofEdge {
  std::string lhsGraphHash;
  std::string rhsGraphHash;
  std::string proofHash;
  std::string proofCertificate;
};

struct PersistentEquivalenceClass {
  std::string classHash;
  std::string resultType;
  std::vector<std::string> memberGraphHashes;
};

// Versioned output of the offline proof stage. The class partition is derived
// entirely from graph records and replayable proof edges, never trusted as an
// independent source of equivalence facts.
struct PersistentEquivalenceClassStore {
  static constexpr unsigned kSchemaVersion = 2;

  unsigned schemaVersion = kSchemaVersion;
  std::string universeHash;
  std::string semanticsHash;
  bool enumerationComplete = false;
  bool fingerprintComplete = false;
  bool proofComplete = false;
  std::vector<PersistentGraphRecord> graphs;
  std::vector<PersistentProofEdge> proofEdges;
  std::vector<PersistentEquivalenceClass> classes;

  bool isComplete() const {
    return enumerationComplete && fingerprintComplete && proofComplete;
  }

  std::string getStableHash() const;
  void writeJSON(llvm::raw_ostream &output,
                 llvm::StringRef precomputedStableHash = {}) const;
  std::string toJSON() const;

  // Builds a canonical store. `base` enables deterministic incremental
  // extension; it must have the same universe and semantics hashes. Current
  // run completeness flags are authoritative for the resulting snapshot.
  static std::optional<PersistentEquivalenceClassStore>
  build(llvm::ArrayRef<OfflineTypedGraph> graphs,
        const OfflineEquivalenceProofResult &proofs,
        llvm::StringRef universeHash, llvm::StringRef semanticsHash,
        bool enumerationComplete, bool fingerprintComplete,
        const PersistentEquivalenceClassStore *base, std::string &error);

  // Reloads an independent snapshot, replays every certificate, reconstructs
  // the union-find partition, and rejects hash/class/version mismatches.
  static std::optional<PersistentEquivalenceClassStore>
  fromJSON(llvm::StringRef serialized, MLIRContext &context,
           const SemanticEquivalenceOracle &oracle, std::string &error);
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_EQUIVALENCE_CLASS_STORE_H

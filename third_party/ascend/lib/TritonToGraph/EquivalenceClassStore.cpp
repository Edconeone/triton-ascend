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

#include "TritonToGraph/EquivalenceClassStore.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static std::string stableHash(StringRef value) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

class StableHashAccumulator {
public:
  void update(StringRef value) {
    for (unsigned char byte : value) {
      hash ^= byte;
      hash *= UINT64_C(1099511628211);
    }
  }

  std::string finish() const {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
  }

private:
  uint64_t hash = UINT64_C(14695981039346656037);
};

static std::string printType(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static std::string printAttributes(DictionaryAttr attributes) {
  if (!attributes)
    return "{}";
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << attributes;
  return storage;
}

static std::string graphContentKey(const PersistentGraphRecord &graph) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << "graph-v1\n" << graph.operationName << '\n' << graph.resultType
         << '\n' << graph.attributes << '\n' << graph.operatorSize << '\n';
  // Effect kind is persisted for safe reconstruction, but the operation name
  // remains the graph identity.  Its registered schema is the authority for
  // the effect contract, so adding the field must not perturb existing graph
  // hashes or online/offline canonical keys.
  for (StringRef operand : graph.operandHashes)
    stream << operand.size() << ':' << operand << '\n';
  return storage;
}

static std::string classContentKey(const PersistentEquivalenceClass &eclass) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << "eclass-v1\n" << eclass.resultType << '\n';
  for (StringRef member : eclass.memberGraphHashes)
    stream << member.size() << ':' << member << '\n';
  return storage;
}

static bool sameGraph(const PersistentGraphRecord &lhs,
                      const PersistentGraphRecord &rhs) {
  return lhs.graphHash == rhs.graphHash &&
         lhs.operationName == rhs.operationName &&
         lhs.resultType == rhs.resultType && lhs.attributes == rhs.attributes &&
         lhs.operatorSize == rhs.operatorSize &&
         lhs.operandHashes == rhs.operandHashes;
}

static bool sameClass(const PersistentEquivalenceClass &lhs,
                      const PersistentEquivalenceClass &rhs) {
  return lhs.classHash == rhs.classHash && lhs.resultType == rhs.resultType &&
         lhs.memberGraphHashes == rhs.memberGraphHashes;
}

class DisjointSet {
public:
  explicit DisjointSet(unsigned size) : parent(size), rank(size, 0) {
    for (unsigned index = 0; index < size; ++index)
      parent[index] = index;
  }

  unsigned find(unsigned value) {
    if (parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  }

  void merge(unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (rank[lhs] < rank[rhs])
      std::swap(lhs, rhs);
    parent[rhs] = lhs;
    if (rank[lhs] == rank[rhs])
      ++rank[lhs];
  }

private:
  std::vector<unsigned> parent;
  std::vector<unsigned> rank;
};

static bool canonicalize(PersistentEquivalenceClassStore &store,
                         std::string &error) {
  llvm::sort(store.graphs, [](const PersistentGraphRecord &lhs,
                              const PersistentGraphRecord &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });

  llvm::StringMap<unsigned> graphIndex;
  for (auto [index, graph] : llvm::enumerate(store.graphs)) {
    if (graph.graphHash.empty() ||
        graph.graphHash != getPersistentGraphHash(graph)) {
      error = "persistent graph hash mismatch";
      return false;
    }
    auto inserted = graphIndex.try_emplace(graph.graphHash, index);
    if (!inserted.second) {
      error = "duplicate persistent graph hash";
      return false;
    }
  }

  for (const PersistentGraphRecord &graph : store.graphs)
    for (StringRef operand : graph.operandHashes)
      if (!graphIndex.contains(operand)) {
        error = "persistent graph references an unknown operand hash";
        return false;
      }

  for (PersistentProofEdge &edge : store.proofEdges) {
    if (edge.rhsGraphHash < edge.lhsGraphHash)
      std::swap(edge.lhsGraphHash, edge.rhsGraphHash);
    if (!graphIndex.contains(edge.lhsGraphHash) ||
        !graphIndex.contains(edge.rhsGraphHash)) {
      error = "proof edge references an unknown graph hash";
      return false;
    }
    if (edge.lhsGraphHash == edge.rhsGraphHash) {
      error = "proof edge must connect two distinct graphs";
      return false;
    }
    if (edge.proofHash.empty() || edge.proofCertificate.empty()) {
      error = "proof edge is missing its hash or certificate";
      return false;
    }
  }
  llvm::sort(store.proofEdges,
             [](const PersistentProofEdge &lhs,
                const PersistentProofEdge &rhs) {
               return std::tie(lhs.lhsGraphHash, lhs.rhsGraphHash,
                               lhs.proofHash, lhs.proofCertificate) <
                      std::tie(rhs.lhsGraphHash, rhs.rhsGraphHash,
                               rhs.proofHash, rhs.proofCertificate);
             });
  store.proofEdges.erase(
      std::unique(store.proofEdges.begin(), store.proofEdges.end(),
                  [](const PersistentProofEdge &lhs,
                     const PersistentProofEdge &rhs) {
                    return lhs.lhsGraphHash == rhs.lhsGraphHash &&
                           lhs.rhsGraphHash == rhs.rhsGraphHash &&
                           lhs.proofHash == rhs.proofHash &&
                           lhs.proofCertificate == rhs.proofCertificate;
                  }),
      store.proofEdges.end());

  DisjointSet sets(store.graphs.size());
  for (const PersistentProofEdge &edge : store.proofEdges) {
    unsigned lhs = graphIndex.lookup(edge.lhsGraphHash);
    unsigned rhs = graphIndex.lookup(edge.rhsGraphHash);
    if (store.graphs[lhs].resultType != store.graphs[rhs].resultType) {
      error = "proof edge connects graphs with different result types";
      return false;
    }
    sets.merge(lhs, rhs);
  }

  std::map<unsigned, std::vector<std::string>> membersByRoot;
  for (unsigned index = 0; index < store.graphs.size(); ++index)
    membersByRoot[sets.find(index)].push_back(store.graphs[index].graphHash);

  store.classes.clear();
  for (auto &[root, members] : membersByRoot) {
    llvm::sort(members);
    PersistentEquivalenceClass eclass;
    eclass.resultType = store.graphs[root].resultType;
    for (StringRef member : members) {
      unsigned index = graphIndex.lookup(member);
      if (store.graphs[index].resultType != eclass.resultType) {
        error = "equivalence class contains different result types";
        return false;
      }
    }
    eclass.memberGraphHashes = std::move(members);
    eclass.classHash = stableHash(classContentKey(eclass));
    store.classes.push_back(std::move(eclass));
  }
  llvm::sort(store.classes,
             [](const PersistentEquivalenceClass &lhs,
                const PersistentEquivalenceClass &rhs) {
               return lhs.memberGraphHashes.front() <
                      rhs.memberGraphHashes.front();
             });
  return true;
}

static llvm::json::Object graphToJSON(const PersistentGraphRecord &graph) {
  llvm::json::Array operands;
  for (StringRef operand : graph.operandHashes)
    operands.push_back(operand);
  llvm::json::Object object;
  object["graph_hash"] = graph.graphHash;
  object["operation_name"] = graph.operationName;
  object["effect_kind"] = static_cast<int64_t>(graph.effectKind);
  object["result_type"] = graph.resultType;
  object["attributes"] = graph.attributes;
  object["operator_size"] = static_cast<int64_t>(graph.operatorSize);
  object["operand_hashes"] = std::move(operands);
  return object;
}

static llvm::json::Object proofToJSON(const PersistentProofEdge &edge) {
  llvm::json::Object object;
  object["lhs_graph_hash"] = edge.lhsGraphHash;
  object["rhs_graph_hash"] = edge.rhsGraphHash;
  object["proof_hash"] = edge.proofHash;
  object["proof_certificate"] = edge.proofCertificate;
  return object;
}

static llvm::json::Object
classToJSON(const PersistentEquivalenceClass &eclass) {
  llvm::json::Array members;
  for (StringRef member : eclass.memberGraphHashes)
    members.push_back(member);
  llvm::json::Object object;
  object["class_hash"] = eclass.classHash;
  object["result_type"] = eclass.resultType;
  object["member_graph_hashes"] = std::move(members);
  return object;
}

static std::optional<std::string>
requiredString(const llvm::json::Object &object, StringRef field,
               std::string &error) {
  std::optional<StringRef> value = object.getString(field);
  if (!value) {
    error = (llvm::Twine("equivalence-class store is missing string field '") +
             field + "'")
                .str();
    return std::nullopt;
  }
  return value->str();
}

static bool requiredBoolean(const llvm::json::Object &object, StringRef field,
                            bool &value, std::string &error) {
  std::optional<bool> parsed = object.getBoolean(field);
  if (!parsed) {
    error = (llvm::Twine("equivalence-class store is missing boolean field '") +
             field + "'")
                .str();
    return false;
  }
  value = *parsed;
  return true;
}

static bool parseStringArray(const llvm::json::Object &object, StringRef field,
                             std::vector<std::string> &values,
                             std::string &error) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array) {
    error = (llvm::Twine("equivalence-class store is missing array field '") +
             field + "'")
                .str();
    return false;
  }
  for (const llvm::json::Value &entry : *array) {
    std::optional<StringRef> value = entry.getAsString();
    if (!value) {
      error = (llvm::Twine("equivalence-class store field '") + field +
               "' must contain only strings")
                  .str();
      return false;
    }
    values.push_back(value->str());
  }
  return true;
}

} // namespace

std::string mlir::triton::cfg::getPersistentGraphHash(
    const PersistentGraphRecord &graph) {
  return stableHash(graphContentKey(graph));
}

std::optional<std::vector<PersistentGraphRecord>>
mlir::triton::cfg::buildPersistentGraphRecords(
    llvm::ArrayRef<OfflineTypedGraph> graphs, std::string &error) {
  std::vector<std::string> graphHashes(graphs.size());
  std::vector<PersistentGraphRecord> records;
  records.reserve(graphs.size());
  for (const OfflineTypedGraph &graph : graphs) {
    if (graph.id != records.size()) {
      error = "enumerated graph IDs must be contiguous and topological";
      return std::nullopt;
    }
    PersistentGraphRecord record;
    record.operationName = graph.operationName;
    record.effectKind = graph.effectKind;
    record.resultType = printType(graph.resultType);
    record.attributes = printAttributes(graph.attributes);
    record.operatorSize = graph.operatorSize;
    for (unsigned operand : graph.operands) {
      if (operand >= graphHashes.size() || graphHashes[operand].empty()) {
        error = "enumerated graph operands are not in topological order";
        return std::nullopt;
      }
      record.operandHashes.push_back(graphHashes[operand]);
    }
    record.graphHash = getPersistentGraphHash(record);
    graphHashes[graph.id] = record.graphHash;
    records.push_back(std::move(record));
  }
  return records;
}

std::optional<std::vector<OfflineTypedGraph>>
mlir::triton::cfg::reconstructOfflineGraphs(
    llvm::ArrayRef<PersistentGraphRecord> records, MLIRContext &context,
    std::string &error) {
  llvm::StringMap<unsigned> recordIndices;
  for (auto [index, record] : llvm::enumerate(records)) {
    if (record.graphHash.empty() ||
        record.graphHash != getPersistentGraphHash(record)) {
      error = "persistent graph hash mismatch during reconstruction";
      return std::nullopt;
    }
    auto inserted = recordIndices.try_emplace(record.graphHash, index);
    if (!inserted.second) {
      error = "duplicate persistent graph hash during reconstruction";
      return std::nullopt;
    }
  }

  std::vector<OfflineTypedGraph> graphs;
  graphs.reserve(records.size());
  llvm::StringMap<unsigned> graphIds;
  llvm::StringMap<uint8_t> visiting;

  std::function<std::optional<unsigned>(StringRef)> materialize =
      [&](StringRef graphHash) -> std::optional<unsigned> {
    if (auto found = graphIds.find(graphHash); found != graphIds.end())
      return found->second;
    if (visiting.contains(graphHash)) {
      error = "persistent graph records contain an operand cycle";
      return std::nullopt;
    }
    auto recordIt = recordIndices.find(graphHash);
    if (recordIt == recordIndices.end()) {
      error = "persistent graph references an unknown operand hash";
      return std::nullopt;
    }
    visiting[graphHash] = 1;
    const PersistentGraphRecord &record = records[recordIt->second];
    SmallVector<unsigned, 4> operands;
    operands.reserve(record.operandHashes.size());
    for (StringRef operandHash : record.operandHashes) {
      std::optional<unsigned> operand = materialize(operandHash);
      if (!operand)
        return std::nullopt;
      operands.push_back(*operand);
    }

    Type resultType = mlir::parseType(record.resultType, &context);
    if (!resultType) {
      error = "cannot parse persistent graph result type '" +
              record.resultType + "'";
      return std::nullopt;
    }
    Attribute parsedAttributes =
        mlir::parseAttribute(record.attributes, &context);
    DictionaryAttr attributes = dyn_cast<DictionaryAttr>(parsedAttributes);
    if (!attributes) {
      error = "cannot parse persistent graph attributes for '" +
              record.graphHash + "'";
      return std::nullopt;
    }

    OfflineTypedGraph graph;
    graph.id = graphs.size();
    graph.operatorSize = record.operatorSize;
    graph.operationName = record.operationName;
    graph.effectKind = record.effectKind;
    graph.resultType = resultType;
    graph.attributes = attributes;
    graph.operands = std::move(operands);
    graph.canonicalKey = record.graphHash;
    graphs.push_back(std::move(graph));
    graphIds.try_emplace(record.graphHash, graphs.back().id);
    visiting.erase(graphHash);
    return graphs.back().id;
  };

  for (const PersistentGraphRecord &record : records)
    if (!materialize(record.graphHash))
      return std::nullopt;
  return graphs;
}

std::string PersistentEquivalenceClassStore::getStableHash() const {
  StableHashAccumulator accumulator;
  auto append = [&](auto &&...values) {
    std::string fragment;
    llvm::raw_string_ostream stream(fragment);
    (stream << ... << values);
    accumulator.update(fragment);
  };
  append("ags-eclass-store-v", schemaVersion, '\n', universeHash, '\n',
         semanticsHash, '\n', enumerationComplete, fingerprintComplete,
         proofComplete, '\n');
  for (const PersistentGraphRecord &graph : graphs) {
    append(graph.graphHash, '\n');
    accumulator.update(graphContentKey(graph));
  }
  for (const PersistentProofEdge &edge : proofEdges) {
    append(edge.lhsGraphHash, '\n', edge.rhsGraphHash, '\n', edge.proofHash,
           '\n', edge.proofCertificate.size(), ':');
    accumulator.update(edge.proofCertificate);
    accumulator.update("\n");
  }
  for (const PersistentEquivalenceClass &eclass : classes) {
    append(eclass.classHash, '\n');
    accumulator.update(classContentKey(eclass));
  }
  return accumulator.finish();
}

void PersistentEquivalenceClassStore::writeJSON(
    llvm::raw_ostream &output, llvm::StringRef precomputedStableHash) const {
  auto writeString = [&](StringRef value) {
    output << llvm::formatv("{0}", llvm::json::Value(value));
  };
  auto writeArray = [&](StringRef name, const auto &values, auto encode) {
    output << "  \"" << name << "\": [";
    bool first = true;
    for (const auto &value : values) {
      output << (first ? "\n    " : ",\n    ");
      first = false;
      output << llvm::formatv("{0}", llvm::json::Value(encode(value)));
    }
    output << (first ? "]" : "\n  ]");
  };

  output << "{\n  \"schema_version\": " << schemaVersion
         << ",\n  \"universe_hash\": ";
  writeString(universeHash);
  output << ",\n  \"semantics_hash\": ";
  writeString(semanticsHash);
  output << ",\n  \"enumeration_complete\": "
         << (enumerationComplete ? "true" : "false")
         << ",\n  \"fingerprint_complete\": "
         << (fingerprintComplete ? "true" : "false")
         << ",\n  \"proof_complete\": "
         << (proofComplete ? "true" : "false")
         << ",\n  \"complete\": " << (isComplete() ? "true" : "false")
         << ",\n  \"store_hash\": ";
  const std::string computedStableHash =
      precomputedStableHash.empty() ? getStableHash() : std::string();
  writeString(precomputedStableHash.empty() ? llvm::StringRef(computedStableHash)
                                             : precomputedStableHash);
  output << ",\n  \"graph_count\": " << graphs.size()
         << ",\n  \"proof_edge_count\": " << proofEdges.size()
         << ",\n  \"eclass_count\": " << classes.size() << ",\n";
  writeArray("graphs", graphs,
             [](const PersistentGraphRecord &graph) {
               return graphToJSON(graph);
             });
  output << ",\n";
  writeArray("proof_edges", proofEdges,
             [](const PersistentProofEdge &edge) {
               return proofToJSON(edge);
             });
  output << ",\n";
  writeArray("eclasses", classes,
             [](const PersistentEquivalenceClass &eclass) {
               return classToJSON(eclass);
             });
  output << "\n}\n";
}

std::string PersistentEquivalenceClassStore::toJSON() const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream);
  return storage;
}

std::optional<PersistentEquivalenceClassStore>
PersistentEquivalenceClassStore::build(
    llvm::ArrayRef<OfflineTypedGraph> enumeratedGraphs,
    const OfflineEquivalenceProofResult &proofs, StringRef universeHash,
    StringRef semanticsHash, bool enumerationComplete,
    bool fingerprintComplete, const PersistentEquivalenceClassStore *base,
    std::string &error) {
  PersistentEquivalenceClassStore store;
  store.universeHash = universeHash.str();
  store.semanticsHash = semanticsHash.str();
  store.enumerationComplete = enumerationComplete;
  store.fingerprintComplete = fingerprintComplete;
  store.proofComplete = proofs.isComplete();
  if (base) {
    if (base->schemaVersion != kSchemaVersion ||
        base->universeHash != universeHash ||
        base->semanticsHash != semanticsHash) {
      error = "incremental equivalence-class store version/hash mismatch";
      return std::nullopt;
    }
    store.graphs = base->graphs;
    store.proofEdges = base->proofEdges;
  }

  std::optional<std::vector<PersistentGraphRecord>> builtGraphs =
      buildPersistentGraphRecords(enumeratedGraphs, error);
  if (!builtGraphs)
    return std::nullopt;
  std::vector<PersistentGraphRecord> newGraphs = std::move(*builtGraphs);
  std::vector<std::string> graphHashes;
  graphHashes.reserve(newGraphs.size());
  for (const PersistentGraphRecord &graph : newGraphs)
    graphHashes.push_back(graph.graphHash);

  llvm::StringMap<unsigned> known;
  for (auto [index, graph] : llvm::enumerate(store.graphs))
    known.try_emplace(graph.graphHash, index);
  for (PersistentGraphRecord &graph : newGraphs) {
    auto found = known.find(graph.graphHash);
    if (found != known.end()) {
      if (!sameGraph(store.graphs[found->second], graph)) {
        error = "stable graph hash collision with different graph content";
        return std::nullopt;
      }
      continue;
    }
    known.try_emplace(graph.graphHash, store.graphs.size());
    store.graphs.push_back(std::move(graph));
  }

  for (const OfflineSemanticProofPair &pair : proofs.pairs) {
    if (pair.status != SemanticProofStatus::Proved)
      continue;
    if (pair.lhsGraphId >= graphHashes.size() ||
        pair.rhsGraphId >= graphHashes.size()) {
      error = "proved pair references an unknown enumerated graph";
      return std::nullopt;
    }
    PersistentProofEdge edge;
    edge.lhsGraphHash = graphHashes[pair.lhsGraphId];
    edge.rhsGraphHash = graphHashes[pair.rhsGraphId];
    // Enumeration IDs can collapse to one persistent structural graph after
    // MLIR attribute normalization (for example low-bit integer constants).
    // Such an identity proof contributes no union edge and is not persisted.
    if (edge.lhsGraphHash == edge.rhsGraphHash)
      continue;
    edge.proofHash = pair.proofHash;
    edge.proofCertificate = pair.getProofCertificate().str();
    store.proofEdges.push_back(std::move(edge));
  }

  if (!canonicalize(store, error))
    return std::nullopt;
  return store;
}

std::optional<PersistentEquivalenceClassStore>
PersistentEquivalenceClassStore::fromJSON(
    StringRef serialized, MLIRContext &context,
    const SemanticEquivalenceOracle &oracle, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(serialized);
  if (!parsed) {
    error = "invalid equivalence-class store JSON: " +
            llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object) {
    error = "equivalence-class store root must be an object";
    return std::nullopt;
  }
  std::optional<int64_t> version = object->getInteger("schema_version");
  if (!version || *version != kSchemaVersion) {
    error = "unsupported or missing equivalence-class store schema version";
    return std::nullopt;
  }

  PersistentEquivalenceClassStore store;
  store.schemaVersion = static_cast<unsigned>(*version);
  std::optional<std::string> universe =
      requiredString(*object, "universe_hash", error);
  std::optional<std::string> semantics =
      requiredString(*object, "semantics_hash", error);
  std::optional<std::string> expectedStoreHash =
      requiredString(*object, "store_hash", error);
  if (!universe || !semantics || !expectedStoreHash ||
      !requiredBoolean(*object, "enumeration_complete",
                       store.enumerationComplete, error) ||
      !requiredBoolean(*object, "fingerprint_complete",
                       store.fingerprintComplete, error) ||
      !requiredBoolean(*object, "proof_complete", store.proofComplete,
                       error))
    return std::nullopt;
  store.universeHash = std::move(*universe);
  store.semanticsHash = std::move(*semantics);

  const llvm::json::Array *graphArray = object->getArray("graphs");
  const llvm::json::Array *proofArray = object->getArray("proof_edges");
  const llvm::json::Array *classArray = object->getArray("eclasses");
  if (!graphArray || !proofArray || !classArray) {
    error = "equivalence-class store is missing graphs/proof_edges/eclasses";
    return std::nullopt;
  }

  for (const llvm::json::Value &value : *graphArray) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "persistent graph record must be an object";
      return std::nullopt;
    }
    PersistentGraphRecord graph;
    std::optional<std::string> hash =
        requiredString(*entry, "graph_hash", error);
    std::optional<std::string> operation =
        requiredString(*entry, "operation_name", error);
    std::optional<std::string> type =
        requiredString(*entry, "result_type", error);
    std::optional<std::string> attributes =
        requiredString(*entry, "attributes", error);
    std::optional<int64_t> size = entry->getInteger("operator_size");
    std::optional<int64_t> effect = entry->getInteger("effect_kind");
    if (!hash || !operation || !type || !attributes || !size || *size < 0 ||
        !effect || *effect < 0 ||
        *effect > static_cast<int64_t>(TTIRCandidateEffectKind::BufferWrite) ||
        !parseStringArray(*entry, "operand_hashes", graph.operandHashes,
                          error)) {
      if (error.empty())
        error = "persistent graph record has an invalid operator_size";
      return std::nullopt;
    }
    graph.graphHash = std::move(*hash);
    graph.operationName = std::move(*operation);
    graph.effectKind = static_cast<TTIRCandidateEffectKind>(*effect);
    graph.resultType = std::move(*type);
    graph.attributes = std::move(*attributes);
    graph.operatorSize = static_cast<unsigned>(*size);
    store.graphs.push_back(std::move(graph));
  }

  for (const llvm::json::Value &value : *proofArray) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "persistent proof edge must be an object";
      return std::nullopt;
    }
    PersistentProofEdge edge;
    std::optional<std::string> lhs =
        requiredString(*entry, "lhs_graph_hash", error);
    std::optional<std::string> rhs =
        requiredString(*entry, "rhs_graph_hash", error);
    std::optional<std::string> proofHash =
        requiredString(*entry, "proof_hash", error);
    std::optional<std::string> certificate =
        requiredString(*entry, "proof_certificate", error);
    if (!lhs || !rhs || !proofHash || !certificate)
      return std::nullopt;
    edge.lhsGraphHash = std::move(*lhs);
    edge.rhsGraphHash = std::move(*rhs);
    edge.proofHash = std::move(*proofHash);
    edge.proofCertificate = std::move(*certificate);

    std::string proofError;
    std::optional<SemanticEquivalenceProof> proof =
        SemanticEquivalenceProof::fromJSON(edge.proofCertificate, context,
                                           proofError);
    if (!proof || proof->getStableHash() != edge.proofHash) {
      error = proofError.empty() ? "persistent proof hash mismatch"
                                 : "persistent proof decode failed: " +
                                       proofError;
      return std::nullopt;
    }
    SemanticProofResult replay = oracle.replay(*proof);
    if (!replay.isProved() || !replay.proof ||
        replay.proof->getStableHash() != edge.proofHash) {
      error = "persistent proof certificate replay failed";
      return std::nullopt;
    }
    store.proofEdges.push_back(std::move(edge));
  }

  std::vector<PersistentEquivalenceClass> serializedClasses;
  for (const llvm::json::Value &value : *classArray) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "persistent equivalence class must be an object";
      return std::nullopt;
    }
    PersistentEquivalenceClass eclass;
    std::optional<std::string> hash =
        requiredString(*entry, "class_hash", error);
    std::optional<std::string> type =
        requiredString(*entry, "result_type", error);
    if (!hash || !type ||
        !parseStringArray(*entry, "member_graph_hashes",
                          eclass.memberGraphHashes, error))
      return std::nullopt;
    eclass.classHash = std::move(*hash);
    eclass.resultType = std::move(*type);
    serializedClasses.push_back(std::move(eclass));
  }

  if (!canonicalize(store, error))
    return std::nullopt;
  llvm::sort(serializedClasses,
             [](const PersistentEquivalenceClass &lhs,
                const PersistentEquivalenceClass &rhs) {
               if (lhs.memberGraphHashes.empty() ||
                   rhs.memberGraphHashes.empty())
                 return lhs.memberGraphHashes.size() <
                        rhs.memberGraphHashes.size();
               return lhs.memberGraphHashes.front() <
                      rhs.memberGraphHashes.front();
             });
  if (serializedClasses.size() != store.classes.size() ||
      !std::equal(serializedClasses.begin(), serializedClasses.end(),
                  store.classes.begin(), sameClass)) {
    error = "serialized equivalence classes do not match certified edges";
    return std::nullopt;
  }

  auto checkCount = [&](StringRef field, size_t actual) {
    std::optional<int64_t> expected = object->getInteger(field);
    if (!expected || *expected < 0 ||
        static_cast<uint64_t>(*expected) != actual) {
      error = (llvm::Twine("equivalence-class store count mismatch for '") +
               field + "'")
                  .str();
      return false;
    }
    return true;
  };
  std::optional<bool> complete = object->getBoolean("complete");
  if (!complete || *complete != store.isComplete() ||
      !checkCount("graph_count", store.graphs.size()) ||
      !checkCount("proof_edge_count", store.proofEdges.size()) ||
      !checkCount("eclass_count", store.classes.size()))
    return std::nullopt;
  if (store.getStableHash() != *expectedStoreHash) {
    error = "equivalence-class store stable hash mismatch";
    return std::nullopt;
  }
  return store;
}

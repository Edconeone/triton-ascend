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

#include "TritonToGraph/RuleDatabase.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>

using namespace mlir::triton::cfg;

namespace {

static std::string stableHash(llvm::StringRef value) {
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
  void update(llvm::StringRef value) {
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

// Exact RuleDB entries need an auditable reference to their certified
// equivalence class, not a private copy of every edge in the class proof
// forest. Keep a single proof hash unchanged for debuggability; summarize a
// larger, sorted proof set by a stable bundle hash. The e-class store hash in
// the RuleDB header identifies the artifact that contains the referenced
// forest, so this remains fail-closed while bounding exact-entry metadata.
static std::vector<std::string>
compactProofReferences(llvm::ArrayRef<std::string> sortedProofHashes) {
  if (sortedProofHashes.size() <= 1)
    return {sortedProofHashes.begin(), sortedProofHashes.end()};
  StableHashAccumulator accumulator;
  accumulator.update("ags-proof-bundle-v1\n");
  for (llvm::StringRef proofHash : sortedProofHashes) {
    accumulator.update(proofHash);
    accumulator.update("\n");
  }
  return {"bundle:" + accumulator.finish()};
}

static std::optional<std::string>
requiredString(const llvm::json::Object &object, llvm::StringRef field,
               std::string &error) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value) {
    error = (llvm::Twine("rule database is missing string field '") + field +
             "'")
                .str();
    return std::nullopt;
  }
  return value->str();
}

static bool requiredUnsigned(const llvm::json::Object &object,
                             llvm::StringRef field, uint64_t &value,
                             std::string &error) {
  std::optional<int64_t> parsed = object.getInteger(field);
  if (!parsed || *parsed < 0) {
    error = (llvm::Twine("rule database is missing unsigned field '") + field +
             "'")
                .str();
    return false;
  }
  value = static_cast<uint64_t>(*parsed);
  return true;
}

static llvm::json::Object graphToJSON(const PersistentGraphRecord &graph) {
  llvm::json::Array operands;
  for (llvm::StringRef operand : graph.operandHashes)
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

static llvm::json::Object costToJSON(const ParetoCandidateCost &cost) {
  llvm::json::Object object;
  object["operator_count"] =
      static_cast<int64_t>(cost.structural.operatorCount);
  object["weighted_operator_cost"] =
      static_cast<int64_t>(cost.structural.weightedOperatorCost);
  object["max_depth"] = static_cast<int64_t>(cost.structural.maxDepth);
  object["conversion_count"] =
      static_cast<int64_t>(cost.structural.conversionCount);
  object["memory_operation_count"] =
      static_cast<int64_t>(cost.structural.memoryOperationCount);
  object["memory_stride_penalty"] =
      static_cast<int64_t>(cost.structural.memoryStridePenalty);
  if (cost.stageCycles)
    object["stage_cycles"] = static_cast<int64_t>(*cost.stageCycles);
  return object;
}

static bool parseStringArray(const llvm::json::Object &object,
                             llvm::StringRef field,
                             std::vector<std::string> &values,
                             std::string &error) {
  const llvm::json::Array *array = object.getArray(field);
  if (!array) {
    error = (llvm::Twine("rule database is missing array field '") + field +
             "'")
                .str();
    return false;
  }
  for (const llvm::json::Value &entry : *array) {
    std::optional<llvm::StringRef> value = entry.getAsString();
    if (!value) {
      error = (llvm::Twine("rule database field '") + field +
               "' must contain strings")
                  .str();
      return false;
    }
    values.push_back(value->str());
  }
  return true;
}

static bool parseCost(const llvm::json::Object &object,
                      ParetoCandidateCost &cost, std::string &error) {
  if (!requiredUnsigned(object, "operator_count",
                        cost.structural.operatorCount, error) ||
      !requiredUnsigned(object, "weighted_operator_cost",
                        cost.structural.weightedOperatorCost, error) ||
      !requiredUnsigned(object, "max_depth", cost.structural.maxDepth,
                        error) ||
      !requiredUnsigned(object, "conversion_count",
                        cost.structural.conversionCount, error) ||
      !requiredUnsigned(object, "memory_operation_count",
                        cost.structural.memoryOperationCount, error) ||
      !requiredUnsigned(object, "memory_stride_penalty",
                        cost.structural.memoryStridePenalty, error))
    return false;
  if (std::optional<int64_t> cycles = object.getInteger("stage_cycles")) {
    if (*cycles < 0) {
      error = "rule database stage_cycles must be non-negative";
      return false;
    }
    cost.stageCycles = static_cast<uint64_t>(*cycles);
  }
  return true;
}

static bool sameStructuralCost(const StructuralCostFeatures &lhs,
                               const StructuralCostFeatures &rhs) {
  return lhs.operatorCount == rhs.operatorCount &&
         lhs.weightedOperatorCost == rhs.weightedOperatorCost &&
         lhs.maxDepth == rhs.maxDepth &&
         lhs.conversionCount == rhs.conversionCount &&
         lhs.memoryOperationCount == rhs.memoryOperationCount &&
         lhs.memoryStridePenalty == rhs.memoryStridePenalty;
}

static bool sameCost(const ParetoCandidateCost &lhs,
                     const ParetoCandidateCost &rhs) {
  return sameStructuralCost(lhs.structural, rhs.structural) &&
         lhs.stageCycles == rhs.stageCycles;
}

static bool sameGraph(const PersistentGraphRecord &lhs,
                      const PersistentGraphRecord &rhs) {
  return lhs.graphHash == rhs.graphHash &&
         lhs.operationName == rhs.operationName &&
         lhs.resultType == rhs.resultType && lhs.attributes == rhs.attributes &&
         lhs.operatorSize == rhs.operatorSize &&
         lhs.operandHashes == rhs.operandHashes;
}

static std::string entryContent(const RuleDatabaseEntry &entry) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << entry.lhsGraphHash << '\n' << entry.classHash << '\n'
         << entry.resultType << '\n';
  for (llvm::StringRef proof : entry.proofHashes)
    stream << "p:" << proof << '\n';
  for (const RuleDatabaseCandidate &candidate : entry.candidates) {
    const StructuralCostFeatures &cost = candidate.cost.structural;
    stream << "c:" << candidate.rootGraphHash << ':' << cost.operatorCount
           << ':' << cost.weightedOperatorCost << ':' << cost.maxDepth << ':'
           << cost.conversionCount << ':' << cost.memoryOperationCount << ':'
           << cost.memoryStridePenalty << ':';
    if (candidate.cost.stageCycles)
      stream << *candidate.cost.stageCycles;
    stream << '\n';
  }
  const StructuralCostFeatures &source = entry.sourceCost.structural;
  stream << "s:" << source.operatorCount << ':'
         << source.weightedOperatorCost << ':' << source.maxDepth << ':'
         << source.conversionCount << ':' << source.memoryOperationCount << ':'
         << source.memoryStridePenalty << ':';
  if (entry.sourceCost.stageCycles)
    stream << *entry.sourceCost.stageCycles;
  stream << '\n';
  return storage;
}

static std::string templateContent(const RuleDatabaseTemplateEntry &entry) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << entry.lhsPatternGraphHash << '\n' << entry.classHash << '\n'
         << entry.resultType << '\n';
  for (llvm::StringRef proof : entry.proofHashes)
    stream << "p:" << proof << '\n';
  for (const RuleDatabaseCandidate &candidate : entry.candidates) {
    const StructuralCostFeatures &cost = candidate.cost.structural;
    stream << "c:" << candidate.rootGraphHash << ':' << cost.operatorCount
           << ':' << cost.weightedOperatorCost << ':' << cost.maxDepth << ':'
           << cost.conversionCount << ':' << cost.memoryOperationCount << ':'
           << cost.memoryStridePenalty << ':';
    if (candidate.cost.stageCycles)
      stream << *candidate.cost.stageCycles;
    stream << '\n';
  }
  const StructuralCostFeatures &source = entry.sourceCost.structural;
  stream << "s:" << source.operatorCount << ':'
         << source.weightedOperatorCost << ':' << source.maxDepth << ':'
         << source.conversionCount << ':' << source.memoryOperationCount << ':'
         << source.memoryStridePenalty << ':';
  if (entry.sourceCost.stageCycles)
    stream << *entry.sourceCost.stageCycles;
  stream << '\n';
  return storage;
}

static std::string semanticEntryContent(
    const RuleDatabaseSemanticEntry &entry) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << entry.semanticKey << '\n' << entry.classHash << '\n'
         << entry.resultType << '\n';
  for (llvm::StringRef proof : entry.proofHashes)
    stream << "p:" << proof << '\n';
  for (const RuleDatabaseCandidate &candidate : entry.candidates) {
    const StructuralCostFeatures &cost = candidate.cost.structural;
    stream << "c:" << candidate.rootGraphHash << ':' << cost.operatorCount
           << ':' << cost.weightedOperatorCost << ':' << cost.maxDepth << ':'
           << cost.conversionCount << ':' << cost.memoryOperationCount << ':'
           << cost.memoryStridePenalty << ':';
    if (candidate.cost.stageCycles)
      stream << *candidate.cost.stageCycles;
    stream << '\n';
  }
  return storage;
}

static bool collectCertificateSemanticKeys(
    llvm::StringRef certificate, llvm::StringSet<> &keys,
    std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(certificate);
  if (!parsed) {
    error = "cannot parse proof certificate while compiling semantic index: " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const llvm::json::Object *object = parsed->getAsObject();
  std::optional<llvm::StringRef> lhs =
      object ? object->getString("lhs_semantic_key") : std::nullopt;
  std::optional<llvm::StringRef> rhs =
      object ? object->getString("rhs_semantic_key") : std::nullopt;
  if (!lhs || lhs->empty() || !rhs || rhs->empty()) {
    error = "proof certificate is missing normalized semantic keys";
    return false;
  }
  keys.insert(*lhs);
  keys.insert(*rhs);
  return true;
}

static bool hasCompilerAction(const PersistentGraphRecord &graph) {
  return llvm::StringRef(graph.attributes).contains("ags.compiler_action");
}

static bool certificateUsesQuantifiedTheorem(llvm::StringRef certificate) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(certificate);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return false;
  }
  const llvm::json::Object *object = parsed->getAsObject();
  const llvm::json::Array *steps = object ? object->getArray("steps") : nullptr;
  if (!steps)
    return false;
  for (const llvm::json::Value &value : *steps) {
    const llvm::json::Object *step = value.getAsObject();
    std::optional<llvm::StringRef> reason =
        step ? step->getString("reason") : std::nullopt;
    if (reason && reason->starts_with("quantified-"))
      return true;
  }
  return false;
}

static bool isTemplateVariable(const PersistentGraphRecord &graph) {
  if (!graph.operandHashes.empty())
    return false;
  llvm::StringRef operation = graph.operationName;
  return operation == "ags.input" || operation == "ags.private_buffer" ||
         operation == "ags.memory" ||
         operation == "ags.partitionable_memory" ||
         operation == "ags.buffer";
}

static bool matchTemplateGraph(
    llvm::StringRef patternHash, llvm::StringRef onlineHash,
    const RuleDatabase &database,
    const llvm::StringMap<const PersistentGraphRecord *> &onlineByHash,
    llvm::StringMap<std::string> &bindings) {
  const PersistentGraphRecord *pattern = database.lookupGraph(patternHash);
  auto foundOnline = onlineByHash.find(onlineHash);
  const PersistentGraphRecord *online =
      foundOnline == onlineByHash.end() ? nullptr : foundOnline->second;
  if (!pattern || !online || pattern->resultType != online->resultType)
    return false;
  if (isTemplateVariable(*pattern)) {
    auto known = bindings.find(patternHash);
    if (known != bindings.end())
      return known->second == onlineHash;
    bindings.try_emplace(patternHash, onlineHash.str());
    return true;
  }
  if (pattern->operationName != online->operationName ||
      pattern->attributes != online->attributes ||
      pattern->operandHashes.size() != online->operandHashes.size())
    return false;
  for (auto [patternOperand, onlineOperand] :
       llvm::zip_equal(pattern->operandHashes, online->operandHashes))
    if (!matchTemplateGraph(patternOperand, onlineOperand, database,
                            onlineByHash, bindings))
      return false;
  return true;
}

static bool containsTemplateVariable(
    const RuleDatabase &database, llvm::StringRef graphHash,
    llvm::StringSet<> &visited) {
  if (!visited.insert(graphHash).second)
    return false;
  const PersistentGraphRecord *graph = database.lookupGraph(graphHash);
  if (!graph)
    return false;
  if (isTemplateVariable(*graph))
    return true;
  return llvm::any_of(graph->operandHashes, [&](llvm::StringRef operand) {
    return containsTemplateVariable(database, operand, visited);
  });
}

static bool collectTemplateVariables(
    const RuleDatabase &database, llvm::StringRef graphHash,
    llvm::StringSet<> &visited, llvm::StringSet<> &variables) {
  if (!visited.insert(graphHash).second)
    return true;
  const PersistentGraphRecord *graph = database.lookupGraph(graphHash);
  if (!graph)
    return false;
  if (isTemplateVariable(*graph)) {
    variables.insert(graphHash);
    return true;
  }
  for (llvm::StringRef operand : graph->operandHashes)
    if (!collectTemplateVariables(database, operand, visited, variables))
      return false;
  return true;
}

static bool candidateUsesOnlyPatternVariables(
    const RuleDatabase &database, llvm::StringRef patternHash,
    llvm::StringRef candidateHash) {
  llvm::StringSet<> patternVisited;
  llvm::StringSet<> patternVariables;
  llvm::StringSet<> candidateVisited;
  llvm::StringSet<> candidateVariables;
  if (!collectTemplateVariables(database, patternHash, patternVisited,
                                patternVariables) ||
      !collectTemplateVariables(database, candidateHash, candidateVisited,
                                candidateVariables))
    return false;
  return !patternVariables.empty() &&
         llvm::all_of(candidateVariables, [&](const auto &variable) {
           return patternVariables.contains(variable.getKey());
         });
}

static std::string proofPairKey(llvm::StringRef lhs, llvm::StringRef rhs) {
  if (rhs < lhs)
    std::swap(lhs, rhs);
  return (llvm::Twine(lhs) + "\n" + rhs).str();
}

struct ProofEdgeIndex {
  llvm::StringMap<llvm::SmallVector<const PersistentProofEdge *, 1>> byPair;
  llvm::StringMap<llvm::SmallVector<const PersistentProofEdge *, 1>> byGraph;
};

static ProofEdgeIndex
buildProofEdgeIndex(const PersistentEquivalenceClassStore &store) {
  ProofEdgeIndex index;
  for (const PersistentProofEdge &edge : store.proofEdges) {
    index.byPair[proofPairKey(edge.lhsGraphHash, edge.rhsGraphHash)]
        .push_back(&edge);
    index.byGraph[edge.lhsGraphHash].push_back(&edge);
    if (edge.rhsGraphHash != edge.lhsGraphHash)
      index.byGraph[edge.rhsGraphHash].push_back(&edge);
  }
  return index;
}

static std::vector<std::string> directProofHashes(
    const ProofEdgeIndex &index, llvm::StringRef lhs, llvm::StringRef rhs) {
  std::vector<std::string> hashes;
  auto found = index.byPair.find(proofPairKey(lhs, rhs));
  if (found == index.byPair.end())
    return hashes;
  for (const PersistentProofEdge *edge : found->second) {
    if (edge->proofHash.empty() || edge->proofCertificate.empty())
      continue;
    llvm::StringSet<> semanticKeys;
    std::string certificateError;
    if (!collectCertificateSemanticKeys(edge->proofCertificate, semanticKeys,
                                        certificateError))
      continue;
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse(edge->proofCertificate);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      continue;
    }
    const llvm::json::Object *object = parsed->getAsObject();
    std::optional<llvm::StringRef> stableHash =
        object ? object->getString("stable_hash") : std::nullopt;
    if (!object || object->getInteger("version") != 2 ||
        !stableHash || *stableHash != edge->proofHash)
      continue;
    hashes.push_back(edge->proofHash);
  }
  llvm::sort(hashes);
  hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  return hashes;
}

static std::vector<std::string> quantifiedProofHashes(
    const ProofEdgeIndex &index, llvm::StringRef lhs, llvm::StringRef rhs) {
  std::vector<std::string> hashes;
  auto found = index.byPair.find(proofPairKey(lhs, rhs));
  if (found == index.byPair.end())
    return hashes;
  for (const PersistentProofEdge *edge : found->second)
    if (certificateUsesQuantifiedTheorem(edge->proofCertificate))
      hashes.push_back(edge->proofHash);
  llvm::sort(hashes);
  hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  return hashes;
}

static bool templateLess(const RuleDatabase &database,
                         const RuleDatabaseTemplateEntry &lhs,
                         const RuleDatabaseTemplateEntry &rhs) {
  const PersistentGraphRecord *lhsRoot =
      database.lookupGraph(lhs.lhsPatternGraphHash);
  const PersistentGraphRecord *rhsRoot =
      database.lookupGraph(rhs.lhsPatternGraphHash);
  llvm::StringRef lhsOperation =
      lhsRoot ? llvm::StringRef(lhsRoot->operationName) : llvm::StringRef();
  llvm::StringRef rhsOperation =
      rhsRoot ? llvm::StringRef(rhsRoot->operationName) : llvm::StringRef();
  return std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>{
             lhsOperation, lhs.resultType, lhs.lhsPatternGraphHash} <
         std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>{
             rhsOperation, rhs.resultType, rhs.lhsPatternGraphHash};
}

} // namespace

std::string mlir::triton::cfg::getRuleDatabaseSemanticKeyDigest(
    llvm::StringRef semanticKey) {
  constexpr llvm::StringLiteral prefix = "sha256:";
  if (semanticKey.starts_with(prefix) &&
      semanticKey.size() == prefix.size() + 64 &&
      llvm::all_of(semanticKey.drop_front(prefix.size()), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f');
      }))
    return semanticKey.str();

  llvm::SHA256 hash;
  hash.update("ags-rule-db-semantic-key-v1\n");
  hash.update(semanticKey);
  std::array<uint8_t, 32> digest = hash.final();
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(prefix);
  result.reserve(prefix.size() + digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

std::string RuleDatabase::getStableHash() const {
  StableHashAccumulator accumulator;
  auto append = [&](auto &&...values) {
    std::string fragment;
    llvm::raw_string_ostream stream(fragment);
    (stream << ... << values);
    accumulator.update(fragment);
  };
  append("ags-rule-db-v", schemaVersion, '\n', eclassStoreHash, '\n',
         universeHash, '\n', semanticsHash, '\n', costContextKey, '\n',
         costModelName, '\n', costModelHash, '\n');
  for (const PersistentGraphRecord &graph : graphs) {
    append(graph.graphHash, '\n', graph.operationName, '\n', graph.resultType,
           '\n', graph.attributes, '\n', graph.operatorSize, '\n');
    for (llvm::StringRef operand : graph.operandHashes)
      append(operand, '\n');
  }
  for (const RuleDatabaseEntry &entry : entries)
    accumulator.update(entryContent(entry));
  for (const RuleDatabaseTemplateEntry &entry : templates) {
    accumulator.update("template\n");
    accumulator.update(templateContent(entry));
  }
  for (const RuleDatabaseSemanticEntry &entry : semanticEntries) {
    accumulator.update("semantic\n");
    accumulator.update(semanticEntryContent(entry));
  }
  return accumulator.finish();
}

void RuleDatabase::writeJSON(llvm::raw_ostream &output) const {
  auto writeString = [&](StringRef value) {
    output << llvm::formatv("{0}", llvm::json::Value(value));
  };
  auto entryToJSON = [](const RuleDatabaseEntry &entry) {
    llvm::json::Array proofHashes;
    for (llvm::StringRef proof : entry.proofHashes)
      proofHashes.push_back(proof);
    llvm::json::Array candidates;
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      llvm::json::Object object;
      object["root_graph_hash"] = candidate.rootGraphHash;
      object["cost"] = costToJSON(candidate.cost);
      candidates.push_back(std::move(object));
    }
    llvm::json::Object object;
    object["lhs_graph_hash"] = entry.lhsGraphHash;
    object["class_hash"] = entry.classHash;
    object["result_type"] = entry.resultType;
    object["proof_hashes"] = std::move(proofHashes);
    object["source_cost"] = costToJSON(entry.sourceCost);
    object["candidates"] = std::move(candidates);
    return object;
  };
  auto templateToJSON = [](const RuleDatabaseTemplateEntry &entry) {
    llvm::json::Array proofHashes;
    for (llvm::StringRef proof : entry.proofHashes)
      proofHashes.push_back(proof);
    llvm::json::Array candidates;
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      llvm::json::Object object;
      object["root_graph_hash"] = candidate.rootGraphHash;
      object["cost"] = costToJSON(candidate.cost);
      candidates.push_back(std::move(object));
    }
    llvm::json::Object object;
    object["lhs_pattern_graph_hash"] = entry.lhsPatternGraphHash;
    object["class_hash"] = entry.classHash;
    object["result_type"] = entry.resultType;
    object["proof_hashes"] = std::move(proofHashes);
    object["source_cost"] = costToJSON(entry.sourceCost);
    object["candidates"] = std::move(candidates);
    return object;
  };
  auto semanticEntryToJSON = [](const RuleDatabaseSemanticEntry &entry) {
    llvm::json::Array proofHashes;
    for (llvm::StringRef proof : entry.proofHashes)
      proofHashes.push_back(proof);
    llvm::json::Array candidates;
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      llvm::json::Object object;
      object["root_graph_hash"] = candidate.rootGraphHash;
      object["cost"] = costToJSON(candidate.cost);
      candidates.push_back(std::move(object));
    }
    llvm::json::Object object;
    object["semantic_key"] = entry.semanticKey;
    object["class_hash"] = entry.classHash;
    object["result_type"] = entry.resultType;
    object["proof_hashes"] = std::move(proofHashes);
    object["candidates"] = std::move(candidates);
    return object;
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
         << ",\n  \"database_hash\": ";
  writeString(getStableHash());
  output << ",\n  \"eclass_store_hash\": ";
  writeString(eclassStoreHash);
  output << ",\n  \"universe_hash\": ";
  writeString(universeHash);
  output << ",\n  \"semantics_hash\": ";
  writeString(semanticsHash);
  output << ",\n  \"cost_context_key\": ";
  writeString(costContextKey);
  output << ",\n  \"cost_model_name\": ";
  writeString(costModelName);
  output << ",\n  \"cost_model_hash\": ";
  writeString(costModelHash);
  output << ",\n  \"graph_count\": " << graphs.size()
         << ",\n  \"entry_count\": " << entries.size()
         << ",\n  \"template_count\": " << templates.size()
         << ",\n  \"semantic_entry_count\": " << semanticEntries.size()
         << ",\n";
  writeArray("graphs", graphs,
             [](const PersistentGraphRecord &graph) {
               return graphToJSON(graph);
             });
  output << ",\n";
  writeArray("entries", entries, entryToJSON);
  output << ",\n";
  writeArray("templates", templates, templateToJSON);
  output << ",\n";
  writeArray("semantic_entries", semanticEntries, semanticEntryToJSON);
  output << "\n}\n";
}

std::string RuleDatabase::toJSON() const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream);
  return storage;
}

const RuleDatabaseEntry *
RuleDatabase::lookup(llvm::StringRef lhsGraphHash) const {
  auto found = llvm::lower_bound(
      entries, lhsGraphHash,
      [](const RuleDatabaseEntry &entry, llvm::StringRef hash) {
        return entry.lhsGraphHash < hash;
      });
  return found != entries.end() && found->lhsGraphHash == lhsGraphHash
             ? &*found
             : nullptr;
}

std::vector<const RuleDatabaseSemanticEntry *>
RuleDatabase::lookupSemantic(llvm::StringRef semanticKey,
                             llvm::StringRef resultType) const {
  std::string digest = getRuleDatabaseSemanticKeyDigest(semanticKey);
  auto first = llvm::lower_bound(
      semanticEntries, digest,
      [](const RuleDatabaseSemanticEntry &entry, llvm::StringRef key) {
        return entry.semanticKey < key;
      });
  std::vector<const RuleDatabaseSemanticEntry *> matches;
  for (auto current = first;
       current != semanticEntries.end() &&
       current->semanticKey == digest;
       ++current)
    if (current->resultType == resultType)
      matches.push_back(&*current);
  return matches;
}

const PersistentGraphRecord *
RuleDatabase::lookupGraph(llvm::StringRef graphHash) const {
  auto found = llvm::lower_bound(
      graphs, graphHash,
      [](const PersistentGraphRecord &graph, llvm::StringRef hash) {
        return graph.graphHash < hash;
      });
  return found != graphs.end() && found->graphHash == graphHash ? &*found
                                                                : nullptr;
}

std::vector<RuleDatabaseTemplateMatch> RuleDatabase::matchTemplates(
    llvm::StringRef onlineRootGraphHash,
    llvm::ArrayRef<PersistentGraphRecord> onlineGraphs) const {
  llvm::StringMap<const PersistentGraphRecord *> onlineByHash;
  for (const PersistentGraphRecord &graph : onlineGraphs)
    onlineByHash.try_emplace(graph.graphHash, &graph);
  auto onlineRoot = onlineByHash.find(onlineRootGraphHash);
  if (onlineRoot == onlineByHash.end())
    return {};
  const std::tuple<llvm::StringRef, llvm::StringRef> rootSignature =
      {onlineRoot->second->operationName, onlineRoot->second->resultType};
  auto begin = std::lower_bound(
      templates.begin(), templates.end(), rootSignature,
      [&](const RuleDatabaseTemplateEntry &entry, const auto &signature) {
        const PersistentGraphRecord *pattern =
            lookupGraph(entry.lhsPatternGraphHash);
        return pattern &&
               std::tuple<llvm::StringRef, llvm::StringRef>{
                   pattern->operationName, entry.resultType} < signature;
      });
  std::vector<RuleDatabaseTemplateMatch> matches;
  for (auto current = begin; current != templates.end(); ++current) {
    const RuleDatabaseTemplateEntry &entry = *current;
    const PersistentGraphRecord *pattern =
        lookupGraph(entry.lhsPatternGraphHash);
    if (!pattern ||
        std::tuple<llvm::StringRef, llvm::StringRef>{
            pattern->operationName, entry.resultType} != rootSignature)
      break;
    llvm::StringMap<std::string> bindings;
    if (!matchTemplateGraph(entry.lhsPatternGraphHash, onlineRootGraphHash,
                            *this, onlineByHash, bindings))
      continue;
    RuleDatabaseTemplateMatch match;
    match.entry = &entry;
    for (const auto &binding : bindings)
      match.bindings.emplace_back(binding.getKey().str(), binding.getValue());
    llvm::sort(match.bindings);
    matches.push_back(std::move(match));
  }
  return matches;
}

std::optional<RuleDatabase> RuleDatabase::compile(
    const PersistentEquivalenceClassStore &store,
    const ParetoExtractionResult &pareto, std::string &error,
    llvm::ArrayRef<CertifiedSemanticGraph> certifiedSemanticGraphs,
    llvm::StringRef prevalidatedStoreHash) {
  if (!store.isComplete()) {
    error = "cannot compile a rule database from an incomplete e-class store";
    return std::nullopt;
  }
  if (!pareto.hasStructuralCostForEveryGraph()) {
    error = "Pareto report has graphs without a valid structural cost";
    return std::nullopt;
  }
  const std::string computedStoreHash =
      prevalidatedStoreHash.empty() ? store.getStableHash() : std::string();
  const llvm::StringRef storeHash =
      prevalidatedStoreHash.empty() ? llvm::StringRef(computedStoreHash)
                                    : prevalidatedStoreHash;
  if (pareto.eclassStoreHash != storeHash) {
    error = "Pareto report and e-class store hash mismatch";
    return std::nullopt;
  }
  bool paretoClassesAligned =
      pareto.classes.size() == store.classes.size();
  if (paretoClassesAligned)
    for (size_t index = 0; index < pareto.classes.size(); ++index)
      if (pareto.classes[index].classHash != store.classes[index].classHash) {
        paretoClassesAligned = false;
        break;
      }
  llvm::StringMap<const PersistentEquivalenceClass *> classes;
  if (!paretoClassesAligned)
    for (const PersistentEquivalenceClass &eclass : store.classes)
      classes.try_emplace(eclass.classHash, &eclass);
  llvm::StringMap<const PersistentGraphRecord *> graphByHash;
  for (const PersistentGraphRecord &graph : store.graphs) {
    if (graph.graphHash != getPersistentGraphHash(graph)) {
      error = "e-class store contains a non-canonical graph hash";
      return std::nullopt;
    }
    graphByHash.try_emplace(graph.graphHash, &graph);
  }
  const ProofEdgeIndex proofEdgeIndex = buildProofEdgeIndex(store);
  llvm::StringMap<const CertifiedSemanticGraph *> certifiedByGraph;
  for (const CertifiedSemanticGraph &certified : certifiedSemanticGraphs) {
    if (!graphByHash.count(certified.graphHash) ||
        certified.semanticKey.empty() || certified.proofHash.empty()) {
      error = "certified semantic graph has an unknown graph or empty proof";
      return std::nullopt;
    }
    auto inserted = certifiedByGraph.try_emplace(certified.graphHash,
                                                  &certified);
    if (!inserted.second &&
        (inserted.first->second->semanticKey != certified.semanticKey ||
         inserted.first->second->proofHash != certified.proofHash)) {
      error = "conflicting semantic certifications for one graph";
      return std::nullopt;
    }
  }

  RuleDatabase database;
  database.eclassStoreHash = pareto.eclassStoreHash;
  database.universeHash = store.universeHash;
  database.semanticsHash = store.semanticsHash;
  database.costContextKey = pareto.costContextKey;
  database.costModelName = pareto.costModelName;
  database.costModelHash = pareto.costModelHash;
  database.graphs = store.graphs;
  auto graphHashLess = [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  };
  if (!llvm::is_sorted(database.graphs, graphHashLess))
    llvm::sort(database.graphs, graphHashLess);

  for (size_t classIndex = 0; classIndex < pareto.classes.size();
       ++classIndex) {
    const ParetoEquivalenceClass &selectedClass = pareto.classes[classIndex];
    const PersistentEquivalenceClass *eclass = nullptr;
    if (paretoClassesAligned) {
      eclass = &store.classes[classIndex];
    } else {
      auto knownClass = classes.find(selectedClass.classHash);
      if (knownClass != classes.end())
        eclass = knownClass->second;
    }
    if (!eclass || selectedClass.candidates.empty()) {
      error = "Pareto report references an unknown or empty e-class";
      return std::nullopt;
    }
    if (eclass->memberGraphHashes.empty()) {
      error = "equivalence class has no member graphs";
      return std::nullopt;
    }

    // More than half a million classes in the complete depth-one universe are
    // singleton identities. They cannot publish an exact/template rewrite,
    // so validate and emit their certified semantic entry without allocating
    // per-class StringSet/StringMap containers or repeatedly binary-searching
    // the global graph table.
    if (eclass->memberGraphHashes.size() == 1) {
      llvm::StringRef member = eclass->memberGraphHashes.front();
      auto knownGraph = graphByHash.find(member);
      if (knownGraph == graphByHash.end() ||
          knownGraph->second->resultType != eclass->resultType) {
        error = "e-class member result type does not match its class";
        return std::nullopt;
      }
      if (selectedClass.allCandidates.size() != 1 ||
          selectedClass.allCandidates.front().graphHash != member) {
        error = "Pareto report does not retain costs for every e-class "
                "member";
        return std::nullopt;
      }
      for (const ParetoCandidateRecord &candidate :
           selectedClass.candidates)
        if (candidate.graphHash != member) {
          error = "Pareto candidate is not a member of its certified e-class";
          return std::nullopt;
        }
      auto certified = certifiedByGraph.find(member);
      if (certified == certifiedByGraph.end())
        continue;
      RuleDatabaseSemanticEntry semanticEntry;
      semanticEntry.semanticKey = getRuleDatabaseSemanticKeyDigest(
          certified->second->semanticKey);
      semanticEntry.classHash = eclass->classHash;
      semanticEntry.resultType = eclass->resultType;
      semanticEntry.proofHashes = {certified->second->proofHash};
      for (const ParetoCandidateRecord &candidate :
           selectedClass.candidates)
        if (!hasCompilerAction(*knownGraph->second))
          semanticEntry.candidates.push_back(
              {candidate.graphHash, candidate.cost});
      if (!semanticEntry.candidates.empty())
        database.semanticEntries.push_back(std::move(semanticEntry));
      continue;
    }

    llvm::StringSet<> members;
    for (llvm::StringRef member : eclass->memberGraphHashes) {
      auto graph = graphByHash.find(member);
      if (graph == graphByHash.end() ||
          graph->second->resultType != eclass->resultType) {
        error = "e-class member result type does not match its class";
        return std::nullopt;
      }
      members.insert(member);
    }
    for (const ParetoCandidateRecord &candidate : selectedClass.candidates)
      if (!members.contains(candidate.graphHash)) {
        error = "Pareto candidate is not a member of its certified e-class";
        return std::nullopt;
      }
    llvm::StringMap<const ParetoCandidateCost *> costsByGraph;
    for (const ParetoCandidateRecord &candidate :
         selectedClass.allCandidates)
      costsByGraph.try_emplace(candidate.graphHash, &candidate.cost);
    if (costsByGraph.size() != eclass->memberGraphHashes.size()) {
      error = "Pareto report does not retain costs for every e-class member";
      return std::nullopt;
    }

    std::vector<std::string> proofHashes;
    llvm::StringSet<> semanticKeys;
    // A singleton class contains no discovered exact rewrite. Publishing its
    // reflexive identity bloats the production artifact and makes exact
    // lookup return a no-op candidate. A separately certified semantic key is
    // still executable knowledge, however, and is compiled below.
    for (llvm::StringRef member : eclass->memberGraphHashes) {
      auto incident = proofEdgeIndex.byGraph.find(member);
      if (incident == proofEdgeIndex.byGraph.end())
        continue;
      for (const PersistentProofEdge *edge : incident->second) {
        // Every edge is indexed by both endpoints. Process it once at the
        // lexicographically smaller endpoint while retaining the fail-closed
        // class-membership check used by the original implementation.
        if (member != std::min(llvm::StringRef(edge->lhsGraphHash),
                               llvm::StringRef(edge->rhsGraphHash)) ||
            !members.contains(edge->lhsGraphHash) ||
            !members.contains(edge->rhsGraphHash))
          continue;
        proofHashes.push_back(edge->proofHash);
        // Persistent stores loaded from disk have already replay-validated
        // every certificate.  Hand-constructed API clients may still carry a
        // legacy certificate without v2 semantic keys; keep exact lookup for
        // those classes but deliberately omit the generalized semantic index.
        std::string semanticKeyError;
        (void)collectCertificateSemanticKeys(edge->proofCertificate,
                                             semanticKeys,
                                             semanticKeyError);
      }
    }
    for (StringRef member : eclass->memberGraphHashes) {
      auto certified = certifiedByGraph.find(member);
      if (certified == certifiedByGraph.end())
        continue;
      semanticKeys.insert(certified->second->semanticKey);
      proofHashes.push_back(certified->second->proofHash);
    }
    llvm::sort(proofHashes);
    proofHashes.erase(std::unique(proofHashes.begin(), proofHashes.end()),
                      proofHashes.end());

    const std::vector<std::string> exactProofReferences =
        compactProofReferences(proofHashes);
    std::vector<RuleDatabaseCandidate> selectedCandidates;
    llvm::StringSet<> selectedCandidateHashes;
    for (const ParetoCandidateRecord &candidate :
         selectedClass.candidates) {
      selectedCandidates.push_back({candidate.graphHash, candidate.cost});
      selectedCandidateHashes.insert(candidate.graphHash);
    }
    llvm::sort(selectedCandidates, [](const auto &lhs, const auto &rhs) {
      return lhs.rootGraphHash < rhs.rootGraphHash;
    });

    for (llvm::StringRef member : eclass->memberGraphHashes) {
      if (!graphByHash.count(member)) {
        error = "e-class references a graph absent from the graph table";
        return std::nullopt;
      }
      RuleDatabaseEntry entry;
      entry.lhsGraphHash = member.str();
      entry.classHash = eclass->classHash;
      entry.resultType = eclass->resultType;
      entry.proofHashes = exactProofReferences;
      auto sourceCost = costsByGraph.find(member);
      if (sourceCost == costsByGraph.end()) {
        error = "Pareto report is missing the LHS source cost";
        return std::nullopt;
      }
      entry.sourceCost = *sourceCost->second;
      entry.candidates = selectedCandidates;

      // Generalize a pure graph only over the exact typed free symbols in a
      // direct replay-validated proof. Compiler actions remain stricter and
      // require an explicit quantified theorem step in that certificate.
      llvm::StringSet<> visitedPattern;
      // A variable leaf is a hole, not a rewrite pattern.  Emitting it as an
      // LHS would match every graph of the same type and turn a proved
      // simplification into an unbounded reverse expansion (x -> x - 0).
      auto patternGraph = graphByHash.find(member);
      const PersistentGraphRecord *patternRoot =
          patternGraph == graphByHash.end() ? nullptr : patternGraph->second;
      if (patternRoot && !isTemplateVariable(*patternRoot) &&
          containsTemplateVariable(database, member, visitedPattern)) {
        RuleDatabaseTemplateEntry templateEntry;
        templateEntry.lhsPatternGraphHash = member.str();
        templateEntry.classHash = eclass->classHash;
        templateEntry.resultType = eclass->resultType;
        templateEntry.sourceCost = entry.sourceCost;
        // Pure symbolic candidates are re-costed after their typed holes are
        // bound online. Keep every proved class member here, including an
        // offline-dominated layout that can become optimal for a concrete
        // affine address graph. Compiler actions have no such substitution
        // semantics and remain restricted to the offline Pareto frontier.
        llvm::StringSet<> emittedTemplateCandidates;
        auto incident = proofEdgeIndex.byGraph.find(member);
        for (const PersistentProofEdge *edge :
             incident == proofEdgeIndex.byGraph.end()
                 ? llvm::ArrayRef<const PersistentProofEdge *>()
                 : llvm::ArrayRef<const PersistentProofEdge *>(
                       incident->second)) {
          llvm::StringRef candidateHash =
              edge->lhsGraphHash == member ? edge->rhsGraphHash
                                           : edge->lhsGraphHash;
          if (!emittedTemplateCandidates.insert(candidateHash).second)
            continue;
          auto candidateCost = costsByGraph.find(candidateHash);
          if (candidateCost == costsByGraph.end())
            continue;
          RuleDatabaseCandidate candidate{candidateHash.str(),
                                          *candidateCost->second};
          auto knownCandidate = graphByHash.find(candidate.rootGraphHash);
          const PersistentGraphRecord *candidateGraph =
              knownCandidate == graphByHash.end() ? nullptr
                                                  : knownCandidate->second;
          if (!candidateGraph ||
              !candidateUsesOnlyPatternVariables(database, member,
                                                 candidate.rootGraphHash))
            continue;
          if (hasCompilerAction(*candidateGraph) &&
              !selectedCandidateHashes.contains(candidate.rootGraphHash))
            continue;
          std::vector<std::string> proofHashes =
              hasCompilerAction(*candidateGraph)
                  ? quantifiedProofHashes(proofEdgeIndex, member,
                                          candidate.rootGraphHash)
                  : directProofHashes(proofEdgeIndex, member,
                                      candidate.rootGraphHash);
          if (proofHashes.empty())
            continue;
          templateEntry.proofHashes.insert(templateEntry.proofHashes.end(),
                                           proofHashes.begin(),
                                           proofHashes.end());
          templateEntry.candidates.push_back(candidate);
        }
        if (!templateEntry.candidates.empty()) {
          llvm::sort(templateEntry.proofHashes);
          templateEntry.proofHashes.erase(
              std::unique(templateEntry.proofHashes.begin(),
                          templateEntry.proofHashes.end()),
              templateEntry.proofHashes.end());
          database.templates.push_back(std::move(templateEntry));
        }
      }
      const bool hasExactAlternative = llvm::any_of(
          selectedCandidates, [&](const RuleDatabaseCandidate &candidate) {
            return candidate.rootGraphHash != member;
          });
      if (hasExactAlternative)
        database.entries.push_back(std::move(entry));
    }

    // A semantic-normal-form entry is an additional index over the same
    // replay-validated e-class.  It does not invent a new equality: every key
    // is copied from a proof certificate endpoint, and every RHS remains one
    // of the class's Pareto candidates.  Compiler actions stay on the stricter
    // quantified-template path because their online preflight is part of the
    // theorem instantiation contract.
    for (const auto &semanticKey : semanticKeys) {
      RuleDatabaseSemanticEntry semanticEntry;
      semanticEntry.semanticKey =
          getRuleDatabaseSemanticKeyDigest(semanticKey.getKey());
      semanticEntry.classHash = eclass->classHash;
      semanticEntry.resultType = eclass->resultType;
      semanticEntry.proofHashes = proofHashes;
      for (const ParetoCandidateRecord &candidate :
           selectedClass.candidates) {
        auto knownCandidate = graphByHash.find(candidate.graphHash);
        const PersistentGraphRecord *candidateGraph =
            knownCandidate == graphByHash.end() ? nullptr
                                                : knownCandidate->second;
        if (!candidateGraph || hasCompilerAction(*candidateGraph))
          continue;
        semanticEntry.candidates.push_back(
            {candidate.graphHash, candidate.cost});
      }
      if (!semanticEntry.candidates.empty()) {
        llvm::sort(semanticEntry.candidates,
                   [](const auto &lhs, const auto &rhs) {
                     return lhs.rootGraphHash < rhs.rootGraphHash;
                   });
        database.semanticEntries.push_back(std::move(semanticEntry));
      }
    }
  }
  llvm::sort(database.entries, [](const auto &lhs, const auto &rhs) {
    return lhs.lhsGraphHash < rhs.lhsGraphHash;
  });
  for (size_t index = 1; index < database.entries.size(); ++index)
    if (database.entries[index - 1].lhsGraphHash ==
        database.entries[index].lhsGraphHash) {
      error = "duplicate canonical LHS in rule database";
      return std::nullopt;
    }
  llvm::sort(database.templates, [&](const auto &lhs, const auto &rhs) {
    return templateLess(database, lhs, rhs);
  });
  llvm::sort(database.semanticEntries,
             [](const auto &lhs, const auto &rhs) {
               return std::tie(lhs.semanticKey, lhs.resultType,
                               lhs.classHash) <
                      std::tie(rhs.semanticKey, rhs.resultType,
                               rhs.classHash);
             });
  // The proof store is intentionally sparse: exact denotation groups keep a
  // spanning forest instead of every transitive equality. Consequently, two
  // persistent e-classes can still publish the same certified Value+Valid
  // normal form. The shard merger already joins that case; do the same for a
  // monolithic build so artifact contents do not depend on partitioning.
  // Equality is not inferred from a fingerprint here: both entries carry the
  // same versioned normalized-denotation digest and replay-backed proof
  // references. Their target/type/cost context is shared by this database.
  std::vector<RuleDatabaseSemanticEntry> coalescedSemanticEntries;
  coalescedSemanticEntries.reserve(database.semanticEntries.size());
  for (RuleDatabaseSemanticEntry &entry : database.semanticEntries) {
    if (coalescedSemanticEntries.empty() ||
        coalescedSemanticEntries.back().semanticKey != entry.semanticKey ||
        coalescedSemanticEntries.back().resultType != entry.resultType) {
      coalescedSemanticEntries.push_back(std::move(entry));
      continue;
    }
    RuleDatabaseSemanticEntry &existing =
        coalescedSemanticEntries.back();
    existing.classHash =
        "semantic:" +
        stableHash(existing.classHash + "\n" + entry.classHash);
    existing.proofHashes.insert(existing.proofHashes.end(),
                                entry.proofHashes.begin(),
                                entry.proofHashes.end());
    for (RuleDatabaseCandidate &candidate : entry.candidates) {
      auto duplicate = llvm::find_if(
          existing.candidates,
          [&](const RuleDatabaseCandidate &known) {
            return known.rootGraphHash == candidate.rootGraphHash;
          });
      if (duplicate == existing.candidates.end()) {
        existing.candidates.push_back(std::move(candidate));
        continue;
      }
      if (!sameCost(duplicate->cost, candidate.cost)) {
        error = "semantic-equivalent classes disagree on candidate cost for '" +
                candidate.rootGraphHash + "'";
        return std::nullopt;
      }
    }
  }
  for (RuleDatabaseSemanticEntry &entry : coalescedSemanticEntries) {
    llvm::sort(entry.proofHashes);
    entry.proofHashes.erase(
        std::unique(entry.proofHashes.begin(), entry.proofHashes.end()),
        entry.proofHashes.end());
    llvm::sort(entry.candidates, [](const auto &lhs, const auto &rhs) {
      return lhs.rootGraphHash < rhs.rootGraphHash;
    });
  }
  database.semanticEntries = std::move(coalescedSemanticEntries);

  // The e-class store is the complete audit artifact; the RuleDB is the
  // executable optimization projection. Retain only graph records reachable
  // from an exact/template LHS or any published RHS. This preserves every
  // materialization operand while omitting hundreds of thousands of
  // unrelated singleton identities.
  llvm::StringSet<> requiredGraphs;
  SmallVector<std::string, 256> graphWorklist;
  auto requireGraph = [&](llvm::StringRef graphHash) {
    if (requiredGraphs.insert(graphHash).second)
      graphWorklist.push_back(graphHash.str());
  };
  for (const RuleDatabaseEntry &entry : database.entries) {
    requireGraph(entry.lhsGraphHash);
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      requireGraph(candidate.rootGraphHash);
  }
  for (const RuleDatabaseTemplateEntry &entry : database.templates) {
    requireGraph(entry.lhsPatternGraphHash);
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      requireGraph(candidate.rootGraphHash);
  }
  for (const RuleDatabaseSemanticEntry &entry : database.semanticEntries)
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      requireGraph(candidate.rootGraphHash);
  while (!graphWorklist.empty()) {
    std::string graphHash = std::move(graphWorklist.back());
    graphWorklist.pop_back();
    auto graph = graphByHash.find(graphHash);
    if (graph == graphByHash.end()) {
      error = "rule database projection references an unknown graph";
      return std::nullopt;
    }
    for (llvm::StringRef operand : graph->second->operandHashes)
      requireGraph(operand);
  }
  std::vector<PersistentGraphRecord> projectedGraphs;
  projectedGraphs.reserve(requiredGraphs.size());
  for (PersistentGraphRecord &graph : database.graphs)
    if (requiredGraphs.contains(graph.graphHash))
      projectedGraphs.push_back(std::move(graph));
  database.graphs = std::move(projectedGraphs);
  return database;
}

std::optional<RuleDatabase> RuleDatabase::quotientFinalLayerBySemanticKey(
    const RuleDatabase &input,
    llvm::ArrayRef<CertifiedSemanticGraph> certifiedSemanticGraphs,
    uint64_t &prunedGraphCount, uint64_t &prunedEntryCount,
    std::string &error) {
  prunedGraphCount = 0;
  prunedEntryCount = 0;
  RuleDatabase database = input;
  if (database.graphs.empty())
    return database;

  llvm::StringMap<const PersistentGraphRecord *> graphByHash;
  unsigned maximumOperatorSize = 0;
  for (const PersistentGraphRecord &graph : database.graphs) {
    graphByHash.try_emplace(graph.graphHash, &graph);
    maximumOperatorSize = std::max(maximumOperatorSize, graph.operatorSize);
  }

  llvm::StringMap<const CertifiedSemanticGraph *> certifiedByGraph;
  for (const CertifiedSemanticGraph &certificate : certifiedSemanticGraphs) {
    if (!graphByHash.count(certificate.graphHash) ||
        certificate.semanticKey.empty() || certificate.proofHash.empty())
      continue;
    auto inserted = certifiedByGraph.try_emplace(certificate.graphHash,
                                                  &certificate);
    if (!inserted.second &&
        (inserted.first->second->semanticKey != certificate.semanticKey ||
         inserted.first->second->proofHash != certificate.proofHash)) {
      error = "conflicting semantic certifications during final-layer "
              "quotient";
      return std::nullopt;
    }
  }

  // A maximal graph can still be a zero-cost/shared operand in a finite
  // universe. Such a node is not a root and cannot be removed without an
  // alias rewrite, so only unreferenced maximal roots are eligible.
  llvm::StringSet<> referencedAsOperand;
  for (const PersistentGraphRecord &graph : database.graphs)
    for (llvm::StringRef operand : graph.operandHashes)
      referencedAsOperand.insert(operand);

  llvm::StringSet<> omitted;
  llvm::StringMap<std::string> representativeBySemanticKey;
  for (const PersistentGraphRecord &graph : database.graphs) {
    if (graph.operatorSize != maximumOperatorSize ||
        referencedAsOperand.contains(graph.graphHash) ||
        hasCompilerAction(graph))
      continue;
    auto certificate = certifiedByGraph.find(graph.graphHash);
    if (certificate == certifiedByGraph.end())
      continue;
    std::string semanticBucket = graph.resultType + "\n" +
                                 std::to_string(
                                     static_cast<unsigned>(graph.effectKind)) +
                                 "\n" + certificate->second->semanticKey;
    auto known = representativeBySemanticKey.find(semanticBucket);
    if (known == representativeBySemanticKey.end()) {
      representativeBySemanticKey.try_emplace(semanticBucket,
                                              graph.graphHash);
      continue;
    }
    if (graph.graphHash < known->second) {
      omitted.insert(known->second);
      known->second = graph.graphHash;
    } else {
      omitted.insert(graph.graphHash);
    }
  }
  prunedGraphCount = omitted.size();
  if (omitted.empty())
    return database;

  // Fail closed if a future graph format violates the maximal-root
  // invariant. The quotient must never leave a retained graph with a missing
  // operand record.
  for (const PersistentGraphRecord &graph : database.graphs) {
    if (omitted.contains(graph.graphHash))
      continue;
    for (llvm::StringRef operand : graph.operandHashes)
      if (omitted.contains(operand)) {
        error = "final-layer semantic quotient would remove a graph operand";
        return std::nullopt;
      }
  }

  std::vector<PersistentGraphRecord> retainedGraphs;
  retainedGraphs.reserve(database.graphs.size() - omitted.size());
  for (PersistentGraphRecord &graph : database.graphs)
    if (!omitted.contains(graph.graphHash))
      retainedGraphs.push_back(std::move(graph));
  database.graphs = std::move(retainedGraphs);

  std::vector<RuleDatabaseEntry> retainedEntries;
  retainedEntries.reserve(database.entries.size());
  for (RuleDatabaseEntry &entry : database.entries) {
    if (omitted.contains(entry.lhsGraphHash)) {
      ++prunedEntryCount;
      continue;
    }
    llvm::erase_if(entry.candidates, [&](const RuleDatabaseCandidate &candidate) {
      return omitted.contains(candidate.rootGraphHash);
    });
    if (entry.candidates.empty()) {
      ++prunedEntryCount;
      continue;
    }
    retainedEntries.push_back(std::move(entry));
  }
  database.entries = std::move(retainedEntries);

  std::vector<RuleDatabaseTemplateEntry> retainedTemplates;
  retainedTemplates.reserve(database.templates.size());
  for (RuleDatabaseTemplateEntry &entry : database.templates) {
    if (omitted.contains(entry.lhsPatternGraphHash)) {
      ++prunedEntryCount;
      continue;
    }
    llvm::erase_if(entry.candidates, [&](const RuleDatabaseCandidate &candidate) {
      return omitted.contains(candidate.rootGraphHash);
    });
    if (entry.candidates.empty()) {
      ++prunedEntryCount;
      continue;
    }
    retainedTemplates.push_back(std::move(entry));
  }
  database.templates = std::move(retainedTemplates);

  std::vector<RuleDatabaseSemanticEntry> retainedSemanticEntries;
  retainedSemanticEntries.reserve(database.semanticEntries.size());
  for (RuleDatabaseSemanticEntry &entry : database.semanticEntries) {
    llvm::erase_if(entry.candidates, [&](const RuleDatabaseCandidate &candidate) {
      return omitted.contains(candidate.rootGraphHash);
    });
    if (entry.candidates.empty())
      continue;
    retainedSemanticEntries.push_back(std::move(entry));
  }
  database.semanticEntries = std::move(retainedSemanticEntries);

  // Reparse the generated artifact so all graph/candidate references and
  // stable ordering invariants are checked by the same loader used online.
  std::optional<RuleDatabase> validated =
      RuleDatabase::fromJSON(database.toJSON(), error);
  if (!validated)
    return std::nullopt;
  return validated;
}

std::optional<RuleDatabase>
RuleDatabase::merge(llvm::ArrayRef<RuleDatabase> shards, std::string &error) {
  if (shards.empty()) {
    error = "cannot merge an empty AGS rule database shard list";
    return std::nullopt;
  }

  SmallVector<std::pair<const RuleDatabase *, std::string>, 8> ordered;
  ordered.reserve(shards.size());
  for (const RuleDatabase &shard : shards)
    ordered.emplace_back(&shard, shard.getStableHash());
  llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.first->universeHash, lhs.first->eclassStoreHash,
                    lhs.second) <
           std::tie(rhs.first->universeHash, rhs.first->eclassStoreHash,
                    rhs.second);
  });

  const RuleDatabase &first = *ordered.front().first;
  RuleDatabase merged;
  merged.schemaVersion = kSchemaVersion;
  merged.semanticsHash = first.semanticsHash;
  merged.costContextKey = first.costContextKey;
  merged.costModelName = first.costModelName;
  merged.costModelHash = first.costModelHash;

  llvm::StringMap<unsigned> graphIndex;
  llvm::StringMap<unsigned> entryIndex;
  llvm::StringMap<unsigned> templateIndex;
  llvm::StringMap<unsigned> semanticIndex;

  for (const auto &[shard, contentHash] : ordered) {
    (void)contentHash;
    if (shard->schemaVersion != kSchemaVersion ||
        shard->semanticsHash != merged.semanticsHash ||
        shard->costContextKey != merged.costContextKey ||
        shard->costModelName != merged.costModelName ||
        shard->costModelHash != merged.costModelHash) {
      error = "AGS rule database shards have incompatible schema, semantics, "
              "cost context, or cost model";
      return std::nullopt;
    }
    for (const PersistentGraphRecord &graph : shard->graphs) {
      auto found = graphIndex.find(graph.graphHash);
      if (found == graphIndex.end()) {
        graphIndex.try_emplace(graph.graphHash,
                               static_cast<unsigned>(merged.graphs.size()));
        merged.graphs.push_back(graph);
        continue;
      }
      if (!sameGraph(merged.graphs[found->second], graph)) {
        error = "AGS rule database shards contain conflicting graph hash '" +
                graph.graphHash + "'";
        return std::nullopt;
      }
    }

    for (const RuleDatabaseEntry &entry : shard->entries) {
      auto found = entryIndex.find(entry.lhsGraphHash);
      if (found == entryIndex.end()) {
        entryIndex.try_emplace(entry.lhsGraphHash,
                               static_cast<unsigned>(merged.entries.size()));
        merged.entries.push_back(entry);
        continue;
      }
      RuleDatabaseEntry &existing = merged.entries[found->second];
      if (existing.resultType != entry.resultType ||
          !sameCost(existing.sourceCost, entry.sourceCost)) {
        error = "AGS rule database shards disagree on source cost/type for '" +
                entry.lhsGraphHash + "'";
        return std::nullopt;
      }
      existing.classHash =
          "bundle:" + stableHash(existing.classHash + "\n" + entry.classHash);
      existing.proofHashes.insert(existing.proofHashes.end(),
                                  entry.proofHashes.begin(),
                                  entry.proofHashes.end());
      for (const RuleDatabaseCandidate &candidate : entry.candidates) {
        auto duplicate = llvm::find_if(
            existing.candidates,
            [&](const RuleDatabaseCandidate &known) {
              return known.rootGraphHash == candidate.rootGraphHash;
            });
        if (duplicate == existing.candidates.end()) {
          existing.candidates.push_back(candidate);
          continue;
        }
        if (!sameCost(duplicate->cost, candidate.cost)) {
          error = "AGS rule database shards disagree on candidate cost for '" +
                  candidate.rootGraphHash + "'";
          return std::nullopt;
        }
      }
    }

    for (const RuleDatabaseTemplateEntry &entry : shard->templates) {
      auto found = templateIndex.find(entry.lhsPatternGraphHash);
      if (found == templateIndex.end()) {
        templateIndex.try_emplace(
            entry.lhsPatternGraphHash,
            static_cast<unsigned>(merged.templates.size()));
        merged.templates.push_back(entry);
        continue;
      }
      RuleDatabaseTemplateEntry &existing = merged.templates[found->second];
      if (existing.resultType != entry.resultType ||
          !sameCost(existing.sourceCost, entry.sourceCost)) {
        error = "AGS rule database shards disagree on template source "
                "cost/type for '" +
                entry.lhsPatternGraphHash + "'";
        return std::nullopt;
      }
      existing.classHash =
          "bundle:" + stableHash(existing.classHash + "\n" + entry.classHash);
      existing.proofHashes.insert(existing.proofHashes.end(),
                                  entry.proofHashes.begin(),
                                  entry.proofHashes.end());
      for (const RuleDatabaseCandidate &candidate : entry.candidates) {
        auto duplicate = llvm::find_if(
            existing.candidates,
            [&](const RuleDatabaseCandidate &known) {
              return known.rootGraphHash == candidate.rootGraphHash;
            });
        if (duplicate == existing.candidates.end()) {
          existing.candidates.push_back(candidate);
          continue;
        }
        if (!sameCost(duplicate->cost, candidate.cost)) {
          error = "AGS rule database shards disagree on template candidate "
                  "cost for '" +
                  candidate.rootGraphHash + "'";
          return std::nullopt;
        }
      }
    }

    for (const RuleDatabaseSemanticEntry &entry :
         shard->semanticEntries) {
      std::string key = entry.resultType + "\n" + entry.semanticKey;
      auto found = semanticIndex.find(key);
      if (found == semanticIndex.end()) {
        semanticIndex.try_emplace(
            key, static_cast<unsigned>(merged.semanticEntries.size()));
        merged.semanticEntries.push_back(entry);
        continue;
      }
      RuleDatabaseSemanticEntry &existing =
          merged.semanticEntries[found->second];
      existing.classHash =
          "bundle:" + stableHash(existing.classHash + "\n" + entry.classHash);
      existing.proofHashes.insert(existing.proofHashes.end(),
                                  entry.proofHashes.begin(),
                                  entry.proofHashes.end());
      for (const RuleDatabaseCandidate &candidate : entry.candidates) {
        auto duplicate = llvm::find_if(
            existing.candidates,
            [&](const RuleDatabaseCandidate &known) {
              return known.rootGraphHash == candidate.rootGraphHash;
            });
        if (duplicate == existing.candidates.end()) {
          existing.candidates.push_back(candidate);
          continue;
        }
        if (!sameCost(duplicate->cost, candidate.cost)) {
          error = "AGS rule database shards disagree on semantic candidate "
                  "cost for '" +
                  candidate.rootGraphHash + "'";
          return std::nullopt;
        }
      }
    }
  }

  llvm::sort(merged.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  std::string universeIdentity = "ags-rule-db-bundle-universe-v2\n";
  for (const PersistentGraphRecord &graph : merged.graphs)
    universeIdentity += graph.graphHash + "\n";
  merged.universeHash = "bundle:" + stableHash(universeIdentity);
  for (RuleDatabaseEntry &entry : merged.entries) {
    llvm::sort(entry.proofHashes);
    entry.proofHashes.erase(
        std::unique(entry.proofHashes.begin(), entry.proofHashes.end()),
        entry.proofHashes.end());
    llvm::sort(entry.candidates, [](const auto &lhs, const auto &rhs) {
      return lhs.rootGraphHash < rhs.rootGraphHash;
    });
    std::string identity = "ags-rule-db-bundle-exact-class-v2\n";
    identity += entry.lhsGraphHash + "\n" + entry.resultType + "\n";
    for (const std::string &proof : entry.proofHashes)
      identity += proof + "\n";
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      identity += candidate.rootGraphHash + "\n";
    entry.classHash = "bundle:" + stableHash(identity);
  }
  llvm::sort(merged.entries, [](const auto &lhs, const auto &rhs) {
    return lhs.lhsGraphHash < rhs.lhsGraphHash;
  });
  for (RuleDatabaseTemplateEntry &entry : merged.templates) {
    llvm::sort(entry.proofHashes);
    entry.proofHashes.erase(
        std::unique(entry.proofHashes.begin(), entry.proofHashes.end()),
        entry.proofHashes.end());
    llvm::sort(entry.candidates, [](const auto &lhs, const auto &rhs) {
      return lhs.rootGraphHash < rhs.rootGraphHash;
    });
    std::string identity = "ags-rule-db-bundle-template-class-v2\n";
    identity +=
        entry.lhsPatternGraphHash + "\n" + entry.resultType + "\n";
    for (const std::string &proof : entry.proofHashes)
      identity += proof + "\n";
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      identity += candidate.rootGraphHash + "\n";
    entry.classHash = "bundle:" + stableHash(identity);
  }
  llvm::sort(merged.templates, [&](const auto &lhs, const auto &rhs) {
    return templateLess(merged, lhs, rhs);
  });
  for (RuleDatabaseSemanticEntry &entry : merged.semanticEntries) {
    llvm::sort(entry.proofHashes);
    entry.proofHashes.erase(
        std::unique(entry.proofHashes.begin(), entry.proofHashes.end()),
        entry.proofHashes.end());
    llvm::sort(entry.candidates, [](const auto &lhs, const auto &rhs) {
      return lhs.rootGraphHash < rhs.rootGraphHash;
    });
    std::string identity = "ags-rule-db-bundle-semantic-class-v2\n";
    identity += entry.semanticKey + "\n" + entry.resultType + "\n";
    for (const std::string &proof : entry.proofHashes)
      identity += proof + "\n";
    for (const RuleDatabaseCandidate &candidate : entry.candidates)
      identity += candidate.rootGraphHash + "\n";
    entry.classHash = "bundle:" + stableHash(identity);
  }
  llvm::sort(merged.semanticEntries,
             [](const auto &lhs, const auto &rhs) {
               return std::tie(lhs.semanticKey, lhs.resultType,
                               lhs.classHash) <
                      std::tie(rhs.semanticKey, rhs.resultType,
                               rhs.classHash);
             });
  std::string eclassIdentity = "ags-rule-db-bundle-eclasses-v2\n";
  for (const RuleDatabaseEntry &entry : merged.entries)
    eclassIdentity += "exact\n" + entry.lhsGraphHash + "\n" +
                      entry.classHash + "\n";
  for (const RuleDatabaseTemplateEntry &entry : merged.templates)
    eclassIdentity += "template\n" + entry.lhsPatternGraphHash + "\n" +
                      entry.classHash + "\n";
  for (const RuleDatabaseSemanticEntry &entry : merged.semanticEntries)
    eclassIdentity += "semantic\n" + entry.semanticKey + "\n" +
                      entry.resultType + "\n" + entry.classHash + "\n";
  merged.eclassStoreHash = "bundle:" + stableHash(eclassIdentity);
  return merged;
}

std::optional<RuleDatabase> RuleDatabase::fromJSON(llvm::StringRef serialized,
                                                    std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(serialized);
  if (!parsed) {
    error = "invalid rule database JSON: " +
            llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object) {
    error = "rule database root must be an object";
    return std::nullopt;
  }
  std::optional<int64_t> version = object->getInteger("schema_version");
  if (!version || *version != kSchemaVersion) {
    error = "unsupported or missing rule database schema version";
    return std::nullopt;
  }
  RuleDatabase database;
  database.schemaVersion = static_cast<unsigned>(*version);
  std::optional<std::string> expectedHash =
      requiredString(*object, "database_hash", error);
  std::optional<std::string> storeHash =
      requiredString(*object, "eclass_store_hash", error);
  std::optional<std::string> universe =
      requiredString(*object, "universe_hash", error);
  std::optional<std::string> semantics =
      requiredString(*object, "semantics_hash", error);
  std::optional<std::string> context =
      requiredString(*object, "cost_context_key", error);
  std::optional<std::string> modelName =
      requiredString(*object, "cost_model_name", error);
  std::optional<std::string> modelHash =
      requiredString(*object, "cost_model_hash", error);
  if (!expectedHash || !storeHash || !universe || !semantics || !context ||
      !modelName || !modelHash)
    return std::nullopt;
  database.eclassStoreHash = std::move(*storeHash);
  database.universeHash = std::move(*universe);
  database.semanticsHash = std::move(*semantics);
  database.costContextKey = std::move(*context);
  database.costModelName = std::move(*modelName);
  database.costModelHash = std::move(*modelHash);

  const llvm::json::Array *graphs = object->getArray("graphs");
  const llvm::json::Array *entries = object->getArray("entries");
  const llvm::json::Array *templates = object->getArray("templates");
  const llvm::json::Array *semanticEntries =
      object->getArray("semantic_entries");
  if (!graphs || !entries || !templates || !semanticEntries) {
    error = "rule database is missing graphs, entries, templates, or semantic "
            "entries";
    return std::nullopt;
  }
  for (const llvm::json::Value &value : *graphs) {
    const llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "rule database graph must be an object";
      return std::nullopt;
    }
    PersistentGraphRecord graph;
    std::optional<std::string> graphHash =
        requiredString(*entry, "graph_hash", error);
    std::optional<std::string> operation =
        requiredString(*entry, "operation_name", error);
    std::optional<std::string> type =
        requiredString(*entry, "result_type", error);
    std::optional<std::string> attributes =
        requiredString(*entry, "attributes", error);
    std::optional<int64_t> effect = entry->getInteger("effect_kind");
    uint64_t operatorSize = 0;
    if (!graphHash || !operation || !type || !attributes ||
        !effect || *effect < 0 ||
        *effect > static_cast<int64_t>(TTIRCandidateEffectKind::BufferWrite) ||
        !requiredUnsigned(*entry, "operator_size", operatorSize, error) ||
        operatorSize > std::numeric_limits<unsigned>::max() ||
        !parseStringArray(*entry, "operand_hashes", graph.operandHashes,
                          error))
      return std::nullopt;
    graph.graphHash = std::move(*graphHash);
    graph.operationName = std::move(*operation);
    graph.effectKind = static_cast<TTIRCandidateEffectKind>(*effect);
    graph.resultType = std::move(*type);
    graph.attributes = std::move(*attributes);
    graph.operatorSize = static_cast<unsigned>(operatorSize);
    database.graphs.push_back(std::move(graph));
  }

  for (const llvm::json::Value &value : *entries) {
    const llvm::json::Object *objectEntry = value.getAsObject();
    if (!objectEntry) {
      error = "rule database entry must be an object";
      return std::nullopt;
    }
    RuleDatabaseEntry entry;
    std::optional<std::string> lhs =
        requiredString(*objectEntry, "lhs_graph_hash", error);
    std::optional<std::string> classHash =
        requiredString(*objectEntry, "class_hash", error);
    std::optional<std::string> type =
        requiredString(*objectEntry, "result_type", error);
    if (!lhs || !classHash || !type ||
        !parseStringArray(*objectEntry, "proof_hashes", entry.proofHashes,
                          error))
      return std::nullopt;
    entry.lhsGraphHash = std::move(*lhs);
    entry.classHash = std::move(*classHash);
    entry.resultType = std::move(*type);
    const llvm::json::Object *sourceCost =
        objectEntry->getObject("source_cost");
    if (!sourceCost || !parseCost(*sourceCost, entry.sourceCost, error)) {
      if (error.empty())
        error = "rule database entry is missing its source cost";
      return std::nullopt;
    }
    const llvm::json::Array *candidates = objectEntry->getArray("candidates");
    if (!candidates || candidates->empty()) {
      error = "rule database entry must contain candidates";
      return std::nullopt;
    }
    for (const llvm::json::Value &candidateValue : *candidates) {
      const llvm::json::Object *candidateObject =
          candidateValue.getAsObject();
      if (!candidateObject) {
        error = "rule database candidate must be an object";
        return std::nullopt;
      }
      RuleDatabaseCandidate candidate;
      std::optional<std::string> root =
          requiredString(*candidateObject, "root_graph_hash", error);
      const llvm::json::Object *cost = candidateObject->getObject("cost");
      if (!root || !cost || !parseCost(*cost, candidate.cost, error)) {
        if (error.empty())
          error = "rule database candidate is missing its cost";
        return std::nullopt;
      }
      candidate.rootGraphHash = std::move(*root);
      entry.candidates.push_back(std::move(candidate));
    }
    database.entries.push_back(std::move(entry));
  }

  for (const llvm::json::Value &value : *templates) {
    const llvm::json::Object *objectEntry = value.getAsObject();
    if (!objectEntry) {
      error = "rule database template must be an object";
      return std::nullopt;
    }
    RuleDatabaseTemplateEntry entry;
    std::optional<std::string> lhs =
        requiredString(*objectEntry, "lhs_pattern_graph_hash", error);
    std::optional<std::string> classHash =
        requiredString(*objectEntry, "class_hash", error);
    std::optional<std::string> type =
        requiredString(*objectEntry, "result_type", error);
    if (!lhs || !classHash || !type ||
        !parseStringArray(*objectEntry, "proof_hashes", entry.proofHashes,
                          error))
      return std::nullopt;
    entry.lhsPatternGraphHash = std::move(*lhs);
    entry.classHash = std::move(*classHash);
    entry.resultType = std::move(*type);
    const llvm::json::Object *sourceCost =
        objectEntry->getObject("source_cost");
    if (!sourceCost || !parseCost(*sourceCost, entry.sourceCost, error)) {
      if (error.empty())
        error = "rule database template is missing its source cost";
      return std::nullopt;
    }
    const llvm::json::Array *candidates = objectEntry->getArray("candidates");
    if (!candidates || candidates->empty()) {
      error = "rule database template must contain candidates";
      return std::nullopt;
    }
    for (const llvm::json::Value &candidateValue : *candidates) {
      const llvm::json::Object *candidateObject =
          candidateValue.getAsObject();
      if (!candidateObject) {
        error = "rule database template candidate must be an object";
        return std::nullopt;
      }
      RuleDatabaseCandidate candidate;
      std::optional<std::string> root =
          requiredString(*candidateObject, "root_graph_hash", error);
      const llvm::json::Object *cost = candidateObject->getObject("cost");
      if (!root || !cost || !parseCost(*cost, candidate.cost, error)) {
        if (error.empty())
          error = "rule database template candidate is missing its cost";
        return std::nullopt;
      }
      candidate.rootGraphHash = std::move(*root);
      entry.candidates.push_back(std::move(candidate));
    }
    database.templates.push_back(std::move(entry));
  }

  for (const llvm::json::Value &value : *semanticEntries) {
    const llvm::json::Object *objectEntry = value.getAsObject();
    if (!objectEntry) {
      error = "rule database semantic entry must be an object";
      return std::nullopt;
    }
    RuleDatabaseSemanticEntry entry;
    std::optional<std::string> semanticKey =
        requiredString(*objectEntry, "semantic_key", error);
    std::optional<std::string> classHash =
        requiredString(*objectEntry, "class_hash", error);
    std::optional<std::string> type =
        requiredString(*objectEntry, "result_type", error);
    if (!semanticKey || semanticKey->empty() || !classHash || !type ||
        !parseStringArray(*objectEntry, "proof_hashes", entry.proofHashes,
                          error))
      return std::nullopt;
    entry.semanticKey = std::move(*semanticKey);
    entry.classHash = std::move(*classHash);
    entry.resultType = std::move(*type);
    const llvm::json::Array *candidates = objectEntry->getArray("candidates");
    if (!candidates || candidates->empty() || entry.proofHashes.empty()) {
      error = "rule database semantic entry must contain proofs and candidates";
      return std::nullopt;
    }
    for (const llvm::json::Value &candidateValue : *candidates) {
      const llvm::json::Object *candidateObject =
          candidateValue.getAsObject();
      if (!candidateObject) {
        error = "rule database semantic candidate must be an object";
        return std::nullopt;
      }
      RuleDatabaseCandidate candidate;
      std::optional<std::string> root =
          requiredString(*candidateObject, "root_graph_hash", error);
      const llvm::json::Object *cost = candidateObject->getObject("cost");
      if (!root || !cost || !parseCost(*cost, candidate.cost, error)) {
        if (error.empty())
          error = "rule database semantic candidate is missing its cost";
        return std::nullopt;
      }
      candidate.rootGraphHash = std::move(*root);
      entry.candidates.push_back(std::move(candidate));
    }
    database.semanticEntries.push_back(std::move(entry));
  }

  llvm::sort(database.graphs,
             [](const auto &lhs, const auto &rhs) {
               return lhs.graphHash < rhs.graphHash;
             });
  llvm::sort(database.entries, [](const auto &lhs, const auto &rhs) {
    return lhs.lhsGraphHash < rhs.lhsGraphHash;
  });
  llvm::sort(database.templates, [&](const auto &lhs, const auto &rhs) {
    return templateLess(database, lhs, rhs);
  });
  llvm::sort(database.semanticEntries,
             [](const auto &lhs, const auto &rhs) {
               return std::tie(lhs.semanticKey, lhs.resultType,
                               lhs.classHash) <
                      std::tie(rhs.semanticKey, rhs.resultType,
                               rhs.classHash);
             });
  uint64_t graphCount = 0;
  uint64_t entryCount = 0;
  uint64_t templateCount = 0;
  uint64_t semanticEntryCount = 0;
  if (!requiredUnsigned(*object, "graph_count", graphCount, error) ||
      !requiredUnsigned(*object, "entry_count", entryCount, error) ||
      !requiredUnsigned(*object, "template_count", templateCount, error) ||
      !requiredUnsigned(*object, "semantic_entry_count", semanticEntryCount,
                        error) ||
      graphCount != database.graphs.size() ||
      entryCount != database.entries.size() ||
      templateCount != database.templates.size() ||
      semanticEntryCount != database.semanticEntries.size()) {
    if (error.empty())
      error = "rule database count mismatch";
    return std::nullopt;
  }
  llvm::StringSet<> knownGraphs;
  for (const PersistentGraphRecord &graph : database.graphs) {
    if (graph.graphHash != getPersistentGraphHash(graph)) {
      error = "rule database contains a non-canonical graph hash";
      return std::nullopt;
    }
    if (!knownGraphs.insert(graph.graphHash).second) {
      error = "duplicate graph hash in rule database";
      return std::nullopt;
    }
    for (llvm::StringRef operand : graph.operandHashes)
      if (!database.lookupGraph(operand)) {
        error = "rule database graph references an unknown operand";
        return std::nullopt;
      }
  }
  llvm::StringSet<> lhsHashes;
  for (const RuleDatabaseEntry &entry : database.entries) {
    const PersistentGraphRecord *lhs = database.lookupGraph(entry.lhsGraphHash);
    if (!lhsHashes.insert(entry.lhsGraphHash).second || !lhs) {
      error = "duplicate or unknown LHS in rule database";
      return std::nullopt;
    }
    if (lhs->resultType != entry.resultType) {
      error = "rule database LHS result type does not match its entry";
      return std::nullopt;
    }
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      const PersistentGraphRecord *root =
          database.lookupGraph(candidate.rootGraphHash);
      if (!root) {
        error = "rule database candidate references an unknown graph";
        return std::nullopt;
      }
      if (root->resultType != entry.resultType) {
        error = "rule database candidate result type does not match its entry";
        return std::nullopt;
      }
    }
  }
  llvm::StringSet<> templatePatterns;
  for (const RuleDatabaseTemplateEntry &entry : database.templates) {
    const PersistentGraphRecord *lhs =
        database.lookupGraph(entry.lhsPatternGraphHash);
    if (!templatePatterns.insert(entry.lhsPatternGraphHash).second || !lhs) {
      error = "duplicate or unknown LHS pattern in rule database";
      return std::nullopt;
    }
    if (lhs->resultType != entry.resultType || entry.proofHashes.empty()) {
      error = "rule database template type/proof metadata is invalid";
      return std::nullopt;
    }
    llvm::StringSet<> visited;
    if (!containsTemplateVariable(database, lhs->graphHash, visited)) {
      error = "rule database template has no typed graph variable";
      return std::nullopt;
    }
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      const PersistentGraphRecord *root =
          database.lookupGraph(candidate.rootGraphHash);
      if (!root || root->resultType != entry.resultType ||
          !candidateUsesOnlyPatternVariables(
              database, entry.lhsPatternGraphHash, root->graphHash)) {
        error = "rule database template candidate is not a closed symbolic "
                "instantiation of its source pattern";
        return std::nullopt;
      }
    }
  }
  llvm::StringSet<> semanticIdentities;
  for (const RuleDatabaseSemanticEntry &entry : database.semanticEntries) {
    std::string identity = entry.resultType + "\n" + entry.semanticKey;
    if (!semanticIdentities.insert(identity).second ||
        entry.semanticKey !=
            getRuleDatabaseSemanticKeyDigest(entry.semanticKey) ||
        entry.proofHashes.empty()) {
      error = "duplicate or invalid semantic entry in rule database";
      return std::nullopt;
    }
    for (const RuleDatabaseCandidate &candidate : entry.candidates) {
      const PersistentGraphRecord *root =
          database.lookupGraph(candidate.rootGraphHash);
      if (!root || root->resultType != entry.resultType ||
          hasCompilerAction(*root)) {
        error = "rule database semantic candidate is not a pure known graph";
        return std::nullopt;
      }
    }
  }
  if (database.getStableHash() != *expectedHash) {
    error = "rule database stable hash mismatch";
    return std::nullopt;
  }
  return database;
}

OnlineRuleLookupResult OnlineRuleMatcher::lookup(
    llvm::StringRef canonicalGraphHash, llvm::StringRef universeHash,
    llvm::StringRef semanticsHash, llvm::StringRef costContextKey) const {
  OnlineRuleLookupResult result;
  if (database.universeHash != universeHash ||
      database.semanticsHash != semanticsHash ||
      database.costContextKey != costContextKey) {
    result.status = OnlineRuleLookupStatus::VersionMismatch;
    result.reason = "Universe, semantics, or cost-context version mismatch";
    return result;
  }
  result.entry = database.lookup(canonicalGraphHash);
  result.status = result.entry ? OnlineRuleLookupStatus::Hit
                               : OnlineRuleLookupStatus::Miss;
  if (!result.entry)
    result.reason = "canonical graph key is absent from the rule database";
  return result;
}

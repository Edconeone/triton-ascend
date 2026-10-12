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

#include "TritonToGraph/GraphOperationCatalog.h"
#include "TritonToGraph/AutoGraphSolver.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static llvm::StringRef stringifyCoverage(OperationSemanticCoverage coverage) {
  switch (coverage) {
  case OperationSemanticCoverage::Missing:
    return "missing";
  case OperationSemanticCoverage::Partial:
    return "partial";
  case OperationSemanticCoverage::Complete:
    return "complete";
  }
  llvm_unreachable("unknown OperationSemanticCoverage");
}

static llvm::StringRef stringifySearchPolicy(TTOpSearchPolicy policy) {
  switch (policy) {
  case TTOpSearchPolicy::Searchable:
    return "searchable";
  case TTOpSearchPolicy::ImportOnly:
    return "import_only";
  case TTOpSearchPolicy::RegionOnly:
    return "region_only";
  case TTOpSearchPolicy::Mixed:
    return "mixed";
  case TTOpSearchPolicy::Incomplete:
    return "incomplete";
  }
  llvm_unreachable("unknown TTOpSearchPolicy");
}

static std::string getDialectNamespace(llvm::StringRef operationName) {
  return operationName.split('.').first.str();
}

} // namespace

GraphOperationCatalog GraphOperationCatalog::build(
    MLIRContext &context, const GraphUniverseSpec &spec,
    const TTOpSemanticsRegistry &semanticsRegistry) {
  llvm::SmallVector<llvm::StringRef, 16> semanticOperations =
      semanticsRegistry.getOperationNames();
  llvm::StringSet<> selected;
  for (const std::string &operation : spec.operations)
    selected.insert(operation);
  llvm::StringSet<> semantic;
  for (llvm::StringRef operation : semanticOperations)
    semantic.insert(operation);

  llvm::StringSet<> operationNames;
  for (const std::string &dialect : spec.catalogDialects) {
    for (RegisteredOperationName operation :
         context.getRegisteredOperationsByDialect(dialect))
      operationNames.insert(operation.getStringRef());
  }
  for (const std::string &operation : spec.operations)
    operationNames.insert(operation);
  for (llvm::StringRef operation : semanticOperations)
    operationNames.insert(operation);

  std::vector<std::string> sortedNames;
  sortedNames.reserve(operationNames.size());
  for (const auto &entry : operationNames)
    sortedNames.push_back(entry.getKey().str());
  llvm::sort(sortedNames);

  GraphOperationCatalog catalog;
  catalog.semanticDefinitionCount = semanticOperations.size();
  catalog.selectedOperationCount = spec.operations.size();
  for (const std::string &operationName : sortedNames) {
    GraphOperationCatalogEntry entry;
    entry.operationName = operationName;
    entry.dialect = getDialectNamespace(operationName);
    entry.pseudoOperation = llvm::StringRef(operationName).starts_with("ags.");
    entry.registered = entry.pseudoOperation ||
                       OperationName(operationName, &context).isRegistered();
    entry.selectedByUniverse = selected.contains(operationName);
    if (entry.selectedByUniverse && !entry.registered)
      ++catalog.unregisteredSelectedCount;
    if (entry.registered && !entry.pseudoOperation)
      ++catalog.registeredOperationCount;

    if (semantic.contains(operationName)) {
      if (!entry.registered && !entry.pseudoOperation)
        ++catalog.orphanSemanticDefinitionCount;
      llvm::ArrayRef<GraphSemanticDomain> implementedSemanticDomains =
          semanticsRegistry.getImplementedDomains(operationName);
      entry.searchPolicy =
          semanticsRegistry.getSearchPolicy(operationName);
      entry.implementedDomains.assign(implementedSemanticDomains.begin(),
                                      implementedSemanticDomains.end());
      for (GraphSemanticDomain required : spec.semanticDomains) {
        if (!llvm::is_contained(implementedSemanticDomains, required))
          entry.missingDomains.push_back(required);
      }
      entry.coverage = entry.missingDomains.empty() &&
                               entry.searchPolicy !=
                                   TTOpSearchPolicy::Incomplete
                           ? OperationSemanticCoverage::Complete
                           : OperationSemanticCoverage::Partial;
      switch (entry.searchPolicy) {
      case TTOpSearchPolicy::Searchable:
        ++catalog.searchableCount;
        break;
      case TTOpSearchPolicy::ImportOnly:
        ++catalog.importOnlyCount;
        break;
      case TTOpSearchPolicy::RegionOnly:
        ++catalog.regionOnlyCount;
        break;
      case TTOpSearchPolicy::Mixed:
        ++catalog.mixedSearchPolicyCount;
        break;
      case TTOpSearchPolicy::Incomplete:
        ++catalog.incompleteSearchPolicyCount;
        break;
      }
    }

    switch (entry.coverage) {
    case OperationSemanticCoverage::Missing:
      ++catalog.missingCount;
      if (entry.selectedByUniverse)
        ++catalog.selectedMissingCount;
      break;
    case OperationSemanticCoverage::Partial:
      ++catalog.partialCount;
      if (entry.selectedByUniverse)
        ++catalog.selectedPartialCount;
      break;
    case OperationSemanticCoverage::Complete:
      ++catalog.completeCount;
      if (entry.selectedByUniverse)
        ++catalog.selectedCompleteCount;
      break;
    }
    catalog.entries.push_back(std::move(entry));
  }
  return catalog;
}

std::string GraphOperationCatalog::toJSON(llvm::StringRef universeHash) const {
  llvm::json::Array operationEntries;
  for (const GraphOperationCatalogEntry &entry : entries) {
    llvm::json::Array implemented;
    for (GraphSemanticDomain domain : entry.implementedDomains)
      implemented.push_back(stringifyGraphSemanticDomain(domain));
    llvm::json::Array missing;
    for (GraphSemanticDomain domain : entry.missingDomains)
      missing.push_back(stringifyGraphSemanticDomain(domain));

    llvm::json::Object object;
    object["operation"] = entry.operationName;
    object["dialect"] = entry.dialect;
    object["registered"] = entry.registered;
    object["pseudo"] = entry.pseudoOperation;
    object["selected"] = entry.selectedByUniverse;
    object["coverage"] = stringifyCoverage(entry.coverage);
    object["search_policy"] = stringifySearchPolicy(entry.searchPolicy);
    object["implemented_domains"] = std::move(implemented);
    object["missing_domains"] = std::move(missing);
    operationEntries.push_back(std::move(object));
  }

  llvm::json::Object summary;
  summary["universe_hash"] = universeHash;
  summary["registered_operation_count"] = registeredOperationCount;
  summary["semantic_definition_count"] = semanticDefinitionCount;
  summary["orphan_semantic_definition_count"] =
      orphanSemanticDefinitionCount;
  summary["selected_operation_count"] = selectedOperationCount;
  summary["unregistered_selected_count"] = unregisteredSelectedCount;
  summary["missing_count"] = missingCount;
  summary["partial_count"] = partialCount;
  summary["complete_count"] = completeCount;
  summary["selected_missing_count"] = selectedMissingCount;
  summary["selected_partial_count"] = selectedPartialCount;
  summary["selected_complete_count"] = selectedCompleteCount;
  summary["searchable_count"] = searchableCount;
  summary["import_only_count"] = importOnlyCount;
  summary["region_only_count"] = regionOnlyCount;
  summary["mixed_search_policy_count"] = mixedSearchPolicyCount;
  summary["incomplete_search_policy_count"] =
      incompleteSearchPolicyCount;
  summary["operations"] = std::move(operationEntries);

  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(summary)));
  return result;
}

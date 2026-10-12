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

#ifndef TRITON_TO_GRAPH_GRAPH_OPERATION_CATALOG_H
#define TRITON_TO_GRAPH_GRAPH_OPERATION_CATALOG_H

#include "TritonToGraph/AutoGraphSolver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace mlir {
class MLIRContext;

namespace triton::cfg {

class TTOpSemanticsRegistry;

enum class OperationSemanticCoverage : uint8_t {
  Missing,
  Partial,
  Complete,
};

struct GraphOperationCatalogEntry {
  std::string operationName;
  std::string dialect;
  bool registered = false;
  bool pseudoOperation = false;
  bool selectedByUniverse = false;
  OperationSemanticCoverage coverage = OperationSemanticCoverage::Missing;
  TTOpSearchPolicy searchPolicy = TTOpSearchPolicy::Incomplete;
  std::vector<GraphSemanticDomain> implementedDomains;
  std::vector<GraphSemanticDomain> missingDomains;
};

struct GraphOperationCatalog {
  std::vector<GraphOperationCatalogEntry> entries;
  unsigned registeredOperationCount = 0;
  unsigned semanticDefinitionCount = 0;
  unsigned orphanSemanticDefinitionCount = 0;
  unsigned selectedOperationCount = 0;
  unsigned unregisteredSelectedCount = 0;
  unsigned missingCount = 0;
  unsigned partialCount = 0;
  unsigned completeCount = 0;
  unsigned selectedMissingCount = 0;
  unsigned selectedPartialCount = 0;
  unsigned selectedCompleteCount = 0;
  unsigned searchableCount = 0;
  unsigned importOnlyCount = 0;
  unsigned regionOnlyCount = 0;
  unsigned mixedSearchPolicyCount = 0;
  unsigned incompleteSearchPolicyCount = 0;

  static GraphOperationCatalog
  build(MLIRContext &context, const GraphUniverseSpec &spec,
        const TTOpSemanticsRegistry &semantics);

  std::string toJSON(llvm::StringRef universeHash) const;
};

} // namespace triton::cfg
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPERATION_CATALOG_H

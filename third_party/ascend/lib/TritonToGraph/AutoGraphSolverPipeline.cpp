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

#include "TritonToGraph/AutoGraphSolverPipeline.h"
#include "TritonToGraph/CanonicalRegion.h"

#include "TritonToGraph/AutoGraphSolver.h"
#include "TritonToGraph/CanonicalGraphKey.h"
#include "TritonToGraph/RuleDatabase.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/AsmParser/AsmParser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

namespace {

static constexpr llvm::StringLiteral kInputOp = "ags.input";
static constexpr llvm::StringLiteral kPrivateBufferOp = "ags.private_buffer";
static constexpr llvm::StringLiteral kResultTypeAttr = "ags.result_type";
static constexpr llvm::StringLiteral kSymbolIdAttr = "ags.symbol_id";
static constexpr llvm::StringLiteral kResultIndexAttr = "ags.result_index";
static constexpr llvm::StringLiteral kOperationResultTypesAttr =
    "ags.operation_result_types";
static constexpr llvm::StringLiteral kStateDomainAttr = "ags.state_domain";
static constexpr llvm::StringLiteral kModuleAttributePatchAttr =
    "ags.module_attribute_patch";
static constexpr llvm::StringLiteral kCompilerActionAttr =
    "ags.compiler_action";
static constexpr llvm::StringLiteral kProofOnlyAttr = "ags.proof_only";
static constexpr llvm::StringLiteral kExploratoryAttr = "ags.exploratory";
static constexpr llvm::StringLiteral kReusableSourceAttr =
    "ags.reusable_source";

static std::string printType(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static Type getResultType(const GraphExprNode &node) {
  if (!node.attributes)
    return {};
  auto type = dyn_cast_or_null<TypeAttr>(node.attributes.get(kResultTypeAttr));
  return type ? type.getValue() : Type{};
}

static bool isPrivateAllocationBoundary(Value value) {
  if (!isa<BaseMemRefType>(value.getType()))
    return false;
  auto effects = dyn_cast_or_null<MemoryEffectOpInterface>(
      value.getDefiningOp());
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance, 4> effectList;
  effects.getEffects(effectList);
  return llvm::any_of(effectList, [&](const auto &effect) {
    return isa<MemoryEffects::Allocate>(effect.getEffect()) &&
           effect.getValue() == value;
  });
}

class TTIRSemanticImporter {
public:
  TTIRSemanticImporter(EquivalenceGraph &graph,
                       const TTOpSemanticsRegistry &semantics, Value rootValue)
      : graph(graph), semantics(semantics), rootValue(rootValue) {}

  std::optional<GraphEClassId> import(Value value,
                                      bool reusableSource = false) {
    auto known = classes.find(value);
    if (known != classes.end())
      return graph.find(known->second);

    Operation *operation = value.getDefiningOp();
    SmallVector<GraphEClassId, 4> operandClasses;
    // A shared producer is normally a region boundary: replacing one root
    // cannot claim that the producer disappears while another user keeps it
    // alive. Constants and canonical index carriers are exceptions. Their
    // denotation is needed by consumers (for example a make_range shared by a
    // correction compare and a store address), while DCE still preserves the
    // producer because its external use remains. General shared-DAG costing is
    // deliberately not inferred from this POC exception.
    bool supported = semantics.supports(operation) &&
        (value == rootValue || value.hasOneUse() ||
         semantics.allowsSharedImport(operation));
    if (supported) {
      for (Value operand : operation->getOperands()) {
        std::optional<GraphEClassId> imported = import(operand);
        if (!imported)
          return std::nullopt;
        operandClasses.push_back(*imported);
      }
    }

    Builder builder(value.getContext());
    SmallVector<NamedAttribute, 8> attributes;
    attributes.push_back(builder.getNamedAttr(kResultTypeAttr,
                                              TypeAttr::get(value.getType())));
    // An already executed effectful or region-bearing producer must be reused
    // as an SSA value, never cloned by the region-free recipe materializer.
    // This is generic effect/region discipline rather than an operation-name
    // special case: a read, allocation, scan, or other non-cloneable result
    // that dominates the root is a zero-incremental-cost island input.  Pure
    // region-free imported operations remain rebuildable so extraction can
    // choose cheaper equivalent operands.
    Operation *rootOperation = rootValue.getDefiningOp();
    bool dominatingNonCloneableSource =
        supported && operation && rootOperation && operation != rootOperation &&
        operation->getBlock() == rootOperation->getBlock() &&
        operation->isBeforeInBlock(rootOperation) &&
        (!isMemoryEffectFree(operation) || operation->getNumRegions() != 0);
    if (reusableSource || dominatingNonCloneableSource)
      attributes.push_back(builder.getNamedAttr(kReusableSourceAttr,
                                                builder.getBoolAttr(true)));
    StringRef operationName = isPrivateAllocationBoundary(value)
                                  ? kPrivateBufferOp
                                  : kInputOp;
    Value sourceValue = value;
    if (supported) {
      operationName = operation->getName().getStringRef();
      attributes.append(operation->getAttrs().begin(),
                        operation->getAttrs().end());
      if (operation->getNumRegions() != 0) {
        std::string regionError;
        std::optional<ArrayAttr> regions =
            captureCanonicalRegions(operation, regionError);
        if (!regions)
          return std::nullopt;
        attributes.push_back(builder.getNamedAttr(kCanonicalRegionsAttr,
                                                   *regions));
      }
    } else {
      attributes.push_back(builder.getNamedAttr(
          kSymbolIdAttr, builder.getI64IntegerAttr(nextSymbolId++)));
    }

    std::optional<GraphEClassId> eclass = graph.addNode(GraphExprNode::create(
        value.getContext(), operationName, operandClasses, attributes,
        sourceValue));
    if (!eclass)
      return std::nullopt;

    const TTOpSemantics *definition = semantics.lookup(operationName);
    if (!definition)
      return std::nullopt;
    SmallVector<TensorDenotation, 4> operandDenotations;
    for (GraphEClassId operand : operandClasses) {
      auto denotation = denotations.find(graph.find(operand));
      if (denotation == denotations.end())
        return std::nullopt;
      operandDenotations.push_back(denotation->second);
    }
    GraphExprNode node = graph.getNodes(*eclass).front();
    FailureOr<TensorDenotation> denotation =
        definition->denote(node, operandDenotations);
    if (failed(denotation))
      return std::nullopt;

    *eclass = graph.find(*eclass);
    classes[value] = *eclass;
    denotations[*eclass] = *denotation;
    sourceValues[*eclass] = value;
    available.push_back(*eclass);
    return eclass;
  }

  DenseMap<GraphEClassId, TensorDenotation> &getDenotations() {
    return denotations;
  }
  const DenseMap<GraphEClassId, Value> &getSourceValues() const {
    return sourceValues;
  }
  ArrayRef<GraphEClassId> getAvailable() const { return available; }

private:
  EquivalenceGraph &graph;
  const TTOpSemanticsRegistry &semantics;
  DenseMap<Value, GraphEClassId> classes;
  DenseMap<GraphEClassId, TensorDenotation> denotations;
  DenseMap<GraphEClassId, Value> sourceValues;
  SmallVector<GraphEClassId, 32> available;
  Value rootValue;
  unsigned nextSymbolId = 0;
};

static std::optional<Value> getStoreObservationMask(Value root) {
  if (!root || !root.hasOneUse())
    return std::nullopt;
  auto store = dyn_cast<triton::StoreOp>(root.use_begin()->getOwner());
  if (!store || store.getValue() != root || !store.getMask())
    return std::nullopt;
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  auto maskType = dyn_cast<RankedTensorType>(store.getMask().getType());
  if (!rootType || !maskType || !rootType.hasStaticShape() ||
      !maskType.hasStaticShape() ||
      rootType.getShape() != maskType.getShape() ||
      !maskType.getElementType().isInteger(1))
    return std::nullopt;
  return store.getMask();
}

static TensorDenotation observeUnderMask(const TensorDenotation &value,
                                         const TensorDenotation &mask) {
  if (!value.isValid() || !mask.isValid() || value.shape != mask.shape)
    return value;
  MLIRContext *context = value.value.getType().getContext();
  Type booleanType = IntegerType::get(context, 1);
  TensorSemanticExpr unobserved = TensorSemanticExpr::operation(
      TensorSemanticKind::Unobserved, value.value.getType());
  TensorSemanticExpr observedValue = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, value.value.getType(),
      {mask.value, value.value, unobserved});
  TensorSemanticExpr observedDefined = TensorSemanticExpr::operation(
      TensorSemanticKind::Select, booleanType,
      {mask.value, value.defined,
       TensorSemanticExpr::boolean(context, true)});
  observedDefined = TensorSemanticExpr::operation(
      TensorSemanticKind::And, booleanType,
      {mask.defined, observedDefined});
  TensorDenotation observed = value;
  observed.value = std::move(observedValue);
  observed.defined = std::move(observedDefined);
  return observed;
}

static Value lookupSourceValue(const EquivalenceGraph &graph,
                               const DenseMap<GraphEClassId, Value> &values,
                               GraphEClassId eclass) {
  eclass = graph.find(eclass);
  for (const auto &entry : values)
    if (graph.find(entry.first) == eclass)
      return entry.second;
  return {};
}

static Type lookupEClassType(const EquivalenceGraph &graph,
                             GraphEClassId eclass) {
  for (const GraphExprNode &node : graph.getNodes(graph.find(eclass)))
    if (Type type = getResultType(node))
      return type;
  return {};
}

static bool containsSemanticExpression(const TensorSemanticExpr &expression,
                                       const TensorSemanticExpr &needle) {
  if (!expression || !needle)
    return false;
  if (expression.structurallyEquals(needle))
    return true;
  return llvm::any_of(expression.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return containsSemanticExpression(operand, needle);
                      });
}

static bool valueDominatesAnchor(Value value, Operation *anchor) {
  if (!value || !anchor)
    return false;
  auto findAncestorInBlock = [&](Block *block) -> Operation * {
    Operation *ancestor = anchor;
    while (ancestor && ancestor->getBlock() != block)
      ancestor = ancestor->getParentOp();
    return ancestor;
  };
  if (auto argument = dyn_cast<BlockArgument>(value))
    return findAncestorInBlock(argument.getOwner()) != nullptr;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  Operation *ancestor = findAncestorInBlock(definition->getBlock());
  return ancestor && definition != ancestor &&
         definition->isBeforeInBlock(ancestor);
}

static bool canHoistValueToAnchorImpl(Value value, Operation *anchor,
                                      SmallPtrSetImpl<Operation *> &active) {
  if (valueDominatesAnchor(value, anchor))
    return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition == anchor ||
      definition->getBlock() != anchor->getBlock() ||
      !isMemoryEffectFree(definition) || definition->getNumRegions() != 0 ||
      !active.insert(definition).second)
    return false;
  bool legal = llvm::all_of(definition->getOperands(), [&](Value operand) {
    return canHoistValueToAnchorImpl(operand, anchor, active);
  });
  active.erase(definition);
  return legal;
}

static bool canReuseValueAtAnchor(Value value, Operation *anchor) {
  SmallPtrSet<Operation *, 16> active;
  return canHoistValueToAnchorImpl(value, anchor, active);
}

// The enumerator is generic over registered executable-op schemas. A schema
// describes one operation in the TTIR language (name, arity and type
// contract); it never describes the source graph that should be replaced.
class SchemaDrivenTTIRLanguage final : public TTIRLanguage {
public:
  explicit SchemaDrivenTTIRLanguage(
      ArrayRef<TTIRCandidateOpSchema> schemas,
      std::function<unsigned(GraphEClassId)> operandPriority = {})
      : schemas(schemas.begin(), schemas.end()),
        operandPriority(std::move(operandPriority)) {
    llvm::stable_sort(this->schemas,
                      [](const TTIRCandidateOpSchema &lhs,
                         const TTIRCandidateOpSchema &rhs) {
                        return lhs.operandCount < rhs.operandCount;
                      });
  }

  void enumerate(Type resultType,
                 ArrayRef<GraphEClassId> availableOperands,
                 const EquivalenceGraph &graph,
                 SmallVectorImpl<GraphExprNode> &candidates,
                 unsigned limit) const override {
    if (limit == 0)
      return;
    const unsigned candidateLimit = candidates.size() + limit;

    Builder builder(resultType.getContext());
    NamedAttribute resultTypeAttribute = builder.getNamedAttr(
        kResultTypeAttr, TypeAttr::get(resultType));

    SmallVector<GraphEClassId, 32> operandPool;
    llvm::DenseSet<GraphEClassId> seen;
    for (GraphEClassId operand : availableOperands) {
      operand = graph.find(operand);
      if (seen.insert(operand).second && lookupEClassType(graph, operand))
        operandPool.push_back(operand);
    }
    // Candidate budgets make operand order semantically important. Expand
    // lower-cost e-classes first so the bounded search approximates best-first
    // enumeration instead of inheriting incidental block/SSA order. This is
    // generic across schemas and prevents a large producer island from
    // starving simpler combinations before they can be proved.
    StructuralGraphCostModel operandCostModel;
    GraphExtractor operandExtractor(operandCostModel);
    DenseMap<GraphEClassId, unsigned> operandCosts;
    for (GraphEClassId operand : operandPool) {
      std::optional<GraphExtraction> plan =
          operandExtractor.extract(graph, operand);
      operandCosts[operand] =
          plan ? plan->cost : std::numeric_limits<unsigned>::max();
    }
    llvm::stable_sort(operandPool, [&](GraphEClassId lhs, GraphEClassId rhs) {
      if (operandPriority) {
        unsigned lhsPriority = operandPriority(lhs);
        unsigned rhsPriority = operandPriority(rhs);
        if (lhsPriority != rhsPriority)
          return lhsPriority < rhsPriority;
      }
      return operandCosts.lookup(lhs) < operandCosts.lookup(rhs);
    });

    auto isEligible = [&](const TTIRCandidateOpSchema &schema) {
      if (!schema.constructible || schema.offlineOnly)
        return false;
      if (schema.canProduceResult &&
          !schema.canProduceResult(resultType, /*attributes=*/{}))
        return false;
      if (schema.requiresWitnessedOperation) {
        bool witnessed = false;
        for (GraphEClassId eclass : graph.getRootEClassIds())
          for (const GraphExprNode &existing : graph.getNodes(eclass))
            witnessed |=
                existing.getOperationName() == schema.operationName;
        if (!witnessed)
          return false;
      }
      return true;
    };
    const unsigned eligibleSchemaCount = std::max(
        1u, static_cast<unsigned>(llvm::count_if(schemas, isEligible)));

    for (const TTIRCandidateOpSchema &schema : schemas) {
      if (!isEligible(schema))
        continue;
      const unsigned schemaLimit = std::min(
          candidateLimit,
          static_cast<unsigned>(candidates.size()) +
              std::max(1u, limit / eligibleSchemaCount));
      SmallVector<GraphEClassId, 4> selectedOperands;
      SmallVector<Type, 4> selectedTypes;
      std::function<void(unsigned)> enumerateOperand =
          [&](unsigned operandIndex) {
            if (candidates.size() >= schemaLimit)
              return;
            if (operandIndex == schema.operandCount) {
              if (!schema.isWellTyped ||
                  !schema.isWellTyped(resultType, selectedTypes,
                                      /*attributes=*/{}))
                return;
              // Attribute domains are seeded by real operations already in
              // the island. This is generic (not an op-name switch) and keeps
              // candidates executable: a transpose reuses a witnessed order,
              // while casts retain witnessed semantic attributes.
              SmallVector<DictionaryAttr, 4> attributeVariants;
              for (GraphEClassId eclass : graph.getRootEClassIds())
                for (const GraphExprNode &existing : graph.getNodes(eclass)) {
                  if (existing.getOperationName() != schema.operationName)
                    continue;
                  SmallVector<NamedAttribute, 8> attributes;
                  attributes.push_back(resultTypeAttribute);
                  if (schema.exploratory)
                    attributes.push_back(builder.getNamedAttr(
                        kExploratoryAttr, builder.getBoolAttr(true)));
                  if (existing.attributes)
                    for (NamedAttribute attribute : existing.attributes)
                      if (!attribute.getName().getValue().starts_with("ags."))
                        attributes.push_back(attribute);
                  DictionaryAttr variant =
                      builder.getDictionaryAttr(attributes);
                  if (!llvm::is_contained(attributeVariants, variant))
                    attributeVariants.push_back(variant);
                }
              if (attributeVariants.empty()) {
                SmallVector<NamedAttribute, 2> attributes = {
                    resultTypeAttribute};
                if (schema.exploratory)
                  attributes.push_back(builder.getNamedAttr(
                      kExploratoryAttr, builder.getBoolAttr(true)));
                attributeVariants.push_back(
                    builder.getDictionaryAttr(attributes));
              }

              for (DictionaryAttr attributes : attributeVariants) {
                SmallVector<NamedAttribute, 8> schemaMetadata(
                    attributes.begin(), attributes.end());
                if (schema.producedStateDomain) {
                  StringRef domain =
                      *schema.producedStateDomain ==
                              TTIRCandidateStateDomain::Memory
                          ? "memory"
                      : *schema.producedStateDomain ==
                              TTIRCandidateStateDomain::Launch
                          ? "launch"
                          : "buffer";
                  schemaMetadata.push_back(builder.getNamedAttr(
                      kStateDomainAttr, builder.getStringAttr(domain)));
                }
                if (schema.proofOnlyMaterialization)
                  schemaMetadata.push_back(builder.getNamedAttr(
                      kProofOnlyAttr, builder.getBoolAttr(true)));
                if (schema.buildModuleAttributePatch) {
                  DictionaryAttr patch =
                      schema.buildModuleAttributePatch(attributes);
                  if (!patch)
                    continue;
                  schemaMetadata.push_back(builder.getNamedAttr(
                      kModuleAttributePatchAttr, patch));
                }
                if (schema.buildCompilerAction) {
                  DictionaryAttr action =
                      schema.buildCompilerAction(attributes);
                  if (action)
                    schemaMetadata.push_back(builder.getNamedAttr(
                        kCompilerActionAttr, action));
                }
                attributes = builder.getDictionaryAttr(schemaMetadata);
                if (schema.operationResultCount > 1) {
                  if (!schema.buildOperationResultTypes)
                    continue;
                  SmallVector<Type, 2> operationResultTypes =
                      schema.buildOperationResultTypes(
                          resultType, selectedTypes, attributes);
                  if (operationResultTypes.size() !=
                      schema.operationResultCount)
                    continue;
                  SmallVector<NamedAttribute, 8> withProjection(
                      attributes.begin(), attributes.end());
                  SmallVector<Attribute, 2> encodedTypes;
                  for (Type type : operationResultTypes)
                    encodedTypes.push_back(TypeAttr::get(type));
                  withProjection.push_back(builder.getNamedAttr(
                      kResultIndexAttr,
                      builder.getI64IntegerAttr(schema.resultIndex)));
                  withProjection.push_back(builder.getNamedAttr(
                      kOperationResultTypesAttr,
                      builder.getArrayAttr(encodedTypes)));
                  attributes = builder.getDictionaryAttr(withProjection);
                }
                if (schema.buildRegions) {
                  ArrayAttr regions = schema.buildRegions(
                      resultType, selectedTypes, attributes);
                  if (!regions)
                    continue;
                  SmallVector<NamedAttribute, 8> withRegions(
                      attributes.begin(), attributes.end());
                  withRegions.push_back(builder.getNamedAttr(
                      kCanonicalRegionsAttr, regions));
                  attributes = builder.getDictionaryAttr(withRegions);
                }
                GraphExprNode candidate = GraphExprNode::create(
                    resultType.getContext(), schema.operationName,
                    selectedOperands, attributes.getValue());
                std::string candidateKey;
                llvm::raw_string_ostream keyStream(candidateKey);
                keyStream << candidate.getOperationName() << ':' << resultType
                          << ':' << attributes;
                for (GraphEClassId operand : candidate.operands)
                  keyStream << ':' << graph.find(operand);
                if (!emitted.insert(keyStream.str()).second)
                  continue;
                bool exists = false;
                for (GraphEClassId eclass : graph.getRootEClassIds())
                  for (const GraphExprNode &existing : graph.getNodes(eclass))
                    if (existing.getOperationName() ==
                            candidate.getOperationName() &&
                        existing.attributes == candidate.attributes &&
                        existing.operands == candidate.operands)
                      exists = true;
                if (!exists)
                  candidates.push_back(std::move(candidate));
                if (candidates.size() >= schemaLimit)
                  return;
              }
              return;
            }
            for (GraphEClassId operand : operandPool) {
              Type operandType = lookupEClassType(graph, operand);
              if (!operandType)
                continue;
              selectedOperands.push_back(operand);
              selectedTypes.push_back(operandType);
              if (!schema.mayBeWellTyped ||
                  schema.mayBeWellTyped(resultType, selectedTypes,
                                        /*attributes=*/{}))
                enumerateOperand(operandIndex + 1);
              selectedTypes.pop_back();
              selectedOperands.pop_back();
              if (candidates.size() >= schemaLimit)
                return;
            }
          };
      enumerateOperand(/*operandIndex=*/0);
      if (candidates.size() >= candidateLimit)
        return;
    }
  }

private:
  SmallVector<TTIRCandidateOpSchema, 4> schemas;
  std::function<unsigned(GraphEClassId)> operandPriority;
  mutable llvm::StringSet<> emitted;
};

enum class SemanticGraphLookupMode {
  SemanticEnumeration,
  RuleDatabaseExact,
  RuleDatabaseSemantic,
  RuleDatabaseTemplate,
};

static StringRef stringifyLookupMode(SemanticGraphLookupMode mode) {
  switch (mode) {
  case SemanticGraphLookupMode::SemanticEnumeration:
    return "semantic-enumeration";
  case SemanticGraphLookupMode::RuleDatabaseExact:
    return "rule-database-exact";
  case SemanticGraphLookupMode::RuleDatabaseSemantic:
    return "rule-database-semantic-normal-form";
  case SemanticGraphLookupMode::RuleDatabaseTemplate:
    return "rule-database-template";
  }
  llvm_unreachable("unknown semantic graph lookup mode");
}

struct SemanticGraphSolution {
  Operation *anchor = nullptr;
  bool operationRoot = false;
  Operation *kernelScope = nullptr;
  SmallVector<Operation *, 4> replacedEffects;
  struct Recipe {
    Value source;
    bool proofOnly = false;
    std::string operationName;
    Type resultType;
    DictionaryAttr attributes;
    SmallVector<std::shared_ptr<Recipe>, 4> operands;
  };
  std::shared_ptr<Recipe> recipe;
  GraphOptimizationRuleId planId = GraphOptimizationRuleId::AutoGraphSolver;
  unsigned originalCost = 0;
  unsigned selectedCost = 0;
  unsigned generatedCandidates = 0;
  unsigned provedEquivalent = 0;
  unsigned proofSteps = 0;
  SemanticGraphLookupMode lookupMode =
      SemanticGraphLookupMode::SemanticEnumeration;
};

struct StateReplacementAnalysis {
  SmallVector<Operation *, 4> replacedEffects;
  // Empty means that source and candidate consume the identical state.  A
  // non-empty hash is the certified common ancestor at which materialization
  // of the candidate state suffix stops.
  std::string commonBoundary;
};

static std::shared_ptr<SemanticGraphSolution::Recipe>
buildRecipe(const EquivalenceGraph &graph,
            const GraphExtraction &extraction, GraphEClassId eclass,
            llvm::DenseSet<GraphEClassId> &active, Operation *anchor) {
  eclass = graph.find(eclass);
  if (!active.insert(eclass).second)
    return {};
  const GraphExprNode *node = extraction.lookup(eclass);
  if (!node) {
    active.erase(eclass);
    return {};
  }
  auto recipe = std::make_shared<SemanticGraphSolution::Recipe>();
  bool reusableSource = node->getOperationName() == kInputOp ||
                        node->getOperationName() == kPrivateBufferOp;
  if (node->attributes)
    if (auto reusable = dyn_cast_or_null<BoolAttr>(
            node->attributes.get(kReusableSourceAttr)))
      reusableSource |= reusable.getValue();
  // Imported executable nodes carry their old SSA result for provenance, but
  // the extractor charges their operands and may select cheaper equivalent
  // children. Reusing such an SSA result would silently restore the old child
  // graph (for example, a modulo-bearing pointer below a masked load). Only a
  // true island boundary or an explicitly reusable value may bypass recursive
  // materialization.
  if (node->input && reusableSource &&
      canReuseValueAtAnchor(node->input, anchor)) {
    recipe->source = node->input;
    active.erase(eclass);
    return recipe;
  }
  if (node->input &&
      (node->getOperationName() == kInputOp ||
       node->getOperationName() == kPrivateBufferOp)) {
    active.erase(eclass);
    return {};
  }
  recipe->operationName = node->getOperationName().str();
  recipe->resultType = getResultType(*node);
  recipe->attributes = node->attributes;
  if (!recipe->resultType) {
    active.erase(eclass);
    return {};
  }
  for (GraphEClassId operand : node->operands) {
    auto child = buildRecipe(graph, extraction, operand, active, anchor);
    if (!child) {
      active.erase(eclass);
      return {};
    }
    recipe->operands.push_back(std::move(child));
  }
  active.erase(eclass);
  return recipe;
}

static bool sameRecipe(const std::shared_ptr<SemanticGraphSolution::Recipe> &lhs,
                       const std::shared_ptr<SemanticGraphSolution::Recipe> &rhs) {
  if (!lhs || !rhs)
    return lhs == rhs;
  if (lhs->source != rhs->source || lhs->proofOnly != rhs->proofOnly ||
      lhs->operationName != rhs->operationName ||
      lhs->resultType != rhs->resultType || lhs->attributes != rhs->attributes ||
      lhs->operands.size() != rhs->operands.size())
    return false;
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip_equal(lhs->operands, rhs->operands))
    if (!sameRecipe(lhsOperand, rhsOperand))
      return false;
  return true;
}

static const TTIRCandidateOpSchema *
findCandidateSchema(ArrayRef<TTIRCandidateOpSchema> schemas,
                    const GraphExprNode &node) {
  unsigned resultIndex = 0;
  if (node.attributes)
    if (auto index = dyn_cast_or_null<IntegerAttr>(
            node.attributes.get(kResultIndexAttr))) {
      if (index.getInt() < 0)
        return nullptr;
      resultIndex = static_cast<unsigned>(index.getInt());
    }
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (schema.constructible && !schema.offlineOnly &&
        node.getOperationName() == schema.operationName &&
        node.operands.size() == schema.operandCount &&
        schema.resultIndex == resultIndex)
      return &schema;
  return nullptr;
}

static std::optional<SemanticGraphSolution>
solveSemantically(Value root, ArrayRef<TTIRCandidateOpSchema> schemas) {
  if (schemas.empty())
    return std::nullopt;
  Operation *anchor = root.getDefiningOp();
  if (!anchor || anchor->getNumResults() != 1 || root.use_empty())
    return std::nullopt;

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  EquivalenceGraph graph(4096);
  TTIRSemanticImporter importer(graph, semantics, root);
  std::optional<GraphEClassId> rootClass = importer.import(root);
  if (!rootClass)
    return std::nullopt;

  std::optional<GraphEClassId> observationMaskClass;
  if (std::optional<Value> mask = getStoreObservationMask(root))
    observationMaskClass = importer.import(
        *mask, /*reusableSource=*/canReuseValueAtAnchor(*mask, anchor));

  // The candidate language may inspect any pure value in the containing
  // block. This forms a real tensor island rather than only the root's
  // producer chain and is essential for discovering alternatives from sibling
  // address/mask calculations. Such siblings are not automatically free: the
  // selected recipe must account for and rebuild their producer graph. Only
  // explicit island boundaries and independently-live observation values are
  // marked reusable. Unsupported/effectful operations stay at the boundary.
  Block *rootBlock = anchor->getBlock();
  if (observationMaskClass && rootBlock)
    for (Operation &operation : *rootBlock) {
      if (!isMemoryEffectFree(&operation) || operation.getNumRegions() != 0)
        continue;
      for (Value result : operation.getResults())
        (void)importer.import(result);
    }

  LLVM_DEBUG({
    auto rootDenotation =
        importer.getDenotations().find(graph.find(*rootClass));
    if (rootDenotation != importer.getDenotations().end()) {
      SemanticNormalizationResult normalized =
          normalizer.normalize(rootDenotation->second);
      llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                   << " root=" << anchor->getName()
                   << " normalized-value=";
      normalized.denotation.value.print(llvm::dbgs());
      llvm::dbgs() << " normalized-valid=";
      normalized.denotation.defined.print(llvm::dbgs());
      llvm::dbgs() << '\n';
    }
  });

  StructuralGraphCostModel costModel;
  GraphExtractor extractor(costModel);
  std::optional<GraphExtraction> original =
      extractor.extract(graph, *rootClass);
  if (!original)
    return std::nullopt;

  // Imported SSA values are executable e-nodes too.  Merge any pair whose
  // Value + Valid denotations prove equal before candidate enumeration.  This
  // is the e-graph analogue of discovering that an existing value already is
  // a cheaper implementation of the root (for example a load followed by a
  // permutation and its inverse).  The loop is generic over operation names;
  // admissibility still comes solely from the semantics registry.
  SemanticEquivalenceOracle importedOracle(normalizer);
  SmallVector<std::pair<GraphEClassId, TensorDenotation>, 32>
      importedDenotations;
  for (const auto &entry : importer.getDenotations())
    importedDenotations.emplace_back(entry.first, entry.second);
  unsigned importedSemanticMerges = 0;
  for (unsigned lhsIndex = 0; lhsIndex < importedDenotations.size();
       ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1;
         rhsIndex < importedDenotations.size(); ++rhsIndex) {
      GraphEClassId lhs = graph.find(importedDenotations[lhsIndex].first);
      GraphEClassId rhs = graph.find(importedDenotations[rhsIndex].first);
      if (lhs == rhs)
        continue;
      std::optional<SemanticEquivalenceProof> proof = importedOracle.prove(
          importedDenotations[lhsIndex].second,
          importedDenotations[rhsIndex].second);
      if (!proof)
        continue;
      graph.merge(lhs, rhs);
      ++importedSemanticMerges;
    }
  }
  graph.rebuild();
  *rootClass = graph.find(*rootClass);

  std::function<TensorDenotation(const TensorDenotation &)> observation;
  if (observationMaskClass) {
    auto maskDenotation =
        importer.getDenotations().find(graph.find(*observationMaskClass));
    if (maskDenotation != importer.getDenotations().end()) {
      TensorDenotation mask = maskDenotation->second;
      observation = [mask = std::move(mask)](
                        const TensorDenotation &candidate) {
        return observeUnderMask(candidate, mask);
      };
    }
  }
  const bool hasObservation = static_cast<bool>(observation);
  std::optional<TensorDenotation> semanticGoal;
  if (hasObservation) {
    auto rootDenotation =
        importer.getDenotations().find(graph.find(*rootClass));
    if (rootDenotation != importer.getDenotations().end())
      semanticGoal = normalizer.normalize(observation(rootDenotation->second))
                         .denotation;
  }
  DenseMap<GraphEClassId, unsigned> semanticPriorities;
  auto operandPriority = [&](GraphEClassId eclass) {
    eclass = graph.find(eclass);
    auto cached = semanticPriorities.find(eclass);
    if (cached != semanticPriorities.end())
      return cached->second;
    unsigned priority = 1;
    auto denotation = importer.getDenotations().find(eclass);
    if (denotation == importer.getDenotations().end())
      for (auto entry = importer.getDenotations().begin();
           entry != importer.getDenotations().end(); ++entry)
        if (graph.find(entry->first) == eclass) {
          denotation = entry;
          break;
        }
    if (semanticGoal && denotation != importer.getDenotations().end()) {
      TensorDenotation normalized =
          normalizer.normalize(denotation->second).denotation;
      if (containsSemanticExpression(semanticGoal->value, normalized.value) ||
          containsSemanticExpression(semanticGoal->defined,
                                     normalized.value))
        priority = 0;
    }
    semanticPriorities[eclass] = priority;
    return priority;
  };
  SchemaDrivenTTIRLanguage language(
      schemas, hasObservation
                   ? std::function<unsigned(GraphEClassId)>(operandPriority)
                   : std::function<unsigned(GraphEClassId)>());
  TypedCandidateEnumerator enumerator(
      language, semantics, normalizer,
      hasObservation
          ? CandidateEnumerationOptions{/*maxCandidates=*/256,
                                        /*maxRounds=*/3}
          : CandidateEnumerationOptions{/*maxCandidates=*/512,
                                        /*maxRounds=*/4},
      hasObservation ? std::optional<GraphEClassId>(*rootClass) : std::nullopt,
      std::move(observation));
  SmallVector<Type, 16> resultTypes;
  auto addResultType = [&](Type type) {
    if (type && !llvm::is_contained(resultTypes, type))
      resultTypes.push_back(type);
  };
  for (GraphEClassId available : importer.getAvailable()) {
    Type type = lookupEClassType(graph, available);
    addResultType(type);
    Type elementType = type;
    if (auto shaped = dyn_cast_or_null<RankedTensorType>(type))
      elementType = shaped.getElementType();
    if (auto integer = dyn_cast_or_null<IntegerType>(elementType);
        integer && integer.getWidth() == 64) {
      Type i32 = IntegerType::get(type.getContext(), 32);
      if (auto shaped = dyn_cast<RankedTensorType>(type))
        addResultType(shaped.clone(i32));
      else
        addResultType(i32);
    }
  }
  // Close the finite type universe under witnessed tensor containers and
  // element types. This permits a type-changing pointwise operation to move
  // across a shape permutation without teaching candidate discovery about
  // either operation. For example, [M,N]xf32 plus [N,M]xf16 witnesses the
  // missing intermediate [M,N]xf16.
  if (!hasObservation) {
    SmallVector<RankedTensorType, 8> containers;
    SmallVector<Type, 8> elementTypes;
    SmallVector<Type, 16> witnessedTypes(resultTypes);
    for (Type type : witnessedTypes) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor || !tensor.hasStaticShape())
        continue;
      if (!llvm::is_contained(containers, tensor))
        containers.push_back(tensor);
      if (!llvm::is_contained(elementTypes, tensor.getElementType()))
        elementTypes.push_back(tensor.getElementType());
    }
    for (RankedTensorType container : containers)
      for (Type elementType : elementTypes)
        addResultType(RankedTensorType::get(container.getShape(), elementType,
                                            container.getEncoding()));
  }
  addResultType(root.getType());
  CandidateEnumerationResult enumeration = enumerator.enumerate(
      graph, resultTypes, importer.getAvailable(),
      importer.getDenotations());
  enumeration.provedEquivalent += importedSemanticMerges;
  graph.rebuild();

  LLVM_DEBUG({
    llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                 << " root=" << anchor->getName()
                 << " generated=" << enumeration.generated
                 << " semantic-merges=" << enumeration.provedEquivalent
                 << " rejected=" << enumeration.rejected
                 << " hit-budget=" << enumeration.hitBudget << '\n';
  });

  *rootClass = graph.find(*rootClass);
  std::optional<GraphExtraction> selected =
      extractor.extract(graph, *rootClass);
  const GraphExprNode *selectedRoot =
      selected ? selected->lookup(selected->root) : nullptr;
  LLVM_DEBUG({
    llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                 << " root=" << anchor->getName()
                 << " original-cost=" << original->cost
                 << " selected-cost="
                 << (selected ? static_cast<int64_t>(selected->cost) : -1)
                 << " selected-root="
                 << (selectedRoot ? selectedRoot->getOperationName()
                                  : StringRef("<none>"))
                 << " selected-source="
                 << (selectedRoot && selectedRoot->input ? 1 : 0)
                 << " root-nodes=" << graph.getNodes(*rootClass).size()
                 << '\n';
  });
  if (!selected || !selectedRoot || selected->cost >= original->cost)
    return std::nullopt;
  if (selectedRoot->input &&
      !canReuseValueAtAnchor(selectedRoot->input, anchor))
    return std::nullopt;
  const TTIRCandidateOpSchema *selectedSchema =
      findCandidateSchema(schemas, *selectedRoot);
  if (!selectedSchema)
    return std::nullopt;
  llvm::DenseSet<GraphEClassId> active;
  std::shared_ptr<SemanticGraphSolution::Recipe> recipe =
      buildRecipe(graph, *selected, selected->root, active, anchor);
  if (!recipe)
    return std::nullopt;

  unsigned proofSteps = 0;
  for (const SemanticMergeRecord &record : enumeration.merges) {
    if (record.candidateOperation == selectedSchema->operationName &&
        graph.find(record.equivalentClass) == *rootClass) {
      proofSteps = static_cast<unsigned>(record.proof.steps.size());
      break;
    }
  }

  return SemanticGraphSolution{
      anchor,
      false,
      nullptr,
      {},
      std::move(recipe),
      selectedSchema->planId,
      original->cost,
      selected->cost,
      enumeration.generated,
      enumeration.provedEquivalent,
      proofSteps,
      SemanticGraphLookupMode::SemanticEnumeration};
}

static bool isStrictlyCheaper(const ParetoCandidateCost &candidate,
                              const ParetoCandidateCost &source) {
  if (candidate.stageCycles || source.stageCycles) {
    if (!candidate.stageCycles || !source.stageCycles)
      return false;
    if (*candidate.stageCycles != *source.stageCycles)
      return *candidate.stageCycles < *source.stageCycles;
    // StageModel is intentionally the primary cost. Its estimates can be
    // coarse, so equal-cycle candidates still use the deterministic complete
    // structural vector as a tie breaker.
  }
  const StructuralCostFeatures &lhs = candidate.structural;
  const StructuralCostFeatures &rhs = source.structural;
  return std::tie(lhs.memoryStridePenalty, lhs.weightedOperatorCost,
                  lhs.operatorCount, lhs.maxDepth, lhs.conversionCount,
                  lhs.memoryOperationCount) <
         std::tie(rhs.memoryStridePenalty, rhs.weightedOperatorCost,
                  rhs.operatorCount, rhs.maxDepth, rhs.conversionCount,
                  rhs.memoryOperationCount);
}

static bool isStrictlyCheaperStructural(
    const StructuralCostFeatures &candidate,
    const StructuralCostFeatures &source) {
  return std::tie(candidate.memoryStridePenalty,
                  candidate.weightedOperatorCost, candidate.operatorCount,
                  candidate.maxDepth, candidate.conversionCount,
                  candidate.memoryOperationCount) <
         std::tie(source.memoryStridePenalty,
                  source.weightedOperatorCost, source.operatorCount,
                  source.maxDepth, source.conversionCount,
                  source.memoryOperationCount);
}

static unsigned solutionCost(const ParetoCandidateCost &cost) {
  if (cost.stageCycles)
    return static_cast<unsigned>(std::min<uint64_t>(
        *cost.stageCycles, std::numeric_limits<unsigned>::max()));

  // This scalar is telemetry only; selection uses the lexicographic feature
  // vector above. Saturate each weighted addition so a malicious or corrupted
  // database cannot wrap a very expensive graph into a deceptively small
  // reported cost.
  const uint64_t limit = std::numeric_limits<unsigned>::max();
  uint64_t value = 0;
  auto add = [&](uint64_t feature, uint64_t weight) {
    if (value >= limit || feature > (limit - value) / weight) {
      value = limit;
      return;
    }
    value += feature * weight;
  };
  add(cost.structural.memoryStridePenalty, uint64_t{1} << 24);
  add(cost.structural.weightedOperatorCost, 1024);
  add(cost.structural.operatorCount, 32);
  add(cost.structural.maxDepth, 4);
  add(cost.structural.conversionCount, 2);
  add(cost.structural.memoryOperationCount, 1);
  return static_cast<unsigned>(
      std::min<uint64_t>(value, std::numeric_limits<unsigned>::max()));
}

static std::shared_ptr<SemanticGraphSolution::Recipe>
buildDatabaseRecipe(const RuleDatabase &database, StringRef graphHash,
                    ArrayRef<Value> boundaryInputs, MLIRContext *context,
                    llvm::StringSet<> &active,
                    StringRef memoryBoundary = {},
                    Operation *structuralWitnessAnchor = nullptr,
                    const llvm::StringMap<Value> *templateBindings = nullptr) {
  if (!active.insert(graphHash).second)
    return {};
  const PersistentGraphRecord *graph = database.lookupGraph(graphHash);
  if (!graph) {
    active.erase(graphHash);
    return {};
  }
  Attribute parsedAttributes = mlir::parseAttribute(graph->attributes, context);
  DictionaryAttr attributes = dyn_cast_or_null<DictionaryAttr>(parsedAttributes);
  Type resultType = mlir::parseType(graph->resultType, context);
  if (!attributes || !resultType) {
    active.erase(graphHash);
    return {};
  }

  auto recipe = std::make_shared<SemanticGraphSolution::Recipe>();
  if (!memoryBoundary.empty() && graphHash == memoryBoundary) {
    if (!isa<NoneType>(resultType)) {
      active.erase(graphHash);
      return {};
    }
    recipe->proofOnly = true;
    recipe->resultType = resultType;
    active.erase(graphHash);
    return recipe;
  }
  if (graph->operationName == kInputOp ||
      graph->operationName == kPrivateBufferOp) {
    Value source;
    if (templateBindings) {
      auto bound = templateBindings->find(graphHash);
      if (bound != templateBindings->end())
        source = bound->second;
    }
    if (!source) {
      auto symbol =
          dyn_cast_or_null<IntegerAttr>(attributes.get(kSymbolIdAttr));
      if (!symbol || symbol.getInt() < 0 ||
          static_cast<uint64_t>(symbol.getInt()) >= boundaryInputs.size()) {
        active.erase(graphHash);
        return {};
      }
      source = boundaryInputs[static_cast<unsigned>(symbol.getInt())];
    }
    const bool expectsPrivate = graph->operationName == kPrivateBufferOp;
    if (isPrivateAllocationBoundary(source) != expectsPrivate) {
      active.erase(graphHash);
      return {};
    }
    // A finite universe may contain several physical layouts for the same
    // logical tensor shape.  The symbolic input ordinal identifies the
    // boundary value, but it does not permit silently changing that value's
    // concrete MLIR type.  Reject a database recipe whose recorded input type
    // does not exactly match the online boundary value.
    if (source.getType() != resultType) {
      active.erase(graphHash);
      return {};
    }
    recipe->source = source;
    recipe->resultType = resultType;
    active.erase(graphHash);
    return recipe;
  }
  if (graph->operationName == "ags.memory" ||
      graph->operationName == "ags.partitionable_memory" ||
      graph->operationName == "ags.launch" ||
      graph->operationName == "ags.buffer") {
    recipe->proofOnly = true;
    recipe->resultType = resultType;
    active.erase(graphHash);
    return recipe;
  }
  recipe->operationName = graph->operationName;
  recipe->resultType = resultType;
  recipe->attributes = attributes;
  if (auto proofOnly =
          dyn_cast_or_null<BoolAttr>(attributes.get(kProofOnlyAttr)))
    recipe->proofOnly = proofOnly.getValue();
  // A compiler action is fully described by its certified attribute payload.
  // Its proof operands (MemoryState, LaunchState, launch bound scalars) are
  // checked by template binding and action preflight, not materialized as IR.
  if (recipe->proofOnly && attributes.get(kCompilerActionAttr)) {
    active.erase(graphHash);
    return recipe;
  }
  // ODS structural bookkeeping is not semantic identity and therefore is not
  // part of the persistent graph hash. Reuse it from any witnessed operation
  // of the same kind in the source block. The generic materializer preserves
  // the certified segment vector and verifies that its sum equals the recipe
  // arity; it never guesses which optional/dynamic segment owns an operand.
  if (structuralWitnessAnchor && structuralWitnessAnchor->getBlock()) {
    for (Operation &witness : *structuralWitnessAnchor->getBlock()) {
      if (witness.getName().getStringRef() != graph->operationName)
        continue;
      SmallVector<NamedAttribute, 8> witnessed(attributes.begin(),
                                               attributes.end());
      for (StringRef name : {StringRef("operandSegmentSizes"),
                             StringRef("resultSegmentSizes")})
        if (!attributes.get(name))
          if (Attribute value = witness.getAttr(name))
            witnessed.push_back(Builder(context).getNamedAttr(name, value));
      recipe->attributes = Builder(context).getDictionaryAttr(witnessed);
      break;
    }
  }
  for (StringRef operandHash : graph->operandHashes) {
    const PersistentGraphRecord *operandGraph =
        database.lookupGraph(operandHash);
    if (!operandGraph) {
      active.erase(graphHash);
      return {};
    }
    Type operandType = mlir::parseType(operandGraph->resultType, context);
    if (!operandType) {
      active.erase(graphHash);
      return {};
    }
    // MemoryState is an edge in the proof graph, not a real TTIR operand.
    // Once canonical lookup has matched the complete state history, recipe
    // materialization may erase the whole state subtree without inspecting
    // or recreating stores/atomics that already executed in the source IR.
    bool carriesCompilerAction = false;
    if (isa<NoneType>(operandType)) {
      Attribute operandAttributes =
          mlir::parseAttribute(operandGraph->attributes, context);
      if (auto dictionary =
              dyn_cast_or_null<DictionaryAttr>(operandAttributes))
        carriesCompilerAction =
            static_cast<bool>(dictionary.get(kModuleAttributePatchAttr));
    }
    if (isa<NoneType>(operandType) && memoryBoundary.empty() &&
        !carriesCompilerAction) {
      auto operand = std::make_shared<SemanticGraphSolution::Recipe>();
      operand->proofOnly = true;
      operand->resultType = operandType;
      recipe->operands.push_back(std::move(operand));
      continue;
    }
    std::shared_ptr<SemanticGraphSolution::Recipe> operand =
        buildDatabaseRecipe(database, operandHash, boundaryInputs, context,
                            active, memoryBoundary, structuralWitnessAnchor,
                            templateBindings);
    if (!operand) {
      active.erase(graphHash);
      return {};
    }
    recipe->operands.push_back(std::move(operand));
  }
  active.erase(graphHash);
  return recipe;
}

static std::optional<llvm::StringMap<Value>> bindTemplateValues(
    const RuleDatabase &database, const RuleDatabaseTemplateMatch &match,
    const CanonicalGraphBuildResult &canonical) {
  llvm::StringMap<Value> onlineValues;
  for (const auto &[hash, value] : canonical.graphValues)
    onlineValues.try_emplace(hash, value);
  llvm::StringMap<Value> bindings;
  for (const auto &[patternHash, onlineHash] : match.bindings) {
    const PersistentGraphRecord *pattern = database.lookupGraph(patternHash);
    if (!pattern)
      return std::nullopt;
    if (pattern->resultType == "none")
      continue;
    auto value = onlineValues.find(onlineHash);
    if (value == onlineValues.end())
      return std::nullopt;
    bindings.try_emplace(patternHash, value->second);
  }
  return bindings;
}

static std::optional<StructuralCostFeatures> estimateBoundTemplateCost(
    const RuleDatabase &database, llvm::StringRef candidateRoot,
    const RuleDatabaseTemplateMatch &match,
    const CanonicalGraphBuildResult &canonical,
    ArrayRef<TTIRCandidateOpSchema> schemas) {
  // Template matching deliberately treats a shared producer or a block
  // argument as an opaque typed hole.  That is the correct semantic boundary,
  // but it is too shallow for layout costing: a store inside scf.for commonly
  // sees only the loop-carried pointer argument even though its affine origin
  // is available as the corresponding init value.  Re-import each bound SSA
  // value as a temporary cost-only root.  CanonicalTTIRGraphBuilder remains
  // the single graph language here; this does not enumerate candidates or
  // change the template binding used by materialization.
  auto getCostOrigin = [](Value value) {
    llvm::SmallPtrSet<const void *, 4> visited;
    while (auto argument = dyn_cast<BlockArgument>(value)) {
      if (!visited.insert(argument.getAsOpaquePointer()).second)
        break;
      Block *owner = argument.getOwner();
      auto loop = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
      if (!loop || loop.getBody() != owner || argument.getArgNumber() == 0)
        break;
      unsigned index = argument.getArgNumber() - 1;
      if (index >= loop.getInitArgs().size())
        break;
      value = loop.getInitArgs()[index];
    }
    return value;
  };

  llvm::StringMap<const PersistentGraphRecord *> onlineGraphs;
  for (const PersistentGraphRecord &graph : canonical.graphs)
    onlineGraphs.try_emplace(graph.graphHash, &graph);
  llvm::StringMap<std::string> substitutions;
  llvm::StringMap<Value> onlineValues;
  for (const auto &[hash, value] : canonical.graphValues)
    onlineValues.try_emplace(hash, value);

  std::vector<PersistentGraphRecord> costGraphs;
  llvm::StringMap<std::string> costRootsByOnlineHash;
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  for (const auto &[patternHash, onlineHash] : match.bindings) {
    substitutions.try_emplace(patternHash, onlineHash);
    auto value = onlineValues.find(onlineHash);
    if (value == onlineValues.end())
      continue;
    std::string error;
    std::optional<CanonicalGraphBuildResult> costGraph =
        CanonicalTTIRGraphBuilder(semantics, schemas)
            .build(getCostOrigin(value->second), error);
    if (!costGraph)
      continue;
    // A bound subgraph may contain an ags.input leaf with the same canonical
    // hash as the template variable being substituted. Keeping the imported
    // hashes unchanged would make that leaf resolve back to the bound root and
    // create an artificial cycle. Namespace this cost-only copy so template
    // substitution applies only at pattern edges, never inside the binding.
    std::string prefix = "ags.bound-cost.";
    prefix += patternHash;
    prefix += ".";
    prefix += onlineHash;
    prefix += ".";
    llvm::StringMap<std::string> renamed;
    for (const PersistentGraphRecord &graph : costGraph->graphs)
      renamed.try_emplace(graph.graphHash, prefix + graph.graphHash);
    for (const PersistentGraphRecord &graph : costGraph->graphs) {
      PersistentGraphRecord copy = graph;
      copy.graphHash = renamed.lookup(graph.graphHash);
      for (std::string &operand : copy.operandHashes)
        if (auto replacement = renamed.find(operand);
            replacement != renamed.end())
          operand = replacement->second;
      costGraphs.push_back(std::move(copy));
    }
    const std::string namespacedRoot =
        renamed.lookup(costGraph->rootGraphHash);
    costRootsByOnlineHash.try_emplace(onlineHash, namespacedRoot);
    substitutions[patternHash] = namespacedRoot;
  }
  llvm::StringMap<const PersistentGraphRecord *> costGraphsByHash;
  for (const PersistentGraphRecord &graph : costGraphs)
    costGraphsByHash.try_emplace(graph.graphHash, &graph);

  auto lookup = [&](llvm::StringRef hash) -> const PersistentGraphRecord * {
    if (auto substitution = substitutions.find(hash);
        substitution != substitutions.end()) {
      auto cost = costGraphsByHash.find(substitution->second);
      if (cost != costGraphsByHash.end())
        return cost->second;
      auto online = onlineGraphs.find(substitution->second);
      return online == onlineGraphs.end() ? nullptr : online->second;
    }
    if (auto costRoot = costRootsByOnlineHash.find(hash);
        costRoot != costRootsByOnlineHash.end()) {
      auto cost = costGraphsByHash.find(costRoot->second);
      if (cost != costGraphsByHash.end())
        return cost->second;
    }
    if (auto cost = costGraphsByHash.find(hash);
        cost != costGraphsByHash.end())
      return cost->second;
    if (auto online = onlineGraphs.find(hash); online != onlineGraphs.end())
      return online->second;
    return database.lookupGraph(hash);
  };
  return estimateStructuralCost(candidateRoot, lookup);
}

static std::optional<StringRef>
getPreviousMemoryState(const RuleDatabase &database,
                       const PersistentGraphRecord &graph);

static std::optional<StateReplacementAnalysis>
getBypassedStateEffects(const RuleDatabase &database,
                        const CanonicalGraphBuildResult &canonical,
                        const PersistentGraphRecord &sourceRoot,
                        const PersistentGraphRecord &candidateRoot,
                        Operation *anchor,
                        ArrayRef<TTIRCandidateOpSchema> schemas);

static bool graphContainsStateToken(const RuleDatabase &database,
                                    StringRef graphHash,
                                    llvm::StringSet<> &visited) {
  if (!visited.insert(graphHash).second)
    return false;
  const PersistentGraphRecord *graph = database.lookupGraph(graphHash);
  if (!graph || graph->resultType == "none")
    return true;
  return llvm::any_of(graph->operandHashes, [&](StringRef operand) {
    return graphContainsStateToken(database, operand, visited);
  });
}

static bool graphIsPureOrReadOnly(
    const RuleDatabase &database, StringRef graphHash,
    ArrayRef<TTIRCandidateOpSchema> schemas, llvm::StringSet<> &visited) {
  if (!visited.insert(graphHash).second)
    return true;
  const PersistentGraphRecord *graph = database.lookupGraph(graphHash);
  if (!graph)
    return false;
  if (graph->operandHashes.empty())
    return graph->operationName == "ags.input" ||
           graph->operationName == "ags.private_buffer" ||
           graph->operationName == "ags.memory" ||
           graph->operationName == "ags.buffer" ||
           graph->operationName == "ags.launch";
  bool safeOperation = false;
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (schema.operationName == graph->operationName &&
        (schema.effectKind == TTIRCandidateEffectKind::Pure ||
         schema.effectKind == TTIRCandidateEffectKind::Read ||
         schema.effectKind == TTIRCandidateEffectKind::BufferRead)) {
      safeOperation = true;
      break;
    }
  if (!safeOperation)
    return false;
  return llvm::all_of(graph->operandHashes, [&](StringRef operand) {
    return graphIsPureOrReadOnly(database, operand, schemas, visited);
  });
}

static std::optional<SemanticGraphSolution> solveFromRuleDatabase(
    Value root, ArrayRef<TTIRCandidateOpSchema> schemas,
    const GraphOptimizationOptions &options, const RuleDatabase &database,
    std::function<bool()> shouldStop = {}) {
  Operation *anchor = root.getDefiningOp();
  if (!anchor || anchor->getNumResults() != 1 || root.use_empty())
    return std::nullopt;
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas, std::move(shouldStop))
          .build(root, error);
  if (!canonical)
    return std::nullopt;
  GraphCostContext context;
  context.target = options.compileOn91095 ? "ascend950dt" : "generic";
  OnlineRuleLookupResult lookup = OnlineRuleMatcher(database).lookup(
      canonical->rootGraphHash, database.universeHash,
      getBuiltinTTOpSemanticsStableHash(), context.getStableKey());
  if (lookup.status == OnlineRuleLookupStatus::VersionMismatch)
    return std::nullopt;

  const RuleDatabaseCandidate *selected = nullptr;
  const RuleDatabaseSemanticEntry *selectedSemanticEntry = nullptr;
  const RuleDatabaseTemplateEntry *selectedTemplateEntry = nullptr;
  std::optional<llvm::StringMap<Value>> selectedTemplateBindings;
  std::optional<StructuralCostFeatures> selectedTemplateSourceStructural;
  std::optional<StructuralCostFeatures> selectedTemplateCandidateStructural;
  std::optional<StructuralCostFeatures> selectedSemanticCandidateStructural;
  const ParetoCandidateCost *selectedSourceCost = nullptr;
  unsigned selectedCandidateCount = 0;
  unsigned selectedProofCount = 0;
  SemanticGraphLookupMode selectedLookupMode =
      SemanticGraphLookupMode::RuleDatabaseExact;
  ParetoCandidateCost semanticSourceCost;
  bool hasSemanticSourceCost = false;
  if (lookup.status == OnlineRuleLookupStatus::Hit && lookup.entry) {
    for (const RuleDatabaseCandidate &candidate : lookup.entry->candidates) {
      if (candidate.rootGraphHash == canonical->rootGraphHash ||
          !isStrictlyCheaper(candidate.cost, lookup.entry->sourceCost))
        continue;
      if (!selected || isStrictlyCheaper(candidate.cost, selected->cost))
        selected = &candidate;
    }
    if (selected) {
      selectedSourceCost = &lookup.entry->sourceCost;
      selectedCandidateCount = lookup.entry->candidates.size();
      selectedProofCount = lookup.entry->proofHashes.size();
    }
  }

  // Symbolic templates are ordinary replay-validated graph equalities whose
  // ags.input leaves are universally quantified typed holes. Matching only
  // binds those holes to existing SSA values; it performs no online search or
  // proof. Compiler actions are handled by the whole-kernel path instead.
  if (database.semanticsHash == getBuiltinTTOpSemanticsStableHash() &&
      database.costContextKey == context.getStableKey()) {
    std::vector<RuleDatabaseTemplateMatch> templateMatches =
        database.matchTemplates(canonical->rootGraphHash, canonical->graphs);
    for (const RuleDatabaseTemplateMatch &match : templateMatches) {
      if (!match.entry)
        continue;
      std::optional<llvm::StringMap<Value>> bindings =
          bindTemplateValues(database, match, *canonical);
      std::optional<StructuralCostFeatures> sourceStructural =
          estimateBoundTemplateCost(database, canonical->rootGraphHash,
                                    match, *canonical, schemas);
      if (!bindings || !sourceStructural)
        continue;
      for (const RuleDatabaseCandidate &candidate :
           match.entry->candidates) {
        const PersistentGraphRecord *candidateRoot =
            database.lookupGraph(candidate.rootGraphHash);
        if (!candidateRoot || candidateRoot->resultType == "none")
          continue;
        Attribute parsed = mlir::parseAttribute(candidateRoot->attributes,
                                                root.getContext());
        auto attributes = dyn_cast_or_null<DictionaryAttr>(parsed);
        std::optional<StructuralCostFeatures> candidateStructural =
            estimateBoundTemplateCost(database, candidate.rootGraphHash,
                                      match, *canonical, schemas);
        if (!attributes || attributes.get(kCompilerActionAttr) ||
            !candidateStructural ||
            !isStrictlyCheaperStructural(*candidateStructural,
                                         *sourceStructural))
          continue;
        llvm::StringSet<> visited;
        if (!graphIsPureOrReadOnly(database, candidate.rootGraphHash,
                                   schemas, visited))
          continue;
        const StructuralCostFeatures *selectedStructural =
            selectedTemplateCandidateStructural
                ? &*selectedTemplateCandidateStructural
                : selected ? &selected->cost.structural : nullptr;
        if (!selectedStructural || isStrictlyCheaperStructural(
                                       *candidateStructural,
                                       *selectedStructural)) {
          selected = &candidate;
          selectedSemanticEntry = nullptr;
          selectedTemplateEntry = match.entry;
          llvm::StringMap<Value> copiedBindings;
          for (const auto &binding : *bindings)
            copiedBindings.try_emplace(binding.getKey(), binding.getValue());
          selectedTemplateBindings = std::move(copiedBindings);
          selectedTemplateSourceStructural = *sourceStructural;
          selectedTemplateCandidateStructural = *candidateStructural;
          selectedSourceCost = &match.entry->sourceCost;
          selectedCandidateCount = match.entry->candidates.size();
          selectedProofCount = match.entry->proofHashes.size();
          selectedLookupMode =
              SemanticGraphLookupMode::RuleDatabaseTemplate;
        }
      }
    }
  }

  // A miss in the exact topology index may still hit an offline-certified
  // semantic normal form. Online work is limited to interpreting and costing
  // the source once followed by an indexed lookup; no alternative graph is
  // generated and no pairwise equivalence proof is run here.
  // Always consult the semantic index, even after an exact/template hit. A
  // partitioned offline universe can place equivalent candidates in different
  // shards; RuleDatabase::merge joins their semantic entries, while an exact
  // entry necessarily contains only the source shard's local candidates.
  if (!canonical->normalizedSemanticKey.empty()) {
    auto lookupSourceGraph = [&](StringRef hash)
        -> const PersistentGraphRecord * {
      auto found = llvm::find_if(canonical->graphs, [&](const auto &graph) {
        return graph.graphHash == hash;
      });
      if (found != canonical->graphs.end())
        return &*found;
      return database.lookupGraph(hash);
    };
    std::optional<StructuralCostFeatures> structural =
        estimateStructuralCost(canonical->rootGraphHash, lookupSourceGraph);
    if (structural) {
      semanticSourceCost.structural = *structural;
      hasSemanticSourceCost = true;
    }
    std::string resultType = printType(root.getType());
    for (const RuleDatabaseSemanticEntry *entry : database.lookupSemantic(
             canonical->normalizedSemanticKey, resultType)) {
      for (const RuleDatabaseCandidate &candidate : entry->candidates) {
        std::optional<StructuralCostFeatures> candidateStructural;
        if (!lookup.entry)
          candidateStructural = estimateStructuralCost(
              candidate.rootGraphHash,
              [&](StringRef hash) { return database.lookupGraph(hash); });
        bool improvesSource = false;
        if (lookup.entry)
          improvesSource =
              isStrictlyCheaper(candidate.cost, lookup.entry->sourceCost);
        else if (hasSemanticSourceCost && candidateStructural)
          improvesSource = isStrictlyCheaperStructural(
              *candidateStructural, semanticSourceCost.structural);
        if (candidate.rootGraphHash == canonical->rootGraphHash ||
            !improvesSource)
          continue;
        llvm::StringSet<> visited;
        if (graphContainsStateToken(database, candidate.rootGraphHash,
                                    visited))
          continue;
        bool improvesSelected = !selected;
        if (selected) {
          if (selectedTemplateCandidateStructural)
            improvesSelected = isStrictlyCheaperStructural(
                lookup.entry ? candidate.cost.structural
                             : *candidateStructural,
                *selectedTemplateCandidateStructural);
          else if (!lookup.entry && selectedSemanticCandidateStructural)
            improvesSelected = isStrictlyCheaperStructural(
                *candidateStructural,
                *selectedSemanticCandidateStructural);
          else
            improvesSelected =
                isStrictlyCheaper(candidate.cost, selected->cost);
        }
        if (improvesSelected) {
          selected = &candidate;
          selectedSemanticEntry = entry;
          selectedTemplateEntry = nullptr;
          selectedTemplateBindings.reset();
          selectedTemplateSourceStructural.reset();
          selectedTemplateCandidateStructural.reset();
          selectedSemanticCandidateStructural = candidateStructural;
          selectedSourceCost =
              lookup.entry ? &lookup.entry->sourceCost : nullptr;
          selectedCandidateCount = entry->candidates.size();
          selectedProofCount = entry->proofHashes.size();
          selectedLookupMode =
              SemanticGraphLookupMode::RuleDatabaseSemantic;
        }
      }
    }
  }
  if (!selected)
    return std::nullopt;

  const PersistentGraphRecord *candidateRoot =
      database.lookupGraph(selected->rootGraphHash);
  if (!candidateRoot)
    return std::nullopt;
  StateReplacementAnalysis stateReplacement;
  if (!selectedSemanticEntry && !selectedTemplateEntry) {
    const PersistentGraphRecord *sourceRoot =
        database.lookupGraph(canonical->rootGraphHash);
    if (!sourceRoot)
      return std::nullopt;
    std::optional<StateReplacementAnalysis> replacement =
        getBypassedStateEffects(database, *canonical, *sourceRoot,
                                *candidateRoot, anchor, schemas);
    if (!replacement)
      return std::nullopt;
    stateReplacement = std::move(*replacement);
  }

  llvm::StringSet<> active;
  std::shared_ptr<SemanticGraphSolution::Recipe> recipe =
      buildDatabaseRecipe(database, selected->rootGraphHash,
                          canonical->boundaryInputs, root.getContext(), active,
                          stateReplacement.commonBoundary, anchor,
                          selectedTemplateBindings
                              ? &*selectedTemplateBindings
                              : nullptr);
  if (!recipe || (recipe->resultType && recipe->resultType != root.getType()))
    return std::nullopt;

  // An unseen semantic source has no offline StageModel score, so compare
  // structural cost on both sides. If the source also has an exact entry,
  // however, its source cost and every merged semantic candidate share one
  // versioned cost model and must participate in the global Stage comparison.
  unsigned originalCost = 0;
  unsigned selectedCost = 0;
  if (selectedSemanticEntry) {
    if (selectedSourceCost) {
      originalCost = solutionCost(*selectedSourceCost);
      selectedCost = solutionCost(selected->cost);
    } else {
      originalCost = solutionCost(semanticSourceCost);
      selectedCost = solutionCost(ParetoCandidateCost{
          *selectedSemanticCandidateStructural, std::nullopt});
    }
  } else if (selectedTemplateEntry) {
    originalCost = solutionCost(ParetoCandidateCost{
        *selectedTemplateSourceStructural, std::nullopt});
    selectedCost = solutionCost(ParetoCandidateCost{
        *selectedTemplateCandidateStructural, std::nullopt});
  } else {
    originalCost = solutionCost(*selectedSourceCost);
    selectedCost = solutionCost(selected->cost);
  }
  return SemanticGraphSolution{
      anchor,
      false,
      nullptr,
      std::move(stateReplacement.replacedEffects),
      std::move(recipe),
      GraphOptimizationRuleId::AutoGraphSolver,
      originalCost,
      selectedCost,
      selectedCandidateCount,
      1,
      selectedProofCount,
      selectedLookupMode};
}

static std::optional<StringRef>
getPreviousMemoryState(const RuleDatabase &database,
                       const PersistentGraphRecord &graph) {
  auto lookup = [&](StringRef hash) { return database.lookupGraph(hash); };
  if (graph.operationName == "ags.memory")
    return std::nullopt;
  if (graph.operationName == "ags.memory_after") {
    if (graph.operandHashes.size() != 1)
      return std::nullopt;
    const PersistentGraphRecord *effect = lookup(graph.operandHashes.front());
    if (!effect || effect->operandHashes.empty())
      return std::nullopt;
    const PersistentGraphRecord *state = lookup(effect->operandHashes.back());
    return state && state->resultType == "none"
               ? std::optional<StringRef>(state->graphHash)
               : std::nullopt;
  }
  if (graph.resultType != "none" || graph.operandHashes.empty())
    return std::nullopt;
  const PersistentGraphRecord *state = lookup(graph.operandHashes.back());
  return state && state->resultType == "none"
             ? std::optional<StringRef>(state->graphHash)
             : std::nullopt;
}

static std::optional<StringRef> getPreviousMemoryState(
    const PersistentGraphRecord &graph,
    llvm::function_ref<const PersistentGraphRecord *(StringRef)> lookup) {
  if (graph.operationName == "ags.memory")
    return std::nullopt;
  if (graph.operationName == "ags.memory_after") {
    if (graph.operandHashes.size() != 1)
      return std::nullopt;
    const PersistentGraphRecord *effect = lookup(graph.operandHashes.front());
    if (!effect || effect->operandHashes.empty())
      return std::nullopt;
    const PersistentGraphRecord *state = lookup(effect->operandHashes.back());
    return state && state->resultType == "none"
               ? std::optional<StringRef>(state->graphHash)
               : std::nullopt;
  }
  if (graph.resultType != "none" || graph.operandHashes.empty())
    return std::nullopt;
  const PersistentGraphRecord *state = lookup(graph.operandHashes.back());
  return state && state->resultType == "none"
             ? std::optional<StringRef>(state->graphHash)
             : std::nullopt;
}

static bool collectEmbeddedStateOperands(const RuleDatabase &database,
                                         const PersistentGraphRecord &graph,
                                         llvm::StringSet<> &visited,
                                         llvm::StringSet<> &states) {
  if (!visited.insert(graph.graphHash).second)
    return true;
  for (StringRef operandHash : graph.operandHashes) {
    const PersistentGraphRecord *operand = database.lookupGraph(operandHash);
    if (!operand)
      return false;
    if (operand->resultType == "none") {
      states.insert(operand->graphHash);
      continue;
    }
    if (!collectEmbeddedStateOperands(database, *operand, visited, states))
      return false;
  }
  return true;
}

struct CommonStateAncestor {
  std::string hash;
  unsigned sourceDepth = 0;
};

static std::optional<CommonStateAncestor>
findCommonStateAncestor(const RuleDatabase &database, StringRef source,
                        StringRef candidate) {
  llvm::StringMap<unsigned> sourceDepth;
  llvm::StringSet<> visited;
  const PersistentGraphRecord *state = database.lookupGraph(source);
  unsigned depth = 0;
  while (state) {
    if (!visited.insert(state->graphHash).second)
      return std::nullopt;
    sourceDepth.try_emplace(state->graphHash, depth++);
    std::optional<StringRef> previous = getPreviousMemoryState(database, *state);
    state = previous ? database.lookupGraph(*previous) : nullptr;
  }

  visited.clear();
  state = database.lookupGraph(candidate);
  while (state) {
    if (!visited.insert(state->graphHash).second)
      return std::nullopt;
    auto known = sourceDepth.find(state->graphHash);
    if (known != sourceDepth.end())
      return CommonStateAncestor{state->graphHash, known->second};
    std::optional<StringRef> previous = getPreviousMemoryState(database, *state);
    state = previous ? database.lookupGraph(*previous) : nullptr;
  }
  return std::nullopt;
}

static bool collectBoundarySymbols(const RuleDatabase &database,
                                   const PersistentGraphRecord &graph,
                                   MLIRContext *context,
                                   llvm::SmallDenseSet<unsigned, 8> &symbols,
                                   llvm::StringSet<> &visited) {
  if (!visited.insert(graph.graphHash).second)
    return true;
  if (graph.operationName == kInputOp ||
      graph.operationName == kPrivateBufferOp) {
    Attribute parsed = mlir::parseAttribute(graph.attributes, context);
    auto attributes = dyn_cast_or_null<DictionaryAttr>(parsed);
    auto symbol = attributes
                      ? dyn_cast_or_null<IntegerAttr>(
                            attributes.get(kSymbolIdAttr))
                      : IntegerAttr{};
    if (!symbol || symbol.getInt() < 0)
      return false;
    symbols.insert(static_cast<unsigned>(symbol.getInt()));
    return true;
  }
  for (StringRef operandHash : graph.operandHashes) {
    const PersistentGraphRecord *operand = database.lookupGraph(operandHash);
    if (!operand || !collectBoundarySymbols(database, *operand, context,
                                            symbols, visited))
      return false;
  }
  return true;
}

static Value getViewRoot(Value value) {
  llvm::SmallDenseSet<Value, 4> visited;
  while (value && visited.insert(value).second) {
    auto view = dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp());
    if (!view)
      break;
    value = view.getViewSource();
  }
  return value;
}

static bool isReadOnlyBufferOperation(
    Operation *operation, ArrayRef<TTIRCandidateOpSchema> schemas) {
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (schema.operationName == operation->getName().getStringRef() &&
        schema.operandCount == operation->getNumOperands() &&
        (!schema.matchesOperation || schema.matchesOperation(operation)))
      return schema.effectKind == TTIRCandidateEffectKind::BufferRead;
  return false;
}

static bool isConfinedToReplacementRoot(
    Operation *operation, Operation *target,
    ArrayRef<TTIRCandidateOpSchema> schemas,
    SmallPtrSetImpl<Operation *> &visited) {
  if (operation == target)
    return true;
  if (!operation || !visited.insert(operation).second)
    return true;
  if ((!isMemoryEffectFree(operation) &&
       !isReadOnlyBufferOperation(operation, schemas)) ||
      operation->getNumRegions() != 0)
    return false;
  for (Value result : operation->getResults())
    for (Operation *user : result.getUsers())
      if (!isConfinedToReplacementRoot(user, target, schemas, visited))
        return false;
  return true;
}

static void collectViewAliases(Value value,
                               llvm::SmallDenseSet<Value, 8> &aliases) {
  if (!value || !aliases.insert(value).second)
    return;
  for (Operation *user : value.getUsers()) {
    auto view = dyn_cast<ViewLikeOpInterface>(user);
    if (!view || view.getViewSource() != value)
      continue;
    for (Value result : user->getResults())
      collectViewAliases(result, aliases);
  }
}

static std::optional<StateReplacementAnalysis>
getBypassedStateEffects(const RuleDatabase &database,
                        const CanonicalGraphBuildResult &canonical,
                        const PersistentGraphRecord &sourceRoot,
                        const PersistentGraphRecord &candidateRoot,
                        Operation *anchor,
                        ArrayRef<TTIRCandidateOpSchema> schemas) {
  llvm::StringSet<> sourceStateSet;
  llvm::StringSet<> candidateStateSet;
  llvm::StringSet<> visited;
  if (!collectEmbeddedStateOperands(database, sourceRoot, visited,
                                    sourceStateSet))
    return std::nullopt;
  visited.clear();
  if (!collectEmbeddedStateOperands(database, candidateRoot, visited,
                                    candidateStateSet))
    return std::nullopt;
  SmallVector<StringRef, 2> sourceStates;
  SmallVector<StringRef, 2> candidateStates;
  for (const auto &state : sourceStateSet)
    sourceStates.push_back(state.getKey());
  for (const auto &state : candidateStateSet)
    candidateStates.push_back(state.getKey());
  if (sourceStates.size() != candidateStates.size())
    return std::nullopt;

  llvm::StringMap<Operation *> sourceOperations;
  for (const auto &[hash, operation] : canonical.stateOperations)
    sourceOperations.try_emplace(hash, operation);
  SmallVector<Operation *, 4> bypassed;
  llvm::SmallPtrSet<Operation *, 8> seenOperations;
  llvm::SmallDenseSet<unsigned, 4> usedCandidateStates;
  std::string commonBoundary;
  for (StringRef sourceState : sourceStates) {
    std::optional<unsigned> selectedCandidate;
    std::optional<CommonStateAncestor> selectedCommon;
    for (auto [index, candidateState] : llvm::enumerate(candidateStates)) {
      if (usedCandidateStates.contains(index))
        continue;
      std::optional<CommonStateAncestor> common = findCommonStateAncestor(
          database, sourceState, candidateState);
      if (!common)
        continue;
      if (!selectedCommon || common->sourceDepth < selectedCommon->sourceDepth) {
        selectedCandidate = index;
        selectedCommon = std::move(common);
      }
    }
    if (!selectedCandidate || !selectedCommon)
      return std::nullopt;
    usedCandidateStates.insert(*selectedCandidate);
    if (commonBoundary.empty())
      commonBoundary = selectedCommon->hash;
    else if (commonBoundary != selectedCommon->hash)
      return std::nullopt;

    const PersistentGraphRecord *state = database.lookupGraph(sourceState);
    for (unsigned index = 0; index < selectedCommon->sourceDepth; ++index) {
      if (!state)
        return std::nullopt;
      auto operation = sourceOperations.find(state->graphHash);
      if (operation == sourceOperations.end() ||
          (operation->second->getNumResults() != 0 &&
           !operation->second->use_empty()))
        return std::nullopt;
      if (seenOperations.insert(operation->second).second)
        bypassed.push_back(operation->second);
      std::optional<StringRef> previous =
          getPreviousMemoryState(database, *state);
      state = previous ? database.lookupGraph(*previous) : nullptr;
    }
  }
  if (bypassed.empty())
    return StateReplacementAnalysis{std::move(bypassed),
                                    std::move(commonBoundary)};

  llvm::SmallDenseSet<unsigned, 8> candidateSymbols;
  visited.clear();
  if (!collectBoundarySymbols(database, candidateRoot, anchor->getContext(),
                              candidateSymbols, visited))
    return std::nullopt;
  llvm::SmallDenseSet<Value, 8> candidateBufferRoots;
  for (unsigned symbol : candidateSymbols) {
    if (symbol >= canonical.boundaryInputs.size())
      return std::nullopt;
    Value boundary = canonical.boundaryInputs[symbol];
    if (isa<BaseMemRefType>(boundary.getType()))
      candidateBufferRoots.insert(getViewRoot(boundary));
  }

  // State-eliding value rewrites are legal only for stage-local buffers whose
  // allocation is private to the eliminated suffix and the value root. A
  // global MemoryState suffix (or an escaping buffer) is observable even when
  // the rewritten value itself is equal, so never infer its deadness from a
  // value certificate alone.
  llvm::SmallPtrSet<Operation *, 8> bypassedSet;
  for (Operation *operation : bypassed)
    bypassedSet.insert(operation);
  SmallVector<Operation *, 4> lifetimeCleanup;
  for (Operation *transition : bypassed) {
    const TTIRCandidateOpSchema *schema = nullptr;
    for (const TTIRCandidateOpSchema &candidate : schemas)
      if (candidate.operationName == transition->getName().getStringRef() &&
          candidate.operandCount == transition->getNumOperands()) {
        schema = &candidate;
        break;
      }
    if (!schema ||
        schema->effectKind != TTIRCandidateEffectKind::BufferWrite)
      return std::nullopt;

    auto transitionEffects = dyn_cast<MemoryEffectOpInterface>(transition);
    if (!transitionEffects)
      return std::nullopt;
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    transitionEffects.getEffects(effects);
    SmallVector<Value, 2> writtenBuffers;
    for (const MemoryEffects::EffectInstance &effect : effects)
      if (isa<MemoryEffects::Write>(effect.getEffect()) && effect.getValue())
        writtenBuffers.push_back(effect.getValue());
    if (writtenBuffers.empty())
      return std::nullopt;

    for (Value buffer : writtenBuffers) {
      Value bufferRoot = getViewRoot(buffer);
      Operation *allocation = bufferRoot.getDefiningOp();
      auto allocationEffects =
          dyn_cast_or_null<MemoryEffectOpInterface>(allocation);
      if (!allocationEffects)
        return std::nullopt;
      SmallVector<MemoryEffects::EffectInstance, 4> allocationEffectList;
      allocationEffects.getEffects(allocationEffectList);
      bool ownsBuffer = llvm::any_of(
          allocationEffectList,
          [&](const MemoryEffects::EffectInstance &effect) {
            return isa<MemoryEffects::Allocate>(effect.getEffect()) &&
                   getViewRoot(effect.getValue()) == bufferRoot;
          });
      if (!ownsBuffer)
        return std::nullopt;

      llvm::SmallDenseSet<Value, 8> aliases;
      collectViewAliases(bufferRoot, aliases);
      SmallVector<Operation *, 2> frees;
      for (Value alias : aliases) {
        for (Operation *user : alias.getUsers()) {
          if (user == anchor || bypassedSet.contains(user) ||
              isa<ViewLikeOpInterface>(user))
            continue;
          auto userEffects = dyn_cast<MemoryEffectOpInterface>(user);
          SmallVector<MemoryEffects::EffectInstance, 4> userEffectList;
          if (userEffects)
            userEffects.getEffects(userEffectList);
          bool onlyFreesBuffer = !userEffectList.empty() && llvm::all_of(
              userEffectList,
              [&](const MemoryEffects::EffectInstance &effect) {
                return isa<MemoryEffects::Free>(effect.getEffect()) &&
                       getViewRoot(effect.getValue()) == bufferRoot;
              });
          if (onlyFreesBuffer) {
            frees.push_back(user);
            continue;
          }
          SmallPtrSet<Operation *, 16> path;
          if (!isConfinedToReplacementRoot(user, anchor, schemas, path))
            return std::nullopt;
        }
      }
      if (!candidateBufferRoots.contains(bufferRoot)) {
        lifetimeCleanup.append(frees.begin(), frees.end());
        lifetimeCleanup.push_back(allocation);
      }
    }
  }

  if (!anchor || llvm::any_of(bypassed, [&](Operation *operation) {
        return operation->getBlock() != anchor->getBlock();
      }))
    return std::nullopt;
  llvm::sort(bypassed, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  for (Operation *operation = bypassed.front(); operation;
       operation = operation->getNextNode()) {
    if (operation != anchor && !bypassedSet.contains(operation) &&
        !isMemoryEffectFree(operation)) {
      SmallPtrSet<Operation *, 16> path;
      if (!isReadOnlyBufferOperation(operation, schemas) ||
          !isConfinedToReplacementRoot(operation, anchor, schemas, path))
        return std::nullopt;
    }
    if (operation == anchor)
      break;
  }
  for (Operation *cleanup : lifetimeCleanup)
    if (!bypassedSet.contains(cleanup)) {
      if (cleanup->getBlock() != anchor->getBlock())
        return std::nullopt;
      bypassedSet.insert(cleanup);
      bypassed.push_back(cleanup);
    }
  llvm::sort(bypassed, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return StateReplacementAnalysis{std::move(bypassed),
                                  std::move(commonBoundary)};
}

static std::optional<StringRef> getImplicitStateOperand(
    const PersistentGraphRecord &graph, TTIRCandidateStateDomain domain,
    ArrayRef<TTIRCandidateOpSchema> schemas) {
  for (const TTIRCandidateOpSchema &schema : schemas) {
    if (schema.operationName != graph.operationName ||
        schema.operandCount + schema.implicitStateDomains.size() !=
            graph.operandHashes.size())
      continue;
    auto state = llvm::find(schema.implicitStateDomains, domain);
    if (state == schema.implicitStateDomains.end())
      continue;
    unsigned index = schema.operandCount +
                     static_cast<unsigned>(std::distance(
                         schema.implicitStateDomains.begin(), state));
    return graph.operandHashes[index];
  }
  return std::nullopt;
}

// A parameterized compiler-action theorem quantifies over an arbitrary
// MemoryState transition.  If detailed whole-kernel import is unavailable
// (for example, the TTIR entry has multiple CFG blocks), build only that
// theorem key: MemoryState variable + exact launch contract.  This summary is
// never used for an ordinary exact rewrite.  The action's generic preflight
// validates every operation and control-flow block before it can be applied.
static std::optional<CanonicalGraphBuildResult>
buildParameterizedKernelKey(Operation *scope, std::string &error) {
  auto function = dyn_cast_or_null<triton::FuncOp>(scope);
  if (!function || function->getNumRegions() != 1 ||
      function->getRegion(0).empty()) {
    error = "parameterized kernel key requires a non-empty tt.func";
    return std::nullopt;
  }

  Builder builder(scope->getContext());
  auto printType = [](Type type) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    stream << type;
    return storage;
  };
  auto printAttributes = [](DictionaryAttr attributes) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    stream << attributes;
    return storage;
  };
  CanonicalGraphBuildResult result;
  auto addGraph = [&](PersistentGraphRecord graph) {
    graph.graphHash = getPersistentGraphHash(graph);
    std::string hash = graph.graphHash;
    if (!llvm::any_of(result.graphs, [&](const auto &known) {
          return known.graphHash == hash;
        }))
      result.graphs.push_back(std::move(graph));
    return hash;
  };

  PersistentGraphRecord memory;
  memory.operationName = "ags.memory";
  memory.resultType = "none";
  memory.attributes = printAttributes(builder.getDictionaryAttr({}));
  std::string memoryHash = addGraph(std::move(memory));

  PersistentGraphRecord launch;
  launch.operationName = "ags.launch";
  launch.resultType = "none";
  launch.attributes = printAttributes(builder.getDictionaryAttr({}));
  std::string launchHash = addGraph(std::move(launch));

  SmallVector<std::pair<int64_t, int64_t>, 3> contracts;
  auto axes = function->getAttrOfType<DenseI32ArrayAttr>(
      "ags.launch_grid_bound_axes");
  auto arguments = function->getAttrOfType<DenseI32ArrayAttr>(
      "ags.launch_grid_bound_arguments");
  if (static_cast<bool>(axes) != static_cast<bool>(arguments) ||
      (axes && axes.size() != arguments.size())) {
    error = "launch grid bound requires equal axes and arguments arrays";
    return std::nullopt;
  }
  if (axes)
    for (auto [axis, argument] :
         llvm::zip(axes.asArrayRef(), arguments.asArrayRef()))
      contracts.emplace_back(axis, argument);
  auto legacyAxis =
      function->getAttrOfType<IntegerAttr>("ags.launch_grid_bound_axis");
  auto legacyArgument =
      function->getAttrOfType<IntegerAttr>("ags.launch_grid_bound_argument");
  if (static_cast<bool>(legacyAxis) != static_cast<bool>(legacyArgument)) {
    error = "launch grid bound requires both legacy axis and argument attrs";
    return std::nullopt;
  }
  if (legacyAxis)
    contracts.emplace_back(legacyAxis.getInt(), legacyArgument.getInt());

  llvm::SmallSet<int64_t, 3> seenAxes;
  for (auto [axis, argument] : contracts) {
    if (axis < 0 || axis > 2 || argument < 0 ||
        static_cast<uint64_t>(argument) >= function.getNumArguments() ||
        !seenAxes.insert(axis).second) {
      error = "invalid or duplicate launch grid bound contract";
      return std::nullopt;
    }
    Value bound = function.getArgument(static_cast<unsigned>(argument));
    if (!isa<IntegerType>(bound.getType())) {
      error = "launch grid bound argument must be a scalar integer";
      return std::nullopt;
    }
    PersistentGraphRecord input;
    input.operationName = "ags.input";
    input.resultType = printType(bound.getType());
    input.attributes = printAttributes(builder.getDictionaryAttr(
        {builder.getNamedAttr("ags.symbol_id",
                              builder.getI64IntegerAttr(argument))}));
    std::string boundHash = addGraph(std::move(input));

    PersistentGraphRecord contract;
    contract.operationName = "ags.launch_bound";
    contract.resultType = "none";
    contract.attributes = printAttributes(builder.getDictionaryAttr(
        {builder.getNamedAttr(kProofOnlyAttr, builder.getBoolAttr(true)),
         builder.getNamedAttr(kStateDomainAttr,
                              builder.getStringAttr("launch")),
         builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))}));
    contract.operandHashes = {boundHash, launchHash};
    contract.operatorSize = 1;
    launchHash = addGraph(std::move(contract));
    result.boundaryInputs.push_back(bound);
  }

  PersistentGraphRecord kernel;
  kernel.operationName = "ags.kernel";
  kernel.resultType = "none";
  kernel.attributes = printAttributes(builder.getDictionaryAttr(
      {builder.getNamedAttr(kProofOnlyAttr, builder.getBoolAttr(true)),
       builder.getNamedAttr(kStateDomainAttr,
                            builder.getStringAttr("memory"))}));
  kernel.operandHashes = {memoryHash, launchHash};
  kernel.operatorSize = 1;
  result.rootGraphHash = addGraph(std::move(kernel));
  result.launchStateHash = launchHash;
  result.stateOperations.emplace_back(
      memoryHash, function.getBody().front().getTerminator());
  llvm::sort(result.graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  return result;
}

// Computes the exact observable MemorySSA suffix that a candidate replaces.
// This is shared by whole-kernel launch plans and remains independent of the
// operations that happen to form the candidate graph.
static std::optional<StateReplacementAnalysis> getReplacedMemorySuffix(
    const RuleDatabase &database,
    const CanonicalGraphBuildResult &canonical, StringRef sourceStateHash,
    StringRef candidateStateHash, Operation *anchor) {
  std::optional<CommonStateAncestor> common = findCommonStateAncestor(
      database, sourceStateHash, candidateStateHash);
  if (!common || !anchor)
    return std::nullopt;

  llvm::StringMap<Operation *> sourceOperations;
  for (const auto &[hash, operation] : canonical.stateOperations)
    sourceOperations.try_emplace(hash, operation);

  SmallVector<Operation *, 4> replaced;
  const PersistentGraphRecord *state =
      database.lookupGraph(sourceStateHash);
  for (unsigned index = 0; index < common->sourceDepth; ++index) {
    if (!state)
      return std::nullopt;
    auto operation = sourceOperations.find(state->graphHash);
    if (operation == sourceOperations.end() ||
        (operation->second->getNumResults() != 0 &&
         !operation->second->use_empty()))
      return std::nullopt;
    replaced.push_back(operation->second);
    std::optional<StringRef> previous =
        getPreviousMemoryState(database, *state);
    state = previous ? database.lookupGraph(*previous) : nullptr;
  }
  if (replaced.empty())
    return std::nullopt;
  std::reverse(replaced.begin(), replaced.end());
  if (replaced.back() != anchor ||
      llvm::any_of(replaced, [&](Operation *operation) {
        return operation->getBlock() != anchor->getBlock();
      }))
    return std::nullopt;

  llvm::SmallPtrSet<Operation *, 8> replacedSet;
  for (Operation *operation : replaced)
    replacedSet.insert(operation);
  for (Operation *operation = replaced.front(); operation;
       operation = operation->getNextNode()) {
    if (!replacedSet.contains(operation) &&
        !isMemoryEffectFree(operation))
      return std::nullopt;
    if (operation == anchor)
      break;
  }
  return StateReplacementAnalysis{std::move(replaced), common->hash};
}

// Generic online lookup for a state transition. Candidate and
// source are reduced to MemorySSA suffixes with a shared certified ancestor;
// the source suffix is replaced atomically by the candidate suffix. No
// operation name or workload topology participates in this decision.
static std::optional<SemanticGraphSolution> solveEffectFromRuleDatabase(
    Operation *root, ArrayRef<TTIRCandidateOpSchema> schemas,
    const GraphOptimizationOptions &options, const RuleDatabase &database,
    std::function<bool()> shouldStop = {}) {
  if (!root)
    return std::nullopt;
  const TTIRCandidateOpSchema *rootSchema = nullptr;
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (schema.operationName == root->getName().getStringRef() &&
        schema.operandCount == root->getNumOperands() &&
        (!schema.matchesOperation || schema.matchesOperation(root))) {
      rootSchema = &schema;
      break;
    }
  const bool isWriteRoot =
      rootSchema && rootSchema->effectKind == TTIRCandidateEffectKind::Write &&
      root->getNumResults() == 0;
  const bool isAtomicRoot =
      rootSchema && rootSchema->effectKind == TTIRCandidateEffectKind::Atomic &&
      root->getNumResults() == 1 && root->use_empty();
  const bool isBufferWriteRoot =
      rootSchema &&
      rootSchema->effectKind == TTIRCandidateEffectKind::BufferWrite &&
      root->getNumResults() == 0;
  if (!isWriteRoot && !isAtomicRoot && !isBufferWriteRoot)
    return std::nullopt;
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas, std::move(shouldStop))
          .build(root, error);
  if (!canonical)
    return std::nullopt;
  GraphCostContext context;
  context.target = options.compileOn91095 ? "ascend950dt" : "generic";
  OnlineRuleLookupResult lookup = OnlineRuleMatcher(database).lookup(
      canonical->rootGraphHash, database.universeHash,
      getBuiltinTTOpSemanticsStableHash(), context.getStableKey());
  if (lookup.status == OnlineRuleLookupStatus::VersionMismatch)
    return std::nullopt;

  auto lookupSourceGraph = [&](StringRef hash) -> const PersistentGraphRecord * {
    auto found = llvm::find_if(canonical->graphs, [&](const auto &graph) {
      return graph.graphHash == hash;
    });
    if (found != canonical->graphs.end())
      return &*found;
    return database.lookupGraph(hash);
  };
  const PersistentGraphRecord *sourceRoot =
      lookupSourceGraph(canonical->rootGraphHash);
  if (!sourceRoot)
    return std::nullopt;
  LLVM_DEBUG({
    llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                 << " mode=effect-database-source"
                 << " root=" << canonical->rootGraphHash
                 << " op=" << sourceRoot->operationName
                 << " exact=" << (lookup.entry ? "hit" : "miss")
                 << " graphs=" << canonical->graphs.size() << '\n';
  });
  std::optional<StructuralCostFeatures> semanticSourceStructural =
      estimateStructuralCost(canonical->rootGraphHash, lookupSourceGraph);

  llvm::StringMap<unsigned> sourceDepth;
  llvm::StringSet<> sourceVisited;
  SmallVector<std::string, 8> sourceStates;
  const PersistentGraphRecord *sourceState = sourceRoot;
  while (sourceState) {
    if (!sourceVisited.insert(sourceState->graphHash).second)
      return std::nullopt;
    sourceDepth.try_emplace(sourceState->graphHash, sourceStates.size());
    sourceStates.push_back(sourceState->graphHash);
    std::optional<StringRef> previous = getPreviousMemoryState(
        *sourceState, lookupSourceGraph);
    sourceState = previous ? lookupSourceGraph(*previous) : nullptr;
  }

  llvm::StringMap<Operation *> sourceOperations;
  for (const auto &[hash, operation] : canonical->stateOperations)
    sourceOperations.try_emplace(hash, operation);

  const RuleDatabaseCandidate *selected = nullptr;
  const RuleDatabaseSemanticEntry *selectedSemanticEntry = nullptr;
  const ParetoCandidateCost *selectedSourceCost = nullptr;
  std::optional<llvm::StringMap<Value>> selectedTemplateBindings;
  std::optional<StructuralCostFeatures> selectedTemplateSourceStructural;
  std::optional<StructuralCostFeatures> selectedTemplateCandidateStructural;
  std::string selectedBoundary;
  SmallVector<Operation *, 4> selectedReplacedEffects;
  unsigned selectedCandidateCount = 0;
  unsigned selectedProofCount = 0;
  SemanticGraphLookupMode selectedLookupMode =
      SemanticGraphLookupMode::RuleDatabaseExact;
  struct CandidateRejectionStats {
    unsigned visited = 0;
    unsigned notCheaper = 0;
    unsigned missingStateRoot = 0;
    unsigned noCommonState = 0;
    unsigned unavailableSourceSuffix = 0;
    unsigned emptySourceSuffix = 0;
    unsigned wrongSourceRoot = 0;
    unsigned effectBarrier = 0;
    unsigned accepted = 0;
  } rejectionStats;
  auto considerCandidate = [&](const RuleDatabaseCandidate &candidate,
                               const RuleDatabaseSemanticEntry *semanticEntry) {
    ++rejectionStats.visited;
    const bool semanticLookup = semanticEntry != nullptr;
    bool improvesSource = false;
    if (lookup.entry)
      improvesSource =
          isStrictlyCheaper(candidate.cost, lookup.entry->sourceCost);
    else if (semanticLookup && semanticSourceStructural)
      improvesSource = isStrictlyCheaperStructural(
          candidate.cost.structural, *semanticSourceStructural);
    if (candidate.rootGraphHash == canonical->rootGraphHash ||
        !improvesSource) {
      ++rejectionStats.notCheaper;
      return;
    }
    const PersistentGraphRecord *candidateState =
        database.lookupGraph(candidate.rootGraphHash);
    if (!candidateState || candidateState->resultType != "none") {
      ++rejectionStats.missingStateRoot;
      return;
    }

    std::string commonState;
    llvm::StringSet<> candidateVisited;
    while (candidateState) {
      if (!candidateVisited.insert(candidateState->graphHash).second)
        break;
      if (sourceDepth.count(candidateState->graphHash)) {
        commonState = candidateState->graphHash;
        break;
      }
      std::optional<StringRef> previous =
          getPreviousMemoryState(database, *candidateState);
      candidateState = previous ? database.lookupGraph(*previous) : nullptr;
    }
    if (commonState.empty()) {
      ++rejectionStats.noCommonState;
      return;
    }

    unsigned commonDepth = sourceDepth.lookup(commonState);
    SmallVector<Operation *, 4> replaced;
    bool sourceSuffixAvailable = true;
    for (unsigned index = 0; index < commonDepth; ++index) {
      auto operation = sourceOperations.find(sourceStates[index]);
      if (operation == sourceOperations.end() ||
          (operation->second->getNumResults() != 0 &&
           !operation->second->use_empty())) {
        sourceSuffixAvailable = false;
        break;
      }
      replaced.push_back(operation->second);
    }
    if (!sourceSuffixAvailable) {
      ++rejectionStats.unavailableSourceSuffix;
      return;
    }
    if (replaced.empty()) {
      ++rejectionStats.emptySourceSuffix;
      return;
    }
    std::reverse(replaced.begin(), replaced.end());
    if (replaced.back() != root) {
      ++rejectionStats.wrongSourceRoot;
      return;
    }

    // A read/unknown effect between two source transitions may observe the
    // intermediate state. Pure calculations may be moved, but such a barrier
    // makes the suffix non-transactional and is rejected.
    llvm::SmallPtrSet<Operation *, 8> replacedSet;
    for (Operation *operation : replaced)
      replacedSet.insert(operation);
    Operation *first = replaced.front();
    if (first->getBlock() != root->getBlock())
      return;
    bool hasEffectBarrier = false;
    for (Operation *operation = first; operation;
         operation = operation->getNextNode()) {
      if (!replacedSet.contains(operation) &&
          !isMemoryEffectFree(operation)) {
        hasEffectBarrier = true;
        break;
      }
      if (operation == root)
        break;
    }
    if (hasEffectBarrier) {
      ++rejectionStats.effectBarrier;
      return;
    }

    bool improvesSelected = !selected;
    if (selected && semanticLookup) {
      if (selectedTemplateCandidateStructural) {
        // A bound template has only a source-specific structural re-cost, so
        // compare the semantic candidate in that same dimension.
        improvesSelected = isStrictlyCheaperStructural(
            candidate.cost.structural,
            *selectedTemplateCandidateStructural);
      } else if (lookup.entry) {
        // An indexed source gives both exact and merged semantic candidates a
        // common versioned StageModel context. Preserve that full comparison
        // across the final-layer shard boundary.
        improvesSelected = isStrictlyCheaper(candidate.cost, selected->cost);
      } else {
        // An unseen semantic source has no offline StageModel score. In that
        // case every candidate participates in one structural comparison.
        improvesSelected = isStrictlyCheaperStructural(
            candidate.cost.structural, selected->cost.structural);
      }
    } else if (selected) {
      improvesSelected = isStrictlyCheaper(candidate.cost, selected->cost);
    }
    if (improvesSelected) {
      ++rejectionStats.accepted;
      selected = &candidate;
      selectedSemanticEntry = semanticEntry;
      selectedTemplateBindings.reset();
      selectedTemplateSourceStructural.reset();
      selectedTemplateCandidateStructural.reset();
      selectedSourceCost =
          lookup.entry ? &lookup.entry->sourceCost : nullptr;
      selectedBoundary = std::move(commonState);
      selectedReplacedEffects = std::move(replaced);
      selectedCandidateCount = semanticEntry
                                   ? semanticEntry->candidates.size()
                                   : lookup.entry->candidates.size();
      selectedProofCount = semanticEntry
                               ? semanticEntry->proofHashes.size()
                               : lookup.entry->proofHashes.size();
      selectedLookupMode =
          semanticEntry ? SemanticGraphLookupMode::RuleDatabaseSemantic
                        : SemanticGraphLookupMode::RuleDatabaseExact;
    }
  };

  if (lookup.status == OnlineRuleLookupStatus::Hit && lookup.entry)
    for (const RuleDatabaseCandidate &candidate : lookup.entry->candidates)
      considerCandidate(candidate, /*semanticEntry=*/nullptr);

  // A state template may replace one matched transition while treating its
  // pointer/value operands and incoming MemoryState as typed holes. Both LHS
  // and RHS must use the same immediate state variable, so the online action
  // is a transactional replacement of the current root only.
  if (database.semanticsHash == getBuiltinTTOpSemanticsStableHash() &&
      database.costContextKey == context.getStableKey()) {
    std::vector<RuleDatabaseTemplateMatch> templateMatches =
        database.matchTemplates(canonical->rootGraphHash,
                                canonical->graphs);
    LLVM_DEBUG(llvm::dbgs()
               << "[graph-optimize-telemetry] component=AutoGraphSolver"
               << " mode=effect-database-template-match"
               << " root=" << canonical->rootGraphHash
               << " matches=" << templateMatches.size() << '\n');
    for (const RuleDatabaseTemplateMatch &match : templateMatches) {
      if (!match.entry)
        continue;
      const PersistentGraphRecord *patternRoot =
          database.lookupGraph(match.entry->lhsPatternGraphHash);
      if (!patternRoot || patternRoot->resultType != "none")
        continue;
      std::optional<StringRef> patternState =
          getPreviousMemoryState(database, *patternRoot);
      const PersistentGraphRecord *stateVariable =
          patternState ? database.lookupGraph(*patternState) : nullptr;
      if (!stateVariable || !stateVariable->operandHashes.empty() ||
          (stateVariable->operationName != "ags.memory" &&
           stateVariable->operationName != "ags.partitionable_memory" &&
           stateVariable->operationName != "ags.buffer"))
        continue;
      std::optional<llvm::StringMap<Value>> bindings =
          bindTemplateValues(database, match, *canonical);
      std::optional<StructuralCostFeatures> sourceStructural =
          estimateBoundTemplateCost(database, canonical->rootGraphHash,
                                    match, *canonical, schemas);
      if (!bindings || !sourceStructural)
        continue;
      for (const RuleDatabaseCandidate &candidate :
           match.entry->candidates) {
        const PersistentGraphRecord *candidateRoot =
            database.lookupGraph(candidate.rootGraphHash);
        std::optional<StructuralCostFeatures> candidateStructural =
            estimateBoundTemplateCost(database, candidate.rootGraphHash,
                                      match, *canonical, schemas);
        if (!candidateRoot || candidateRoot->resultType != "none" ||
            !candidateStructural ||
            !isStrictlyCheaperStructural(*candidateStructural,
                                         *sourceStructural))
          continue;
        Attribute parsed = mlir::parseAttribute(candidateRoot->attributes,
                                                root->getContext());
        auto attributes = dyn_cast_or_null<DictionaryAttr>(parsed);
        if (!attributes || attributes.get(kCompilerActionAttr))
          continue;
        std::optional<StringRef> candidateState =
            getPreviousMemoryState(database, *candidateRoot);
        if (!candidateState || *candidateState != *patternState)
          continue;
        const StructuralCostFeatures *selectedStructural =
            selectedTemplateCandidateStructural
                ? &*selectedTemplateCandidateStructural
                : selected ? &selected->cost.structural : nullptr;
        if (!selectedStructural || isStrictlyCheaperStructural(
                                       *candidateStructural,
                                       *selectedStructural)) {
          selected = &candidate;
          selectedSemanticEntry = nullptr;
          selectedSourceCost = &match.entry->sourceCost;
          llvm::StringMap<Value> copiedBindings;
          for (const auto &binding : *bindings)
            copiedBindings.try_emplace(binding.getKey(), binding.getValue());
          selectedTemplateBindings = std::move(copiedBindings);
          selectedTemplateSourceStructural = *sourceStructural;
          selectedTemplateCandidateStructural = *candidateStructural;
          selectedBoundary = patternState->str();
          selectedReplacedEffects = {root};
          selectedCandidateCount = match.entry->candidates.size();
          selectedProofCount = match.entry->proofHashes.size();
          selectedLookupMode =
              SemanticGraphLookupMode::RuleDatabaseTemplate;
        }
      }
    }
  }

  // The merged semantic entry is the cross-shard candidate union. Restricting
  // it to exact misses would make the selected result depend on the partition
  // that happened to own the online source graph.
  auto semanticEntries = database.lookupSemantic(
      canonical->normalizedSemanticKey, sourceRoot->resultType);
  if (!canonical->normalizedSemanticKey.empty())
    for (const RuleDatabaseSemanticEntry *entry : semanticEntries)
      for (const RuleDatabaseCandidate &candidate : entry->candidates)
        considerCandidate(candidate, entry);
  LLVM_DEBUG({
    llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                 << " mode=effect-database-candidates"
                 << " root=" << canonical->rootGraphHash
                 << " semantic-key-bytes="
                 << canonical->normalizedSemanticKey.size()
                 << " semantic-key=" << canonical->normalizedSemanticKey
                 << " semantic-entries=" << semanticEntries.size()
                 << " visited=" << rejectionStats.visited
                 << " not-cheaper=" << rejectionStats.notCheaper
                 << " missing-state-root=" << rejectionStats.missingStateRoot
                 << " no-common-state=" << rejectionStats.noCommonState
                 << " unavailable-source-suffix="
                 << rejectionStats.unavailableSourceSuffix
                 << " empty-source-suffix="
                 << rejectionStats.emptySourceSuffix
                 << " wrong-source-root=" << rejectionStats.wrongSourceRoot
                 << " effect-barrier=" << rejectionStats.effectBarrier
                 << " accepted=" << rejectionStats.accepted << '\n';
  });
  if (!selected)
    return std::nullopt;

  llvm::StringSet<> active;
  std::shared_ptr<SemanticGraphSolution::Recipe> recipe =
      buildDatabaseRecipe(database, selected->rootGraphHash,
                          canonical->boundaryInputs, root->getContext(), active,
                          selectedBoundary, root,
                          selectedTemplateBindings
                              ? &*selectedTemplateBindings
                              : nullptr);
  if (!recipe || (!recipe->proofOnly && !isa<NoneType>(recipe->resultType)))
    return std::nullopt;

  const bool templateLookup =
      selectedLookupMode == SemanticGraphLookupMode::RuleDatabaseTemplate;
  unsigned originalCost = 0;
  unsigned selectedCost = 0;
  if (selectedSemanticEntry) {
    if (selectedSourceCost) {
      originalCost = solutionCost(*selectedSourceCost);
      selectedCost = solutionCost(selected->cost);
    } else {
      originalCost = solutionCost(ParetoCandidateCost{
          *semanticSourceStructural, std::nullopt});
      selectedCost = solutionCost(ParetoCandidateCost{
          selected->cost.structural, std::nullopt});
    }
  } else if (templateLookup) {
    originalCost = solutionCost(ParetoCandidateCost{
        *selectedTemplateSourceStructural, std::nullopt});
    selectedCost = solutionCost(ParetoCandidateCost{
        *selectedTemplateCandidateStructural, std::nullopt});
  } else {
    originalCost = solutionCost(*selectedSourceCost);
    selectedCost = solutionCost(selected->cost);
  }
  return SemanticGraphSolution{
      root,
      true,
      nullptr,
      std::move(selectedReplacedEffects),
      std::move(recipe),
      GraphOptimizationRuleId::AutoGraphSolver,
      originalCost,
      selectedCost,
      selectedCandidateCount,
      1,
      selectedProofCount,
      selectedLookupMode};
}

static bool canApplyCompilerAction(Operation *scope,
                                   DictionaryAttr candidateAttributes);

static std::optional<SemanticGraphSolution> solveKernelFromRuleDatabase(
    Operation *scope, ArrayRef<TTIRCandidateOpSchema> schemas,
    const GraphOptimizationOptions &options, const RuleDatabase &database,
    std::function<bool()> shouldStop = {}) {
  if (!scope)
    return std::nullopt;
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  bool parameterizedKeyOnly = false;
  std::optional<CanonicalGraphBuildResult> canonical =
      CanonicalTTIRGraphBuilder(semantics, schemas, shouldStop)
          .buildKernel(scope, error);
  if (!canonical) {
    error.clear();
    canonical = buildParameterizedKernelKey(scope, error);
    parameterizedKeyOnly = static_cast<bool>(canonical);
  }
  if (!canonical || canonical->stateOperations.empty())
    return std::nullopt;
  GraphCostContext context;
  context.target = options.compileOn91095 ? "ascend950dt" : "generic";
  OnlineRuleLookupResult exactLookup = OnlineRuleMatcher(database).lookup(
      canonical->rootGraphHash, database.universeHash,
      getBuiltinTTOpSemanticsStableHash(), context.getStableKey());
  const PersistentGraphRecord *sourceRoot = nullptr;
  auto sourceFound = llvm::find_if(canonical->graphs, [&](const auto &graph) {
    return graph.graphHash == canonical->rootGraphHash;
  });
  if (sourceFound != canonical->graphs.end())
    sourceRoot = &*sourceFound;
  if (!sourceRoot || sourceRoot->operationName != "ags.kernel")
    return std::nullopt;
  std::optional<StringRef> sourceMemory = getImplicitStateOperand(
      *sourceRoot, TTIRCandidateStateDomain::Memory, schemas);
  if (!sourceMemory)
    return std::nullopt;

  Operation *anchor = canonical->stateOperations.back().second;
  const RuleDatabaseCandidate *selected = nullptr;
  std::optional<StateReplacementAnalysis> selectedReplacement;
  const ParetoCandidateCost *selectedSourceCost = nullptr;
  unsigned selectedCandidateCount = 0;
  unsigned selectedProofCount = 0;
  SemanticGraphLookupMode selectedLookupMode =
      SemanticGraphLookupMode::RuleDatabaseExact;

  auto considerCandidate = [&](const RuleDatabaseCandidate &candidate,
                               const ParetoCandidateCost &sourceCost,
                               unsigned candidateCount,
                               unsigned proofCount,
                               bool parameterizedTemplate) {
    if (candidate.rootGraphHash == canonical->rootGraphHash ||
        !isStrictlyCheaper(candidate.cost, sourceCost))
      return;
    const PersistentGraphRecord *candidateRoot =
        database.lookupGraph(candidate.rootGraphHash);
    if (!candidateRoot)
      return;
    Attribute parsedCandidateAttributes =
        mlir::parseAttribute(candidateRoot->attributes, scope->getContext());
    auto candidateAttributes =
        dyn_cast_or_null<DictionaryAttr>(parsedCandidateAttributes);
    const bool hasCompilerAction =
        candidateAttributes &&
        isa_and_nonnull<DictionaryAttr>(
            candidateAttributes.get(kCompilerActionAttr));
    if (candidateRoot->operationName != "ags.kernel" && !hasCompilerAction)
      return;
    if (hasCompilerAction &&
        !canApplyCompilerAction(scope, candidateAttributes))
      return;

    std::optional<StateReplacementAnalysis> replacement;
    if (parameterizedTemplate) {
      // The template matcher already bound the arbitrary MemoryState variable
      // to this kernel root.  The replayed candidate is a proof-only compiler
      // action, so it replaces no MemorySSA operation itself.
      if (!hasCompilerAction)
        return;
      replacement = StateReplacementAnalysis{{}, sourceMemory->str()};
    } else {
      std::optional<StringRef> candidateMemory = getImplicitStateOperand(
          *candidateRoot, TTIRCandidateStateDomain::Memory, schemas);
      if (!candidateMemory)
        return;
      if (hasCompilerAction && *candidateMemory == *sourceMemory)
        replacement = StateReplacementAnalysis{{}, sourceMemory->str()};
      else
        replacement = getReplacedMemorySuffix(
            database, *canonical, *sourceMemory, *candidateMemory, anchor);
    }
    if (!replacement)
      return;
    if (!selected || isStrictlyCheaper(candidate.cost, selected->cost)) {
      selected = &candidate;
      selectedSourceCost = &sourceCost;
      selectedReplacement = std::move(replacement);
      selectedCandidateCount = candidateCount;
      selectedProofCount = proofCount;
      selectedLookupMode =
          parameterizedTemplate
              ? SemanticGraphLookupMode::RuleDatabaseTemplate
              : SemanticGraphLookupMode::RuleDatabaseExact;
    }
  };

  if (!parameterizedKeyOnly &&
      exactLookup.status == OnlineRuleLookupStatus::Hit && exactLookup.entry)
    for (const RuleDatabaseCandidate &candidate :
         exactLookup.entry->candidates)
      considerCandidate(candidate, exactLookup.entry->sourceCost,
                        static_cast<unsigned>(
                            exactLookup.entry->candidates.size()),
                        static_cast<unsigned>(
                            exactLookup.entry->proofHashes.size()),
                        /*parameterizedTemplate=*/false);

  if (database.semanticsHash == getBuiltinTTOpSemanticsStableHash() &&
      database.costContextKey == context.getStableKey()) {
    for (const RuleDatabaseTemplateMatch &match : database.matchTemplates(
             canonical->rootGraphHash, canonical->graphs)) {
      if (!match.entry)
        continue;
      for (const RuleDatabaseCandidate &candidate : match.entry->candidates)
        considerCandidate(
            candidate, match.entry->sourceCost,
            static_cast<unsigned>(match.entry->candidates.size()),
            static_cast<unsigned>(match.entry->proofHashes.size()),
            /*parameterizedTemplate=*/true);
    }
  }

  if (!selected || !selectedSourceCost || !selectedReplacement)
    return std::nullopt;

  llvm::StringSet<> active;
  std::shared_ptr<SemanticGraphSolution::Recipe> recipe =
      buildDatabaseRecipe(database, selected->rootGraphHash,
                          canonical->boundaryInputs, scope->getContext(),
                          active, selectedReplacement->commonBoundary, anchor);
  if (!recipe || !recipe->proofOnly ||
      !isa<NoneType>(recipe->resultType))
    return std::nullopt;

  unsigned originalCost = solutionCost(*selectedSourceCost);
  unsigned selectedCost = solutionCost(selected->cost);
  return SemanticGraphSolution{
      anchor,
      true,
      scope,
      std::move(selectedReplacement->replacedEffects),
      std::move(recipe),
      GraphOptimizationRuleId::AutoGraphSolver,
      originalCost,
      selectedCost,
      selectedCandidateCount,
      1,
      selectedProofCount,
      selectedLookupMode};
}

static std::optional<SemanticGraphSolution> solveConfigured(
    Value root, ArrayRef<TTIRCandidateOpSchema> schemas,
    const GraphOptimizationOptions &options,
    const std::shared_ptr<const RuleDatabase> &database,
    std::function<bool()> shouldStop = {}) {
  if (database)
    return solveFromRuleDatabase(root, schemas, options, *database,
                                 std::move(shouldStop));
  return solveSemantically(root, schemas);
}

static bool sameSolution(const SemanticGraphSolution &lhs,
                         const SemanticGraphSolution &rhs) {
  return lhs.anchor == rhs.anchor &&
         lhs.operationRoot == rhs.operationRoot &&
         lhs.kernelScope == rhs.kernelScope &&
         lhs.replacedEffects == rhs.replacedEffects &&
         sameRecipe(lhs.recipe, rhs.recipe) && lhs.planId == rhs.planId &&
         lhs.lookupMode == rhs.lookupMode &&
         lhs.originalCost == rhs.originalCost &&
         lhs.selectedCost == rhs.selectedCost;
}

static LogicalResult
materializeEffectRecipe(const SemanticGraphSolution::Recipe &recipe,
                        Location location, IRRewriter &rewriter);

static FailureOr<Value>
materializeRecipe(const SemanticGraphSolution::Recipe &recipe,
                  Location location, IRRewriter &rewriter) {
  if (recipe.source)
    return recipe.source;
  if (recipe.proofOnly)
    return failure();
  SmallVector<Value, 4> operands;
  for (const auto &operandRecipe : recipe.operands) {
    if (operandRecipe->proofOnly)
      continue;
    if (isa<NoneType>(operandRecipe->resultType)) {
      if (failed(materializeEffectRecipe(*operandRecipe, location, rewriter)))
        return failure();
      continue;
    }
    FailureOr<Value> operand =
        materializeRecipe(*operandRecipe, location, rewriter);
    if (failed(operand))
      return failure();
    operands.push_back(*operand);
  }
  OperationState state(location, recipe.operationName);
  state.addOperands(operands);
  SmallVector<Type, 2> operationResultTypes{recipe.resultType};
  unsigned resultIndex = 0;
  ArrayAttr canonicalRegions;
  if (recipe.attributes)
    for (NamedAttribute attribute : recipe.attributes) {
      if (attribute.getName().getValue() == kCanonicalRegionsAttr) {
        canonicalRegions = dyn_cast<ArrayAttr>(attribute.getValue());
        if (!canonicalRegions)
          return failure();
        continue;
      }
      if (attribute.getName().getValue() == kResultIndexAttr) {
        auto index = dyn_cast<IntegerAttr>(attribute.getValue());
        if (!index || index.getInt() < 0)
          return failure();
        resultIndex = static_cast<unsigned>(index.getInt());
        continue;
      }
      if (attribute.getName().getValue() == kOperationResultTypesAttr) {
        auto types = dyn_cast<ArrayAttr>(attribute.getValue());
        if (!types || types.empty())
          return failure();
        operationResultTypes.clear();
        for (Attribute encodedType : types) {
          auto type = dyn_cast<TypeAttr>(encodedType);
          if (!type)
            return failure();
          operationResultTypes.push_back(type.getValue());
        }
        continue;
      }
      if (attribute.getName().getValue().starts_with("ags."))
        continue;
      // AttrSizedOperandSegments records which optional/dynamic segment owns
      // each operand. Arity alone cannot reconstruct that information, so
      // preserve the certified vector and validate it before creating IR.
      if (attribute.getName().getValue() == "operandSegmentSizes") {
        auto oldSegments = dyn_cast<DenseI32ArrayAttr>(attribute.getValue());
        if (!oldSegments)
          return failure();
        uint64_t operandCount = 0;
        for (int32_t segment : oldSegments.asArrayRef()) {
          if (segment < 0)
            return failure();
          operandCount += static_cast<uint64_t>(segment);
        }
        if (operandCount != operands.size())
          return failure();
        state.addAttribute(attribute.getName(), oldSegments);
        continue;
      }
      state.addAttribute(attribute.getName(), attribute.getValue());
    }
  if (resultIndex >= operationResultTypes.size() ||
      operationResultTypes[resultIndex] != recipe.resultType)
    return failure();
  state.addTypes(operationResultTypes);
  if (canonicalRegions)
    for (size_t index = 0; index < canonicalRegions.size(); ++index)
      state.addRegion();
  Operation *operation = rewriter.create(state);
  if (canonicalRegions) {
    std::string regionError;
    if (failed(materializeCanonicalRegions(operation, canonicalRegions,
                                           rewriter, regionError))) {
      rewriter.eraseOp(operation);
      return failure();
    }
  }
  if (operation->getNumResults() != operationResultTypes.size()) {
    rewriter.eraseOp(operation);
    return failure();
  }
  return operation->getResult(resultIndex);
}

static LogicalResult
materializeEffectRecipe(const SemanticGraphSolution::Recipe &recipe,
                        Location location, IRRewriter &rewriter) {
  if (recipe.source)
    return failure();
  if (recipe.proofOnly) {
    for (const auto &operand : recipe.operands) {
      if (operand->proofOnly)
        continue;
      if (!isa<NoneType>(operand->resultType) ||
          failed(materializeEffectRecipe(*operand, location, rewriter)))
        return failure();
    }
    return success();
  }
  if (!isa<NoneType>(recipe.resultType))
    return failure();

  // Compiler/module actions are proof-only state transitions. Their
  // dictionary is collected and committed after the IR replacement succeeds;
  // no pseudo operation is inserted into the user's module here.
  if (recipe.attributes &&
      recipe.attributes.get(kModuleAttributePatchAttr))
    return isa<DictionaryAttr>(
               recipe.attributes.get(kModuleAttributePatchAttr))
               ? success()
               : failure();

  // `ags.memory_after` is a proof-language projection, not a real MLIR op.
  // Materializing it means executing its single result-producing atomic
  // child and intentionally discarding that child's unused SSA result while
  // retaining the memory effect.
  if (recipe.operationName == "ags.memory_after") {
    if (recipe.operands.size() != 1 || recipe.operands.front()->proofOnly ||
        isa<NoneType>(recipe.operands.front()->resultType))
      return failure();
    return succeeded(materializeRecipe(*recipe.operands.front(), location,
                                       rewriter))
               ? success()
               : failure();
  }

  SmallVector<Value, 4> operands;
  for (const auto &operandRecipe : recipe.operands) {
    if (operandRecipe->proofOnly)
      continue;
    if (isa<NoneType>(operandRecipe->resultType)) {
      if (failed(materializeEffectRecipe(*operandRecipe, location, rewriter)))
        return failure();
      continue;
    }
    FailureOr<Value> operand =
        materializeRecipe(*operandRecipe, location, rewriter);
    if (failed(operand))
      return failure();
    operands.push_back(*operand);
  }
  OperationState state(location, recipe.operationName);
  state.addOperands(operands);
  ArrayAttr canonicalRegions;
  if (recipe.attributes)
    for (NamedAttribute attribute : recipe.attributes) {
      StringRef name = attribute.getName().getValue();
      if (name == kCanonicalRegionsAttr) {
        canonicalRegions = dyn_cast<ArrayAttr>(attribute.getValue());
        if (!canonicalRegions)
          return failure();
        continue;
      }
      if (name.starts_with("ags."))
        continue;
      if (name == "operandSegmentSizes") {
        auto oldSegments = dyn_cast<DenseI32ArrayAttr>(attribute.getValue());
        if (!oldSegments)
          return failure();
        uint64_t operandCount = 0;
        for (int32_t segment : oldSegments.asArrayRef()) {
          if (segment < 0)
            return failure();
          operandCount += static_cast<uint64_t>(segment);
        }
        if (operandCount != operands.size())
          return failure();
        state.addAttribute(attribute.getName(), oldSegments);
        continue;
      }
      state.addAttribute(attribute.getName(), attribute.getValue());
    }
  if (canonicalRegions)
    for (size_t index = 0; index < canonicalRegions.size(); ++index)
      state.addRegion();
  Operation *operation = rewriter.create(state);
  if (canonicalRegions) {
    std::string regionError;
    if (failed(materializeCanonicalRegions(operation, canonicalRegions,
                                           rewriter, regionError))) {
      rewriter.eraseOp(operation);
      return failure();
    }
  }
  if (operation->getNumResults() != 0) {
    rewriter.eraseOp(operation);
    return failure();
  }
  return success();
}

// Validate every condition that can make recursive materialization fail
// before moving a source operation or creating any replacement IR.  This
// gives the rewrite transaction an all-or-nothing boundary even though MLIR's
// generic OperationState builder itself is not transactional.
static bool isRecipeMaterializable(
    const SemanticGraphSolution::Recipe &recipe, bool effectPosition) {
  if (recipe.source)
    return !effectPosition && !recipe.proofOnly &&
           recipe.resultType && !isa<NoneType>(recipe.resultType) &&
           recipe.source.getType() == recipe.resultType;
  if (recipe.proofOnly) {
    if (!effectPosition)
      return false;
    if (recipe.attributes) {
      if (Attribute patch =
              recipe.attributes.get(kModuleAttributePatchAttr))
        if (!isa<DictionaryAttr>(patch))
          return false;
      if (Attribute action = recipe.attributes.get(kCompilerActionAttr))
        if (!isa<DictionaryAttr>(action))
          return false;
    }
    return llvm::all_of(recipe.operands, [&](const auto &operand) {
      if (!isa<NoneType>(operand->resultType))
        return false;
      return operand->proofOnly ||
             isRecipeMaterializable(*operand, /*effectPosition=*/true);
    });
  }
  if (recipe.operationName.empty() ||
      isa<NoneType>(recipe.resultType) != effectPosition)
    return false;

  unsigned materializedOperands = 0;
  for (const auto &operand : recipe.operands) {
    const bool operandIsEffect = isa<NoneType>(operand->resultType);
    if (operand->proofOnly) {
      if (!operandIsEffect)
        return false;
      continue;
    }
    if (!isRecipeMaterializable(*operand, operandIsEffect))
      return false;
    if (!operandIsEffect)
      ++materializedOperands;
  }

  if (!recipe.attributes)
    return true;
  if (Attribute regions = recipe.attributes.get(kCanonicalRegionsAttr)) {
    auto array = dyn_cast<ArrayAttr>(regions);
    std::string regionError;
    if (!array || failed(validateCanonicalRegions(array, regionError)))
      return false;
  }
  if (Attribute patch = recipe.attributes.get(kModuleAttributePatchAttr))
    if (!isa<DictionaryAttr>(patch) || !effectPosition)
      return false;
  if (Attribute action = recipe.attributes.get(kCompilerActionAttr))
    if (!isa<DictionaryAttr>(action) || !effectPosition || !recipe.proofOnly)
      return false;
  Attribute encodedResultTypes =
      recipe.attributes.get(kOperationResultTypesAttr);
  Attribute encodedResultIndex = recipe.attributes.get(kResultIndexAttr);
  if (static_cast<bool>(encodedResultTypes) !=
      static_cast<bool>(encodedResultIndex))
    return false;
  if (encodedResultTypes) {
    auto types = dyn_cast<ArrayAttr>(encodedResultTypes);
    auto index = dyn_cast<IntegerAttr>(encodedResultIndex);
    if (!types || types.empty() || !index || index.getInt() < 0 ||
        static_cast<uint64_t>(index.getInt()) >= types.size())
      return false;
    for (Attribute type : types)
      if (!isa<TypeAttr>(type))
        return false;
    if (cast<TypeAttr>(
            types[static_cast<unsigned>(index.getInt())]).getValue() !=
        recipe.resultType)
      return false;
  }
  Attribute segments = recipe.attributes.get("operandSegmentSizes");
  if (!segments)
    return true;
  auto oldSegments = dyn_cast<DenseI32ArrayAttr>(segments);
  if (!oldSegments)
    return false;
  uint64_t operandCount = 0;
  for (int32_t segment : oldSegments.asArrayRef()) {
    if (segment < 0)
      return false;
    operandCount += static_cast<uint64_t>(segment);
  }
  return operandCount == materializedOperands;
}

static bool collectModuleAttributePatch(
    const SemanticGraphSolution::Recipe &recipe,
    SmallVectorImpl<NamedAttribute> &patch) {
  if (recipe.attributes)
    if (auto local = dyn_cast_or_null<DictionaryAttr>(
            recipe.attributes.get(kModuleAttributePatchAttr)))
      for (NamedAttribute attribute : local) {
        auto known = llvm::find_if(patch, [&](NamedAttribute existing) {
          return existing.getName() == attribute.getName();
        });
        if (known == patch.end())
          patch.push_back(attribute);
        else if (known->getValue() != attribute.getValue())
          return false;
      }
  for (const auto &operand : recipe.operands)
    if (!collectModuleAttributePatch(*operand, patch))
      return false;
  return true;
}

static bool canApplyModuleAttributePatch(ModuleOp module,
                                         ArrayRef<NamedAttribute> patch) {
  if (!module)
    return patch.empty();
  for (NamedAttribute attribute : patch)
    if (Attribute existing = module->getAttr(attribute.getName()))
      if (existing != attribute.getValue())
        return false;
  return true;
}

static void applyModuleAttributePatch(ModuleOp module,
                                      ArrayRef<NamedAttribute> patch) {
  for (NamedAttribute attribute : patch)
    module->setAttr(attribute.getName(), attribute.getValue());
}

static LogicalResult hoistValueBeforeAnchor(
    Value value, Operation *anchor, IRRewriter &rewriter,
    SmallPtrSetImpl<Operation *> &hoisted) {
  if (valueDominatesAnchor(value, anchor))
    return success();
  Operation *definition = value.getDefiningOp();
  if (!definition || definition == anchor ||
      definition->getBlock() != anchor->getBlock() ||
      !isMemoryEffectFree(definition) || definition->getNumRegions() != 0)
    return failure();
  if (!hoisted.insert(definition).second)
    return success();
  for (Value operand : definition->getOperands())
    if (failed(hoistValueBeforeAnchor(operand, anchor, rewriter, hoisted)))
      return failure();
  rewriter.moveOpBefore(definition, anchor);
  return success();
}

static bool canHoistValueBeforeAnchor(
    Value value, Operation *anchor, SmallPtrSetImpl<Operation *> &visited) {
  if (valueDominatesAnchor(value, anchor))
    return true;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition == anchor ||
      definition->getBlock() != anchor->getBlock() ||
      !isMemoryEffectFree(definition) || definition->getNumRegions() != 0)
    return false;
  if (!visited.insert(definition).second)
    return true;
  return llvm::all_of(definition->getOperands(), [&](Value operand) {
    return canHoistValueBeforeAnchor(operand, anchor, visited);
  });
}

static bool canHoistRecipeSources(
    const SemanticGraphSolution::Recipe &recipe, Operation *anchor,
    SmallPtrSetImpl<Operation *> &visited) {
  if (recipe.source)
    return canHoistValueBeforeAnchor(recipe.source, anchor, visited);
  if (recipe.proofOnly)
    return llvm::all_of(recipe.operands, [&](const auto &operand) {
      return canHoistRecipeSources(*operand, anchor, visited);
    });
  return llvm::all_of(recipe.operands, [&](const auto &operand) {
    return canHoistRecipeSources(*operand, anchor, visited);
  });
}

static LogicalResult hoistRecipeSources(
    const SemanticGraphSolution::Recipe &recipe, Operation *anchor,
    IRRewriter &rewriter, SmallPtrSetImpl<Operation *> &hoisted) {
  if (recipe.source)
    return hoistValueBeforeAnchor(recipe.source, anchor, rewriter, hoisted);
  if (recipe.proofOnly) {
    for (const auto &operand : recipe.operands)
      if (!operand->proofOnly &&
          failed(hoistRecipeSources(*operand, anchor, rewriter, hoisted)))
        return failure();
    return success();
  }
  for (const auto &operand : recipe.operands)
    if (failed(hoistRecipeSources(*operand, anchor, rewriter, hoisted)))
      return failure();
  return success();
}

static bool isDeadPrivateBufferRead(Operation *operation) {
  if (!operation || operation->getNumResults() == 0 ||
      !operation->use_empty() || operation->getNumRegions() != 0)
    return false;
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance, 4> effects;
  interface.getEffects(effects);
  if (effects.empty())
    return false;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()) || !effect.getValue())
      return false;
    Value root = getViewRoot(effect.getValue());
    auto allocation = dyn_cast_or_null<MemoryEffectOpInterface>(
        root.getDefiningOp());
    if (!allocation)
      return false;
    SmallVector<MemoryEffects::EffectInstance, 4> allocationEffects;
    allocation.getEffects(allocationEffects);
    if (!llvm::any_of(allocationEffects,
                      [&](const MemoryEffects::EffectInstance &candidate) {
                        return isa<MemoryEffects::Allocate>(
                                   candidate.getEffect()) &&
                               getViewRoot(candidate.getValue()) == root;
                      }))
      return false;
  }
  return true;
}

// These operations are reached only from producers of an effect/value slice
// that a replayed RuleDB proof has replaced. A non-volatile operation whose
// complete effect set consists of reads cannot be observed after all of its
// SSA results become dead. Keeping it would retain obsolete tt.load address
// and layout DAGs after an otherwise successful whole-kernel transaction.
static bool isDeadCertifiedRead(Operation *operation) {
  if (!operation || !operation->use_empty() || operation->getNumRegions() != 0)
    return false;
  if (auto isVolatile = operation->getAttrOfType<BoolAttr>("isVolatile"))
    if (isVolatile.getValue())
      return false;
  auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance, 4> effects;
  interface.getEffects(effects);
  return !effects.empty() &&
         llvm::all_of(effects, [](const auto &effect) {
           return isa<MemoryEffects::Read>(effect.getEffect());
         });
}

static bool isDeadRemovableProducer(Operation *operation) {
  return operation && operation->use_empty() &&
         (isMemoryEffectFree(operation) ||
          isDeadPrivateBufferRead(operation) ||
          isDeadCertifiedRead(operation));
}

static void eraseDeadPureProducers(IRRewriter &rewriter,
                                   ArrayRef<Operation *> roots,
                                   Block *rewrittenBlock) {
  SmallVector<Operation *, 32> worklist(roots);
  SmallPtrSet<Operation *, 32> visited;
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (!operation || !visited.insert(operation).second ||
        !isDeadRemovableProducer(operation) || operation->getNumRegions() > 1)
      continue;
    SmallVector<Operation *, 4> producers;
    for (Value operand : operation->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        producers.push_back(producer);
    // Overflow sanitization places a provenance-marked assertion immediately
    // before the integer operation it checks. If the checked operation becomes
    // dead because extraction selected another graph, that assertion no
    // longer protects any executable computation and would otherwise pin the
    // entire obsolete producer slice in place. User assertions, including
    // those with the same message, do not carry this private marker.
    Operation *preceding = operation->getPrevNode();
    if (preceding && preceding->getName().getStringRef() == "tt.assert" &&
        preceding->hasAttr("tt.auto_overflow_assert")) {
      for (Value operand : preceding->getOperands())
        if (Operation *producer = operand.getDefiningOp())
          producers.push_back(producer);
      rewriter.eraseOp(preceding);
    }
    rewriter.eraseOp(operation);
    worklist.append(producers.begin(), producers.end());
  }

  // Contextual solving may import sibling values from the whole block.  A
  // selected candidate can therefore make a producer chain dead even when it
  // was not reachable from the rewritten root's immediate operands at the
  // time the recursive worklist was seeded.  Sweep the rewritten block in
  // reverse SSA order so dead consumers disappear before their producers.
  // This is generic dead-code cleanup; it does not know which TTIR operation
  // or optimization caused the chain to become unused.
  if (!rewrittenBlock)
    return;
  SmallVector<Operation *, 64> blockOperations;
  for (Operation &operation : *rewrittenBlock)
    blockOperations.push_back(&operation);
  for (Operation *operation : llvm::reverse(blockOperations)) {
    if (!operation || operation->getNumResults() == 0 ||
        !isDeadRemovableProducer(operation) ||
        operation->getNumRegions() != 0)
      continue;
    Operation *preceding = operation->getPrevNode();
    if (preceding && preceding->getName().getStringRef() == "tt.assert" &&
        preceding->hasAttr("tt.auto_overflow_assert"))
      rewriter.eraseOp(preceding);
    rewriter.eraseOp(operation);
  }
}

struct LaunchPartitionAction {
  int64_t axis = 0;
  int64_t factor = 0;
  int64_t ceilDiv = 0;
};

static std::optional<LaunchPartitionAction>
getLaunchPartitionAction(const SemanticGraphSolution::Recipe &recipe) {
  if (!recipe.attributes)
    return std::nullopt;
  auto action = dyn_cast_or_null<DictionaryAttr>(
      recipe.attributes.get(kCompilerActionAttr));
  auto kind = action ? dyn_cast_or_null<StringAttr>(action.get("kind"))
                     : StringAttr{};
  auto axis = action ? dyn_cast_or_null<IntegerAttr>(action.get("axis"))
                     : IntegerAttr{};
  auto factor = action ? dyn_cast_or_null<IntegerAttr>(action.get("factor"))
                       : IntegerAttr{};
  auto ceilDiv =
      action ? dyn_cast_or_null<IntegerAttr>(action.get("ceil_div"))
             : IntegerAttr{};
  if (!kind || kind.getValue() != "launch_partition" || !axis || !factor ||
      !ceilDiv)
    return std::nullopt;
  return LaunchPartitionAction{axis.getInt(), factor.getInt(),
                               ceilDiv.getInt()};
}

static bool isMappedProgramId(Operation *operation, int64_t axis) {
  auto programId = dyn_cast_or_null<triton::GetProgramIdOp>(operation);
  return programId && static_cast<int64_t>(programId.getAxis()) == axis;
}

static bool valueDependsOnProgramId(Value value, int64_t axis,
                                    llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (isMappedProgramId(definition, axis))
    return true;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnProgramId(operand, axis, visited);
  });
}

static bool isLosslessIntegerCarrier(Value value, Value source,
                                     llvm::DenseSet<Value> &visited) {
  if (value == source)
    return true;
  if (!value || !visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getNumOperands() != 1)
    return false;
  // Only widening or representation-only casts preserve the implication used
  // by a tail guard.  Arithmetic, truncation and arbitrary expressions are
  // intentionally rejected: `pid - 1 < bound`, for example, does not prove
  // that the original pid is in range.
  StringRef name = definition->getName().getStringRef();
  if (name != "arith.extui" && name != "arith.extsi")
    return false;
  Type sourceType = definition->getOperand(0).getType();
  Type resultType = value.getType();
  auto sourceInteger = dyn_cast<IntegerType>(sourceType);
  auto resultInteger = dyn_cast<IntegerType>(resultType);
  if (sourceInteger && resultInteger &&
      resultInteger.getWidth() < sourceInteger.getWidth())
    return false;
  return isLosslessIntegerCarrier(definition->getOperand(0), source, visited);
}

static bool isProgramIdCarrier(Value value, int64_t axis,
                               llvm::DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (isMappedProgramId(definition, axis))
    return true;
  if (definition->getNumOperands() != 1)
    return false;
  StringRef name = definition->getName().getStringRef();
  if (name != "arith.extui" && name != "arith.extsi")
    return false;
  Type sourceType = definition->getOperand(0).getType();
  Type resultType = value.getType();
  auto sourceInteger = dyn_cast<IntegerType>(sourceType);
  auto resultInteger = dyn_cast<IntegerType>(resultType);
  if (sourceInteger && resultInteger &&
      resultInteger.getWidth() < sourceInteger.getWidth())
    return false;
  return isProgramIdCarrier(definition->getOperand(0), axis, visited);
}

enum class TailPredicatePolarity : uint8_t { None, TrueIsSafe, FalseIsSafe };

static TailPredicatePolarity classifyTailPredicate(Value condition,
                                                    int64_t axis,
                                                    Value bound) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare)
    return TailPredicatePolarity::None;
  llvm::DenseSet<Value> lhsPidVisited;
  llvm::DenseSet<Value> rhsPidVisited;
  llvm::DenseSet<Value> lhsBoundVisited;
  llvm::DenseSet<Value> rhsBoundVisited;
  const bool lhsPid =
      isProgramIdCarrier(compare.getLhs(), axis, lhsPidVisited);
  const bool rhsPid =
      isProgramIdCarrier(compare.getRhs(), axis, rhsPidVisited);
  const bool lhsBound =
      isLosslessIntegerCarrier(compare.getLhs(), bound, lhsBoundVisited);
  const bool rhsBound =
      isLosslessIntegerCarrier(compare.getRhs(), bound, rhsBoundVisited);
  using Predicate = arith::CmpIPredicate;
  Predicate predicate = compare.getPredicate();
  if (lhsPid && rhsBound && !rhsPid) {
    if (predicate == Predicate::slt || predicate == Predicate::ult)
      return TailPredicatePolarity::TrueIsSafe;
    if (predicate == Predicate::sge || predicate == Predicate::uge)
      return TailPredicatePolarity::FalseIsSafe;
  }
  if (rhsPid && lhsBound && !lhsPid) {
    if (predicate == Predicate::sgt || predicate == Predicate::ugt)
      return TailPredicatePolarity::TrueIsSafe;
    if (predicate == Predicate::sle || predicate == Predicate::ule)
      return TailPredicatePolarity::FalseIsSafe;
  }
  return TailPredicatePolarity::None;
}

static bool maskContainsTailPredicate(Value mask, int64_t axis, Value bound,
                                      llvm::DenseSet<Value> &visited) {
  if (!mask || !visited.insert(mask).second)
    return false;
  if (classifyTailPredicate(mask, axis, bound) ==
      TailPredicatePolarity::TrueIsSafe)
    return true;
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return false;
  // Conjunction and shape-only propagation preserve a true=>in-bounds
  // implication.  Disjunction is intentionally excluded.
  if (isa<arith::AndIOp>(definition))
    return llvm::any_of(definition->getOperands(), [&](Value operand) {
      return maskContainsTailPredicate(operand, axis, bound, visited);
    });
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "arith.extui" ||
      name == "arith.extsi" || name == "arith.trunci")
    return definition->getNumOperands() == 1 &&
           maskContainsTailPredicate(definition->getOperand(0), axis, bound,
                                     visited);
  return false;
}

static std::optional<Value> getLaunchBoundArgument(Operation *scope,
                                                   int64_t axis) {
  auto function = dyn_cast_or_null<triton::FuncOp>(scope);
  if (!function || function.getBody().empty())
    return std::nullopt;
  auto axes = function->getAttrOfType<DenseI32ArrayAttr>(
      "ags.launch_grid_bound_axes");
  auto arguments = function->getAttrOfType<DenseI32ArrayAttr>(
      "ags.launch_grid_bound_arguments");
  if (axes && arguments && axes.size() == arguments.size())
    for (auto [candidateAxis, argument] :
         llvm::zip(axes.asArrayRef(), arguments.asArrayRef()))
      if (candidateAxis == axis && argument >= 0 &&
          static_cast<uint64_t>(argument) < function.getNumArguments())
        return function.getArgument(static_cast<unsigned>(argument));
  auto legacyAxis =
      function->getAttrOfType<IntegerAttr>("ags.launch_grid_bound_axis");
  auto legacyArgument =
      function->getAttrOfType<IntegerAttr>("ags.launch_grid_bound_argument");
  if (legacyAxis && legacyArgument && legacyAxis.getInt() == axis &&
      legacyArgument.getInt() >= 0 &&
      static_cast<uint64_t>(legacyArgument.getInt()) <
          function.getNumArguments())
    return function.getArgument(
        static_cast<unsigned>(legacyArgument.getInt()));
  return std::nullopt;
}

static bool isTailGuardedMemoryOperation(Operation *operation, Operation *scope,
                                         int64_t axis, Value bound,
                                         const DominanceInfo &dominance) {
  llvm::DenseSet<Value> dependencyVisited;
  const bool laneDependent =
      llvm::any_of(operation->getOperands(), [&](Value operand) {
        return valueDependsOnProgramId(operand, axis, dependencyVisited);
      });
  if (!laneDependent)
    return true;

  bool dominatedByGuard = false;
  scope->walk([&](cf::CondBranchOp branch) {
    if (dominatedByGuard)
      return;
    TailPredicatePolarity polarity =
        classifyTailPredicate(branch.getCondition(), axis, bound);
    Block *safe = polarity == TailPredicatePolarity::TrueIsSafe
                      ? branch.getTrueDest()
                      : polarity == TailPredicatePolarity::FalseIsSafe
                            ? branch.getFalseDest()
                            : nullptr;
    dominatedByGuard =
        safe && dominance.dominates(safe, operation->getBlock());
  });
  if (dominatedByGuard)
    return true;

  Value mask;
  if (auto load = dyn_cast<triton::LoadOp>(operation))
    mask = load.getMask();
  else if (auto store = dyn_cast<triton::StoreOp>(operation))
    mask = store.getMask();
  if (!mask)
    return false;
  llvm::DenseSet<Value> maskVisited;
  return maskContainsTailPredicate(mask, axis, bound, maskVisited);
}

static bool canApplyLaunchPartition(Operation *scope,
                                    const LaunchPartitionAction &action) {
  if (!scope || action.axis < 0 || action.axis > 2 || action.factor <= 1 ||
      action.factor > 64 ||
      (action.ceilDiv != 0 && action.ceilDiv != 1) ||
      scope->getNumRegions() != 1 || scope->getRegion(0).empty())
    return false;
  Region &body = scope->getRegion(0);
  Block &entry = body.front();
  if (!entry.hasNoPredecessors())
    return false;
  std::optional<Value> tailBound;
  if (action.ceilDiv == 1) {
    tailBound = getLaunchBoundArgument(scope, action.axis);
    if (!tailBound)
      return false;
  }
  DominanceInfo dominance(scope);
  bool safe = true;
  bool sawMappedProgramId = false;
  scope->walk([&](Operation *operation) {
    if (!safe || operation == scope)
      return;
    if (auto returnOp = dyn_cast<triton::ReturnOp>(operation)) {
      safe = returnOp->getNumOperands() == 0;
      return;
    }
    if (auto numPrograms = dyn_cast<triton::GetNumProgramsOp>(operation))
      if (static_cast<int64_t>(numPrograms.getAxis()) == action.axis)
        safe = false;
    if (!safe)
      return;
    if (isMappedProgramId(operation, action.axis)) {
      sawMappedProgramId = true;
      safe = operation->getNumResults() == 1 &&
             isa<IntegerType>(operation->getResult(0).getType());
      return;
    }
    StringRef name = operation->getName().getStringRef();
    if (name.starts_with("tt.atomic_") || name == "tt.call" ||
        name == "tt.extern_elementwise" || name == "tt.print" ||
        name == "tt.assert") {
      safe = false;
      return;
    }
    if (isMemoryEffectFree(operation))
      return;
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
      safe = false;
      return;
    }
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    interface.getEffects(effects);
    if (effects.empty() || llvm::any_of(effects, [](const auto &effect) {
          return !isa<MemoryEffects::Read, MemoryEffects::Write>(
              effect.getEffect());
        })) {
      safe = false;
      return;
    }
    if (tailBound && operation->getNumRegions() == 0 &&
        !isTailGuardedMemoryOperation(operation, scope, action.axis,
                                      *tailBound, dominance))
      safe = false;
  });
  return safe && sawMappedProgramId;
}

static bool canApplyCompilerAction(Operation *scope,
                                   DictionaryAttr candidateAttributes) {
  if (!candidateAttributes)
    return false;
  auto action = dyn_cast_or_null<DictionaryAttr>(
      candidateAttributes.get(kCompilerActionAttr));
  auto kind = action ? dyn_cast_or_null<StringAttr>(action.get("kind"))
                     : StringAttr{};
  auto axis = action ? dyn_cast_or_null<IntegerAttr>(action.get("axis"))
                     : IntegerAttr{};
  auto factor = action ? dyn_cast_or_null<IntegerAttr>(action.get("factor"))
                       : IntegerAttr{};
  auto ceilDiv =
      action ? dyn_cast_or_null<IntegerAttr>(action.get("ceil_div"))
             : IntegerAttr{};
  return kind && kind.getValue() == "launch_partition" && axis && factor &&
         ceilDiv &&
         canApplyLaunchPartition(
             scope, LaunchPartitionAction{axis.getInt(), factor.getInt(),
                                          ceilDiv.getInt()});
}

static LogicalResult applyLaunchPartition(Operation *scope,
                                          const LaunchPartitionAction &action,
                                          IRRewriter &rewriter) {
  if (!canApplyLaunchPartition(scope, action))
    return failure();
  auto function = dyn_cast<triton::FuncOp>(scope);
  if (!function)
    return failure();
  (void)rewriter;

  ModuleOp sandbox = ModuleOp::create(function.getLoc());
  sandbox.getBody()->push_back(function->clone());
  auto transformed = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
  if (!transformed)
    return failure();

  Region &sourceBody = function.getBody();
  Region &targetBody = transformed.getBody();
  targetBody.dropAllReferences();
  targetBody.getBlocks().clear();

  SmallVector<triton::GetProgramIdOp, 4> originalProgramIds;
  function.walk([&](triton::GetProgramIdOp programId) {
    if (static_cast<int64_t>(programId.getAxis()) == action.axis)
      originalProgramIds.push_back(programId);
  });
  SmallVector<Block *, 16> laneEntries;
  SmallVector<SmallVector<Block *, 8>, 16> laneBlocks;
  SmallVector<SmallVector<Value, 4>, 16> laneProgramIds;
  for (int64_t lane = 0; lane < action.factor; ++lane) {
    IRMapping mapping;
    sourceBody.cloneInto(&targetBody, mapping);
    Block *entry = mapping.lookupOrNull(&sourceBody.front());
    if (!entry)
      return failure();
    laneEntries.push_back(entry);
    SmallVector<Block *, 8> blocks;
    for (Block &sourceBlock : sourceBody) {
      Block *clonedBlock = mapping.lookupOrNull(&sourceBlock);
      if (!clonedBlock)
        return failure();
      blocks.push_back(clonedBlock);
    }
    laneBlocks.push_back(std::move(blocks));
    SmallVector<Value, 4> programIds;
    for (triton::GetProgramIdOp programId : originalProgramIds) {
      Value cloned = mapping.lookupOrNull(programId.getResult());
      if (!cloned)
        return failure();
      programIds.push_back(cloned);
    }
    laneProgramIds.push_back(std::move(programIds));
  }

  auto *wrapper = new Block();
  targetBody.getBlocks().push_front(wrapper);
  for (BlockArgument argument : function.getArguments())
    wrapper->addArgument(argument.getType(), argument.getLoc());
  for (Block *entry : laneEntries) {
    if (entry->getNumArguments() != wrapper->getNumArguments())
      return failure();
    for (auto [argument, replacement] :
         llvm::zip_equal(entry->getArguments(), wrapper->getArguments()))
      argument.replaceAllUsesWith(replacement);
    entry->eraseArguments([](BlockArgument) { return true; });
  }

  OpBuilder wrapperBuilder = OpBuilder::atBlockEnd(wrapper);
  SmallVector<Value, 4> physicalProgramIds;
  for (triton::GetProgramIdOp programId : originalProgramIds) {
    Operation *physical = programId->clone();
    wrapperBuilder.insert(physical);
    physicalProgramIds.push_back(physical->getResult(0));
  }
  wrapperBuilder.create<cf::BranchOp>(function.getLoc(), laneEntries.front());

  for (int64_t lane = 0; lane < action.factor; ++lane) {
    OpBuilder laneBuilder = OpBuilder::atBlockBegin(laneEntries[lane]);
    for (auto [clonedProgramId, physicalProgramId] :
         llvm::zip_equal(laneProgramIds[lane], physicalProgramIds)) {
      auto integerType = cast<IntegerType>(physicalProgramId.getType());
      Location location = clonedProgramId.getLoc();
      Value factor = laneBuilder.create<arith::ConstantOp>(
          location, integerType,
          IntegerAttr::get(integerType, action.factor));
      Value laneValue = laneBuilder.create<arith::ConstantOp>(
          location, integerType, IntegerAttr::get(integerType, lane));
      Value base = laneBuilder.create<arith::MulIOp>(
          location, physicalProgramId, factor);
      Value logicalProgramId = laneBuilder.create<arith::AddIOp>(
          location, base, laneValue);
      Operation *clonedOperation = clonedProgramId.getDefiningOp();
      clonedProgramId.replaceAllUsesWith(logicalProgramId);
      clonedOperation->erase();
    }
    if (lane + 1 == action.factor)
      continue;
    for (Block *block : laneBlocks[lane]) {
      auto returnOp =
          dyn_cast_or_null<triton::ReturnOp>(block->getTerminator());
      if (!returnOp)
        continue;
      OpBuilder returnBuilder(returnOp);
      returnBuilder.create<cf::BranchOp>(returnOp.getLoc(),
                                         laneEntries[lane + 1]);
      returnOp.erase();
    }
  }

  if (failed(mlir::verify(sandbox.getOperation())))
    return failure();
  function.getBody().takeBody(transformed.getBody());
  return success();
}

class SemanticGraphRewritePlan final : public RewritePlan {
public:
  SemanticGraphRewritePlan(SemanticGraphSolution solution, unsigned epoch,
                           const GraphOptimizationOptions &options,
                           std::shared_ptr<const RuleDatabase> database = {})
      : solution(std::move(solution)), epoch(epoch), options(options),
        database(std::move(database)) {}

  GraphOptimizationRuleId getRuleId() const override {
    return solution.planId;
  }
  unsigned getBenefit() const override {
    // originalCost/selectedCost are scalar telemetry.  Structural selection is
    // lexicographic (stride, weighted operators, node count, depth, ...), so a
    // proven improvement may legitimately have a larger telemetry scalar.
    // Keep the rewrite plan profitable without unsigned underflow in that
    // case; the full cost vector already decided whether the candidate wins.
    if (solution.originalCost > solution.selectedCost)
      return solution.originalCost - solution.selectedCost;
    return 1;
  }
  Operation *getAnchor() const override { return solution.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (!solution.anchor ||
        solution.anchor->getParentOfType<triton::FuncOp>() !=
            context.getFunction())
      return failure();
    SmallVector<TTIRCandidateOpSchema, 4> schemas =
        createBuiltinTTIRCandidateOpSchemas(options);
    std::optional<SemanticGraphSolution> current;
    if (solution.kernelScope) {
      if (!database)
        return failure();
      current = solveKernelFromRuleDatabase(solution.kernelScope, schemas,
                                            options, *database);
    } else if (solution.operationRoot) {
      if (!database)
        return failure();
      current = solveEffectFromRuleDatabase(solution.anchor, schemas, options,
                                            *database);
    } else {
      current = solveConfigured(solution.anchor->getResult(0), schemas,
                                options, database);
    }
    return current && sameSolution(solution, *current) ? success() : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    SmallVector<TTIRCandidateOpSchema, 4> schemas =
        createBuiltinTTIRCandidateOpSchemas(options);
    std::optional<SemanticGraphSolution> current;
    if (solution.kernelScope) {
      if (!database)
        return failure();
      current = solveKernelFromRuleDatabase(solution.kernelScope, schemas,
                                            options, *database);
    } else if (solution.operationRoot) {
      if (!database)
        return failure();
      current = solveEffectFromRuleDatabase(solution.anchor, schemas, options,
                                            *database);
    } else {
      current = solveConfigured(solution.anchor->getResult(0), schemas,
                                options, database);
    }
    if (!current || !sameSolution(solution, *current))
      return failure();

    if (!isRecipeMaterializable(*solution.recipe,
                                solution.operationRoot))
      return failure();
    SmallVector<NamedAttribute, 4> modulePatch;
    if (!collectModuleAttributePatch(*solution.recipe, modulePatch))
      return failure();
    ModuleOp module = solution.anchor->getParentOfType<ModuleOp>();
    if (!canApplyModuleAttributePatch(module, modulePatch))
      return failure();
    if (solution.recipe->attributes &&
        solution.recipe->attributes.get(kCompilerActionAttr)) {
      std::optional<LaunchPartitionAction> action =
          getLaunchPartitionAction(*solution.recipe);
      if (!action)
        return failure();
      if (!solution.kernelScope ||
          !canApplyLaunchPartition(solution.kernelScope, *action) ||
          failed(applyLaunchPartition(solution.kernelScope, *action,
                                      rewriter)))
        return failure();
      applyModuleAttributePatch(module, modulePatch);
      return success();
    }

    SmallVector<Operation *, 8> oldProducers;
    SmallVector<Operation *, 4> replaced;
    replaced.append(solution.replacedEffects.begin(),
                    solution.replacedEffects.end());
    if (!solution.operationRoot)
      replaced.push_back(solution.anchor);
    for (Operation *effect : replaced)
      for (Value operand : effect->getOperands())
        if (Operation *producer = operand.getDefiningOp())
          oldProducers.push_back(producer);
    Operation *insertionAnchor =
        solution.operationRoot ? solution.replacedEffects.front()
                               : solution.anchor;
    SmallPtrSet<Operation *, 16> preflight;
    if (!canHoistRecipeSources(*solution.recipe, insertionAnchor, preflight))
      return failure();
    SmallPtrSet<Operation *, 16> hoisted;
    if (failed(hoistRecipeSources(*solution.recipe, insertionAnchor, rewriter,
                                  hoisted)))
      return failure();
    Block *rewrittenBlock = solution.anchor->getBlock();
    rewriter.setInsertionPoint(insertionAnchor);
    if (solution.operationRoot) {
      if (failed(materializeEffectRecipe(*solution.recipe,
                                         solution.anchor->getLoc(), rewriter)))
        return failure();
      for (Operation *effect : llvm::reverse(solution.replacedEffects)) {
        if (!effect->use_empty())
          return failure();
        rewriter.eraseOp(effect);
      }
      eraseDeadPureProducers(rewriter, oldProducers, rewrittenBlock);
      applyModuleAttributePatch(module, modulePatch);
      return success();
    }
    FailureOr<Value> replacement = materializeRecipe(
        *solution.recipe, solution.anchor->getLoc(), rewriter);
    if (failed(replacement) || replacement->getType() !=
                                   solution.anchor->getResult(0).getType())
      return failure();
    rewriter.replaceOp(solution.anchor, *replacement);
    SmallVector<Operation *, 4> deferredLifetimeCleanup;
    for (Operation *effect : llvm::reverse(solution.replacedEffects)) {
      if (!effect->use_empty()) {
        deferredLifetimeCleanup.push_back(effect);
        continue;
      }
      rewriter.eraseOp(effect);
    }
    eraseDeadPureProducers(rewriter, oldProducers, rewrittenBlock);
    for (Operation *cleanup : deferredLifetimeCleanup) {
      if (!cleanup->use_empty()) {
        cleanup->emitError(
            "certified buffer rewrite left a private lifetime use alive");
        return failure();
      }
      rewriter.eraseOp(cleanup);
    }
    applyModuleAttributePatch(module, modulePatch);
    return success();
  }

private:
  SemanticGraphSolution solution;
  unsigned epoch;
  GraphOptimizationOptions options;
  std::shared_ptr<const RuleDatabase> database;
};

} // namespace

// Start at maximal semantic islands, not at every intermediate SSA value.  A
// value is an island boundary when at least one consumer cannot be represented
// by the registered semantics (stores, returns and other effect/control sinks
// are the common cases).  If a maximal root has no cheaper equivalent, the
// driver descends one semantic level and tries its producers.  It stops below
// a producer as soon as that producer has a solution; the next rewrite epoch
// starts from the maximal root again.  This preserves multi-step discovery
// without eagerly running the exponential search at every node.
static bool isSemanticIslandBoundary(Value value,
                                     const TTOpSemanticsRegistry &semantics) {
  if (!value || value.use_empty())
    return true;
  return llvm::any_of(value.getUsers(), [&](Operation *user) {
    return user->getNumResults() != 1 || !semantics.supports(user);
  });
}

static FailureOr<std::shared_ptr<const RuleDatabase>>
loadAutoGraphRuleDatabase(StringRef path, Operation *diagnosticAnchor) {
  static std::mutex databaseCacheMutex;
  static llvm::StringMap<std::shared_ptr<const RuleDatabase>> databaseCache;
  {
    std::lock_guard<std::mutex> lock(databaseCacheMutex);
    auto cached = databaseCache.find(path);
    if (cached != databaseCache.end())
      return cached->second;
  }

  SmallVector<StringRef, 8> shardPaths;
  path.split(shardPaths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  std::vector<RuleDatabase> shards;
  shards.reserve(shardPaths.size());
  for (StringRef shardPath : shardPaths) {
    shardPath = shardPath.trim();
    if (shardPath.empty()) {
      diagnosticAnchor->emitError(
          "AGS rule database shard list contains an empty path");
      return failure();
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
        llvm::MemoryBuffer::getFile(shardPath);
    if (!input) {
      diagnosticAnchor->emitError()
          << "cannot read AGS rule database shard '" << shardPath
          << "': " << input.getError().message();
      return failure();
    }
    std::string databaseError;
    std::optional<RuleDatabase> loaded =
        RuleDatabase::fromJSON(input.get()->getBuffer(), databaseError);
    if (!loaded) {
      diagnosticAnchor->emitError()
          << "invalid AGS rule database shard '" << shardPath
          << "': " << databaseError;
      return failure();
    }
    if (loaded->semanticsHash != getBuiltinTTOpSemanticsStableHash()) {
      diagnosticAnchor->emitError()
          << "AGS rule database shard '" << shardPath
          << "' semantics version does not match this compiler";
      return failure();
    }
    shards.push_back(std::move(*loaded));
  }

  std::string databaseError;
  std::optional<RuleDatabase> loaded;
  if (shards.size() == 1)
    loaded = std::move(shards.front());
  else
    loaded = RuleDatabase::merge(shards, databaseError);
  if (!loaded) {
    diagnosticAnchor->emitError()
        << "cannot merge AGS rule database shards: " << databaseError;
    return failure();
  }
  std::shared_ptr<const RuleDatabase> parsed =
      std::make_shared<RuleDatabase>(std::move(*loaded));
  std::lock_guard<std::mutex> lock(databaseCacheMutex);
  auto inserted = databaseCache.try_emplace(path, std::move(parsed));
  return inserted.first->second;
}

LogicalResult mlir::triton::cfg::populateAutoGraphSolverCandidates(
    GraphOptimizationContext &context,
    SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans,
    const GraphOptimizationOptions &options) {
  // The production architecture is offline exploration plus online lookup.
  // A missing RuleDB therefore disables AGS instead of silently moving the
  // exponential candidate search into user compilation.  The old in-process
  // solver remains available only to focused IR tests through an explicit
  // pass option; backend/compiler.py never enables that option.
  if (options.autoGraphRuleDatabase.empty() &&
      !options.allowOnlineSemanticEnumeration)
    return success();
  const auto onlineStart = std::chrono::steady_clock::now();
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  if (schemas.empty())
    return success();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  std::shared_ptr<const RuleDatabase> database;
  if (!options.autoGraphRuleDatabase.empty()) {
    FailureOr<std::shared_ptr<const RuleDatabase>> loaded =
        loadAutoGraphRuleDatabase(options.autoGraphRuleDatabase,
                                  context.getFunction());
    if (failed(loaded))
      return failure();
    database = *loaded;
  }

  uint64_t roots = 0;
  uint64_t proved = 0;
  bool budgetExceeded = false;
  auto elapsedMilliseconds = [&]() -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - onlineStart)
            .count());
  };
  auto shouldStopOnlineWork = [&]() {
    return options.maxAutoGraphOnlineMilliseconds != 0 &&
           elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds;
  };
  if (database) {
    ++roots;
    std::optional<SemanticGraphSolution> kernel =
        solveKernelFromRuleDatabase(context.getFunction(), schemas, options,
                                    *database, shouldStopOnlineWork);
    if (options.maxAutoGraphOnlineMilliseconds != 0 &&
        elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds) {
      context.getFunction().emitError()
          << "AutoGraphSolver exceeded max-auto-graph-online-ms while "
             "matching the whole-kernel root";
      return failure();
    }
    if (kernel) {
      ++proved;
      plans.push_back(std::make_unique<SemanticGraphRewritePlan>(
          *kernel, context.getEpoch(), options, database));
      LLVM_DEBUG({
        llvm::dbgs()
            << "[graph-optimize-telemetry] component=AutoGraphSolver"
            << " mode=rule-database-kernel-lookup roots=" << roots
            << " proved=" << proved
            << " elapsed-ms=" << elapsedMilliseconds() << '\n';
      });
      return success();
    }
  }
  llvm::DenseSet<Value> visited;
  std::function<void(Value)> solveRootOrDescend = [&](Value result) {
    if (budgetExceeded || !result || !visited.insert(result).second)
      return;

    bool canProduce = llvm::any_of(
        schemas, [&](const TTIRCandidateOpSchema &schema) {
          return schema.constructible &&
                 (database || !schema.offlineOnly) &&
                 (!schema.canProduceResult ||
                  schema.canProduceResult(result.getType(),
                                          /*attributes=*/{}));
        });
    if (canProduce) {
      ++roots;
      std::optional<SemanticGraphSolution> solution =
          solveConfigured(result, schemas, options, database,
                          shouldStopOnlineWork);
      if (options.maxAutoGraphOnlineMilliseconds != 0 &&
          elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds) {
        budgetExceeded = true;
        return;
      }
      if (solution) {
        ++proved;
        LLVM_DEBUG({
          llvm::dbgs()
              << "[graph-optimize-telemetry] component=AutoGraphSolver"
              << " selected-op=" << solution->recipe->operationName
              << " generated=" << solution->generatedCandidates
              << " semantic-merges=" << solution->provedEquivalent
              << " proof-steps=" << solution->proofSteps
              << " lookup-mode=" << stringifyLookupMode(solution->lookupMode)
              << " original-cost=" << solution->originalCost
              << " selected-cost=" << solution->selectedCost << '\n';
        });
        plans.push_back(std::make_unique<SemanticGraphRewritePlan>(
            *solution, context.getEpoch(), options, database));
        return;
      }
    }

    Operation *producer = result.getDefiningOp();
    if (!producer || !semantics.supports(producer))
      return;
    for (Value operand : producer->getOperands()) {
      Operation *operandProducer = operand.getDefiningOp();
      if (!operandProducer || operandProducer->getNumResults() != 1 ||
          !semantics.supports(operandProducer))
        continue;
      solveRootOrDescend(operand);
    }
  };

  context.getFunction().walk([&](Operation *operation) {
    if (budgetExceeded)
      return;
    bool unusedAtomicStateRoot = false;
    if (operation->getNumResults() == 1 && operation->use_empty())
      unusedAtomicStateRoot = llvm::any_of(
          schemas, [&](const TTIRCandidateOpSchema &schema) {
            return schema.operationName ==
                       operation->getName().getStringRef() &&
                   schema.operandCount == operation->getNumOperands() &&
                   (!schema.matchesOperation ||
                    schema.matchesOperation(operation)) &&
                   schema.effectKind == TTIRCandidateEffectKind::Atomic;
          });
    if (operation->getNumResults() == 0 || unusedAtomicStateRoot) {
      if (!database || !semantics.supports(operation))
        return;
      ++roots;
      std::optional<SemanticGraphSolution> solution =
          solveEffectFromRuleDatabase(operation, schemas, options, *database,
                                      shouldStopOnlineWork);
      if (options.maxAutoGraphOnlineMilliseconds != 0 &&
          elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds) {
        budgetExceeded = true;
        return;
      }
      if (solution) {
        ++proved;
        plans.push_back(std::make_unique<SemanticGraphRewritePlan>(
            *solution, context.getEpoch(), options, database));
      }
      return;
    }
    if (operation->getNumResults() != 1)
      return;
    Value result = operation->getResult(0);
    if (!isSemanticIslandBoundary(result, semantics))
      return;
    solveRootOrDescend(result);
  });

  if (budgetExceeded) {
    context.getFunction().emitError()
        << "AutoGraphSolver exceeded max-auto-graph-online-ms: "
        << elapsedMilliseconds() << " > "
        << options.maxAutoGraphOnlineMilliseconds;
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[graph-optimize-telemetry] component=AutoGraphSolver"
                 << " mode="
                 << (database ? "rule-database-lookup"
                              : "semantic-enumeration")
                 << " semantics=ttop-denotation-v2"
                 << " cost-model="
                 << (database ? database->costModelName
                              : "structural-node-count-v1")
                 << " roots=" << roots << " proved=" << proved
                 << " elapsed-ms=" << elapsedMilliseconds() << '\n';
  });
  return success();
}

LogicalResult mlir::triton::cfg::applyAutoGraphSolverDatabase(
    Operation *scope, const GraphOptimizationOptions &options) {
  if (!scope || options.autoGraphRuleDatabase.empty())
    return success();
  FailureOr<std::shared_ptr<const RuleDatabase>> loaded =
      loadAutoGraphRuleDatabase(options.autoGraphRuleDatabase, scope);
  if (failed(loaded))
    return failure();
  std::shared_ptr<const RuleDatabase> database = *loaded;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  SmallVector<Operation *, 4> kernelScopes;
  if (isa<triton::FuncOp>(scope)) {
    kernelScopes.push_back(scope);
  } else {
    scope->walk([&](triton::FuncOp function) {
      kernelScopes.push_back(function.getOperation());
    });
  }
  const auto start = std::chrono::steady_clock::now();
  auto elapsedMilliseconds = [&]() -> uint64_t {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };
  auto shouldStopOnlineWork = [&]() {
    return options.maxAutoGraphOnlineMilliseconds != 0 &&
           elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds;
  };

  unsigned applied = 0;
  while (applied < options.maxRewritesPerFunction) {
    std::optional<SemanticGraphSolution> selected;
    for (Operation *kernelScope : kernelScopes) {
      selected = solveKernelFromRuleDatabase(kernelScope, schemas, options,
                                             *database, shouldStopOnlineWork);
      if (selected)
        break;
    }
    if (!selected)
      scope->walk([&](Operation *operation) -> WalkResult {
        if (operation == scope || !semantics.supports(operation))
          return WalkResult::advance();
        if (operation->getNumResults() == 0) {
          selected = solveEffectFromRuleDatabase(operation, schemas, options,
                                                 *database,
                                                 shouldStopOnlineWork);
        } else if (operation->getNumResults() == 1) {
          selected = solveFromRuleDatabase(operation->getResult(0), schemas,
                                           options, *database,
                                           shouldStopOnlineWork);
        }
        return selected ? WalkResult::interrupt() : WalkResult::advance();
      });
    if (options.maxAutoGraphOnlineMilliseconds != 0 &&
        elapsedMilliseconds() > options.maxAutoGraphOnlineMilliseconds) {
      scope->emitError()
          << "AutoGraphSolver stage lookup exceeded "
          << "max-auto-graph-online-ms: " << elapsedMilliseconds() << " > "
          << options.maxAutoGraphOnlineMilliseconds;
      return failure();
    }
    if (!selected)
      break;
    SemanticGraphRewritePlan plan(*selected, applied, options, database);
    IRRewriter rewriter(scope->getContext());
    if (failed(plan.apply(rewriter))) {
      scope->emitError("AutoGraphSolver stage recipe failed revalidation or "
                       "transactional materialization");
      return failure();
    }
    ++applied;
  }
  LLVM_DEBUG(llvm::dbgs()
             << "[graph-optimize-telemetry] component=AutoGraphSolver"
             << " mode=stage-rule-database-lookup"
             << " rewrites=" << applied
             << " elapsed-ms=" << elapsedMilliseconds() << '\n';);
  return success();
}

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

#include "TritonToGraph/CanonicalGraphKey.h"
#include "TritonToGraph/CanonicalRegion.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/AsmParser/AsmParser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

using namespace mlir;
using namespace mlir::triton::cfg;

#define DEBUG_TYPE "graph-optimize"

namespace {

static constexpr llvm::StringLiteral kResultIndexAttr = "ags.result_index";
static constexpr llvm::StringLiteral kOperationResultTypesAttr =
    "ags.operation_result_types";
static constexpr llvm::StringLiteral kStateDomainAttr = "ags.state_domain";
static constexpr llvm::StringLiteral kCallSummaryRegionsAttr =
    "ags.call_summary_regions";

static StringRef stringifyStateDomain(TTIRCandidateStateDomain domain) {
  switch (domain) {
  case TTIRCandidateStateDomain::Memory:
    return "memory";
  case TTIRCandidateStateDomain::Launch:
    return "launch";
  case TTIRCandidateStateDomain::Buffer:
    return "buffer";
  }
  llvm_unreachable("unknown candidate state domain");
}

static std::string printType(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static std::string printAttributes(DictionaryAttr attributes) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << attributes;
  return storage;
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

static bool hasOnlyAllocationEffects(Operation *operation) {
  auto effects = dyn_cast_or_null<MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance, 4> effectList;
  effects.getEffects(effectList);
  return !effectList.empty() &&
         llvm::all_of(effectList, [](const auto &effect) {
           return isa<MemoryEffects::Allocate>(effect.getEffect());
         });
}

static bool isZeroCostGraphTerminal(StringRef operationName) {
  return operationName == "ags.input" ||
         operationName == "ags.private_buffer" ||
         operationName == "ags.memory" ||
         operationName == "ags.partitionable_memory" ||
         operationName == "ags.launch" ||
         operationName == "ags.buffer" ||
         operationName == "arith.constant";
}

static std::optional<std::string> buildNormalizedSemanticKey(
    const CanonicalGraphBuildResult &graph,
    const TTOpSemanticsRegistry &semantics, MLIRContext *context,
    const std::function<bool()> &shouldStop) {
  llvm::StringMap<const PersistentGraphRecord *> records;
  for (const PersistentGraphRecord &record : graph.graphs)
    records.try_emplace(record.graphHash, &record);
  llvm::StringMap<TensorDenotation> denotations;
  llvm::StringSet<> active;
  TensorSemanticNormalizer normalizer(shouldStop);
  std::function<std::optional<TensorDenotation>(StringRef)> denote =
      [&](StringRef hash) -> std::optional<TensorDenotation> {
    if (shouldStop && shouldStop())
      return std::nullopt;
    auto known = denotations.find(hash);
    if (known != denotations.end())
      return known->second;
    if (!active.insert(hash).second)
      return std::nullopt;
    auto found = records.find(hash);
    if (found == records.end()) {
      active.erase(hash);
      return std::nullopt;
    }
    const PersistentGraphRecord &record = *found->second;
    const TTOpSemantics *definition =
        semantics.lookup(record.operationName);
    Type resultType = mlir::parseType(record.resultType, context);
    Attribute parsedAttributes =
        mlir::parseAttribute(record.attributes, context);
    auto dictionary = dyn_cast_or_null<DictionaryAttr>(parsedAttributes);
    if (!definition || !resultType || !dictionary) {
      active.erase(hash);
      return std::nullopt;
    }
    SmallVector<TensorDenotation, 4> operands;
    SmallVector<GraphEClassId, 4> operandIds;
    for (auto [index, operandHash] :
         llvm::enumerate(record.operandHashes)) {
      std::optional<TensorDenotation> operand = denote(operandHash);
      if (!operand) {
        active.erase(hash);
        return std::nullopt;
      }
      operands.push_back(std::move(*operand));
      operandIds.push_back(static_cast<GraphEClassId>(index));
    }
    Builder builder(context);
    SmallVector<NamedAttribute, 8> attributes(dictionary.begin(),
                                             dictionary.end());
    attributes.push_back(builder.getNamedAttr(
        "ags.result_type", TypeAttr::get(resultType)));
    GraphExprNode node = GraphExprNode::create(
        context, record.operationName, operandIds, attributes);
    FailureOr<TensorDenotation> result =
        definition->denote(node, operands);
    active.erase(hash);
    if (failed(result))
      return std::nullopt;
    TensorDenotation normalized = normalizer.normalize(*result).denotation;
    if (normalizer.wasInterrupted())
      return std::nullopt;
    denotations.try_emplace(hash, normalized);
    return normalized;
  };

  std::optional<TensorDenotation> root = denote(graph.rootGraphHash);
  if (!root)
    return std::nullopt;
  std::string key = getStableTensorDenotationKey(*root);
  return key;
}

} // namespace

CanonicalTTIRGraphBuilder::CanonicalTTIRGraphBuilder(
    const TTOpSemanticsRegistry &semantics,
    llvm::ArrayRef<TTIRCandidateOpSchema> schemas,
    std::function<bool()> shouldStop)
    : semantics(semantics), schemas(schemas),
      shouldStop(std::move(shouldStop)) {}

bool CanonicalTTIRGraphBuilder::shouldInterrupt(std::string &error) const {
  if (!interrupted && (!shouldStop || !shouldStop()))
    return false;
  interrupted = true;
  error = "canonical graph construction interrupted";
  return true;
}

std::optional<CanonicalGraphBuildResult>
CanonicalTTIRGraphBuilder::build(Value root, std::string &error) const {
  std::optional<CanonicalGraphBuildResult> result =
      buildImpl(root, /*operationRoot=*/nullptr, error);
  if (result)
    if (std::optional<std::string> key =
            buildNormalizedSemanticKey(*result, semantics, root.getContext(),
                                       shouldStop))
      result->normalizedSemanticKey = std::move(*key);
  if (shouldInterrupt(error))
    return std::nullopt;
  return result;
}

std::optional<CanonicalGraphBuildResult>
CanonicalTTIRGraphBuilder::build(Operation *root, std::string &error) const {
  std::optional<CanonicalGraphBuildResult> result =
      buildImpl(/*valueRoot=*/{}, root, error);
  if (result)
    if (std::optional<std::string> key = buildNormalizedSemanticKey(
            *result, semantics, root->getContext(), shouldStop))
      result->normalizedSemanticKey = std::move(*key);
  if (shouldInterrupt(error))
    return std::nullopt;
  return result;
}

std::optional<CanonicalGraphBuildResult>
CanonicalTTIRGraphBuilder::buildKernel(Operation *scope,
                                       std::string &error) const {
  if (!scope || scope->getNumRegions() != 1 ||
      !scope->getRegion(0).hasOneBlock()) {
    error = "whole-kernel canonicalization requires one straight-line block";
    return std::nullopt;
  }

  Operation *finalMemoryTransition = nullptr;
  for (Operation &operation : scope->getRegion(0).front()) {
    if (shouldInterrupt(error))
      return std::nullopt;
    const TTIRCandidateOpSchema *schema = nullptr;
    for (const TTIRCandidateOpSchema &candidate : schemas)
      if (candidate.operationName == operation.getName().getStringRef() &&
          candidate.operandCount == operation.getNumOperands()) {
        schema = &candidate;
        break;
      }
    if (!schema) {
      if (!isMemoryEffectFree(&operation)) {
        error = "unknown memory effect in whole-kernel scope: " +
                operation.getName().getStringRef().str();
        return std::nullopt;
      }
      continue;
    }
    if (schema->effectKind == TTIRCandidateEffectKind::Write ||
        schema->effectKind == TTIRCandidateEffectKind::Observable) {
      if (operation.getNumResults() > 1 || !semantics.supports(&operation)) {
        error = "unsupported observable transition in whole-kernel scope";
        return std::nullopt;
      }
      finalMemoryTransition = &operation;
      continue;
    }
    if (schema->effectKind == TTIRCandidateEffectKind::Atomic) {
      if (operation.getNumResults() != 1 || !semantics.supports(&operation)) {
        error = "unsupported atomic in whole-kernel scope";
        return std::nullopt;
      }
      finalMemoryTransition = &operation;
    }
  }
  if (!finalMemoryTransition) {
    error = "whole-kernel scope has no observable memory transition";
    return std::nullopt;
  }
  if (finalMemoryTransition->getNumResults() == 1 &&
      !finalMemoryTransition->use_empty()) {
    error = "final whole-kernel atomic result is still observable";
    return std::nullopt;
  }

  std::optional<CanonicalGraphBuildResult> result = buildImpl(
      /*valueRoot=*/{}, finalMemoryTransition, error,
      /*includeLaunchState=*/true);
  if (!result)
    return std::nullopt;

  auto findGraph = [&](StringRef hash) -> const PersistentGraphRecord * {
    auto found = llvm::find_if(result->graphs, [&](const auto &graph) {
      return graph.graphHash == hash;
    });
    return found == result->graphs.end() ? nullptr : &*found;
  };
  const PersistentGraphRecord *memoryRoot = findGraph(result->rootGraphHash);
  if (!memoryRoot) {
    error = "whole-kernel MemoryState root is missing";
    return std::nullopt;
  }
  std::string launchHash = result->launchStateHash;
  if (launchHash.empty()) {
    PersistentGraphRecord launch;
    launch.operationName = "ags.launch";
    launch.resultType = "none";
    launch.attributes =
        printAttributes(Builder(scope->getContext()).getDictionaryAttr({}));
    launch.operatorSize = 0;
    launch.graphHash = getPersistentGraphHash(launch);
    launchHash = launch.graphHash;
    result->graphs.push_back(std::move(launch));
  }

  Builder builder(scope->getContext());
  PersistentGraphRecord kernel;
  kernel.operationName = "ags.kernel";
  kernel.resultType = "none";
  kernel.attributes = printAttributes(builder.getDictionaryAttr(
      {builder.getNamedAttr("ags.proof_only", builder.getBoolAttr(true)),
       builder.getNamedAttr("ags.state_domain",
                            builder.getStringAttr("memory"))}));
  kernel.operandHashes = {result->rootGraphHash, launchHash};
  llvm::StringMap<const PersistentGraphRecord *> graphByHash;
  for (const PersistentGraphRecord &graph : result->graphs)
    graphByHash[graph.graphHash] = &graph;
  llvm::StringSet<> reachableOperators;
  std::function<bool(StringRef)> collectReachable = [&](StringRef hash) {
    if (shouldInterrupt(error))
      return false;
    auto graph = graphByHash.find(hash);
    if (graph == graphByHash.end())
      return false;
    if (!isZeroCostGraphTerminal(graph->second->operationName) &&
        !reachableOperators.insert(hash).second)
      return true;
    for (StringRef operand : graph->second->operandHashes)
      if (!collectReachable(operand))
        return false;
    return true;
  };
  if (!collectReachable(result->rootGraphHash) ||
      !collectReachable(launchHash)) {
    if (error.empty())
      error = "whole-kernel root has a missing reachable graph";
    return std::nullopt;
  }
  kernel.operatorSize = reachableOperators.size() + 1;
  kernel.graphHash = getPersistentGraphHash(kernel);
  result->rootGraphHash = kernel.graphHash;
  if (!findGraph(kernel.graphHash))
    result->graphs.push_back(std::move(kernel));
  llvm::sort(result->graphs, [](const auto &lhs, const auto &rhs) {
    return lhs.graphHash < rhs.graphHash;
  });
  // Whole-kernel lookup currently has only exact and parameterized-template
  // indexes.  Building a normalized denotation here is both unused and
  // potentially exponential: a shared tensor DAG is expanded into a value
  // expression tree, so wide address/mask graphs can spend minutes copying an
  // otherwise irrelevant semantic key.  Per-value and per-effect builders
  // still compute normalizedSemanticKey for the semantic RuleDB index.
  return result;
}

std::optional<CanonicalGraphBuildResult>
CanonicalTTIRGraphBuilder::buildImpl(Value root, Operation *operationRoot,
                                     std::string &error,
                                     bool includeLaunchState) const {
  if (shouldInterrupt(error))
    return std::nullopt;
  if (!root && !operationRoot) {
    error = "cannot canonicalize an empty TTIR value";
    return std::nullopt;
  }
  if (root && operationRoot) {
    error = "canonical TTIR graph must have exactly one root kind";
    return std::nullopt;
  }
  MLIRContext *context = root ? root.getContext() : operationRoot->getContext();

  llvm::StringMap<llvm::StringSet<>> semanticAttributes;
  llvm::StringMap<SmallVector<const TTIRCandidateOpSchema *, 2>>
      schemasByOperation;
  for (const TTIRCandidateOpSchema &schema : schemas) {
    schemasByOperation[schema.operationName].push_back(&schema);
    for (const TTIRCandidateAttributeBinding &binding :
         schema.attributeBindings)
      semanticAttributes[schema.operationName].insert(binding.attributeName);
    for (const std::string &attribute : schema.semanticAttributeNames)
      semanticAttributes[schema.operationName].insert(attribute);
  }

  CanonicalGraphBuildResult result;
  DenseMap<Value, std::string> hashes;
  DenseMap<Value, unsigned> sizes;
  llvm::StringMap<unsigned> graphIndices;
  llvm::StringMap<unsigned> graphSizes;
  llvm::StringMap<SmallVector<std::string, 8>> reachableGraphOperators;
  unsigned nextInput = 0;
  std::function<std::optional<std::string>(Value, bool)> import;

  auto addGraph = [&](PersistentGraphRecord graph) {
    llvm::StringSet<> reachableSet;
    for (StringRef operandHash : graph.operandHashes) {
      auto known = reachableGraphOperators.find(operandHash);
      assert(known != reachableGraphOperators.end() &&
             "operand graph must be canonicalized before its user");
      for (const std::string &reachable : known->second)
        reachableSet.insert(reachable);
    }
    graph.operatorSize = reachableSet.size() +
                         (isZeroCostGraphTerminal(graph.operationName) ? 0
                                                                      : 1);
    graph.graphHash = getPersistentGraphHash(graph);
    graphSizes[graph.graphHash] = graph.operatorSize;
    SmallVector<std::string, 8> reachable;
    reachable.reserve(graph.operatorSize);
    for (const auto &entry : reachableSet)
      reachable.push_back(entry.getKey().str());
    if (!isZeroCostGraphTerminal(graph.operationName))
      reachable.push_back(graph.graphHash);
    llvm::sort(reachable);
    reachableGraphOperators.try_emplace(graph.graphHash,
                                        std::move(reachable));
    auto inserted =
        graphIndices.try_emplace(graph.graphHash, result.graphs.size());
    if (inserted.second)
      result.graphs.push_back(std::move(graph));
    return inserted.first->getKey().str();
  };

  std::optional<std::string> initialMemory;
  auto addInitialMemory = [&]() -> std::string {
    if (initialMemory)
      return *initialMemory;
    PersistentGraphRecord memory;
    memory.operationName = "ags.memory";
    memory.resultType = "none";
    memory.attributes = printAttributes(
        Builder(context).getDictionaryAttr({}));
    memory.operatorSize = 0;
    initialMemory = addGraph(std::move(memory));
    return *initialMemory;
  };

  std::optional<std::string> initialLaunch;
  auto addInitialLaunch = [&]() -> std::string {
    if (initialLaunch)
      return *initialLaunch;
    PersistentGraphRecord launch;
    launch.operationName = "ags.launch";
    launch.resultType = "none";
    launch.attributes =
        printAttributes(Builder(context).getDictionaryAttr({}));
    launch.operatorSize = 0;
    std::string state = addGraph(std::move(launch));

    Operation *parent = operationRoot;
    if (!parent && root) {
      if (Operation *definition = root.getDefiningOp())
        parent = definition;
      else if (Block *block = root.getParentBlock())
        parent = block->getParentOp();
    }
    triton::FuncOp function;
    if (parent) {
      function = dyn_cast<triton::FuncOp>(parent);
      if (!function)
        function = parent->getParentOfType<triton::FuncOp>();
    }
    if (function) {
      SmallVector<std::pair<int64_t, int64_t>, 3> contracts;
      auto axes = function->getAttrOfType<DenseI32ArrayAttr>(
          "ags.launch_grid_bound_axes");
      auto arguments = function->getAttrOfType<DenseI32ArrayAttr>(
          "ags.launch_grid_bound_arguments");
      if (static_cast<bool>(axes) != static_cast<bool>(arguments) ||
          (axes && axes.size() != arguments.size())) {
        error = "launch grid bound requires equal axes and arguments arrays";
        return {};
      }
      if (axes)
        for (auto [axisValue, argumentValue] :
             llvm::zip(axes.asArrayRef(), arguments.asArrayRef()))
          contracts.emplace_back(axisValue, argumentValue);

      auto legacyAxis = function->getAttrOfType<IntegerAttr>(
          "ags.launch_grid_bound_axis");
      auto legacyArgument = function->getAttrOfType<IntegerAttr>(
          "ags.launch_grid_bound_argument");
      if (static_cast<bool>(legacyAxis) != static_cast<bool>(legacyArgument)) {
        error = "launch grid bound requires both legacy axis and argument attrs";
        return {};
      }
      if (legacyAxis)
        contracts.emplace_back(legacyAxis.getInt(), legacyArgument.getInt());

      llvm::SmallSet<int64_t, 3> seenAxes;
      for (auto [axisValue, argumentValue] : contracts) {
        if (axisValue < 0 || axisValue > 2 || argumentValue < 0 ||
            static_cast<uint64_t>(argumentValue) >=
                function.getNumArguments()) {
          error = "invalid launch grid bound contract";
          return {};
        }
        Value bound = function.getArgument(
            static_cast<unsigned>(argumentValue));
        Type boundType = bound.getType();
        if (!isa<IntegerType>(boundType)) {
          error = "launch grid bound argument must be a scalar integer";
          return {};
        }
        std::optional<std::string> boundHash = import(bound, false);
        if (!boundHash)
          return {};
        Builder builder(context);
        PersistentGraphRecord contract;
        contract.operationName = "ags.launch_bound";
        contract.resultType = "none";
        contract.attributes = printAttributes(builder.getDictionaryAttr(
            {builder.getNamedAttr("ags.proof_only",
                                  builder.getBoolAttr(true)),
             builder.getNamedAttr(kStateDomainAttr,
                                  builder.getStringAttr("launch")),
             builder.getNamedAttr("axis",
                                  builder.getI32IntegerAttr(axisValue))}));
        contract.operandHashes = {*boundHash, state};
        state = addGraph(std::move(contract));
        if (!seenAxes.insert(axisValue).second) {
          error = "duplicate launch grid bound axis";
          return {};
        }
      }
    }
    initialLaunch = std::move(state);
    return *initialLaunch;
  };

  std::optional<std::string> initialBuffer;
  auto addInitialBuffer = [&]() -> std::string {
    if (initialBuffer)
      return *initialBuffer;
    PersistentGraphRecord buffer;
    buffer.operationName = "ags.buffer";
    buffer.resultType = "none";
    buffer.attributes =
        printAttributes(Builder(context).getDictionaryAttr({}));
    buffer.operatorSize = 0;
    initialBuffer = addGraph(std::move(buffer));
    return *initialBuffer;
  };

  auto findSchema = [&](Operation *operation)
      -> const TTIRCandidateOpSchema * {
    if (!operation)
      return nullptr;
    auto found = schemasByOperation.find(
        operation->getName().getStringRef());
    if (found == schemasByOperation.end())
      return nullptr;
    for (const TTIRCandidateOpSchema *schema : found->second)
      if (schema->operandCount == operation->getNumOperands() &&
          (!schema->matchesOperation || schema->matchesOperation(operation)))
        return schema;
    return nullptr;
  };

  auto getSemanticAttributes = [&](Operation *operation) {
    SmallVector<NamedAttribute, 4> attributes;
    StringRef operationName = operation->getName().getStringRef();
    if (operationName == "arith.constant") {
      if (Attribute valueAttribute = operation->getAttr("value"))
        attributes.push_back(Builder(operation->getContext())
                                 .getNamedAttr("value", valueAttribute));
      return attributes;
    }
    const TTIRCandidateOpSchema *schema = findSchema(operation);
    auto isExplicitlyBound = [&](StringRef name) {
      return schema && llvm::any_of(
                           schema->attributeBindings,
                           [&](const TTIRCandidateAttributeBinding &binding) {
                             return binding.attributeName == name;
                           });
    };
    auto isUnboundAbsentDefault = [&](NamedAttribute attribute) {
      if (isExplicitlyBound(attribute.getName().getValue()))
        return false;
      if (auto boolean = dyn_cast<BoolAttr>(attribute.getValue()))
        return !boolean.getValue();
      if (auto array = dyn_cast<DenseI32ArrayAttr>(attribute.getValue()))
        return array.empty();
      if (auto array = dyn_cast<ArrayAttr>(attribute.getValue()))
        return array.empty();
      return false;
    };
    auto selected = semanticAttributes.find(operationName);
    if (selected != semanticAttributes.end())
      for (NamedAttribute attribute : operation->getAttrs())
        if (selected->second.contains(attribute.getName().getValue()) &&
            !isUnboundAbsentDefault(attribute))
          attributes.push_back(attribute);
    if (schema)
      if (schema->producedStateDomain)
        attributes.push_back(Builder(operation->getContext()).getNamedAttr(
            kStateDomainAttr,
            Builder(operation->getContext())
                .getStringAttr(
                    stringifyStateDomain(*schema->producedStateDomain))));
    if (schema && schema->effectKind == TTIRCandidateEffectKind::Pure) {
      if (auto call = dyn_cast<CallOpInterface>(operation)) {
        Operation *callable = call.resolveCallable();
        std::string summaryError;
        std::optional<ArrayAttr> summary =
            callable ? captureCanonicalRegions(callable, summaryError)
                     : std::nullopt;
        if (!summary) {
          error = "cannot build pure call summary for " +
                  operationName.str() + ": " +
                  (summaryError.empty() ? "unresolved callable"
                                        : summaryError);
          return attributes;
        }
        attributes.push_back(Builder(operation->getContext()).getNamedAttr(
            kCallSummaryRegionsAttr, *summary));
      }
    }
    if (operation->getNumRegions() != 0) {
      std::string regionError;
      std::optional<ArrayAttr> regions =
          captureCanonicalRegions(operation, regionError);
      if (!regions) {
        error = "cannot canonicalize regions of " + operationName.str() +
                ": " + regionError;
        return attributes;
      }
      attributes.push_back(Builder(operation->getContext()).getNamedAttr(
          kCanonicalRegionsAttr, *regions));
    }
    return attributes;
  };

  DenseMap<Operation *, std::string> stateBeforeCache;
  DenseMap<Operation *, std::string> bufferStateBeforeCache;
  std::function<std::optional<std::string>(Operation *)> getStateBefore;
  std::function<std::optional<std::string>(Operation *)> getBufferStateBefore;

  getStateBefore = [&](Operation *target) -> std::optional<std::string> {
    if (shouldInterrupt(error))
      return std::nullopt;
    if (!target || !target->getBlock()) {
      error = "cannot construct MemorySSA outside a straight-line block";
      return std::nullopt;
    }
    if (auto known = stateBeforeCache.find(target);
        known != stateBeforeCache.end())
      return known->second;

    std::string state = addInitialMemory();
    for (Operation &operation : *target->getBlock()) {
      if (shouldInterrupt(error))
        return std::nullopt;
      if (&operation == target) {
        stateBeforeCache[target] = state;
        return state;
      }

      const TTIRCandidateOpSchema *schema = findSchema(&operation);
      if (!schema) {
        // Allocation establishes the identity represented by an
        // ags.private_buffer terminal. It does not mutate the contents of any
        // pre-existing state; reads/writes/frees still fail closed when no
        // executable state semantics is available.
        if (hasOnlyAllocationEffects(&operation))
          continue;
        if (!isMemoryEffectFree(&operation)) {
          error = "unknown memory effect before semantic root: " +
                  operation.getName().getStringRef().str();
          return std::nullopt;
        }
        continue;
      }
      if (schema->effectKind == TTIRCandidateEffectKind::Pure ||
          schema->effectKind == TTIRCandidateEffectKind::Read ||
          schema->effectKind == TTIRCandidateEffectKind::BufferRead ||
          schema->effectKind == TTIRCandidateEffectKind::BufferWrite)
        continue;

      if (!semantics.supports(&operation)) {
        error = "unsupported stateful operation before semantic root: " +
                operation.getName().getStringRef().str();
        return std::nullopt;
      }

      PersistentGraphRecord effect;
      effect.operationName = operation.getName().getStringRef().str();
      effect.effectKind = schema->effectKind;
      const bool directStateTransition =
          schema->effectKind == TTIRCandidateEffectKind::Write ||
          (schema->effectKind == TTIRCandidateEffectKind::Observable &&
           operation.getNumResults() == 0);
      if (directStateTransition) {
        if (operation.getNumResults() != 0) {
          error = "observable state transition requires a zero-result TTIR op";
          return std::nullopt;
        }
        effect.resultType = "none";
      } else {
        if (operation.getNumResults() != 1) {
          error = "atomic semantic currently requires one TTIR result";
          return std::nullopt;
        }
        effect.resultType = printType(operation.getResult(0).getType());
      }

      unsigned operatorSize = 1;
      for (Value operand : operation.getOperands()) {
        std::optional<std::string> operandHash = import(operand, false);
        if (!operandHash)
          return std::nullopt;
        effect.operandHashes.push_back(*operandHash);
        operatorSize += sizes.lookup(operand);
      }
      effect.operandHashes.push_back(state);
      operatorSize += graphSizes.lookup(state);
      effect.operatorSize = operatorSize;
      effect.attributes = printAttributes(Builder(context)
                                              .getDictionaryAttr(
                                                  getSemanticAttributes(
                                                      &operation)));
      if (!error.empty())
        return std::nullopt;
      std::string effectHash = addGraph(std::move(effect));
      if (directStateTransition) {
        state = effectHash;
        result.stateOperations.emplace_back(state, &operation);
        continue;
      }

      hashes[operation.getResult(0)] = effectHash;
      sizes[operation.getResult(0)] = graphSizes.lookup(effectHash);
      result.graphValues.emplace_back(effectHash, operation.getResult(0));
      PersistentGraphRecord projection;
      projection.operationName = "ags.memory_after";
      projection.resultType = "none";
      Builder builder(context);
      projection.attributes = printAttributes(builder.getDictionaryAttr(
          {builder.getNamedAttr(kStateDomainAttr,
                                builder.getStringAttr("memory"))}));
      projection.operatorSize = operatorSize + 1;
      projection.operandHashes.push_back(effectHash);
      state = addGraph(std::move(projection));
      result.stateOperations.emplace_back(state, &operation);
    }
    error = "semantic root is not in its parent block";
    return std::nullopt;
  };

  getBufferStateBefore = [&](Operation *target)
      -> std::optional<std::string> {
    if (shouldInterrupt(error))
      return std::nullopt;
    if (!target || !target->getBlock()) {
      error = "cannot construct BufferSSA outside a straight-line block";
      return std::nullopt;
    }
    if (auto known = bufferStateBeforeCache.find(target);
        known != bufferStateBeforeCache.end())
      return known->second;

    std::string state = addInitialBuffer();
    for (Operation &operation : *target->getBlock()) {
      if (shouldInterrupt(error))
        return std::nullopt;
      if (&operation == target) {
        bufferStateBeforeCache[target] = state;
        return state;
      }
      const TTIRCandidateOpSchema *schema = findSchema(&operation);
      if (!schema) {
        if (hasOnlyAllocationEffects(&operation))
          continue;
        if (!isMemoryEffectFree(&operation)) {
          error = "unknown buffer effect before semantic root: " +
                  operation.getName().getStringRef().str();
          return std::nullopt;
        }
        continue;
      }
      if (schema->effectKind != TTIRCandidateEffectKind::BufferWrite)
        continue;
      if (!semantics.supports(&operation) || operation.getNumResults() != 0) {
        error = "unsupported BufferState transition before semantic root: " +
                operation.getName().getStringRef().str();
        return std::nullopt;
      }

      PersistentGraphRecord effect;
      effect.operationName = operation.getName().getStringRef().str();
      effect.effectKind = schema->effectKind;
      effect.resultType = "none";
      unsigned operatorSize = 1;
      for (Value operand : operation.getOperands()) {
        std::optional<std::string> operandHash = import(operand, false);
        if (!operandHash)
          return std::nullopt;
        effect.operandHashes.push_back(*operandHash);
        operatorSize += sizes.lookup(operand);
      }
      effect.operandHashes.push_back(state);
      operatorSize += graphSizes.lookup(state);
      effect.operatorSize = operatorSize;
      effect.attributes = printAttributes(
          Builder(context).getDictionaryAttr(
              getSemanticAttributes(&operation)));
      if (!error.empty())
        return std::nullopt;
      state = addGraph(std::move(effect));
      result.stateOperations.emplace_back(state, &operation);
    }
    error = "semantic root is not in its parent block";
    return std::nullopt;
  };

  import =
      [&](Value value, bool isRoot) -> std::optional<std::string> {
    if (shouldInterrupt(error))
      return std::nullopt;
    auto known = hashes.find(value);
    if (known != hashes.end())
      return known->second;

    Operation *operation = value.getDefiningOp();
    bool supported = operation && semantics.supports(operation) &&
                     (isRoot || value.hasOneUse() ||
                      semantics.allowsSharedImport(operation));
    PersistentGraphRecord graph;
    graph.resultType = printType(value.getType());

    if (!supported) {
      graph.operationName = isPrivateAllocationBoundary(value)
                                ? "ags.private_buffer"
                                : "ags.input";
      Builder builder(value.getContext());
      graph.attributes = printAttributes(builder.getDictionaryAttr(
          {builder.getNamedAttr("ags.symbol_id",
                                builder.getI64IntegerAttr(nextInput++))}));
      graph.operatorSize = 0;
      result.boundaryInputs.push_back(value);
    } else {
      graph.operationName = operation->getName().getStringRef().str();
      unsigned operatorSize = graph.operationName == "arith.constant" ? 0 : 1;
      for (Value operand : operation->getOperands()) {
        std::optional<std::string> operandHash = import(operand, false);
        if (!operandHash)
          return std::nullopt;
        graph.operandHashes.push_back(*operandHash);
        operatorSize += sizes.lookup(operand);
      }
      const TTIRCandidateOpSchema *schema = findSchema(operation);
      if (schema) {
        graph.effectKind = schema->effectKind;
        for (TTIRCandidateStateDomain domain :
             schema->implicitStateDomains) {
          std::optional<std::string> state;
          if (domain == TTIRCandidateStateDomain::Memory) {
            state = getStateBefore(operation);
          } else if (domain == TTIRCandidateStateDomain::Launch) {
            state = addInitialLaunch();
          } else {
            state = getBufferStateBefore(operation);
          }
          if (!state || !error.empty())
            return std::nullopt;
          graph.operandHashes.push_back(*state);
          operatorSize += graphSizes.lookup(*state);
        }
      }
      graph.operatorSize = operatorSize;

      SmallVector<NamedAttribute, 8> attributes =
          getSemanticAttributes(operation);
      if (operation->getNumResults() > 1) {
        auto selected = dyn_cast<OpResult>(value);
        if (!selected) {
          error = "multi-result semantic value is not an OpResult";
          return std::nullopt;
        }
        Builder builder(value.getContext());
        SmallVector<Attribute, 2> resultTypes;
        for (Type type : operation->getResultTypes())
          resultTypes.push_back(TypeAttr::get(type));
        attributes.push_back(builder.getNamedAttr(
            kResultIndexAttr,
            builder.getI64IntegerAttr(selected.getResultNumber())));
        attributes.push_back(builder.getNamedAttr(
            kOperationResultTypesAttr, builder.getArrayAttr(resultTypes)));
      }
      graph.attributes = printAttributes(
          Builder(value.getContext()).getDictionaryAttr(attributes));
      if (!error.empty())
        return std::nullopt;
    }

    hashes[value] = addGraph(std::move(graph));
    result.graphValues.emplace_back(hashes.lookup(value), value);
    sizes[value] = graphSizes.lookup(hashes.lookup(value));
    return hashes.lookup(value);
  };

  std::optional<std::string> rootHash;
  if (operationRoot) {
    const TTIRCandidateOpSchema *schema = findSchema(operationRoot);
    const bool isWrite = schema &&
                         schema->effectKind == TTIRCandidateEffectKind::Write &&
                         operationRoot->getNumResults() == 0;
    const bool isObservable =
        schema &&
        schema->effectKind == TTIRCandidateEffectKind::Observable &&
        ((operationRoot->getNumResults() == 0) ||
         (operationRoot->getNumResults() == 1 && operationRoot->use_empty()));
    const bool isAtomic =
        schema && schema->effectKind == TTIRCandidateEffectKind::Atomic &&
        operationRoot->getNumResults() == 1 && operationRoot->use_empty();
    const bool isBufferWrite =
        schema &&
        schema->effectKind == TTIRCandidateEffectKind::BufferWrite &&
        operationRoot->getNumResults() == 0;
    if ((!isWrite && !isObservable && !isAtomic && !isBufferWrite) ||
        !semantics.supports(operationRoot)) {
      error = "operation root is not a supported write or unused-result "
              "atomic/buffer transition";
      return std::nullopt;
    }
    std::optional<std::string> state =
        isBufferWrite ? getBufferStateBefore(operationRoot)
                      : getStateBefore(operationRoot);
    if (!state)
      return std::nullopt;
    PersistentGraphRecord effect;
    effect.operationName = operationRoot->getName().getStringRef().str();
    effect.effectKind = schema->effectKind;
    effect.resultType =
        (isWrite || (isObservable && operationRoot->getNumResults() == 0) ||
         isBufferWrite)
            ? "none"
            : printType(operationRoot->getResult(0).getType());
    unsigned operatorSize = 1;
    for (Value operand : operationRoot->getOperands()) {
      if (shouldInterrupt(error))
        return std::nullopt;
      std::optional<std::string> operandHash = import(operand, false);
      if (!operandHash)
        return std::nullopt;
      effect.operandHashes.push_back(*operandHash);
      operatorSize += sizes.lookup(operand);
    }
    effect.operandHashes.push_back(*state);
    operatorSize += graphSizes.lookup(*state);
    effect.operatorSize = operatorSize;
    effect.attributes = printAttributes(
        Builder(context).getDictionaryAttr(
            getSemanticAttributes(operationRoot)));
    if (!error.empty())
      return std::nullopt;
    std::string effectHash = addGraph(std::move(effect));
    operatorSize = graphSizes.lookup(effectHash);
    if (isWrite || (isObservable && operationRoot->getNumResults() == 0) ||
        isBufferWrite) {
      rootHash = effectHash;
    } else {
      PersistentGraphRecord projection;
      projection.operationName = "ags.memory_after";
      projection.resultType = "none";
      Builder builder(context);
      projection.attributes = printAttributes(builder.getDictionaryAttr(
          {builder.getNamedAttr(kStateDomainAttr,
                                builder.getStringAttr("memory"))}));
      projection.operatorSize = operatorSize + 1;
      projection.operandHashes.push_back(effectHash);
      rootHash = addGraph(std::move(projection));
    }
    result.stateOperations.emplace_back(*rootHash, operationRoot);
  } else {
    rootHash = import(root, true);
  }
  if (!rootHash) {
    if (error.empty())
      error = "failed to import TTIR semantic island";
    return std::nullopt;
  }
  if (includeLaunchState) {
    std::string launchState = addInitialLaunch();
    if (launchState.empty() || !error.empty())
      return std::nullopt;
    result.launchStateHash = std::move(launchState);
  } else if (initialLaunch) {
    result.launchStateHash = *initialLaunch;
  }
  result.rootGraphHash = std::move(*rootHash);
  llvm::sort(result.graphs,
             [](const auto &lhs, const auto &rhs) {
               return lhs.graphHash < rhs.graphHash;
             });
  return result;
}

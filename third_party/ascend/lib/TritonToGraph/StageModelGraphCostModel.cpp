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

#include "TritonToGraph/StageModelGraphCostModel.h"
#include "TritonToGraph/CanonicalRegion.h"

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <utility>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static bool isBoundaryTerminal(llvm::StringRef operation) {
  return operation == "ags.input" || operation == "ags.private_buffer" ||
         operation == "ags.memory" ||
         operation == "ags.partitionable_memory" ||
         operation == "ags.launch" || operation == "ags.buffer";
}

static bool isStateTerminal(llvm::StringRef operation) {
  return operation == "ags.memory" ||
         operation == "ags.partitionable_memory" ||
         operation == "ags.launch" || operation == "ags.buffer";
}

// Keep the bridge's accepted source language synchronized with the patterns
// installed by createConvertTritonToAscendPass(). The offline semantic
// universe is intentionally wider than StageModel. Preflighting here makes an
// unsupported registered OP an ordinary unknown cost instead of letting its
// verifier or dialect hook abort the complete offline search.
static bool isStageModelSourceOperation(llvm::StringRef operation) {
  if (operation == "arith.constant")
    return true;
  if (operation == "arith.cmpf" || operation == "arith.cmpi" ||
      operation == "arith.select" || operation == "arith.negf")
    return true;
  if (operation.starts_with("arith.ext") ||
      operation.starts_with("arith.trunc") ||
      operation.starts_with("arith.sitofp") ||
      operation.starts_with("arith.uitofp") ||
      operation.starts_with("arith.fptosi") ||
      operation.starts_with("arith.fptoui") ||
      operation.starts_with("arith.bitcast") ||
      operation.starts_with("arith.index_cast"))
    return true;
  if (llvm::StringSwitch<bool>(operation)
          .Cases("arith.addf", "arith.addi", "arith.andi", true)
          .Cases("arith.ori", "arith.xori", "arith.shli", true)
          .Cases("arith.shrsi", "arith.shrui", "arith.subf", true)
          .Cases("arith.subi", "arith.mulf", "arith.muli", true)
          .Cases("arith.divf", "arith.divsi", "arith.divui", true)
          .Cases("arith.remf", "arith.remsi", "arith.remui", true)
          .Cases("arith.ceildivsi", "arith.ceildivui", "arith.floordivsi",
                 true)
          .Cases("arith.maxnumf", "arith.maximumf", "arith.maxsi", true)
          .Cases("arith.maxui", "arith.minnumf", "arith.minimumf", true)
          .Cases("arith.minsi", "arith.minui", true)
          .Default(false))
    return true;
  if (llvm::StringSwitch<bool>(operation)
          .Cases("math.exp", "math.exp2", "math.log", true)
          .Cases("math.log2", "math.sqrt", "math.rsqrt", true)
          .Cases("math.tanh", "math.absf", "math.absi", true)
          .Default(false))
    return true;
  return llvm::StringSwitch<bool>(operation)
      .Cases("tt.dot", "tt.load", "tt.store", true)
      .Cases("tt.reduce", "tt.histogram", "tt.scan", true)
      .Cases("tt.broadcast", "tt.addptr", "tt.splat", true)
      .Cases("tt.make_range", "tt.expand_dims", "tt.get_program_id", true)
      .Cases("tt.get_num_programs", "tt.int_to_ptr", "tt.ptr_to_int", true)
      .Cases("tt.bitcast", "tt.trans", "tt.reshape", true)
      .Default(false);
}

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

static void appendDiagnostic(std::string &message, Diagnostic &diagnostic) {
  std::string rendered;
  llvm::raw_string_ostream stream(rendered);
  diagnostic.print(stream);
  if (!message.empty())
    message += "; ";
  message += rendered;
}

static bool collectReachable(
    llvm::StringRef graphHash,
    const llvm::StringMap<const PersistentGraphRecord *> &graphs,
    llvm::StringSet<> &active, llvm::StringSet<> &visited,
    SmallVector<const PersistentGraphRecord *, 32> &postorder,
    std::string &error) {
  if (visited.contains(graphHash))
    return true;
  auto found = graphs.find(graphHash);
  if (found == graphs.end()) {
    error = "candidate DAG references an unknown graph hash";
    return false;
  }
  if (!active.insert(graphHash).second) {
    error = "candidate DAG contains a cycle";
    return false;
  }
  for (llvm::StringRef operand : found->second->operandHashes)
    if (!collectReachable(operand, graphs, active, visited, postorder, error))
      return false;
  active.erase(graphHash);
  visited.insert(graphHash);
  postorder.push_back(found->second);
  return true;
}

} // namespace

AscendStageModelGraphCostModel::AscendStageModelGraphCostModel(
    std::string hardwareConfigPath)
    : hardwareConfigPath(std::move(hardwareConfigPath)) {
  if (this->hardwareConfigPath.empty()) {
    hardwareConfigHash = stableHash("builtin-ascend-910b");
    return;
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> config =
      llvm::MemoryBuffer::getFile(this->hardwareConfigPath);
  if (!config) {
    configurationError = "cannot read StageModel hardware config '" +
                         this->hardwareConfigPath + "': " +
                         config.getError().message();
    return;
  }
  hardwareConfigHash = stableHash(config.get()->getBuffer());
}

AscendStageModelGraphCostModel::~AscendStageModelGraphCostModel() = default;

std::string AscendStageModelGraphCostModel::getVersionHash() const {
  return stableHash((llvm::Twine(getName()) + ":" + hardwareConfigHash).str());
}

std::optional<uint64_t>
AscendStageModelGraphCostModel::estimateStageCycles(
    const PersistentEquivalenceClassStore &store,
    const PersistentGraphRecord &root, const GraphCostContext &context,
    std::string &unknownReason) const {
  if (!configurationError.empty()) {
    unknownReason = configurationError;
    return std::nullopt;
  }
  if (hardwareConfigPath.empty() && context.target != "ascend910b") {
    unknownReason = "the built-in StageModel configuration is Ascend 910B, "
                    "but the cost context target is '" +
                    context.target + "'; provide an explicit hardware config";
    return std::nullopt;
  }

  // Pareto extraction is serial and evaluates many roots from one store.
  // Reuse the dialect-loaded context while every temporary module still owns
  // and destroys its operations within this call.
  if (!materializationContext) {
    materializationContext = std::make_unique<MLIRContext>();
    materializationContext->getOrLoadDialect<arith::ArithDialect>();
    materializationContext
        ->getOrLoadDialect<bufferization::BufferizationDialect>();
    materializationContext->getOrLoadDialect<func::FuncDialect>();
    materializationContext->getOrLoadDialect<linalg::LinalgDialect>();
    materializationContext->getOrLoadDialect<math::MathDialect>();
    materializationContext->getOrLoadDialect<memref::MemRefDialect>();
    materializationContext->getOrLoadDialect<scf::SCFDialect>();
    materializationContext->getOrLoadDialect<tensor::TensorDialect>();
    materializationContext->getOrLoadDialect<triton::TritonDialect>();
    materializationContext->getOrLoadDialect<triton::gpu::TritonGPUDialect>();
    materializationContext->getOrLoadDialect<ascend::AscendModelDialect>();
  }
  MLIRContext &mlirContext = *materializationContext;

  if (indexedStore != &store) {
    indexedGraphs.clear();
    for (const PersistentGraphRecord &graph : store.graphs)
      indexedGraphs.try_emplace(graph.graphHash, &graph);
    indexedStore = &store;
  }
  llvm::StringSet<> active;
  llvm::StringSet<> visited;
  SmallVector<const PersistentGraphRecord *, 32> postorder;
  if (!collectReachable(root.graphHash, indexedGraphs, active, visited,
                        postorder,
                        unknownReason))
    return std::nullopt;

  // The offline graph context does not carry concrete bindings for dynamic
  // tensor dimensions. Besides making byte/cycle counts unknowable, asking
  // TritonGPU to verify a distributed layout against a dynamic shape reaches
  // ensureLayoutNotLargerThan(), whose contract requires positive concrete
  // extents. Tensor pointers similarly require the dedicated Triton tensor-
  // pointer decomposition pipeline; feeding one directly to generic Triton
  // operation verifiers reaches verifySameEncoding(), whose contract excludes
  // tensor-pointer types. Fail closed before constructing such IR so the
  // complete e-class consistently falls back to structural cost.
  for (const PersistentGraphRecord *graph : postorder) {
    if (!isBoundaryTerminal(graph->operationName) &&
        !isStageModelSourceOperation(graph->operationName)) {
      unknownReason = "StageModel has no conversion for source operation '" +
                      graph->operationName + "'";
      return std::nullopt;
    }
    Type type = parseType(graph->resultType, &mlirContext);
    if (!type) {
      unknownReason = "cannot parse candidate type '" + graph->resultType +
                      "'";
      return std::nullopt;
    }
    if (auto tensorType = dyn_cast<RankedTensorType>(type);
        tensorType && !tensorType.hasStaticShape()) {
      unknownReason =
          "StageModel requires concrete bindings for dynamic tensor shapes";
      return std::nullopt;
    }
    if (triton::isTensorPointerType(type)) {
      unknownReason =
          "StageModel requires tensor-pointer decomposition before costing";
      return std::nullopt;
    }
  }

  SmallVector<const PersistentGraphRecord *, 8> inputs;
  for (const PersistentGraphRecord *graph : postorder)
    if (isBoundaryTerminal(graph->operationName))
      inputs.push_back(graph);
  llvm::sort(inputs, [](const auto *lhs, const auto *rhs) {
    return lhs->graphHash < rhs->graphHash;
  });

  SmallVector<Type, 8> inputTypes;
  for (const PersistentGraphRecord *input : inputs) {
    Type type = parseType(input->resultType, &mlirContext);
    if (!type) {
      unknownReason = "cannot parse candidate input type '" +
                      input->resultType + "'";
      return std::nullopt;
    }
    if (!isa<NoneType>(type) && !isStateTerminal(input->operationName))
      inputTypes.push_back(type);
  }
  Type rootType = parseType(root.resultType, &mlirContext);
  if (!rootType) {
    unknownReason = "cannot parse candidate root type '" + root.resultType +
                    "'";
    return std::nullopt;
  }

  OpBuilder builder(&mlirContext);
  Location location = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(location);
  SmallVector<Type, 1> resultTypes;
  if (!isa<NoneType>(rootType))
    resultTypes.push_back(rootType);
  auto function = func::FuncOp::create(
      location, "ags_stage_candidate",
      builder.getFunctionType(inputTypes, resultTypes));
  module->push_back(function);
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::StringMap<Value> values;
  llvm::StringSet<> materialized;
  unsigned argumentIndex = 0;
  for (const PersistentGraphRecord *input : inputs) {
    Type type = parseType(input->resultType, &mlirContext);
    if (!type) {
      unknownReason = "cannot parse candidate input type '" +
                      input->resultType + "'";
      return std::nullopt;
    }
    if (!isa<NoneType>(type) && !isStateTerminal(input->operationName))
      values.try_emplace(input->graphHash,
                         entry->getArgument(argumentIndex++));
    materialized.insert(input->graphHash);
  }

  std::string diagnostics;
  ScopedDiagnosticHandler diagnosticHandler(
      &mlirContext, [&](Diagnostic &diagnostic) {
        appendDiagnostic(diagnostics, diagnostic);
        return success();
      });
  for (const PersistentGraphRecord *graph : postorder) {
    if (isBoundaryTerminal(graph->operationName))
      continue;
    if (graph->operationName == "ags.memory_after") {
      if (graph->operandHashes.size() != 1 ||
          !materialized.contains(graph->operandHashes.front())) {
        unknownReason =
            "memory projection references an unmaterialized effect";
        return std::nullopt;
      }
      materialized.insert(graph->graphHash);
      continue;
    }
    if (llvm::StringRef(graph->operationName).starts_with("ags.")) {
      unknownReason = "StageModel cannot materialize proof-only operation '" +
                      graph->operationName + "'";
      return std::nullopt;
    }
    Type resultType = parseType(graph->resultType, &mlirContext);
    Attribute parsedAttributes =
        parseAttribute(graph->attributes, &mlirContext);
    auto attributes = dyn_cast_or_null<DictionaryAttr>(parsedAttributes);
    if (!resultType || !attributes) {
      unknownReason = "cannot parse type or attribute dictionary for graph " +
                      graph->graphHash;
      return std::nullopt;
    }
    SmallVector<Value, 4> operands;
    for (llvm::StringRef operandHash : graph->operandHashes) {
      auto operandGraph = indexedGraphs.find(operandHash);
      if (operandGraph == indexedGraphs.end()) {
        unknownReason = "candidate DAG references an unknown operand";
        return std::nullopt;
      }
      Type operandType =
          parseType(operandGraph->second->resultType, &mlirContext);
      if (!operandType) {
        unknownReason = "cannot parse candidate operand type '" +
                        operandGraph->second->resultType + "'";
        return std::nullopt;
      }
      if (isa<NoneType>(operandType)) {
        if (!materialized.contains(operandHash)) {
          unknownReason = "candidate state operand is not materialized";
          return std::nullopt;
        }
        continue;
      }
      auto operand = values.find(operandHash);
      if (operand == values.end()) {
        unknownReason = "candidate DAG is not in topological order";
        return std::nullopt;
      }
      operands.push_back(operand->second);
    }
    OperationState state(location, graph->operationName);
    state.addOperands(operands);
    if (!isa<NoneType>(resultType))
      state.addTypes(resultType);
    ArrayAttr canonicalRegions =
        attributes.getAs<ArrayAttr>(kCanonicalRegionsAttr);
    if (attributes.get(kCanonicalRegionsAttr) && !canonicalRegions) {
      unknownReason = "candidate has a malformed canonical region recipe";
      return std::nullopt;
    }
    if (canonicalRegions)
      for (size_t index = 0; index < canonicalRegions.size(); ++index)
        state.addRegion();
    for (NamedAttribute attribute : attributes)
      if (!attribute.getName().getValue().starts_with("ags."))
        state.addAttribute(attribute.getName(), attribute.getValue());
    state.addAttribute("ags.stage_source_hash",
                       builder.getStringAttr(graph->graphHash));
    Operation *operation = builder.create(state);
    if (canonicalRegions) {
      IRRewriter regionRewriter(&mlirContext);
      std::string regionError;
      if (failed(materializeCanonicalRegions(
              operation, canonicalRegions, regionRewriter, regionError))) {
        unknownReason = "cannot materialize candidate regions: " + regionError;
        return std::nullopt;
      }
    }
    if (isa<NoneType>(resultType)) {
      if (operation->getNumResults() != 0) {
        unknownReason = "state candidate operation unexpectedly has a result";
        return std::nullopt;
      }
    } else {
      if (operation->getNumResults() != 1) {
        unknownReason = "candidate operation does not produce one result";
        return std::nullopt;
      }
      values.try_emplace(graph->graphHash, operation->getResult(0));
    }
    materialized.insert(graph->graphHash);
  }
  if (isa<NoneType>(rootType)) {
    if (!materialized.contains(root.graphHash)) {
      unknownReason = "candidate state root was not materialized";
      return std::nullopt;
    }
    builder.create<func::ReturnOp>(location);
  } else {
    auto rootValue = values.find(root.graphHash);
    if (rootValue == values.end()) {
      unknownReason = "candidate root was not materialized";
      return std::nullopt;
    }
    builder.create<func::ReturnOp>(location, rootValue->second);
  }
  if (failed(verify(*module))) {
    unknownReason = "materialized candidate failed MLIR verification";
    if (!diagnostics.empty())
      unknownReason += ": " + diagnostics;
    return std::nullopt;
  }

  PassManager conversion(&mlirContext);
  conversion.addPass(ascend::createConvertTritonToAscendPass());
  if (failed(conversion.run(*module))) {
    unknownReason = "ConvertTritonToAscend failed";
    if (!diagnostics.empty())
      unknownReason += ": " + diagnostics;
    return std::nullopt;
  }

  Operation *unmodeled = nullptr;
  module->walk([&](Operation *operation) {
    if (unmodeled || !operation->hasAttr("ags.stage_source_hash"))
      return;
    if (operation->getName().getStringRef() == "arith.constant") {
      operation->removeAttr("ags.stage_source_hash");
      return;
    }
    unmodeled = operation;
  });
  if (unmodeled) {
    unknownReason = "StageModel did not lower candidate operation '" +
                    unmodeled->getName().getStringRef().str() + "'";
    return std::nullopt;
  }

  ascend::EstimateCyclesPassOptions estimateOptions;
  estimateOptions.hardwareConfigPath = hardwareConfigPath;
  ascend::PipelineAnalysisPassOptions pipelineOptions;
  pipelineOptions.hardwareConfigPath = hardwareConfigPath;
  PassManager costPipeline(&mlirContext);
  costPipeline.addPass(ascend::createInsertDataTransfersPass());
  costPipeline.addPass(ascend::createAssignOpIDsPass());
  costPipeline.addPass(ascend::createEstimateCyclesPass(estimateOptions));
  costPipeline.addPass(ascend::createPipelineAnalysisPass(pipelineOptions));
  if (failed(costPipeline.run(*module))) {
    unknownReason = "Ascend StageModel pipeline failed";
    if (!diagnostics.empty())
      unknownReason += ": " + diagnostics;
    return std::nullopt;
  }
  auto cycles =
      (*module)->getAttrOfType<IntegerAttr>("ascend.scheduled_cycles");
  if (!cycles || cycles.getInt() < 0) {
    unknownReason = "Ascend StageModel did not produce non-negative "
                    "ascend.scheduled_cycles";
    return std::nullopt;
  }
  return static_cast<uint64_t>(cycles.getInt());
}

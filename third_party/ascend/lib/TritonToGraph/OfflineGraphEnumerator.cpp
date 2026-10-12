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

#include "TritonToGraph/OfflineGraphEnumerator.h"
#include "TritonToGraph/CanonicalRegion.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <tuple>
#include <thread>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static constexpr StringLiteral kFinalLayerPartitionScheme =
    "hierarchical-short-operand-prefix-v3";

template <typename Function>
static void parallelForWork(unsigned count, unsigned requestedJobs,
                            Function &&function) {
  const unsigned jobs = std::min(count, std::max(1u, requestedJobs));
  if (jobs <= 1 || count <= 1) {
    for (unsigned index = 0; index < count; ++index)
      function(index);
    return;
  }

  std::atomic<unsigned> next{0};
  std::vector<std::thread> workers;
  workers.reserve(jobs);
  for (unsigned worker = 0; worker < jobs; ++worker) {
    workers.emplace_back([&] {
      while (true) {
        unsigned index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count)
          return;
        function(index);
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
}

static constexpr llvm::StringLiteral kResultIndexAttr = "ags.result_index";
static constexpr llvm::StringLiteral kOperationResultTypesAttr =
    "ags.operation_result_types";
static constexpr llvm::StringLiteral kStateDomainAttr = "ags.state_domain";
static constexpr llvm::StringLiteral kModuleAttributePatchAttr =
    "ags.module_attribute_patch";
static constexpr llvm::StringLiteral kCompilerActionAttr =
    "ags.compiler_action";
static constexpr llvm::StringLiteral kProofOnlyAttr = "ags.proof_only";

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

static StringRef stringifyStateDomain(TTIRCandidateStateDomain domain) {
  switch (domain) {
  case TTIRCandidateStateDomain::Memory:
    return "memory";
  case TTIRCandidateStateDomain::Launch:
    return "launch";
  case TTIRCandidateStateDomain::Buffer:
    return "buffer";
  }
  llvm_unreachable("unknown AGS state domain");
}

static std::string printType(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static std::string makeKey(StringRef operationName, Type resultType,
                           DictionaryAttr attributes,
                           ArrayRef<unsigned> operands) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << operationName << '|' << resultType;
  if (attributes && !attributes.empty())
    stream << "|attrs=" << attributes;
  for (unsigned operand : operands)
    stream << '|' << operand;
  return storage;
}

// FNV-1a is a deterministic partition key; it is not used as graph identity.
static uint64_t getFinalLayerPartitionHash(StringRef schemaPrefix,
                                           ArrayRef<unsigned> operandPrefix) {
  uint64_t hash = UINT64_C(14695981039346656037);
  auto mix = [&](unsigned char byte) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  };
  for (unsigned char byte : schemaPrefix)
    mix(byte);
  mix(static_cast<unsigned char>(operandPrefix.size()));
  for (unsigned operand : operandPrefix)
    for (unsigned shift = 0; shift != 32; shift += 8)
      mix(static_cast<unsigned char>((operand >> shift) & 0xff));
  return hash;
}

struct FinalLayerPartitionRange {
  unsigned begin = 0;
  unsigned size = 1;
};

// Wide schemas use hierarchical fan-out. The first operand maps a prefix to a
// group of approximately sqrt(P) shards, so the other P-sqrt(P) shards prune
// before walking the second operand. The second operand then chooses one owner
// inside that group. This splits a heavy low-cardinality first operand without
// making every shard repeat the complete two-operand prefix.
static FinalLayerPartitionRange
getWideFinalLayerPartitionRange(StringRef schemaPrefix, unsigned firstOperand,
                                unsigned partitionCount) {
  unsigned fanout = 1;
  while (static_cast<uint64_t>(fanout) * fanout * 4 <= partitionCount)
    fanout *= 2;
  const unsigned groupCount = (partitionCount + fanout - 1) / fanout;
  const unsigned group = static_cast<unsigned>(
      getFinalLayerPartitionHash(schemaPrefix, {firstOperand}) % groupCount);
  const unsigned begin = group * fanout;
  return {begin, std::min(fanout, partitionCount - begin)};
}

static unsigned getFinalLayerPartition(StringRef schemaPrefix,
                                       ArrayRef<unsigned> operandPrefix,
                                       unsigned partitionCount) {
  if (operandPrefix.size() < 2)
    return static_cast<unsigned>(
        getFinalLayerPartitionHash(schemaPrefix, operandPrefix) %
        partitionCount);
  FinalLayerPartitionRange range = getWideFinalLayerPartitionRange(
      schemaPrefix, operandPrefix.front(), partitionCount);
  return range.begin + static_cast<unsigned>(
                           getFinalLayerPartitionHash(schemaPrefix,
                                                      operandPrefix) %
                           range.size);
}

// Semantic normal forms can be much larger than their graphs. Build a cheap
// process-local structural bucket directly from the shared expression DAG;
// printing the complete semantic key for millions of graphs dominated broad
// prefix replay. The proof oracle still compares every representative in a
// bucket, so a 64-bit collision cannot merge non-equivalent graphs.
static uint64_t getQuotientBucket(Type type,
                                  const TensorDenotation &denotation) {
  llvm::hash_code hash = llvm::hash_combine(
      type.getAsOpaquePointer(), denotation.shape.size(),
      denotation.layoutIdentity, denotation.resultIndices.size(),
      denotation.isMemoryStateToken, denotation.isLaunchStateToken,
      denotation.isBufferStateToken);
  for (int64_t dimension : denotation.shape)
    hash = llvm::hash_combine(hash, dimension);
  for (const std::string &symbol : denotation.shapeSymbols)
    hash = llvm::hash_combine(hash, symbol);
  auto combineExpression = [&](const TensorSemanticExpr &expression) {
    hash = llvm::hash_combine(hash, static_cast<bool>(expression),
                              expression.getStructuralHash());
  };
  for (const TensorSemanticExpr &index : denotation.resultIndices)
    combineExpression(index);
  combineExpression(denotation.value);
  combineExpression(denotation.defined);
  combineExpression(denotation.effect);
  combineExpression(denotation.memory);
  combineExpression(denotation.memoryDefined);
  combineExpression(denotation.launch);
  combineExpression(denotation.launchDefined);
  combineExpression(denotation.buffer);
  combineExpression(denotation.bufferDefined);
  return static_cast<uint64_t>(static_cast<size_t>(hash));
}

static bool structurallyEqualDenotations(const TensorDenotation &lhs,
                                         const TensorDenotation &rhs) {
  auto equalExpression = [](const TensorSemanticExpr &left,
                            const TensorSemanticExpr &right) {
    if (static_cast<bool>(left) != static_cast<bool>(right))
      return false;
    return !left || left.structurallyEquals(right);
  };
  if (lhs.shape != rhs.shape || lhs.shapeSymbols != rhs.shapeSymbols ||
      lhs.layoutIdentity != rhs.layoutIdentity ||
      lhs.resultIndices.size() != rhs.resultIndices.size() ||
      lhs.isMemoryStateToken != rhs.isMemoryStateToken ||
      lhs.isLaunchStateToken != rhs.isLaunchStateToken ||
      lhs.isBufferStateToken != rhs.isBufferStateToken)
    return false;
  for (auto [left, right] : llvm::zip(lhs.resultIndices, rhs.resultIndices))
    if (!equalExpression(left, right))
      return false;
  return equalExpression(lhs.value, rhs.value) &&
         equalExpression(lhs.defined, rhs.defined) &&
         equalExpression(lhs.effect, rhs.effect) &&
         equalExpression(lhs.memory, rhs.memory) &&
         equalExpression(lhs.memoryDefined, rhs.memoryDefined) &&
         equalExpression(lhs.launch, rhs.launch) &&
         equalExpression(lhs.launchDefined, rhs.launchDefined) &&
         equalExpression(lhs.buffer, rhs.buffer) &&
         equalExpression(lhs.bufferDefined, rhs.bufferDefined);
}

// Prefix replay needs random access to every surviving operand e-class, but a
// full TensorDenotation is intentionally convenient rather than compact: its
// inline shape, symbolic-shape and index storage is several hundred bytes even
// when most domains are empty. Millions of individually allocated copies can
// therefore exceed the production RSS budget before final-layer enumeration.
// Keep type-like metadata in a small interned table and allocate uncommon state
// domains only when present. Materialization reconstructs the public value at
// the existing semantic boundary, so the denoter and proof oracle remain
// completely generic.
struct StoredTensorDenotationExtra {
  TensorSemanticExpr effect;
  TensorSemanticExpr memory;
  TensorSemanticExpr memoryDefined;
  TensorSemanticExpr launch;
  TensorSemanticExpr launchDefined;
  TensorSemanticExpr buffer;
  TensorSemanticExpr bufferDefined;
};

struct StoredTensorDenotation {
  unsigned metadataId = 0;
  uint8_t stateFlags = 0;
  TensorSemanticExpr value;
  TensorSemanticExpr defined;
  std::unique_ptr<StoredTensorDenotationExtra> extra;
};

class OfflineDenotationStorage {
public:
  std::unique_ptr<StoredTensorDenotation>
  store(const TensorDenotation &denotation) {
    auto stored = std::make_unique<StoredTensorDenotation>();
    SmallVector<TensorSemanticExpr, 2> resultIndices;
    for (const TensorSemanticExpr &index : denotation.resultIndices)
      resultIndices.push_back(storeExpression(index));
    stored->metadataId = internMetadata(denotation, resultIndices);
    stored->value = storeExpression(denotation.value);
    stored->defined = storeExpression(denotation.defined);
    if (denotation.effect || denotation.memory || denotation.memoryDefined ||
        denotation.launch || denotation.launchDefined || denotation.buffer ||
        denotation.bufferDefined) {
      stored->extra = std::make_unique<StoredTensorDenotationExtra>();
      stored->extra->effect = storeExpression(denotation.effect);
      stored->extra->memory = storeExpression(denotation.memory);
      stored->extra->memoryDefined =
          storeExpression(denotation.memoryDefined);
      stored->extra->launch = storeExpression(denotation.launch);
      stored->extra->launchDefined =
          storeExpression(denotation.launchDefined);
      stored->extra->buffer = storeExpression(denotation.buffer);
      stored->extra->bufferDefined =
          storeExpression(denotation.bufferDefined);
    }
    if (denotation.isMemoryStateToken)
      stored->stateFlags |= memoryStateFlag;
    if (denotation.isLaunchStateToken)
      stored->stateFlags |= launchStateFlag;
    if (denotation.isBufferStateToken)
      stored->stateFlags |= bufferStateFlag;
    return stored;
  }

  TensorDenotation materialize(const StoredTensorDenotation &stored) const {
    assert(stored.metadataId < metadata.size());
    const DenotationMetadata &storedMetadata = metadata[stored.metadataId];
    TensorDenotation denotation;
    denotation.shape.append(storedMetadata.shape.begin(),
                            storedMetadata.shape.end());
    denotation.shapeSymbols.append(storedMetadata.shapeSymbols.begin(),
                                   storedMetadata.shapeSymbols.end());
    denotation.layoutIdentity = storedMetadata.layoutIdentity;
    denotation.resultIndices.append(storedMetadata.resultIndices.begin(),
                                    storedMetadata.resultIndices.end());
    denotation.value = stored.value;
    denotation.defined = stored.defined;
    if (stored.extra) {
      denotation.effect = stored.extra->effect;
      denotation.memory = stored.extra->memory;
      denotation.memoryDefined = stored.extra->memoryDefined;
      denotation.launch = stored.extra->launch;
      denotation.launchDefined = stored.extra->launchDefined;
      denotation.buffer = stored.extra->buffer;
      denotation.bufferDefined = stored.extra->bufferDefined;
    }
    denotation.isMemoryStateToken = stored.stateFlags & memoryStateFlag;
    denotation.isLaunchStateToken = stored.stateFlags & launchStateFlag;
    denotation.isBufferStateToken = stored.stateFlags & bufferStateFlag;
    return denotation;
  }

private:
  struct DenotationMetadata {
    SmallVector<int64_t, 4> shape;
    SmallVector<std::string, 4> shapeSymbols;
    std::string layoutIdentity;
    SmallVector<TensorSemanticExpr, 2> resultIndices;
  };

  static constexpr uint8_t memoryStateFlag = 1u << 0;
  static constexpr uint8_t launchStateFlag = 1u << 1;
  static constexpr uint8_t bufferStateFlag = 1u << 2;

  unsigned internMetadata(
      const TensorDenotation &denotation,
      ArrayRef<TensorSemanticExpr> resultIndices) {
    llvm::hash_code hash = llvm::hash_combine(
        denotation.shape.size(), denotation.shapeSymbols.size(),
        StringRef(denotation.layoutIdentity), resultIndices.size());
    for (int64_t dimension : denotation.shape)
      hash = llvm::hash_combine(hash, dimension);
    for (const std::string &symbol : denotation.shapeSymbols)
      hash = llvm::hash_combine(hash, StringRef(symbol));
    for (const TensorSemanticExpr &index : resultIndices)
      hash = llvm::hash_combine(hash, index.getStructuralHash());
    const uint64_t key =
        static_cast<uint64_t>(static_cast<size_t>(hash));
    if (auto found = metadataBuckets.find(key);
        found != metadataBuckets.end()) {
      for (unsigned id : found->second) {
        const DenotationMetadata &candidate = metadata[id];
        bool indicesEqual =
            candidate.resultIndices.size() == resultIndices.size();
        if (indicesEqual)
          for (auto [candidateIndex, resultIndex] :
               llvm::zip_equal(candidate.resultIndices, resultIndices))
            if (!candidateIndex.structurallyEquals(resultIndex)) {
              indicesEqual = false;
              break;
            }
        if (ArrayRef<int64_t>(candidate.shape) ==
                ArrayRef<int64_t>(denotation.shape) &&
            ArrayRef<std::string>(candidate.shapeSymbols) ==
                ArrayRef<std::string>(denotation.shapeSymbols) &&
            candidate.layoutIdentity == denotation.layoutIdentity &&
            indicesEqual)
          return id;
      }
    }
    const unsigned id = metadata.size();
    DenotationMetadata &added = metadata.emplace_back();
    added.shape.append(denotation.shape.begin(), denotation.shape.end());
    added.shapeSymbols.append(denotation.shapeSymbols.begin(),
                              denotation.shapeSymbols.end());
    added.layoutIdentity = denotation.layoutIdentity;
    added.resultIndices.append(resultIndices.begin(), resultIndices.end());
    metadataBuckets[key].push_back(id);
    return id;
  }

  TensorSemanticExpr internTreeLeaves(
      const TensorSemanticExpr &expression) {
    if (!expression ||
        storedExpressionRoots.contains(expression.getAsOpaquePointer()))
      return expression;
    if (expression.getOperands().empty())
      return internLeaf(expression);

    SmallVector<TensorSemanticExpr, 4> operands;
    bool changed = false;
    for (const TensorSemanticExpr &operand : expression.getOperands()) {
      TensorSemanticExpr canonical = internTreeLeaves(operand);
      changed |= canonical.getAsOpaquePointer() != operand.getAsOpaquePointer();
      operands.push_back(std::move(canonical));
    }
    if (!changed)
      return expression;
    return TensorSemanticExpr::operation(
        expression.getKind(), expression.getType(), operands,
        expression.getPayload());
  }

  TensorSemanticExpr storeExpression(
      const TensorSemanticExpr &expression) {
    TensorSemanticExpr stored = internTreeLeaves(expression);
    if (stored)
      storedExpressionRoots.insert(stored.getAsOpaquePointer());
    return stored;
  }

  TensorSemanticExpr internLeaf(const TensorSemanticExpr &expression) {
    if (!expression || !expression.getOperands().empty())
      return expression;
    const uint64_t key = expression.getStructuralHash();
    if (auto found = leafBuckets.find(key); found != leafBuckets.end())
      for (const TensorSemanticExpr &candidate : found->second)
        if (candidate.structurallyEquals(expression))
          return candidate;
    leafBuckets[key].push_back(expression);
    return expression;
  }

  std::vector<DenotationMetadata> metadata;
  llvm::DenseMap<uint64_t, SmallVector<unsigned, 1>> metadataBuckets;
  llvm::DenseMap<uint64_t, SmallVector<TensorSemanticExpr, 1>> leafBuckets;
  llvm::DenseSet<const void *> storedExpressionRoots;
};

static bool isTerminalOperation(StringRef operationName) {
  return operationName == "ags.input" ||
         operationName == "ags.private_buffer" ||
         operationName == "ags.memory" ||
         operationName == "ags.launch" ||
         operationName == "ags.buffer" || operationName == "arith.constant";
}

static StringRef
getStateTerminalName(TTIRCandidateStateDomain domain) {
  switch (domain) {
  case TTIRCandidateStateDomain::Memory:
    return "ags.memory";
  case TTIRCandidateStateDomain::Launch:
    return "ags.launch";
  case TTIRCandidateStateDomain::Buffer:
    return "ags.buffer";
  }
  llvm_unreachable("unknown AGS state domain");
}

static bool graphProvidesState(const OfflineTypedGraph &graph,
                               TTIRCandidateStateDomain domain) {
  if (!isa<NoneType>(graph.resultType))
    return false;
  if (graph.attributes)
    if (auto produced = dyn_cast_or_null<StringAttr>(
            graph.attributes.get(kStateDomainAttr)))
      if (produced.getValue() == stringifyStateDomain(domain))
        return true;
  if (graph.operationName == getStateTerminalName(domain))
    return true;
  // Memory writes directly produce the next MemoryState. Atomic results use
  // the declarative ags.memory_after projection instead.
  if (domain == TTIRCandidateStateDomain::Memory)
    return graph.effectKind == TTIRCandidateEffectKind::Write ||
           graph.effectKind == TTIRCandidateEffectKind::Observable ||
           graph.operationName == "ags.memory_after";
  return domain == TTIRCandidateStateDomain::Buffer &&
         graph.effectKind == TTIRCandidateEffectKind::BufferWrite;
}

static void sortAndUnique(std::vector<std::string> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

static std::string getCandidateSchemaKey(const TTIRCandidateOpSchema &schema) {
  return (llvm::Twine(schema.operationName) + "/" +
          llvm::Twine(schema.operandCount) + "#result" +
          llvm::Twine(schema.resultIndex) + "@clause" +
          llvm::Twine(schema.semanticClauseIndex))
      .str();
}

static Attribute materializeAttribute(
    Builder &builder, const GraphAttributeValue &value,
    TTIRCandidateAttributeBinding::StorageKind storageKind) {
  switch (value.kind) {
  case GraphAttributeValue::Kind::Integer:
    return storageKind == TTIRCandidateAttributeBinding::StorageKind::I32
               ? Attribute(builder.getI32IntegerAttr(value.integer))
               : Attribute(builder.getI64IntegerAttr(value.integer));
  case GraphAttributeValue::Kind::Boolean:
    return builder.getBoolAttr(value.boolean);
  case GraphAttributeValue::Kind::IntegerArray: {
    if (storageKind ==
        TTIRCandidateAttributeBinding::StorageKind::I64Array) {
      SmallVector<int64_t, 4> extended(value.integerArray.begin(),
                                       value.integerArray.end());
      return builder.getDenseI64ArrayAttr(extended);
    }
    return builder.getDenseI32ArrayAttr(value.integerArray);
  }
  }
  llvm_unreachable("unknown graph attribute value kind");
}

} // namespace

uint64_t AbstractSemanticSignature::getBucketHash() const {
  llvm::hash_code hash = llvm::hash_combine(
      resultType, shape.size(), layoutIdentity, indexHash, valueHash,
      definednessHash, effectHash, memoryHash, launchHash, bufferHash,
      stateFlags, static_cast<uint8_t>(precision));
  for (int64_t dimension : shape)
    hash = llvm::hash_combine(hash, dimension);
  for (const std::string &symbol : shapeSymbols)
    hash = llvm::hash_combine(hash, symbol);
  return static_cast<uint64_t>(static_cast<size_t>(hash));
}

bool AbstractSemanticSignature::structurallyEquals(
    const AbstractSemanticSignature &other) const {
  return resultType == other.resultType && shape == other.shape &&
         shapeSymbols == other.shapeSymbols &&
         layoutIdentity == other.layoutIdentity && indexHash == other.indexHash &&
         valueHash == other.valueHash &&
         definednessHash == other.definednessHash &&
         effectHash == other.effectHash && memoryHash == other.memoryHash &&
         launchHash == other.launchHash && bufferHash == other.bufferHash &&
         stateFlags == other.stateFlags && precision == other.precision;
}

AbstractSemanticSignature mlir::triton::cfg::buildAbstractSemanticSignature(
    Type resultType, const TensorDenotation &denotation) {
  AbstractSemanticSignature signature;
  signature.resultType = printType(resultType);
  signature.shape = denotation.shape;
  signature.shapeSymbols = denotation.shapeSymbols;
  signature.layoutIdentity = denotation.layoutIdentity;
  auto hashExpressions = [](ArrayRef<TensorSemanticExpr> expressions) {
    llvm::hash_code hash = llvm::hash_combine(expressions.size());
    for (const TensorSemanticExpr &expression : expressions)
      hash = llvm::hash_combine(hash, static_cast<bool>(expression),
                                expression.getStructuralHash());
    return static_cast<uint64_t>(static_cast<size_t>(hash));
  };
  signature.indexHash = hashExpressions(denotation.resultIndices);
  signature.valueHash = denotation.value
                            ? denotation.value.getStructuralHash()
                            : 0;
  signature.definednessHash = denotation.defined
                                  ? denotation.defined.getStructuralHash()
                                  : 0;
  signature.effectHash = denotation.effect
                             ? denotation.effect.getStructuralHash()
                             : 0;
  signature.memoryHash = llvm::hash_combine(
      denotation.memory ? denotation.memory.getStructuralHash() : 0,
      denotation.memoryDefined
          ? denotation.memoryDefined.getStructuralHash()
          : 0);
  signature.launchHash = llvm::hash_combine(
      denotation.launch ? denotation.launch.getStructuralHash() : 0,
      denotation.launchDefined
          ? denotation.launchDefined.getStructuralHash()
          : 0);
  signature.bufferHash = llvm::hash_combine(
      denotation.buffer ? denotation.buffer.getStructuralHash() : 0,
      denotation.bufferDefined
          ? denotation.bufferDefined.getStructuralHash()
          : 0);
  if (denotation.isMemoryStateToken)
    signature.stateFlags |= 1u << 0;
  if (denotation.isLaunchStateToken)
    signature.stateFlags |= 1u << 1;
  if (denotation.isBufferStateToken)
    signature.stateFlags |= 1u << 2;
  return signature;
}

OfflineTypeDomain mlir::triton::cfg::buildOfflineTypeDomain(
    MLIRContext &context, const GraphUniverseSpec &spec) {
  OfflineTypeDomain domain;
  SmallVector<Type, 8> elementTypes;
  for (unsigned bits : spec.integerBits)
    elementTypes.push_back(IntegerType::get(&context, bits));
  for (unsigned bits : spec.floatingBits) {
    if (bits == 16) {
      elementTypes.push_back(Float16Type::get(&context));
      elementTypes.push_back(BFloat16Type::get(&context));
    } else if (bits == 32) {
      elementTypes.push_back(Float32Type::get(&context));
    } else if (bits == 64) {
      elementTypes.push_back(Float64Type::get(&context));
    } else {
      domain.unsupportedUniverseFeatures.push_back(
          (llvm::Twine("floating-point bit width ") + llvm::Twine(bits))
              .str());
    }
  }

  SmallVector<Type, 16> scalarTypes(elementTypes.begin(), elementTypes.end());
  for (Type pointeeType : elementTypes)
    for (unsigned addressSpace : spec.pointerAddressSpaces)
      scalarTypes.push_back(
          triton::PointerType::get(pointeeType, addressSpace));
  if (spec.includeIndexType)
    scalarTypes.push_back(IndexType::get(&context));

  domain.types.append(scalarTypes.begin(), scalarTypes.end());
  if (spec.bounds.allowEffects ||
      spec.hasDomain(GraphSemanticDomain::Launch) ||
      spec.hasDomain(GraphSemanticDomain::Buffer))
    domain.types.push_back(NoneType::get(&context));
  for (const std::vector<int64_t> &shape : spec.tensorShapes) {
    SmallVector<int64_t, 4> mlirShape;
    llvm::transform(shape, std::back_inserter(mlirShape), [](int64_t dim) {
      return dim == GraphUniverseSpec::kDynamicDimension
                 ? ShapedType::kDynamic
                 : dim;
    });
    SmallVector<Attribute, 4> encodings;
    if (spec.tensorEncodings.empty()) {
      encodings.push_back(Attribute{});
    } else {
      for (const std::string &spelling : spec.tensorEncodings) {
        Attribute encoding = mlir::parseAttribute(spelling, &context);
        if (!encoding) {
          domain.unsupportedUniverseFeatures.push_back(
              "cannot parse tensor encoding attribute: " + spelling);
          continue;
        }
        encodings.push_back(encoding);
      }
    }
    for (Type elementType : scalarTypes)
      for (Attribute encoding : encodings)
        domain.types.push_back(
            RankedTensorType::get(mlirShape, elementType, encoding));
  }

  for (const std::string &spelling : spec.explicitTypes) {
    Type type = mlir::parseType(spelling, &context);
    if (!type) {
      domain.unsupportedUniverseFeatures.push_back(
          "cannot parse explicit MLIR type: " + spelling);
      continue;
    }
    domain.types.push_back(type);
  }

  return domain;
}

OfflineEnumerationResult OfflineTypedGraphEnumerator::enumerate(
    const GraphUniverseSpec &spec, ArrayRef<TTIRCandidateOpSchema> schemas,
    ArrayRef<Type> resultTypes) const {
  OfflineEnumerationResult result;
  result.graphCountBySize.resize(spec.bounds.maxNodes + 1, 0);
  result.exactQuotientClassesBySize.resize(spec.bounds.maxNodes + 1, 0);
  result.finalLayerPartitionCount =
      spec.bounds.finalLayerPartitionCount;
  result.finalLayerPartitionIndex =
      spec.bounds.finalLayerPartitionIndex;
  result.finalLayerPartitionScheme = kFinalLayerPartitionScheme.str();
  GraphUniverseSpec partitionFamily = spec;
  // The fixed family name binds the partition algorithm into every shard and
  // prefix-cache identity. Artifacts produced by another sharding scheme can
  // therefore never be accepted as members of this exact union.
  partitionFamily.name =
      "ags-final-layer-partition-family:hierarchical-short-operand-prefix-v3";
  partitionFamily.bounds.finalLayerPartitionIndex = 0;
  result.finalLayerPartitionFamilyHash = partitionFamily.getStableHash();
  const bool useOperandQuotient =
      options.quotientEquivalentOperands && options.semantics &&
      options.normalizer;
  result.quotientEquivalentOperands = useOperandQuotient;
  const bool useFinalLayerSemanticQuotient =
      options.finalLayerSemanticQuotient && useOperandQuotient &&
      !options.schemaReachabilityOnly;
  const bool useSemanticFrontierCatalog =
      options.semanticFrontierCatalogOnly && options.semantics &&
      options.normalizer &&
      !options.schemaReachabilityOnly;
  result.finalLayerSemanticQuotient = useFinalLayerSemanticQuotient;
  result.semanticFrontierCatalogOnly = useSemanticFrontierCatalog;
  result.semanticCatalogCompressed = useFinalLayerSemanticQuotient ||
                                     useSemanticFrontierCatalog;
  result.schemaReachabilityOnly = options.schemaReachabilityOnly;
  result.semanticValidationJobs = std::max(1u, options.semanticValidationJobs);
  if (options.quotientEquivalentOperands && !useOperandQuotient)
    result.unsupportedUniverseFeatures.push_back(
        "operand quotient requires semantics and normalizer");
  if (options.finalLayerSemanticQuotient && !useFinalLayerSemanticQuotient)
    result.unsupportedUniverseFeatures.push_back(
        "final-layer semantic quotient requires operand quotient, semantics, "
        "normalizer, and exhaustive enumeration");
  if (options.semanticFrontierCatalogOnly && !useSemanticFrontierCatalog)
    result.unsupportedUniverseFeatures.push_back(
        "semantic frontier catalog requires semantics, normalizer, and "
        "exhaustive enumeration");

  // Multi-result operations are enumerated as complete-signature semantic
  // projections. `allowControlFlow` is a permission, not a requirement to
  // invent arbitrary CFGs: region-bearing operations are complete only when
  // their schema supplies a finite canonical region constructor. A selected
  // operation without such a schema is reported by the normal schema gap.

  llvm::StringSet<> selectedOperations;
  for (const std::string &operation : spec.operations)
    selectedOperations.insert(operation);

  SmallVector<Type, 16> uniqueTypes;
  for (Type type : resultTypes) {
    if (!type || llvm::is_contained(uniqueTypes, type))
      continue;
    uniqueTypes.push_back(type);
    result.resultTypes.push_back(printType(type));
  }
  llvm::sort(result.resultTypes);
  if (uniqueTypes.empty()) {
    result.unsupportedUniverseFeatures.push_back("empty finite type domain");
    return result;
  }
  Builder builder(uniqueTypes.front().getContext());

  SmallVector<Type, 8> boundaryInputSymbolTypes;
  if (!spec.inputSymbolTypes.empty()) {
    for (const std::string &spelling : spec.inputSymbolTypes) {
      Type type = mlir::parseType(spelling, builder.getContext());
      if (!type) {
        result.unsupportedUniverseFeatures.push_back(
            "cannot parse input symbol type: " + spelling);
        continue;
      }
      if (!llvm::is_contained(uniqueTypes, type)) {
        result.unsupportedUniverseFeatures.push_back(
            "input symbol type is outside the finite type domain: " +
            spelling);
        continue;
      }
      boundaryInputSymbolTypes.push_back(type);
    }
  }

  llvm::StringMap<SmallVector<const TTIRCandidateOpSchema *, 2>> schemaByName;
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (selectedOperations.contains(schema.operationName))
      schemaByName[schema.operationName].push_back(&schema);

  for (const TTIRCandidateOpSchema &schema : schemas) {
    if (!selectedOperations.contains(schema.operationName))
      continue;
    if (!schema.constructible)
      continue;
    if (schema.effectKind != TTIRCandidateEffectKind::Pure &&
        !spec.bounds.allowEffects)
      result.unsupportedUniverseFeatures.push_back(
          schema.operationName + " requires allow_effects=true");
    for (TTIRCandidateStateDomain domain : schema.implicitStateDomains) {
      StringRef terminal = getStateTerminalName(domain);
      if (!selectedOperations.contains(terminal))
        result.unsupportedUniverseFeatures.push_back(
            (llvm::Twine(schema.operationName) + " requires the " + terminal +
             " terminal")
                .str());
    }
  }
  sortAndUnique(result.unsupportedUniverseFeatures);

  llvm::DenseSet<const TTIRCandidateOpSchema *> enumerableSchemaSet;
  llvm::DenseMap<const TTIRCandidateOpSchema *,
                 SmallVector<DictionaryAttr, 8>>
      attributeAssignments;
  struct SchemaTypingWitness {
    DictionaryAttr attributes;
    Type resultType;
    SmallVector<Type, 8> operandTypes;
  };
  llvm::DenseMap<const TTIRCandidateOpSchema *, SchemaTypingWitness>
      schemaTypingWitnesses;
  for (const std::string &operation : spec.operations) {
    if (isTerminalOperation(operation))
      continue;
    auto found = schemaByName.find(operation);
    if (found == schemaByName.end()) {
      result.missingCandidateSchemaOperations.push_back(operation);
      continue;
    }
    for (const TTIRCandidateOpSchema *schema : found->second) {
      std::string overload =
          (llvm::Twine(operation) + "/" + llvm::Twine(schema->operandCount))
              .str();
      if (!schema->constructible) {
        if (schema->intentionalImportOnly || schema->structuralRegionOnly)
          result.nonSearchableSemanticSchemas.push_back(
              getCandidateSchemaKey(*schema));
        else
          result.missingCandidateSchemaOperations.push_back(overload);
        continue;
      }
      if (!spec.allowsOperationOperandCount(operation,
                                            schema->operandCount)) {
        result.inapplicableCandidateSchemas.push_back(
            getCandidateSchemaKey(*schema));
        continue;
      }
      unsigned semanticArity = schema->operandCount +
                               schema->implicitStateDomains.size();
      if (schema->buildRegions && !spec.bounds.allowControlFlow) {
        result.unsupportedUniverseFeatures.push_back(
            (llvm::Twine(overload) +
             " requires allow_control_flow=true")
                .str());
        continue;
      }
      if (schema->operationResultCount > spec.bounds.maxResults) {
        result.unsupportedUniverseFeatures.push_back(
            (llvm::Twine(overload) + " has " +
             llvm::Twine(schema->operationResultCount) +
             " results, exceeding max_results")
                .str());
        continue;
      }
      if (semanticArity > spec.bounds.maxOperands) {
        // The finite universe deliberately excludes schemas outside its
        // operand bound.  Variadic clauses generate such sibling arities by
        // construction; report them as inapplicable to this universe rather
        // than making an otherwise closed reachability audit fail.
        result.inapplicableCandidateSchemas.push_back(
            getCandidateSchemaKey(*schema));
        continue;
      }
      if (spec.bounds.maxNodes < 2 &&
          llvm::any_of(schema->operandEffectRequirements,
                       [](const auto &requirement) {
                         // Boundary terminals are pure. A required effectful
                         // producer contributes at least one operator below
                         // this root, so it cannot exist in a depth-1 DAG.
                         return requirement.effectKind !=
                                TTIRCandidateEffectKind::Pure;
                       })) {
        result.inapplicableCandidateSchemas.push_back(
            getCandidateSchemaKey(*schema));
        continue;
      }
      {
        bool domainsComplete = true;
        for (const TTIRCandidateAttributeBinding &binding :
             schema->attributeBindings)
          if (!spec.findAttributeDomain(binding.domainName)) {
            domainsComplete = false;
            result.missingAttributeDomainOperations.push_back(
                (llvm::Twine(overload) + ":" + binding.domainName).str());
          }
        if (!domainsComplete)
          continue;

        SmallVector<DictionaryAttr, 8> assignments;
        SmallVector<NamedAttribute, 4> selected;
        std::function<void(unsigned)> enumerateAttribute =
            [&](unsigned bindingIndex) {
              if (bindingIndex == schema->attributeBindings.size()) {
                assignments.push_back(builder.getDictionaryAttr(selected));
                return;
              }
              const TTIRCandidateAttributeBinding &binding =
                  schema->attributeBindings[bindingIndex];
              const GraphAttributeDomain *domain =
                  spec.findAttributeDomain(binding.domainName);
              assert(domain && "domain completeness checked above");
              for (const GraphAttributeValue &value : domain->values) {
                selected.push_back(builder.getNamedAttr(
                    binding.attributeName,
                    materializeAttribute(builder, value,
                                         binding.storageKind)));
                enumerateAttribute(bindingIndex + 1);
                selected.pop_back();
              }
            };
        enumerateAttribute(/*bindingIndex=*/0);
        {
          bool applicable = false;
          SmallVector<Type, 8> operandTypes(schema->operandCount, Type{});
          for (DictionaryAttr attributes : assignments) {
            for (Type resultType : uniqueTypes) {
              if (schema->canProduceResult &&
                  !schema->canProduceResult(resultType, attributes))
                continue;
              std::function<bool(unsigned)> findLegalTyping =
                  [&](unsigned assignedCount) {
                    if (assignedCount == schema->operandCount) {
                      if (schema->isWellTyped &&
                          !schema->isWellTyped(resultType, operandTypes,
                                               attributes))
                        return false;
                      if (schema->buildOperationResultTypes &&
                          schema->buildOperationResultTypes(
                                     resultType, operandTypes, attributes)
                              .empty())
                        return false;
                      return true;
                    }

                    // Hand-built test/extension schemas may only expose the
                    // legacy prefix predicate. Preserve its complete ordered
                    // search; generated `.ttsem` schemas use the arbitrary
                    // partial-assignment callback below.
                    if (!schema->mayBeWellTypedAssignment) {
                      const unsigned operandIndex = assignedCount;
                      for (Type operandType : uniqueTypes) {
                        operandTypes[operandIndex] = operandType;
                        ++result.schemaTypingAssignmentsTested;
                        bool prefixLegal =
                            !schema->mayBeWellTyped ||
                            schema->mayBeWellTyped(
                                resultType,
                                ArrayRef<Type>(operandTypes)
                                    .take_front(operandIndex + 1),
                                attributes);
                        if (prefixLegal &&
                            findLegalTyping(assignedCount + 1))
                          return true;
                        operandTypes[operandIndex] = Type{};
                      }
                      return false;
                    }

                    // Generic minimum-remaining-values search. The semantic
                    // constraint itself filters every unassigned operand's
                    // finite type domain; no operation-specific ordering is
                    // embedded in the enumerator.
                    unsigned selectedOperand = schema->operandCount;
                    SmallVector<Type, 16> selectedDomain;
                    for (unsigned operandIndex = 0;
                         operandIndex < schema->operandCount; ++operandIndex) {
                      if (operandTypes[operandIndex])
                        continue;
                      SmallVector<Type, 16> legalTypes;
                      for (Type operandType : uniqueTypes) {
                        operandTypes[operandIndex] = operandType;
                        ++result.schemaTypingAssignmentsTested;
                        bool partialLegal = schema->mayBeWellTypedAssignment(
                            resultType, operandTypes, attributes);
                        operandTypes[operandIndex] = Type{};
                        if (partialLegal)
                          legalTypes.push_back(operandType);
                      }
                      if (legalTypes.empty())
                        return false;
                      if (selectedOperand == schema->operandCount ||
                          legalTypes.size() < selectedDomain.size()) {
                        selectedOperand = operandIndex;
                        selectedDomain = std::move(legalTypes);
                      }
                    }
                    assert(selectedOperand != schema->operandCount &&
                           "an unassigned operand must remain");
                    for (Type operandType : selectedDomain) {
                      operandTypes[selectedOperand] = operandType;
                      if (findLegalTyping(assignedCount + 1))
                        return true;
                      operandTypes[selectedOperand] = Type{};
                    }
                    return false;
                  };
              if (findLegalTyping(/*assignedCount=*/0)) {
                applicable = true;
                schemaTypingWitnesses[schema] =
                    SchemaTypingWitness{attributes, resultType, operandTypes};
                break;
              }
              llvm::fill(operandTypes, Type{});
            }
            if (applicable)
              break;
          }
          if (!applicable) {
            result.inapplicableCandidateSchemas.push_back(
                getCandidateSchemaKey(*schema));
            continue;
          }
        }
        attributeAssignments[schema] = std::move(assignments);
        enumerableSchemaSet.insert(schema);
      }
    }
  }
  sortAndUnique(result.missingCandidateSchemaOperations);
  sortAndUnique(result.missingAttributeDomainOperations);
  sortAndUnique(result.inapplicableCandidateSchemas);
  sortAndUnique(result.nonSearchableSemanticSchemas);
  sortAndUnique(result.unsupportedUniverseFeatures);

  llvm::StringMap<unsigned> emittedGraphIds;
  llvm::StringMap<uint64_t> emittedByOperation;
  // Sparse compact ownership makes non-frontier entries one pointer each and
  // avoids repeating large inline TensorDenotation metadata for every
  // surviving representative.
  OfflineDenotationStorage denotationStorage;
  std::vector<std::unique_ptr<StoredTensorDenotation>> quotientDenotations;
  std::vector<bool> operandFrontier;
  using CandidateSchemaList =
      SmallVector<const TTIRCandidateOpSchema *, 1>;
  // Cached prefixes contain millions of lower-layer graphs but no local
  // candidate-schema witnesses. Keep that overwhelmingly sparse relation out
  // of every graph's inline storage.
  std::vector<std::unique_ptr<CandidateSchemaList>> schemasByGraph;
  std::vector<unsigned> pendingFinalSemanticValidation;
  llvm::DenseMap<uint64_t, SmallVector<unsigned, 2>> quotientClasses;
  // Final-layer roots cannot be expanded, so they have their own semantic
  // classes. Keeping this separate from lower-layer operand classes is
  // important: a root that is equivalent to a smaller graph is still a valid
  // root candidate and must not be dropped for that reason.
  llvm::DenseMap<uint64_t, SmallVector<unsigned, 2>> finalLayerQuotientClasses;
  llvm::DenseMap<uint64_t, SmallVector<unsigned, 2>>
      finalLayerAbstractSignatureClasses;
  std::vector<std::optional<AbstractSemanticSignature>>
      finalLayerAbstractSignatures;
  std::vector<std::unique_ptr<StoredTensorDenotation>>
      finalLayerRawDenotations;
  uint64_t cancellationPollCounter = 0;
  auto shouldStop = [&]() {
    if (!options.shouldStop)
      return false;
    // Poll often enough to make a caller's budget effective without putting a
    // clock/system call on every recursive Cartesian-product edge.
    if ((++cancellationPollCounter & 1023) != 0)
      return false;
    if (!options.shouldStop())
      return false;
    result.interrupted = true;
    return true;
  };
  auto exceedsEffectfulNodeBound =
      [&](ArrayRef<unsigned> reachableOperators,
          TTIRCandidateEffectKind rootEffect) {
        unsigned effectfulNodes =
            rootEffect == TTIRCandidateEffectKind::Pure ? 0 : 1;
        for (unsigned graphId : reachableOperators) {
          if (graphId >= result.graphs.size())
            continue;
          if (result.graphs[graphId].effectKind !=
              TTIRCandidateEffectKind::Pure)
            ++effectfulNodes;
        }
        return effectfulNodes > spec.bounds.getMaxEffectfulNodes();
      };
  auto addGraph = [&](StringRef operationName,
                      TTIRCandidateEffectKind effectKind, Type resultType,
                      DictionaryAttr attributes,
                      ArrayRef<unsigned> operands,
                      StringRef keyOperationName,
                      bool *accepted = nullptr,
                      const TTIRCandidateOpSchema *candidateSchema = nullptr) {
    if (accepted)
      *accepted = false;
    if (shouldStop())
      return false;
    std::string key =
        makeKey(keyOperationName, resultType, attributes, operands);
    if (auto existing = emittedGraphIds.find(key);
        existing != emittedGraphIds.end()) {
      ++result.exactDuplicates;
      if (candidateSchema) {
        std::unique_ptr<CandidateSchemaList> &schemas =
            schemasByGraph[existing->second];
        if (!schemas)
          schemas = std::make_unique<CandidateSchemaList>();
        if (!llvm::is_contained(*schemas, candidateSchema))
          schemas->push_back(candidateSchema);
      }
      if (accepted)
        *accepted = true;
      return true;
    }
    if (result.graphs.size() >= options.maxGraphs) {
      result.hitGraphLimit = true;
      return false;
    }
    OfflineTypedGraph graph;
    graph.id = result.graphs.size();
    graph.operationName = operationName.str();
    graph.effectKind = effectKind;
    graph.resultType = resultType;
    graph.attributes = attributes;
    graph.operands.assign(operands.begin(), operands.end());
    for (unsigned operandId : operands) {
      assert(operandId < result.graphs.size() &&
             "operand must precede its DAG user");
      for (unsigned reachable :
           result.graphs[operandId].reachableOperatorIds)
        if (!llvm::is_contained(graph.reachableOperatorIds, reachable))
          graph.reachableOperatorIds.push_back(reachable);
    }
    if (!isTerminalOperation(operationName))
      graph.reachableOperatorIds.push_back(graph.id);
    llvm::sort(graph.reachableOperatorIds);
    graph.operatorSize = graph.reachableOperatorIds.size();
    graph.canonicalKey = std::move(key);
    if (exceedsEffectfulNodeBound(
            ArrayRef<unsigned>(graph.reachableOperatorIds).drop_back(
                isTerminalOperation(operationName) ? 0 : 1),
            isTerminalOperation(operationName)
                ? TTIRCandidateEffectKind::Pure
                : effectKind))
      return true;

    std::optional<TensorDenotation> normalizedDenotation;
    std::optional<TensorDenotation> finalLayerDenotation;
    const bool deferSemanticValidation =
        useOperandQuotient && !options.schemaReachabilityOnly &&
        graph.operatorSize == spec.bounds.maxNodes &&
        (options.semanticValidationJobs > 1 ||
         options.recordFinalLayerOperandFrontier);
    const bool needsSemanticDenotation =
        (useOperandQuotient || useSemanticFrontierCatalog) &&
        !deferSemanticValidation &&
        (graph.operatorSize < spec.bounds.maxNodes ||
         useFinalLayerSemanticQuotient);
    if (needsSemanticDenotation) {
      const TTOpSemantics *definition =
          options.semantics->lookup(graph.operationName);
      SmallVector<TensorDenotation, 4> operandDenotations;
      bool denotationAvailable = definition != nullptr;
      for (unsigned operandId : graph.operands) {
        if (operandId >= quotientDenotations.size() ||
            !quotientDenotations[operandId]) {
          denotationAvailable = false;
          break;
        }
        operandDenotations.push_back(
            denotationStorage.materialize(*quotientDenotations[operandId]));
      }
      if (!denotationAvailable) {
        ++result.rejectedBySemantics;
        return true;
      }
      Builder semanticBuilder(graph.resultType.getContext());
      SmallVector<NamedAttribute, 8> semanticAttributes;
      if (graph.attributes)
        semanticAttributes.append(graph.attributes.begin(),
                                  graph.attributes.end());
      semanticAttributes.push_back(semanticBuilder.getNamedAttr(
          "ags.result_type", TypeAttr::get(graph.resultType)));
      GraphExprNode node = GraphExprNode::create(
          graph.resultType.getContext(), graph.operationName, graph.operands,
          semanticAttributes);
      FailureOr<TensorDenotation> denotation =
          definition->denote(node, operandDenotations);
      if (failed(denotation)) {
        ++result.rejectedBySemantics;
        return true;
      }
      if (graph.operatorSize < spec.bounds.maxNodes)
        normalizedDenotation =
            options.normalizer->normalize(*denotation).denotation;
      if (useFinalLayerSemanticQuotient &&
          graph.operatorSize == spec.bounds.maxNodes)
        finalLayerDenotation = *denotation;
    }

    // In semantic-frontier mode, intermediate syntactic spellings are not
    // part of the retained catalog once an earlier expansion representative
    // has already been proved equivalent.  This check happens before the
    // graph is appended to `result.graphs`, so it bounds both catalog memory
    // and the number of graph records that later prefix/RuleDB stages must
    // carry.  The mode is deliberately restricted to pure, materializable
    // nodes; compiler actions, proof-only nodes, and terminal symbols keep
    // their full records even when their value denotations coincide.
    if (useSemanticFrontierCatalog && normalizedDenotation &&
        graph.operatorSize > 0 &&
        graph.operatorSize < spec.bounds.maxNodes &&
        graph.effectKind == TTIRCandidateEffectKind::Pure &&
        (!graph.attributes ||
         (!graph.attributes.get(kCompilerActionAttr) &&
          !graph.attributes.get(kModuleAttributePatchAttr) &&
          !graph.attributes.get(kProofOnlyAttr)))) {
      const uint64_t semanticKey =
          getQuotientBucket(graph.resultType, *normalizedDenotation);
      auto representatives = quotientClasses.find(semanticKey);
      if (representatives != quotientClasses.end()) {
        SemanticEquivalenceOracle oracle(*options.normalizer);
        for (unsigned representative : representatives->second) {
          if (representative >= quotientDenotations.size() ||
              !quotientDenotations[representative])
            continue;
          TensorDenotation representativeDenotation =
              denotationStorage.materialize(
                  *quotientDenotations[representative]);
          if (!oracle.compareNormalized(*normalizedDenotation,
                                        representativeDenotation)
                   .isProved())
            continue;
          ++result.semanticFrontierCatalogPruned;
          if (accepted)
            *accepted = true;
          return true;
        }
      }
    }

    if (useFinalLayerSemanticQuotient && finalLayerDenotation &&
        graph.operatorSize == spec.bounds.maxNodes) {
      AbstractSemanticSignature signature =
          buildAbstractSemanticSignature(graph.resultType,
                                         *finalLayerDenotation);
      SmallVector<unsigned, 2> &representatives =
          finalLayerAbstractSignatureClasses[signature.getBucketHash()];
      for (unsigned representative : representatives) {
        if (representative >= finalLayerAbstractSignatures.size() ||
            !finalLayerAbstractSignatures[representative] ||
            !finalLayerAbstractSignatures[representative]
                 ->structurallyEquals(signature) ||
            representative >= finalLayerRawDenotations.size() ||
            !finalLayerRawDenotations[representative])
          continue;
        TensorDenotation representativeDenotation =
            denotationStorage.materialize(
                *finalLayerRawDenotations[representative]);
        if (!structurallyEqualDenotations(*finalLayerDenotation,
                                          representativeDenotation))
          continue;
        ++result.finalLayerAbstractSignaturePruned;
        ++result.finalLayerSemanticQuotientPruned;
        if (accepted)
          *accepted = true;
        return true;
      }
    }

    if (useFinalLayerSemanticQuotient && finalLayerDenotation &&
        graph.operatorSize == spec.bounds.maxNodes && !normalizedDenotation)
      normalizedDenotation =
          options.normalizer->normalize(*finalLayerDenotation).denotation;

    if (useFinalLayerSemanticQuotient && normalizedDenotation &&
        graph.operatorSize == spec.bounds.maxNodes) {
      const uint64_t semanticKey =
          getQuotientBucket(graph.resultType, *normalizedDenotation);
      SmallVector<unsigned, 2> &representatives =
          finalLayerQuotientClasses[semanticKey];
      SemanticEquivalenceOracle oracle(*options.normalizer);
      for (unsigned representative : representatives) {
        if (representative >= quotientDenotations.size() ||
            !quotientDenotations[representative])
          continue;
        TensorDenotation representativeDenotation = denotationStorage.materialize(
            *quotientDenotations[representative]);
        // Normalization may canonicalize distinct raw spellings into the
        // same expression DAG even when their raw abstract signatures differ.
        // Prove that cheap structural case before entering the considerably
        // more expensive semantic oracle.  This is exact: the oracle remains
        // the fallback for structurally different normalized denotations.
        if (structurallyEqualDenotations(*normalizedDenotation,
                                         representativeDenotation)) {
          ++result.finalLayerNormalizedStructuralPruned;
          ++result.finalLayerSemanticQuotientPruned;
          if (accepted)
            *accepted = true;
          return true;
        }
        SemanticProofResult proof = oracle.compareNormalized(
            *normalizedDenotation, representativeDenotation);
        if (!proof.isProved())
          continue;
        ++result.finalLayerSemanticQuotientPruned;
        // The schema has a semantically valid representative even though this
        // particular spelling was omitted from the compressed catalog.
        if (accepted)
          *accepted = true;
        return true;
      }
    }

    result.graphs.push_back(std::move(graph));
    emittedGraphIds.try_emplace(result.graphs.back().canonicalKey,
                                result.graphs.back().id);
    if (accepted)
      *accepted = true;
    ++emittedByOperation[operationName];
    operandFrontier.push_back(true);
    quotientDenotations.push_back(nullptr);
    finalLayerAbstractSignatures.push_back(std::nullopt);
    finalLayerRawDenotations.push_back(nullptr);
    schemasByGraph.push_back(nullptr);
    if (candidateSchema) {
      schemasByGraph.back() = std::make_unique<CandidateSchemaList>();
      schemasByGraph.back()->push_back(candidateSchema);
    }
    if (deferSemanticValidation)
      pendingFinalSemanticValidation.push_back(result.graphs.back().id);
    // Quotient denotations are needed only for graphs that can become an
    // operand of a later layer. Computing them for the final layer cannot
    // prune any expansion and dominates broad depth-1 universes.
    if ((useOperandQuotient || useSemanticFrontierCatalog) &&
        normalizedDenotation &&
        result.graphs.back().operatorSize < spec.bounds.maxNodes) {
      const OfflineTypedGraph &added = result.graphs.back();
      uint64_t semanticKey =
          getQuotientBucket(added.resultType, *normalizedDenotation);
      SmallVector<unsigned, 2> &representatives =
          quotientClasses[semanticKey];
      if (useOperandQuotient) {
        SemanticEquivalenceOracle oracle(*options.normalizer);
        for (unsigned representative : representatives) {
          if (!quotientDenotations[representative])
            continue;
          TensorDenotation representativeDenotation =
              denotationStorage.materialize(
                  *quotientDenotations[representative]);
          SemanticProofResult proof = oracle.compare(
              *normalizedDenotation, representativeDenotation);
          if (!proof.isProved())
            continue;
          operandFrontier[added.id] = false;
          ++result.operandQuotientPruned;
          break;
        }
      }
      if (!useOperandQuotient || operandFrontier[added.id]) {
        representatives.push_back(added.id);
        ++result.exactQuotientClassesBySize[added.operatorSize];
        quotientDenotations[added.id] =
            denotationStorage.store(*normalizedDenotation);
      }
    }
    if (useFinalLayerSemanticQuotient && normalizedDenotation &&
        result.graphs.back().operatorSize == spec.bounds.maxNodes) {
      const OfflineTypedGraph &added = result.graphs.back();
      const uint64_t semanticKey =
          getQuotientBucket(added.resultType, *normalizedDenotation);
      finalLayerQuotientClasses[semanticKey].push_back(added.id);
      quotientDenotations[added.id] =
          denotationStorage.store(*normalizedDenotation);
      ++result.exactQuotientClassesBySize[added.operatorSize];
    }
    if (useFinalLayerSemanticQuotient && finalLayerDenotation &&
        result.graphs.back().operatorSize == spec.bounds.maxNodes) {
      AbstractSemanticSignature signature =
          buildAbstractSemanticSignature(result.graphs.back().resultType,
                                         *finalLayerDenotation);
      finalLayerAbstractSignatureClasses[signature.getBucketHash()].push_back(
          result.graphs.back().id);
      finalLayerAbstractSignatures[result.graphs.back().id] =
          std::move(signature);
      finalLayerRawDenotations[result.graphs.back().id] =
          denotationStorage.store(*finalLayerDenotation);
    }
    ++result.graphCountBySize[result.graphs.back().operatorSize];
    return true;
  };

  OfflineEnumerationPrefixCache *prefixCache = options.prefixCache;
  if (prefixCache) {
    if (options.schemaReachabilityOnly) {
      result.unsupportedUniverseFeatures.push_back(
          "enumeration prefix cache is incompatible with schema-only mode");
      return result;
    }
    if (useSemanticFrontierCatalog) {
      result.unsupportedUniverseFeatures.push_back(
          "semantic frontier catalog is incompatible with prefix cache");
      return result;
    }
    if (prefixCache->maximumOperatorSize >= spec.bounds.maxNodes) {
      result.unsupportedUniverseFeatures.push_back(
          "enumeration prefix cache reaches the requested final layer");
      return result;
    }
    if (prefixCache->quotientEquivalentOperands != useOperandQuotient) {
      result.unsupportedUniverseFeatures.push_back(
          "enumeration prefix quotient mode mismatch");
      return result;
    }
    if (prefixCache->graphs.size() > options.maxGraphs) {
      result.hitGraphLimit = true;
      return result;
    }

    result.reusedPrefixGraphCount = prefixCache->graphs.size();
    result.reusedPrefixCacheHash = prefixCache->validatedStableHash.empty()
                                       ? prefixCache->getStableHash()
                                       : prefixCache->validatedStableHash;
    if (prefixCache->operandFrontier.size() != prefixCache->graphs.size()) {
      result.unsupportedUniverseFeatures.push_back(
          "enumeration prefix operand frontier size mismatch");
      return result;
    }
    if (options.consumePrefixCache)
      result.graphs = std::move(prefixCache->graphs);
    else
      result.graphs = prefixCache->graphs;
    quotientDenotations.resize(result.graphs.size());
    operandFrontier.assign(prefixCache->operandFrontier.begin(),
                           prefixCache->operandFrontier.end());
    result.reusedPrefixOperandFrontierGraphCount =
        llvm::count(operandFrontier, true);
    result.operandQuotientPruned += llvm::count(operandFrontier, false);
    schemasByGraph.resize(result.graphs.size());
    for (const OfflineTypedGraph &graph : result.graphs) {
      if (graph.id >= result.graphs.size() ||
          graph.operatorSize > prefixCache->maximumOperatorSize) {
        result.unsupportedUniverseFeatures.push_back(
            "enumeration prefix graph violates the cached layer bound");
        return result;
      }
      // A final-layer graph cannot have the same canonical key as a lower
      // layer graph: identical root/attributes/operands imply the same
      // reachable operator set and therefore the same operator size.  Keep
      // only newly emitted final-layer keys instead of duplicating every
      // cached canonical string in this shard's hash table.
      ++emittedByOperation[graph.operationName];
      ++result.graphCountBySize[graph.operatorSize];
    }

    if (useOperandQuotient) {
      SemanticEquivalenceOracle oracle(*options.normalizer);
      std::vector<std::vector<unsigned>> graphsByOperatorSize(
          prefixCache->maximumOperatorSize + 1);
      for (const OfflineTypedGraph &graph : result.graphs)
        if (operandFrontier[graph.id])
          graphsByOperatorSize[graph.operatorSize].push_back(graph.id);

      // Every non-terminal DAG has a strictly larger operator size than each
      // operand, so one size layer can be denoted in parallel after all lower
      // layers have been classified.  Merge each batch into quotient classes
      // in graph-id order afterwards; parallel evaluation therefore cannot
      // change representatives, graph order, or stable output hashes.
      // Bound transient normalized DAG ownership independently from the
      // retained quotient frontier.  Merge order remains graph-id order, so a
      // smaller batch changes neither representatives nor stable artifacts.
      // Compact retained denotations leave enough headroom for a wider batch;
      // this amortizes worker creation and allocator synchronization without
      // retaining an entire multi-million-graph layer at once.
      constexpr unsigned replayBatchSize = 32768;
      for (ArrayRef<unsigned> layer : graphsByOperatorSize) {
        for (unsigned base = 0; base < layer.size();
             base += replayBatchSize) {
          if (options.shouldStop && options.shouldStop()) {
            result.interrupted = true;
            return result;
          }
          const unsigned batchSize =
              std::min<unsigned>(replayBatchSize, layer.size() - base);
          std::vector<std::unique_ptr<TensorDenotation>> normalizedBatch(
              batchSize);
          parallelForWork(batchSize, options.semanticValidationJobs,
                          [&](unsigned batchIndex) {
            const OfflineTypedGraph &graph =
                result.graphs[layer[base + batchIndex]];
            const TTOpSemantics *definition =
                options.semantics->lookup(graph.operationName);
            if (!definition)
              return;
            SmallVector<TensorDenotation, 4> operandDenotations;
            for (unsigned operandId : graph.operands) {
              if (operandId >= quotientDenotations.size() ||
                  !quotientDenotations[operandId])
                return;
              operandDenotations.push_back(denotationStorage.materialize(
                  *quotientDenotations[operandId]));
            }
            Builder semanticBuilder(graph.resultType.getContext());
            SmallVector<NamedAttribute, 8> semanticAttributes;
            if (graph.attributes)
              semanticAttributes.append(graph.attributes.begin(),
                                        graph.attributes.end());
            semanticAttributes.push_back(semanticBuilder.getNamedAttr(
                "ags.result_type", TypeAttr::get(graph.resultType)));
            GraphExprNode node = GraphExprNode::create(
                graph.resultType.getContext(), graph.operationName,
                graph.operands, semanticAttributes);
            FailureOr<TensorDenotation> denotation =
                definition->denote(node, operandDenotations);
            if (failed(denotation))
              return;
            normalizedBatch[batchIndex] = std::make_unique<TensorDenotation>(
                options.normalizer->normalize(*denotation).denotation);
          });

          for (unsigned batchIndex = 0; batchIndex < batchSize;
               ++batchIndex) {
            const OfflineTypedGraph &graph =
                result.graphs[layer[base + batchIndex]];
            if (!normalizedBatch[batchIndex]) {
              result.unsupportedUniverseFeatures.push_back(
                  "cached prefix semantic replay failed for " +
                  graph.operationName);
              return result;
            }
            TensorDenotation &normalizedDenotation =
                *normalizedBatch[batchIndex];
            uint64_t semanticKey =
                getQuotientBucket(graph.resultType, normalizedDenotation);
            SmallVector<unsigned, 2> &representatives =
                quotientClasses[semanticKey];
            bool duplicatesCachedRepresentative = false;
            for (unsigned representative : representatives) {
              TensorDenotation representativeDenotation =
                  denotationStorage.materialize(
                      *quotientDenotations[representative]);
              SemanticProofResult proof = oracle.compareNormalized(
                  normalizedDenotation, representativeDenotation);
              if (!proof.isProved())
                continue;
              // A monotonically strengthened semantic normalizer can prove
              // two representatives equivalent even though the older prefix
              // cache retained both.  Keeping that older frontier is a safe
              // over-approximation: it may enumerate redundant candidates,
              // but it cannot omit a graph.  Do not compact it here because
              // stable graph ids are required to replay the proof-progress
              // cache produced with that prefix.
              duplicatesCachedRepresentative = true;
              break;
            }
            if (!duplicatesCachedRepresentative)
              ++result.exactQuotientClassesBySize[graph.operatorSize];
            if (!duplicatesCachedRepresentative)
              representatives.push_back(graph.id);
            quotientDenotations[graph.id] =
                denotationStorage.store(normalizedDenotation);
          }
        }
      }
    }
  }

  // Distinct input ordinals are encoded in the terminal operation name used
  // by the canonical key. They remain ags.input semantically and do not add an
  // operation node to the bounded DAG.
  if (!prefixCache && selectedOperations.contains("ags.input")) {
    auto addInput = [&](Type type, unsigned input) {
        std::string terminal = ("ags.input#" + llvm::Twine(input)).str();
        DictionaryAttr attributes = builder.getDictionaryAttr(
            {builder.getNamedAttr("ags.symbol_id",
                                  builder.getI64IntegerAttr(input))});
        if (!addGraph("ags.input",
                      TTIRCandidateEffectKind::Pure, type,
                      attributes, /*operands=*/{}, terminal))
          return false;
        return true;
    };
    if (!spec.inputSymbolTypes.empty()) {
      for (auto [input, type] : llvm::enumerate(boundaryInputSymbolTypes))
        if (!addInput(type, input))
          return result;
    } else {
      for (Type type : uniqueTypes) {
        if (isa<NoneType>(type))
          continue;
        for (unsigned input = 0; input < spec.bounds.getMaxInputSymbols();
             ++input)
          if (!addInput(type, input))
            return result;
      }
    }
  }

  if (!prefixCache && selectedOperations.contains("ags.private_buffer"))
    for (Type type : uniqueTypes) {
      if (!isa<BaseMemRefType>(type))
        continue;
      for (unsigned input = 0; input < spec.bounds.maxOperands; ++input) {
        std::string terminal =
            ("ags.private_buffer#" + llvm::Twine(input)).str();
        DictionaryAttr attributes = builder.getDictionaryAttr(
            {builder.getNamedAttr("ags.symbol_id",
                                  builder.getI64IntegerAttr(input))});
        if (!addGraph("ags.private_buffer",
                      TTIRCandidateEffectKind::Pure, type, attributes,
                      /*operands=*/{}, terminal))
          return result;
      }
    }

  if (!prefixCache && selectedOperations.contains("ags.memory")) {
    Type memoryType = NoneType::get(builder.getContext());
    if (llvm::is_contained(uniqueTypes, memoryType) &&
        !addGraph("ags.memory",
                  TTIRCandidateEffectKind::Pure, memoryType,
                  builder.getDictionaryAttr({}), /*operands=*/{},
                  "ags.memory#initial"))
      return result;
  }
  for (StringRef terminal : {StringRef("ags.launch"),
                             StringRef("ags.buffer")}) {
    if (prefixCache)
      continue;
    if (!selectedOperations.contains(terminal))
      continue;
    Type stateType = NoneType::get(builder.getContext());
    if (llvm::is_contained(uniqueTypes, stateType) &&
        !addGraph(terminal,
                  TTIRCandidateEffectKind::Pure, stateType,
                  builder.getDictionaryAttr({}), /*operands=*/{}, terminal))
      return result;
  }

  if (!prefixCache && selectedOperations.contains("arith.constant"))
    for (Type type : uniqueTypes) {
      Type elementType = type;
      if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        // A tensor arith.constant owns a DenseElementsAttr, whose shape must
        // be static. Dynamic tensors remain available as boundary inputs.
        if (!tensorType.hasStaticShape())
          continue;
        elementType = tensorType.getElementType();
      }
      if (!isa<IntegerType>(elementType))
        continue;
      for (int64_t value : spec.integerConstants) {
        std::string terminal =
            ("arith.constant#" + llvm::Twine(value)).str();
        Attribute constant;
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
          auto integerType = cast<IntegerType>(tensorType.getElementType());
          constant = DenseElementsAttr::get(
              tensorType, builder.getIntegerAttr(integerType, value));
        } else {
          constant = builder.getIntegerAttr(cast<IntegerType>(type), value);
        }
        DictionaryAttr attributes = builder.getDictionaryAttr(
            {builder.getNamedAttr("value", constant)});
        if (!addGraph("arith.constant",
                      TTIRCandidateEffectKind::Pure, type,
                      attributes, /*operands=*/{}, terminal))
          return result;
      }
    }

  SmallVector<const TTIRCandidateOpSchema *, 32> enumerableSchemas;
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (enumerableSchemaSet.contains(&schema))
      enumerableSchemas.push_back(&schema);
  llvm::sort(enumerableSchemas,
             [](const TTIRCandidateOpSchema *lhs,
                const TTIRCandidateOpSchema *rhs) {
               return std::tie(lhs->operationName, lhs->operandCount,
                               lhs->resultIndex,
                               lhs->semanticClauseIndex) <
                      std::tie(rhs->operationName, rhs->operandCount,
                               rhs->resultIndex,
                               rhs->semanticClauseIndex);
             });

  llvm::DenseSet<const TTIRCandidateOpSchema *> realizedSchemas;
  if (prefixCache) {
    llvm::StringSet<> cachedSchemas;
    for (const std::string &key : prefixCache->realizedCandidateSchemas)
      cachedSchemas.insert(key);
    for (const TTIRCandidateOpSchema *schema : enumerableSchemas)
      if (cachedSchemas.contains(getCandidateSchemaKey(*schema)))
        realizedSchemas.insert(schema);
  }

  if (options.schemaReachabilityOnly) {
    // A reachability audit needs one finite, well-typed witness per schema;
    // it must not execute the exhaustive OP x operand Cartesian product.  The
    // type solver above records one legal signature.  Forward-chain those
    // signatures through the smallest currently available typed operands.
    // This is generic over schemas and state domains, so adding a new OP only
    // extends the declarative semantics file.
    for (unsigned round = 1; round <= spec.bounds.maxNodes; ++round) {
      const unsigned availableCount = result.graphs.size();
      bool madeProgress = false;
      for (const TTIRCandidateOpSchema *schema : enumerableSchemas) {
        if (realizedSchemas.contains(schema))
          continue;
        auto witnessIt = schemaTypingWitnesses.find(schema);
        if (witnessIt == schemaTypingWitnesses.end())
          continue;
        const SchemaTypingWitness &witness = witnessIt->second;
        SmallVector<unsigned, 8> operands;
        SmallVector<unsigned, 8> reachableOperators;

        auto chooseOperand =
            [&](Type requiredType,
                std::optional<TTIRCandidateEffectKind> requiredEffect,
                std::optional<TTIRCandidateStateDomain> requiredState)
            -> std::optional<unsigned> {
          std::optional<unsigned> best;
          size_t bestReachableCount = std::numeric_limits<size_t>::max();
          for (unsigned graphId = 0; graphId < availableCount; ++graphId) {
            const OfflineTypedGraph &graph = result.graphs[graphId];
            if (graph.resultType != requiredType)
              continue;
            if (requiredEffect && graph.effectKind != *requiredEffect)
              continue;
            if (requiredState && !graphProvidesState(graph, *requiredState))
              continue;
            SmallVector<unsigned, 8> combined = reachableOperators;
            for (unsigned reachable : graph.reachableOperatorIds)
              if (!llvm::is_contained(combined, reachable))
                combined.push_back(reachable);
            if (combined.size() < bestReachableCount) {
              best = graphId;
              bestReachableCount = combined.size();
            }
          }
          return best;
        };

        bool operandsAvailable = true;
        for (unsigned operandIndex = 0;
             operandIndex < witness.operandTypes.size(); ++operandIndex) {
          Type operandType = witness.operandTypes[operandIndex];
          std::optional<TTIRCandidateEffectKind> requiredEffect;
          auto required = llvm::find_if(
              schema->operandEffectRequirements, [&](const auto &requirement) {
                return requirement.operandIndex == operandIndex;
              });
          if (required != schema->operandEffectRequirements.end())
            requiredEffect = required->effectKind;
          std::optional<unsigned> selected = chooseOperand(
              operandType, requiredEffect, std::nullopt);
          if (!selected) {
            operandsAvailable = false;
            break;
          }
          operands.push_back(*selected);
          for (unsigned reachable : result.graphs[*selected].reachableOperatorIds)
            if (!llvm::is_contained(reachableOperators, reachable))
              reachableOperators.push_back(reachable);
        }
        if (!operandsAvailable)
          continue;
        Type stateType = NoneType::get(builder.getContext());
        for (TTIRCandidateStateDomain state : schema->implicitStateDomains) {
          std::optional<unsigned> selected =
              chooseOperand(stateType, std::nullopt, state);
          if (!selected) {
            operandsAvailable = false;
            break;
          }
          operands.push_back(*selected);
          for (unsigned reachable : result.graphs[*selected].reachableOperatorIds)
            if (!llvm::is_contained(reachableOperators, reachable))
              reachableOperators.push_back(reachable);
        }
        if (!operandsAvailable ||
            reachableOperators.size() + 1 > spec.bounds.maxNodes)
          continue;

        SmallVector<NamedAttribute, 8> semanticAttributes(
            witness.attributes.begin(), witness.attributes.end());
        if (schema->producedStateDomain)
          semanticAttributes.push_back(builder.getNamedAttr(
              kStateDomainAttr,
              builder.getStringAttr(
                  stringifyStateDomain(*schema->producedStateDomain))));
        if (schema->proofOnlyMaterialization)
          semanticAttributes.push_back(builder.getNamedAttr(
              kProofOnlyAttr, builder.getBoolAttr(true)));
        if (schema->buildModuleAttributePatch) {
          DictionaryAttr patch =
              schema->buildModuleAttributePatch(witness.attributes);
          if (!patch)
            continue;
          semanticAttributes.push_back(
              builder.getNamedAttr(kModuleAttributePatchAttr, patch));
        }
        if (schema->buildCompilerAction) {
          DictionaryAttr action =
              schema->buildCompilerAction(witness.attributes);
          if (action)
            semanticAttributes.push_back(
                builder.getNamedAttr(kCompilerActionAttr, action));
        }
        if (schema->operationResultCount > 1) {
          if (!schema->buildOperationResultTypes)
            continue;
          SmallVector<Type, 2> operationResultTypes =
              schema->buildOperationResultTypes(
                  witness.resultType, witness.operandTypes,
                  witness.attributes);
          if (operationResultTypes.size() != schema->operationResultCount)
            continue;
          SmallVector<Attribute, 2> encodedTypes;
          for (Type type : operationResultTypes)
            encodedTypes.push_back(TypeAttr::get(type));
          semanticAttributes.push_back(builder.getNamedAttr(
              kResultIndexAttr,
              builder.getI64IntegerAttr(schema->resultIndex)));
          semanticAttributes.push_back(builder.getNamedAttr(
              kOperationResultTypesAttr, builder.getArrayAttr(encodedTypes)));
        }
        if (schema->buildRegions) {
          ArrayAttr regions = schema->buildRegions(
              witness.resultType, witness.operandTypes, witness.attributes);
          if (!regions)
            continue;
          semanticAttributes.push_back(
              builder.getNamedAttr(kCanonicalRegionsAttr, regions));
        }
        DictionaryAttr graphAttributes =
            builder.getDictionaryAttr(semanticAttributes);
        bool accepted = false;
        if (!addGraph(schema->operationName, schema->effectKind,
                      witness.resultType, graphAttributes, operands,
                      schema->operationName, &accepted, schema))
          return result;
        if (accepted) {
          realizedSchemas.insert(schema);
          madeProgress = true;
        }
      }
      if (llvm::all_of(enumerableSchemas,
                       [&](const TTIRCandidateOpSchema *schema) {
                         return realizedSchemas.contains(schema);
                       }) ||
          !madeProgress)
        break;
    }
  } else {
  const unsigned firstEnumeratedSize =
      prefixCache ? prefixCache->maximumOperatorSize + 1 : 1;
  for (unsigned size = firstEnumeratedSize; size <= spec.bounds.maxNodes;
       ++size) {
    // Snapshot the previous layers. Every operand of a size-N root contains
    // at most N-1 distinct operators, while repeated operands may share the
    // same reachable nodes. This is a DAG bound, not a tree-size partition.
    const unsigned availableCount = result.graphs.size();
    SmallVector<unsigned, 64> availableOperands;
    availableOperands.reserve(availableCount);
    std::vector<uint8_t> isAvailableOperand(availableCount, 0);
    for (unsigned graphId = 0; graphId < availableCount; ++graphId)
      if (!useOperandQuotient || operandFrontier[graphId]) {
        availableOperands.push_back(graphId);
        isAvailableOperand[graphId] = 1;
      }
    llvm::DenseMap<Type, SmallVector<unsigned, 8>> availableOperandsByType;
    llvm::DenseMap<Type, SmallVector<unsigned, 8>>
        availableTerminalOperandsByType;
    SmallVector<Type, 16> availableOperandTypes;
    for (unsigned graphId : availableOperands) {
      Type type = result.graphs[graphId].resultType;
      auto [iterator, inserted] = availableOperandsByType.try_emplace(type);
      if (inserted)
        availableOperandTypes.push_back(type);
      iterator->second.push_back(graphId);
      if (result.graphs[graphId].reachableOperatorIds.empty())
        availableTerminalOperandsByType[type].push_back(graphId);
    }
    for (const TTIRCandidateOpSchema *schema : enumerableSchemas) {
      if (options.schemaReachabilityOnly &&
          realizedSchemas.contains(schema))
        continue;
      for (DictionaryAttr attributes : attributeAssignments.lookup(schema)) {
        if (options.schemaReachabilityOnly &&
            realizedSchemas.contains(schema))
          break;
        for (Type resultType : uniqueTypes) {
          if (options.schemaReachabilityOnly &&
              realizedSchemas.contains(schema))
            break;
          if (schema->canProduceResult &&
              !schema->canProduceResult(resultType, attributes))
            continue;
          const std::string finalLayerPartitionPrefix = makeKey(
              getCandidateSchemaKey(*schema), resultType, attributes,
              /*operands=*/{});
          const unsigned enumeratedOperandCount =
              schema->operandCount + schema->implicitStateDomains.size();
          SmallVector<unsigned, 4> operands(
              enumeratedOperandCount, std::numeric_limits<unsigned>::max());
          SmallVector<Type, 4> operandTypes(enumeratedOperandCount, Type{});
          auto hasCanonicalInputPrefix = [&]() {
            if (!useOperandQuotient ||
                !options.alphaNormalizeFinalLayerInputs ||
                spec.bounds.maxNodes != 1 || size != 1)
              return true;
            llvm::DenseMap<Type, SmallVector<uint64_t, 4>> seenSymbols;
            for (unsigned operandIndex = 0;
                 operandIndex < enumeratedOperandCount; ++operandIndex) {
              unsigned graphId = operands[operandIndex];
              if (graphId == std::numeric_limits<unsigned>::max())
                break;
              const OfflineTypedGraph &operand = result.graphs[graphId];
              if (operand.operationName != "ags.input")
                continue;
              auto symbol = operand.attributes
                                ? dyn_cast_or_null<IntegerAttr>(
                                      operand.attributes.get("ags.symbol_id"))
                                : IntegerAttr{};
              if (!symbol)
                return false;
              uint64_t ordinal = symbol.getValue().getLimitedValue();
              SmallVector<uint64_t, 4> &seen =
                  seenSymbols[operand.resultType];
              if (llvm::is_contained(seen, ordinal))
                continue;
              if (ordinal != seen.size())
                return false;
              seen.push_back(ordinal);
            }
            return true;
          };
          std::function<bool(unsigned, SmallVector<unsigned, 8>)>
              enumerateOperand =
              [&](unsigned assignedCount,
                  SmallVector<unsigned, 8> reachableOperators) {
                if (shouldStop())
                  return false;
                if (options.schemaReachabilityOnly &&
                    realizedSchemas.contains(schema))
                  return true;
                if (result.hitGraphLimit)
                  return false;
                if (assignedCount == enumeratedOperandCount) {
                  if (reachableOperators.size() + 1 != size)
                    return true;
                  ++result.attemptedCandidates;
                  if (schema->isWellTyped &&
                      !schema->isWellTyped(resultType,
                                           ArrayRef<Type>(operandTypes)
                                               .take_front(
                                                   schema->operandCount),
                                           attributes)) {
                    ++result.rejectedByType;
                    return true;
                  }
                  DictionaryAttr graphAttributes = attributes;
                  SmallVector<NamedAttribute, 8> semanticAttributes(
                      attributes.begin(), attributes.end());
                  if (schema->producedStateDomain)
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kStateDomainAttr,
                        builder.getStringAttr(stringifyStateDomain(
                            *schema->producedStateDomain))));
                  if (schema->proofOnlyMaterialization)
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kProofOnlyAttr, builder.getBoolAttr(true)));
                  if (schema->buildModuleAttributePatch) {
                    DictionaryAttr patch =
                        schema->buildModuleAttributePatch(attributes);
                    if (!patch) {
                      ++result.rejectedByType;
                      return true;
                    }
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kModuleAttributePatchAttr, patch));
                  }
                  if (schema->buildCompilerAction) {
                    DictionaryAttr action =
                        schema->buildCompilerAction(attributes);
                    if (action)
                      semanticAttributes.push_back(builder.getNamedAttr(
                          kCompilerActionAttr, action));
                  }
                  if (schema->operationResultCount > 1) {
                    if (!schema->buildOperationResultTypes) {
                      ++result.rejectedByType;
                      return true;
                    }
                    SmallVector<Type, 2> operationResultTypes =
                        schema->buildOperationResultTypes(
                            resultType,
                            ArrayRef<Type>(operandTypes).take_front(
                                schema->operandCount),
                            attributes);
                    if (operationResultTypes.size() !=
                        schema->operationResultCount) {
                      ++result.rejectedByType;
                      return true;
                    }
                    SmallVector<Attribute, 2> encodedTypes;
                    for (Type type : operationResultTypes)
                      encodedTypes.push_back(TypeAttr::get(type));
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kResultIndexAttr,
                        builder.getI64IntegerAttr(schema->resultIndex)));
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kOperationResultTypesAttr,
                        builder.getArrayAttr(encodedTypes)));
                  }
                  if (schema->buildRegions) {
                    ArrayAttr regions = schema->buildRegions(
                        resultType,
                        ArrayRef<Type>(operandTypes).take_front(
                            schema->operandCount),
                        attributes);
                    if (!regions) {
                      ++result.rejectedByType;
                      return true;
                    }
                    semanticAttributes.push_back(builder.getNamedAttr(
                        kCanonicalRegionsAttr, regions));
                  }
                  graphAttributes =
                      builder.getDictionaryAttr(semanticAttributes);
                  bool accepted = false;
                  bool continued = addGraph(schema->operationName,
                                             schema->effectKind, resultType,
                                             graphAttributes, operands,
                                             schema->operationName, &accepted,
                                             schema);
                  if (accepted)
                    realizedSchemas.insert(schema);
                  return continued;
                }
                // Choose the operand position with the smallest currently
                // legal graph domain. This is the same generic fail-first CSP
                // search used by schema preflight, now over typed DAGs rather
                // than Types. Operand identity and materialization order stay
                // unchanged because assignments are stored by semantic index.
                unsigned selectedOperand = enumeratedOperandCount;
                SmallVector<unsigned, 64> selectedGraphs;
                for (unsigned operandIndex = 0;
                     operandIndex < enumeratedOperandCount; ++operandIndex) {
                  if (operands[operandIndex] !=
                      std::numeric_limits<unsigned>::max())
                    continue;
                  if (!schema->mayBeWellTypedAssignment &&
                      operandIndex != assignedCount)
                    continue;
                  SmallVector<unsigned, 64> legalGraphs;
                  const bool stateOperand =
                      operandIndex >= schema->operandCount;
                  for (Type operandType : availableOperandTypes) {
                    operandTypes[operandIndex] = operandType;
                    bool typeLegal =
                        stateOperand ||
                        (schema->mayBeWellTypedAssignment
                             ? schema->mayBeWellTypedAssignment(
                                   resultType,
                                   ArrayRef<Type>(operandTypes)
                                       .take_front(schema->operandCount),
                                   attributes)
                             : (!schema->mayBeWellTyped ||
                                schema->mayBeWellTyped(
                                    resultType,
                                    ArrayRef<Type>(operandTypes)
                                        .take_front(operandIndex + 1),
                                    attributes)));
                    operandTypes[operandIndex] = Type{};
                    if (!typeLegal)
                      continue;
                    const auto &typedOperands =
                        availableOperandsByType.find(operandType)->second;
                    SmallVector<unsigned, 64> domainGraphs;
                    if (reachableOperators.size() + 1 == size) {
                      // The root itself consumes the last available operator
                      // node. A non-terminal operand is now legal iff its own
                      // graph id is already reachable. Because every graph's
                      // reachable set contains its root id, terminals plus
                      // those existing ids are the exact remaining domain.
                      // Sorting restores the same graph-id order as the full
                      // typed frontier, preserving deterministic enumeration.
                      auto terminal =
                          availableTerminalOperandsByType.find(operandType);
                      if (terminal != availableTerminalOperandsByType.end())
                        domainGraphs.append(terminal->second.begin(),
                                            terminal->second.end());
                      for (unsigned reachable : reachableOperators)
                        if (reachable < availableCount &&
                            isAvailableOperand[reachable] &&
                            result.graphs[reachable].resultType == operandType)
                          domainGraphs.push_back(reachable);
                      llvm::sort(domainGraphs);
                      domainGraphs.erase(
                          std::unique(domainGraphs.begin(), domainGraphs.end()),
                          domainGraphs.end());
                      ++result.saturatedReachabilityDomains;
                      result.saturatedReachabilityCandidatesSkipped +=
                          typedOperands.size() - domainGraphs.size();
                    } else {
                      domainGraphs.append(typedOperands.begin(),
                                          typedOperands.end());
                    }
                    for (unsigned graphId : domainGraphs) {
                      if (shouldStop())
                        return false;
                      const OfflineTypedGraph &operand = result.graphs[graphId];
                      if (stateOperand) {
                        TTIRCandidateStateDomain domain =
                            schema->implicitStateDomains[operandIndex -
                                                         schema->operandCount];
                        if (!graphProvidesState(operand, domain))
                          continue;
                      } else {
                        auto required = llvm::find_if(
                            schema->operandEffectRequirements,
                            [&](const auto &requirement) {
                              return requirement.operandIndex == operandIndex;
                            });
                        if (required !=
                                schema->operandEffectRequirements.end() &&
                            operand.effectKind != required->effectKind)
                          continue;
                      }
                      legalGraphs.push_back(graphId);
                    }
                  }
                  if (legalGraphs.empty())
                    return true;
                  if (selectedOperand == enumeratedOperandCount ||
                      legalGraphs.size() < selectedGraphs.size()) {
                    selectedOperand = operandIndex;
                    selectedGraphs = std::move(legalGraphs);
                  }
                }
                assert(selectedOperand != enumeratedOperandCount &&
                       "an unassigned graph operand must remain");
                for (unsigned graphId : selectedGraphs) {
                  if (options.schemaReachabilityOnly &&
                      realizedSchemas.contains(schema))
                    break;
                  const OfflineTypedGraph &operand = result.graphs[graphId];
                  operands[selectedOperand] = graphId;
                  operandTypes[selectedOperand] = operand.resultType;
                  const unsigned partitionPrefixLength =
                      std::min<unsigned>(enumeratedOperandCount,
                                         enumeratedOperandCount >= 3 ? 2 : 1);
                  bool rejectPartitionPrefix = false;
                  if (!options.schemaReachabilityOnly &&
                      size == spec.bounds.maxNodes &&
                      spec.bounds.finalLayerPartitionCount > 1 &&
                      selectedOperand < partitionPrefixLength &&
                      partitionPrefixLength > 0 &&
                      operands[0] != std::numeric_limits<unsigned>::max()) {
                    if (partitionPrefixLength == 1) {
                      rejectPartitionPrefix =
                          getFinalLayerPartition(
                              finalLayerPartitionPrefix,
                              ArrayRef<unsigned>(operands).take_front(1),
                              spec.bounds.finalLayerPartitionCount) !=
                          spec.bounds.finalLayerPartitionIndex;
                    } else {
                      FinalLayerPartitionRange range =
                          getWideFinalLayerPartitionRange(
                              finalLayerPartitionPrefix, operands[0],
                              spec.bounds.finalLayerPartitionCount);
                      const unsigned partitionIndex =
                          spec.bounds.finalLayerPartitionIndex;
                      rejectPartitionPrefix =
                          partitionIndex < range.begin ||
                          partitionIndex >= range.begin + range.size;
                      if (!rejectPartitionPrefix &&
                          operands[1] !=
                              std::numeric_limits<unsigned>::max())
                        rejectPartitionPrefix =
                            getFinalLayerPartition(
                                finalLayerPartitionPrefix,
                                ArrayRef<unsigned>(operands).take_front(2),
                                spec.bounds.finalLayerPartitionCount) !=
                            partitionIndex;
                    }
                  }
                  if (rejectPartitionPrefix) {
                    ++result.finalLayerPartitionPrefixesPruned;
                    operandTypes[selectedOperand] = Type{};
                    operands[selectedOperand] =
                        std::numeric_limits<unsigned>::max();
                    continue;
                  }
                  if (!hasCanonicalInputPrefix()) {
                    ++result.inputRenamingQuotientPruned;
                    operandTypes[selectedOperand] = Type{};
                    operands[selectedOperand] =
                        std::numeric_limits<unsigned>::max();
                    continue;
                  }
                  SmallVector<unsigned, 8> nextReachable =
                      reachableOperators;
                  for (unsigned reachable : operand.reachableOperatorIds)
                    if (!llvm::is_contained(nextReachable, reachable))
                      nextReachable.push_back(reachable);
                  if (exceedsEffectfulNodeBound(nextReachable,
                                                schema->effectKind)) {
                    operandTypes[selectedOperand] = Type{};
                    operands[selectedOperand] =
                        std::numeric_limits<unsigned>::max();
                    continue;
                  }
                  if (nextReachable.size() + 1 > size) {
                    operandTypes[selectedOperand] = Type{};
                    operands[selectedOperand] =
                        std::numeric_limits<unsigned>::max();
                    continue;
                  }
                  if (!enumerateOperand(assignedCount + 1,
                                        std::move(nextReachable)))
                    return false;
                  operandTypes[selectedOperand] = Type{};
                  operands[selectedOperand] =
                      std::numeric_limits<unsigned>::max();
                }
                return true;
              };
          if (!enumerateOperand(/*assignedCount=*/0,
                                /*reachableOperators=*/{}))
            goto finalize_enumeration;
        }
      }
    }
  }
  }

finalize_enumeration:
  if (!pendingFinalSemanticValidation.empty()) {
    result.semanticValidationGraphCount =
        pendingFinalSemanticValidation.size();
    std::vector<uint8_t> valid(pendingFinalSemanticValidation.size(), 0);
    constexpr unsigned validationBatchSize = 65536;
    const bool recordOperandFrontier =
        options.recordFinalLayerOperandFrontier && useOperandQuotient;
    std::vector<uint8_t> keep(result.graphs.size(), 1);
    SemanticEquivalenceOracle finalLayerOracle(*options.normalizer);
    for (unsigned base = 0; base < pendingFinalSemanticValidation.size();
         base += validationBatchSize) {
      if (options.shouldStop && options.shouldStop()) {
        result.interrupted = true;
        return result;
      }
      const unsigned batchSize = std::min<unsigned>(
          validationBatchSize, pendingFinalSemanticValidation.size() - base);
      const bool processFinalLayerSemanticQuotient =
          useFinalLayerSemanticQuotient;
      // In the deferred parallel path every candidate is normalized before
      // merge.  A raw denotation would duplicate that retained semantic
      // state without adding a proof: normalized structural equality is
      // already exact, and the oracle handles the remaining cases.  Keep the
      // raw abstract-signature path for serial/small callers where it is a
      // useful cheap prefilter and diagnostic metric.
      const bool retainRawFinalSignature =
          processFinalLayerSemanticQuotient &&
          options.semanticValidationJobs <= 1;
      std::vector<std::unique_ptr<TensorDenotation>> rawBatch;
      std::vector<std::unique_ptr<TensorDenotation>> normalizedBatch;
      if (retainRawFinalSignature)
        rawBatch.resize(batchSize);
      if (processFinalLayerSemanticQuotient || recordOperandFrontier)
        normalizedBatch.resize(batchSize);
      parallelForWork(batchSize, options.semanticValidationJobs,
                      [&](unsigned batchIndex) {
        const unsigned index = base + batchIndex;
        const OfflineTypedGraph &graph =
            result.graphs[pendingFinalSemanticValidation[index]];
        const TTOpSemantics *definition =
            options.semantics->lookup(graph.operationName);
        if (!definition)
          return;
        SmallVector<TensorDenotation, 4> operandDenotations;
        for (unsigned operandId : graph.operands) {
          if (operandId >= quotientDenotations.size() ||
              !quotientDenotations[operandId])
            return;
          operandDenotations.push_back(
              denotationStorage.materialize(*quotientDenotations[operandId]));
        }
        Builder semanticBuilder(graph.resultType.getContext());
        SmallVector<NamedAttribute, 8> semanticAttributes;
        if (graph.attributes)
          semanticAttributes.append(graph.attributes.begin(),
                                    graph.attributes.end());
        semanticAttributes.push_back(semanticBuilder.getNamedAttr(
            "ags.result_type", TypeAttr::get(graph.resultType)));
        GraphExprNode node = GraphExprNode::create(
            graph.resultType.getContext(), graph.operationName,
            graph.operands, semanticAttributes);
        FailureOr<TensorDenotation> denotation =
            definition->denote(node, operandDenotations);
        if (failed(denotation))
          return;
        valid[index] = 1;
        if (retainRawFinalSignature)
          rawBatch[batchIndex] =
              std::make_unique<TensorDenotation>(*denotation);
        if (processFinalLayerSemanticQuotient || recordOperandFrontier)
          normalizedBatch[batchIndex] = std::make_unique<TensorDenotation>(
              options.normalizer->normalize(*denotation).denotation);
      });
      for (unsigned batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        if (!valid[base + batchIndex])
          continue;
        const unsigned graphId =
            pendingFinalSemanticValidation[base + batchIndex];
        const OfflineTypedGraph &graph = result.graphs[graphId];

        // Final-layer semantic quotienting used to run inline in addGraph,
        // which made every raw denotation and oracle comparison serial even
        // when semanticValidationJobs was large.  Denote the batch in
        // parallel, then merge in graph-id order so representative selection
        // and all persisted hashes remain deterministic.
        if (processFinalLayerSemanticQuotient) {
          if (!normalizedBatch[batchIndex] ||
              (retainRawFinalSignature && !rawBatch[batchIndex])) {
            valid[base + batchIndex] = 0;
            continue;
          }
          bool pruned = false;
          std::optional<AbstractSemanticSignature> signature;
          if (retainRawFinalSignature) {
            TensorDenotation &rawDenotation = *rawBatch[batchIndex];
            signature = buildAbstractSemanticSignature(graph.resultType,
                                                       rawDenotation);
            SmallVector<unsigned, 2> &abstractRepresentatives =
                finalLayerAbstractSignatureClasses[signature->getBucketHash()];
            for (unsigned representative : abstractRepresentatives) {
              if (representative >= finalLayerAbstractSignatures.size() ||
                  !finalLayerAbstractSignatures[representative] ||
                  !finalLayerAbstractSignatures[representative]
                       ->structurallyEquals(*signature) ||
                  representative >= finalLayerRawDenotations.size() ||
                  !finalLayerRawDenotations[representative])
                continue;
              TensorDenotation representativeDenotation =
                  denotationStorage.materialize(
                      *finalLayerRawDenotations[representative]);
              if (!structurallyEqualDenotations(rawDenotation,
                                                representativeDenotation))
                continue;
              ++result.finalLayerAbstractSignaturePruned;
              ++result.finalLayerSemanticQuotientPruned;
              pruned = true;
              break;
            }
          }
          if (!pruned) {
            TensorDenotation &normalizedDenotation =
                *normalizedBatch[batchIndex];
            const uint64_t semanticKey =
                getQuotientBucket(graph.resultType, normalizedDenotation);
            SmallVector<unsigned, 2> &representatives =
                finalLayerQuotientClasses[semanticKey];
            for (unsigned representative : representatives) {
              if (representative >= quotientDenotations.size() ||
                  !quotientDenotations[representative])
                continue;
              TensorDenotation representativeDenotation =
                  denotationStorage.materialize(
                      *quotientDenotations[representative]);
              if (structurallyEqualDenotations(normalizedDenotation,
                                                representativeDenotation)) {
                ++result.finalLayerNormalizedStructuralPruned;
                ++result.finalLayerSemanticQuotientPruned;
                pruned = true;
                break;
              }
              SemanticProofResult proof = finalLayerOracle.compareNormalized(
                  normalizedDenotation, representativeDenotation);
              if (!proof.isProved())
                continue;
              ++result.finalLayerSemanticQuotientPruned;
              pruned = true;
              break;
            }
            if (!pruned) {
              representatives.push_back(graphId);
              ++result.exactQuotientClassesBySize[graph.operatorSize];
              quotientDenotations[graphId] =
                  denotationStorage.store(normalizedDenotation);
              if (signature) {
                finalLayerAbstractSignatureClasses[signature->getBucketHash()]
                    .push_back(graphId);
                finalLayerAbstractSignatures[graphId] = std::move(*signature);
                finalLayerRawDenotations[graphId] =
                    denotationStorage.store(*rawBatch[batchIndex]);
              }
            }
          }
          if (pruned)
            keep[graphId] = 0;
          continue;
        }

        if (!recordOperandFrontier || !normalizedBatch[batchIndex])
          continue;
        // Prefix generation already denotes every graph in this layer.  Reuse
        // that work to persist only the proof-certified operand frontier; later
        // shards re-denote those representatives instead of replaying graphs
        // that were proved redundant here.  Ordered merging preserves the exact
        // representative choice of the ordinary serial quotient path.
        TensorDenotation &normalizedDenotation =
            *normalizedBatch[batchIndex];
        const uint64_t semanticKey =
            getQuotientBucket(graph.resultType, normalizedDenotation);
        SmallVector<unsigned, 2> &representatives =
            quotientClasses[semanticKey];
        for (unsigned representative : representatives) {
          TensorDenotation representativeDenotation =
              denotationStorage.materialize(
                  *quotientDenotations[representative]);
          SemanticProofResult proof = finalLayerOracle.compareNormalized(
              normalizedDenotation, representativeDenotation);
          if (!proof.isProved())
            continue;
          operandFrontier[graphId] = false;
          ++result.operandQuotientPruned;
          break;
        }
        if (!operandFrontier[graphId])
          continue;
        representatives.push_back(graphId);
        ++result.exactQuotientClassesBySize[graph.operatorSize];
        quotientDenotations[graphId] =
            denotationStorage.store(normalizedDenotation);
      }
    }

    uint64_t rejectedPending = 0;
    uint64_t prunedPending = 0;
    llvm::DenseSet<const TTIRCandidateOpSchema *> validPendingSchemas;
    for (auto [index, graphId] :
         llvm::enumerate(pendingFinalSemanticValidation)) {
      if (valid[index]) {
        if (schemasByGraph[graphId])
          for (const TTIRCandidateOpSchema *schema :
               *schemasByGraph[graphId])
            validPendingSchemas.insert(schema);
        if (!keep[graphId])
          ++prunedPending;
        continue;
      }
      keep[graphId] = 0;
      ++result.rejectedBySemantics;
      ++rejectedPending;
    }

    std::vector<OfflineTypedGraph> retainedGraphs;
    std::vector<std::unique_ptr<StoredTensorDenotation>>
        retainedDenotations;
    std::vector<bool> retainedFrontier;
    std::vector<std::unique_ptr<CandidateSchemaList>> retainedSchemas;
    retainedGraphs.reserve(result.graphs.size() - rejectedPending -
                           prunedPending);
    retainedDenotations.reserve(retainedGraphs.capacity());
    retainedFrontier.reserve(retainedGraphs.capacity());
    retainedSchemas.reserve(retainedGraphs.capacity());
    for (unsigned oldId = 0; oldId < result.graphs.size(); ++oldId) {
      if (!keep[oldId])
        continue;
      OfflineTypedGraph graph = std::move(result.graphs[oldId]);
      const unsigned newId = retainedGraphs.size();
      for (unsigned &reachable : graph.reachableOperatorIds)
        if (reachable == oldId)
          reachable = newId;
      graph.id = newId;
      retainedGraphs.push_back(std::move(graph));
      retainedDenotations.push_back(std::move(quotientDenotations[oldId]));
      retainedFrontier.push_back(operandFrontier[oldId]);
      retainedSchemas.push_back(std::move(schemasByGraph[oldId]));
    }
    result.graphs = std::move(retainedGraphs);
    quotientDenotations = std::move(retainedDenotations);
    operandFrontier = std::move(retainedFrontier);
    schemasByGraph = std::move(retainedSchemas);

    std::fill(result.graphCountBySize.begin(), result.graphCountBySize.end(),
              0);
    emittedByOperation.clear();
    realizedSchemas.clear();
    for (const TTIRCandidateOpSchema *schema : validPendingSchemas)
      realizedSchemas.insert(schema);
    for (const OfflineTypedGraph &graph : result.graphs) {
      ++result.graphCountBySize[graph.operatorSize];
      ++emittedByOperation[graph.operationName];
      if (schemasByGraph[graph.id])
        for (const TTIRCandidateOpSchema *schema :
             *schemasByGraph[graph.id])
          realizedSchemas.insert(schema);
    }
    if (prefixCache) {
      llvm::StringSet<> cachedSchemas;
      for (const std::string &key : prefixCache->realizedCandidateSchemas)
        cachedSchemas.insert(key);
      for (const TTIRCandidateOpSchema *schema : enumerableSchemas)
        if (cachedSchemas.contains(getCandidateSchemaKey(*schema)))
          realizedSchemas.insert(schema);
    }
  }

  for (const std::string &operation : spec.operations) {
    uint64_t count = emittedByOperation.lookup(operation);
    result.graphCountByOperation.emplace_back(operation, count);
    bool hasApplicableSchema = false;
    bool hasRealizedSchema = false;
    if (auto found = schemaByName.find(operation); found != schemaByName.end())
      hasApplicableSchema = llvm::any_of(
          found->second, [&](const TTIRCandidateOpSchema *schema) {
            return enumerableSchemaSet.contains(schema);
          });
    if (auto found = schemaByName.find(operation); found != schemaByName.end())
      hasRealizedSchema = llvm::any_of(
          found->second, [&](const TTIRCandidateOpSchema *schema) {
            return realizedSchemas.contains(schema);
          });
    if (count == 0 &&
        (isTerminalOperation(operation) || hasApplicableSchema) &&
        !hasRealizedSchema)
      result.unrealizedCandidateSchemaOperations.push_back(operation);
  }
  for (const TTIRCandidateOpSchema *schema : enumerableSchemas) {
    uint64_t count = realizedSchemas.contains(schema) ? 1 : 0;
    std::string key = getCandidateSchemaKey(*schema);
    result.graphCountBySchema.emplace_back(key, count);
    if (count == 0)
      result.unrealizedCandidateSchemas.push_back(std::move(key));
  }
  if (prefixCache)
    result.realizedPrefixCandidateSchemas =
        prefixCache->realizedCandidateSchemas;
  for (const OfflineTypedGraph &graph : result.graphs) {
    if (graph.operatorSize >= spec.bounds.maxNodes)
      continue;
    if (schemasByGraph[graph.id])
      for (const TTIRCandidateOpSchema *schema :
           *schemasByGraph[graph.id])
        result.realizedPrefixCandidateSchemas.push_back(
            getCandidateSchemaKey(*schema));
  }
  sortAndUnique(result.realizedPrefixCandidateSchemas);
  std::vector<StringRef> completePrefixKeys;
  completePrefixKeys.reserve(result.graphs.size());
  for (const OfflineTypedGraph &graph : result.graphs)
    if (graph.operatorSize < spec.bounds.maxNodes)
      completePrefixKeys.push_back(graph.canonicalKey);
  llvm::sort(completePrefixKeys);
  std::string completePrefix;
  llvm::raw_string_ostream prefixStream(completePrefix);
  prefixStream << "ags-complete-prefix-v1\n";
  for (StringRef key : completePrefixKeys)
    prefixStream << key.size() << ':' << key << '\n';
  prefixStream.flush();
  result.completePrefixGraphCount = completePrefixKeys.size();
  result.completePrefixHash = stableHash(completePrefix);
  result.operandFrontier.reserve(operandFrontier.size());
  for (bool retained : operandFrontier)
    result.operandFrontier.push_back(retained ? 1 : 0);
  return result;
}

std::string
OfflineEnumerationResult::toJSON(StringRef universeHash) const {
  llvm::json::Array layers;
  const bool frontierReportComplete = operandFrontier.size() == graphs.size();
  for (auto [size, count] : llvm::enumerate(graphCountBySize)) {
    uint64_t frontierRepresentatives = 0;
    if (frontierReportComplete) {
      for (const OfflineTypedGraph &graph : graphs)
        if (graph.operatorSize == size && graph.id < operandFrontier.size() &&
            operandFrontier[graph.id])
          ++frontierRepresentatives;
    }
    const uint64_t frontierPruned =
        frontierReportComplete && count >= frontierRepresentatives
            ? count - frontierRepresentatives
            : 0;
    const uint64_t exactQuotientClasses =
        size < exactQuotientClassesBySize.size()
            ? exactQuotientClassesBySize[size]
            : 0;
    llvm::json::Object layer;
    layer["operator_size"] = static_cast<int64_t>(size);
    layer["graph_count"] = count;
    // Keep the original graph_count key for existing consumers.  These
    // explicit names make the Phase 0/1 frontier accounting auditable without
    // changing enumeration or quotient behavior.
    layer["catalog_graphs"] = count;
    layer["frontier_representatives"] =
        static_cast<int64_t>(frontierRepresentatives);
    layer["exact_quotient_classes"] =
        static_cast<int64_t>(exactQuotientClasses);
    layer["frontier_pruned"] = static_cast<int64_t>(frontierPruned);
    layer["retained_for_rule_output"] = count;
    layers.push_back(std::move(layer));
  }
  llvm::json::Array types;
  for (const std::string &type : resultTypes)
    types.push_back(type);
  llvm::json::Array missingSchemas;
  for (const std::string &operation : missingCandidateSchemaOperations)
    missingSchemas.push_back(operation);
  llvm::json::Array missingAttributes;
  for (const std::string &operation : missingAttributeDomainOperations)
    missingAttributes.push_back(operation);
  llvm::json::Array unrealizedSchemas;
  for (const std::string &operation : unrealizedCandidateSchemaOperations)
    unrealizedSchemas.push_back(operation);
  llvm::json::Array unrealizedSchemaOverloads;
  for (const std::string &schema : unrealizedCandidateSchemas)
    unrealizedSchemaOverloads.push_back(schema);
  llvm::json::Array inapplicableSchemaOverloads;
  for (const std::string &schema : inapplicableCandidateSchemas)
    inapplicableSchemaOverloads.push_back(schema);
  llvm::json::Array nonSearchableSchemas;
  for (const std::string &schema : nonSearchableSemanticSchemas)
    nonSearchableSchemas.push_back(schema);
  llvm::json::Object operationCounts;
  for (const auto &[operation, count] : graphCountByOperation)
    operationCounts[operation] = static_cast<int64_t>(count);
  llvm::json::Object schemaCounts;
  for (const auto &[schema, count] : graphCountBySchema)
    schemaCounts[schema] = static_cast<int64_t>(count);
  llvm::json::Array unsupportedFeatures;
  for (const std::string &feature : unsupportedUniverseFeatures)
    unsupportedFeatures.push_back(feature);

  llvm::json::Object report;
  report["universe_hash"] = universeHash;
  report["complete"] = isComplete();
  report["schema_reachability_only"] = schemaReachabilityOnly;
  report["schema_reachability_complete"] =
      isSchemaReachabilityComplete();
  report["graph_count"] = static_cast<int64_t>(graphs.size());
  report["attempted_candidates"] =
      static_cast<int64_t>(attemptedCandidates);
  report["rejected_by_type"] = static_cast<int64_t>(rejectedByType);
  report["rejected_by_semantics"] =
      static_cast<int64_t>(rejectedBySemantics);
  report["semantic_validation_graph_count"] =
      static_cast<int64_t>(semanticValidationGraphCount);
  report["semantic_validation_jobs"] =
      static_cast<int64_t>(semanticValidationJobs);
  report["schema_typing_assignments_tested"] =
      static_cast<int64_t>(schemaTypingAssignmentsTested);
  report["exact_duplicates"] = static_cast<int64_t>(exactDuplicates);
  report["quotient_equivalent_operands"] = quotientEquivalentOperands;
  report["final_layer_semantic_quotient"] = finalLayerSemanticQuotient;
  report["semantic_frontier_catalog_only"] = semanticFrontierCatalogOnly;
  report["semantic_catalog_compressed"] = semanticCatalogCompressed;
  report["semantic_coverage_complete"] = isSemanticCoverageComplete();
  report["operand_quotient_pruned"] =
      static_cast<int64_t>(operandQuotientPruned);
  report["final_layer_semantic_quotient_pruned"] =
      static_cast<int64_t>(finalLayerSemanticQuotientPruned);
  report["final_layer_abstract_signature_pruned"] =
      static_cast<int64_t>(finalLayerAbstractSignaturePruned);
  report["final_layer_normalized_structural_pruned"] =
      static_cast<int64_t>(finalLayerNormalizedStructuralPruned);
  report["semantic_frontier_catalog_pruned"] =
      static_cast<int64_t>(semanticFrontierCatalogPruned);
  report["saturated_reachability_domains"] =
      static_cast<int64_t>(saturatedReachabilityDomains);
  report["saturated_reachability_candidates_skipped"] =
      static_cast<int64_t>(saturatedReachabilityCandidatesSkipped);
  report["input_renaming_quotient_pruned"] =
      static_cast<int64_t>(inputRenamingQuotientPruned);
  report["final_layer_partition_prefixes_pruned"] =
      static_cast<int64_t>(finalLayerPartitionPrefixesPruned);
  report["final_layer_partition_count"] =
      static_cast<int64_t>(finalLayerPartitionCount);
  report["final_layer_partition_index"] =
      static_cast<int64_t>(finalLayerPartitionIndex);
  report["final_layer_partition_scheme"] = finalLayerPartitionScheme;
  report["final_layer_partition_family_hash"] =
      finalLayerPartitionFamilyHash;
  report["complete_prefix_graph_count"] =
      static_cast<int64_t>(completePrefixGraphCount);
  report["complete_prefix_hash"] = completePrefixHash;
  report["reused_prefix_graph_count"] =
      static_cast<int64_t>(reusedPrefixGraphCount);
  report["reused_prefix_operand_frontier_graph_count"] =
      static_cast<int64_t>(reusedPrefixOperandFrontierGraphCount);
  report["operand_frontier_graph_count"] = static_cast<int64_t>(
      llvm::count(operandFrontier, static_cast<uint8_t>(1)));
  report["frontier_report_complete"] = frontierReportComplete;
  report["reused_prefix_cache_hash"] = reusedPrefixCacheHash;
  report["hit_graph_limit"] = hitGraphLimit;
  report["interrupted"] = interrupted;
  report["result_types"] = std::move(types);
  report["layers"] = std::move(layers);
  report["missing_candidate_schema_operations"] =
      std::move(missingSchemas);
  report["missing_attribute_domain_operations"] =
      std::move(missingAttributes);
  report["unrealized_candidate_schema_operations"] =
      std::move(unrealizedSchemas);
  report["unrealized_candidate_schemas"] =
      std::move(unrealizedSchemaOverloads);
  report["inapplicable_candidate_schemas"] =
      std::move(inapplicableSchemaOverloads);
  report["non_searchable_semantic_schemas"] =
      std::move(nonSearchableSchemas);
  report["graph_count_by_operation"] = std::move(operationCounts);
  report["graph_count_by_schema"] = std::move(schemaCounts);
  report["unsupported_universe_features"] =
      std::move(unsupportedFeatures);

  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(report)));
  return storage;
}

static std::string getPrefixCacheStableHash(
    unsigned schemaVersion, StringRef partitionFamilyHash,
    StringRef semanticsHash, unsigned maximumOperatorSize,
    bool quotientEquivalentOperands, ArrayRef<OfflineTypedGraph> graphs,
    ArrayRef<uint8_t> operandFrontier,
    ArrayRef<std::string> realizedCandidateSchemas) {
  assert(graphs.size() == operandFrontier.size() &&
         "prefix graph/frontier sizes must match");
  uint64_t hash = UINT64_C(14695981039346656037);
  auto update = [&](StringRef value) {
    for (unsigned char byte : value) {
      hash ^= byte;
      hash *= UINT64_C(1099511628211);
    }
  };
  std::string header;
  llvm::raw_string_ostream headerStream(header);
  headerStream << "ags-enumeration-prefix-v" << schemaVersion << '\n'
               << partitionFamilyHash.size() << ':' << partitionFamilyHash
               << '\n' << semanticsHash.size() << ':' << semanticsHash
               << '\n' << maximumOperatorSize << '\n'
               << quotientEquivalentOperands << '\n';
  headerStream.flush();
  update(header);
  for (auto [index, graph] : llvm::enumerate(graphs)) {
    std::string line;
    llvm::raw_string_ostream stream(line);
    stream << graph.id << '|' << graph.operatorSize << '|'
           << static_cast<unsigned>(graph.effectKind) << '|'
           << graph.operationName.size() << ':' << graph.operationName << '|'
           << graph.resultType << '|';
    if (graph.attributes)
      stream << graph.attributes;
    stream << '|';
    for (unsigned operand : graph.operands)
      stream << operand << ',';
    stream << '|';
    for (unsigned reachable : graph.reachableOperatorIds)
      stream << reachable << ',';
    stream << '|' << graph.canonicalKey.size() << ':' << graph.canonicalKey
           << "|frontier="
           << static_cast<unsigned>(operandFrontier[index]) << '\n';
    stream.flush();
    update(line);
  }
  for (const std::string &schema : realizedCandidateSchemas) {
    std::string line = (llvm::Twine("schema=") + llvm::Twine(schema.size()) +
                        ":" + schema + "\n")
                           .str();
    update(line);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

std::string OfflineEnumerationPrefixCache::getStableHash() const {
  return getPrefixCacheStableHash(
      schemaVersion, partitionFamilyHash, semanticsHash,
      maximumOperatorSize, quotientEquivalentOperands, graphs,
      operandFrontier,
      realizedCandidateSchemas);
}

static constexpr StringLiteral kPrefixCacheJSONLinesMagic =
    "AGS_ENUMERATION_PREFIX_JSONL_V2\n";

static llvm::json::Object
makePrefixCacheGraphRecord(const OfflineTypedGraph &graph,
                           bool operandFrontier) {
  llvm::json::Array operands;
  for (unsigned operand : graph.operands)
    operands.push_back(static_cast<int64_t>(operand));
  llvm::json::Array reachable;
  for (unsigned id : graph.reachableOperatorIds)
    reachable.push_back(static_cast<int64_t>(id));
  std::string attributes;
  llvm::raw_string_ostream attributeStream(attributes);
  attributeStream << graph.attributes;
  attributeStream.flush();
  llvm::json::Object object;
  object["record"] = "graph";
  object["id"] = static_cast<int64_t>(graph.id);
  object["operator_size"] = static_cast<int64_t>(graph.operatorSize);
  object["operation_name"] = graph.operationName;
  object["effect_kind"] = static_cast<int64_t>(graph.effectKind);
  object["result_type"] = printType(graph.resultType);
  object["attributes"] = attributes;
  object["operands"] = std::move(operands);
  object["reachable_operator_ids"] = std::move(reachable);
  object["canonical_key"] = graph.canonicalKey;
  object["operand_frontier"] = operandFrontier;
  return object;
}

static void writePrefixCacheJSONLines(
    llvm::raw_ostream &output, unsigned schemaVersion,
    StringRef partitionFamilyHash, StringRef semanticsHash,
    unsigned maximumOperatorSize, bool quotientEquivalentOperands,
    ArrayRef<OfflineTypedGraph> graphs,
    ArrayRef<uint8_t> operandFrontier,
    ArrayRef<std::string> realizedCandidateSchemas, StringRef stableHash) {
  output << kPrefixCacheJSONLinesMagic;
  llvm::json::Object header;
  header["record"] = "header";
  header["schema_version"] = static_cast<int64_t>(schemaVersion);
  header["partition_family_hash"] = partitionFamilyHash;
  header["semantics_hash"] = semanticsHash;
  header["maximum_operator_size"] =
      static_cast<int64_t>(maximumOperatorSize);
  header["quotient_equivalent_operands"] = quotientEquivalentOperands;
  header["graph_count"] = static_cast<int64_t>(graphs.size());
  header["schema_count"] =
      static_cast<int64_t>(realizedCandidateSchemas.size());
  header["stable_hash"] = stableHash;
  output << llvm::formatv("{0}\n", llvm::json::Value(std::move(header)));
  for (const std::string &schema : realizedCandidateSchemas) {
    llvm::json::Object record;
    record["record"] = "schema";
    record["value"] = schema;
    output << llvm::formatv("{0}\n", llvm::json::Value(std::move(record)));
  }
  for (auto [index, graph] : llvm::enumerate(graphs))
    output << llvm::formatv(
        "{0}\n", llvm::json::Value(makePrefixCacheGraphRecord(
                      graph, operandFrontier[index] != 0)));
}

void OfflineEnumerationPrefixCache::writeJSON(
    llvm::raw_ostream &output) const {
  const std::string hash = getStableHash();
  writePrefixCacheJSONLines(
      output, schemaVersion, partitionFamilyHash, semanticsHash,
      maximumOperatorSize, quotientEquivalentOperands, graphs,
      operandFrontier,
      realizedCandidateSchemas, hash);
}

bool OfflineEnumerationPrefixCache::writeJSONFromEnumeration(
    llvm::raw_ostream &output,
    const OfflineEnumerationResult &enumeration, StringRef semanticsHash,
    unsigned maximumOperatorSize, std::string &error) {
  if (!enumeration.isComplete()) {
    error = "cannot cache an incomplete enumeration prefix";
    return false;
  }
  size_t prefixCount = 0;
  while (prefixCount < enumeration.graphs.size() &&
         enumeration.graphs[prefixCount].operatorSize <= maximumOperatorSize)
    ++prefixCount;
  for (size_t index = prefixCount; index < enumeration.graphs.size(); ++index)
    if (enumeration.graphs[index].operatorSize <= maximumOperatorSize) {
      error = "enumeration prefix is not contiguous";
      return false;
    }
  const bool cachesEntireEnumeration =
      prefixCount == enumeration.graphs.size();
  if (!cachesEntireEnumeration &&
      prefixCount != enumeration.completePrefixGraphCount) {
    error = "enumeration prefix graph count does not match its certificate";
    return false;
  }
  std::vector<std::string> schemas;
  if (cachesEntireEnumeration) {
    for (const auto &[schema, count] : enumeration.graphCountBySchema)
      if (count != 0)
        schemas.push_back(schema);
  } else {
    schemas = enumeration.realizedPrefixCandidateSchemas;
  }
  sortAndUnique(schemas);
  if (enumeration.operandFrontier.size() < prefixCount) {
    error = "enumeration prefix has no complete operand frontier";
    return false;
  }
  ArrayRef<OfflineTypedGraph> graphs(enumeration.graphs.data(), prefixCount);
  ArrayRef<uint8_t> operandFrontier(enumeration.operandFrontier.data(),
                                    prefixCount);
  const std::string hash = getPrefixCacheStableHash(
      kSchemaVersion, enumeration.finalLayerPartitionFamilyHash,
      semanticsHash, maximumOperatorSize,
      enumeration.quotientEquivalentOperands, graphs, operandFrontier,
      schemas);
  writePrefixCacheJSONLines(
      output, kSchemaVersion, enumeration.finalLayerPartitionFamilyHash,
      semanticsHash, maximumOperatorSize,
      enumeration.quotientEquivalentOperands, graphs, operandFrontier,
      schemas, hash);
  return true;
}

std::string OfflineEnumerationPrefixCache::toJSON() const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream);
  return storage;
}

std::optional<OfflineEnumerationPrefixCache>
OfflineEnumerationPrefixCache::fromEnumeration(
    const OfflineEnumerationResult &enumeration, StringRef semanticsHash,
    unsigned maximumOperatorSize, std::string &error) {
  if (!enumeration.isComplete()) {
    error = "cannot cache an incomplete enumeration prefix";
    return std::nullopt;
  }
  OfflineEnumerationPrefixCache cache;
  cache.partitionFamilyHash = enumeration.finalLayerPartitionFamilyHash;
  cache.semanticsHash = semanticsHash.str();
  cache.maximumOperatorSize = maximumOperatorSize;
  cache.quotientEquivalentOperands =
      enumeration.quotientEquivalentOperands;
  for (const OfflineTypedGraph &graph : enumeration.graphs) {
    if (graph.operatorSize > maximumOperatorSize)
      continue;
    if (graph.id != cache.graphs.size()) {
      error = "enumeration prefix is not a contiguous topological prefix";
      return std::nullopt;
    }
    cache.graphs.push_back(graph);
    if (graph.id >= enumeration.operandFrontier.size()) {
      error = "enumeration prefix has no complete operand frontier";
      return std::nullopt;
    }
    cache.operandFrontier.push_back(
        enumeration.operandFrontier[graph.id]);
  }
  const bool cachesEntireEnumeration =
      cache.graphs.size() == enumeration.graphs.size();
  if (!cachesEntireEnumeration &&
      cache.graphs.size() != enumeration.completePrefixGraphCount) {
    error = "enumeration prefix graph count does not match its certificate";
    return std::nullopt;
  }
  if (cachesEntireEnumeration) {
    for (const auto &[schema, count] : enumeration.graphCountBySchema)
      if (count != 0)
        cache.realizedCandidateSchemas.push_back(schema);
  } else {
    cache.realizedCandidateSchemas =
        enumeration.realizedPrefixCandidateSchemas;
  }
  sortAndUnique(cache.realizedCandidateSchemas);
  return cache;
}

static bool appendPrefixCacheGraphRecord(
    const llvm::json::Object &object,
    OfflineEnumerationPrefixCache &cache, MLIRContext &context,
    std::string &error) {
  if (std::optional<StringRef> record = object.getString("record");
      record && *record != "graph") {
    error = "enumeration prefix graph has the wrong record kind";
    return false;
  }
  std::optional<int64_t> id = object.getInteger("id");
  std::optional<int64_t> operatorSize = object.getInteger("operator_size");
  std::optional<StringRef> operation = object.getString("operation_name");
  std::optional<int64_t> effect = object.getInteger("effect_kind");
  std::optional<StringRef> typeText = object.getString("result_type");
  std::optional<StringRef> attributeText = object.getString("attributes");
  const llvm::json::Array *operands = object.getArray("operands");
  const llvm::json::Array *reachable =
      object.getArray("reachable_operator_ids");
  std::optional<StringRef> canonical = object.getString("canonical_key");
  std::optional<bool> operandFrontier =
      object.getBoolean("operand_frontier");
  if (!id || *id < 0 || !operatorSize || *operatorSize < 0 || !operation ||
      !effect || *effect < 0 ||
      *effect > static_cast<int64_t>(TTIRCandidateEffectKind::BufferWrite) ||
      !typeText || !attributeText || !operands || !reachable || !canonical ||
      !operandFrontier) {
    error = "enumeration prefix graph is missing a required field";
    return false;
  }
  if (static_cast<size_t>(*id) != cache.graphs.size()) {
    error = "enumeration prefix graph IDs must be contiguous";
    return false;
  }
  Type type = mlir::parseType(*typeText, &context);
  Attribute attribute = mlir::parseAttribute(*attributeText, &context);
  DictionaryAttr dictionary = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!type || !dictionary) {
    error = "enumeration prefix graph has an invalid type or attribute";
    return false;
  }
  OfflineTypedGraph graph;
  graph.id = static_cast<unsigned>(*id);
  graph.operatorSize = static_cast<unsigned>(*operatorSize);
  graph.operationName = operation->str();
  graph.effectKind = static_cast<TTIRCandidateEffectKind>(*effect);
  graph.resultType = type;
  graph.attributes = dictionary;
  graph.canonicalKey = canonical->str();
  for (const llvm::json::Value &operandValue : *operands) {
    std::optional<int64_t> operand = operandValue.getAsInteger();
    if (!operand || *operand < 0 ||
        static_cast<uint64_t>(*operand) >= cache.graphs.size()) {
      error = "enumeration prefix operand is not topological";
      return false;
    }
    graph.operands.push_back(static_cast<unsigned>(*operand));
  }
  for (const llvm::json::Value &reachableValue : *reachable) {
    std::optional<int64_t> reachableId = reachableValue.getAsInteger();
    if (!reachableId || *reachableId < 0 ||
        static_cast<uint64_t>(*reachableId) > cache.graphs.size()) {
      error = "enumeration prefix reachable ID is invalid";
      return false;
    }
    graph.reachableOperatorIds.push_back(
        static_cast<unsigned>(*reachableId));
  }
  if (graph.operatorSize > cache.maximumOperatorSize ||
      graph.reachableOperatorIds.size() != graph.operatorSize) {
    error = "enumeration prefix graph violates its operator-size bound";
    return false;
  }
  cache.graphs.push_back(std::move(graph));
  cache.operandFrontier.push_back(*operandFrontier ? 1 : 0);
  return true;
}

std::optional<OfflineEnumerationPrefixCache>
OfflineEnumerationPrefixCache::fromJSON(StringRef serialized,
                                        MLIRContext &context,
                                        std::string &error,
                                        size_t maxGraphs) {
  if (serialized.consume_front(kPrefixCacheJSONLinesMagic)) {
    auto takeLine = [&]() -> std::optional<StringRef> {
      if (serialized.empty())
        return std::nullopt;
      size_t end = serialized.find('\n');
      StringRef line = end == StringRef::npos
                           ? serialized
                           : serialized.take_front(end);
      serialized = end == StringRef::npos
                       ? StringRef{}
                       : serialized.drop_front(end + 1);
      return line.rtrim("\r");
    };
    std::optional<StringRef> headerLine = takeLine();
    if (!headerLine) {
      error = "enumeration prefix JSON Lines cache has no header";
      return std::nullopt;
    }
    llvm::Expected<llvm::json::Value> parsedHeader =
        llvm::json::parse(*headerLine);
    if (!parsedHeader) {
      error = "invalid enumeration prefix JSON Lines header: " +
              llvm::toString(parsedHeader.takeError());
      return std::nullopt;
    }
    const llvm::json::Object *header = parsedHeader->getAsObject();
    std::optional<StringRef> record =
        header ? header->getString("record") : std::nullopt;
    std::optional<int64_t> version =
        header ? header->getInteger("schema_version") : std::nullopt;
    std::optional<StringRef> family =
        header ? header->getString("partition_family_hash") : std::nullopt;
    std::optional<StringRef> semantics =
        header ? header->getString("semantics_hash") : std::nullopt;
    std::optional<int64_t> maximum =
        header ? header->getInteger("maximum_operator_size") : std::nullopt;
    std::optional<bool> quotient =
        header ? header->getBoolean("quotient_equivalent_operands")
               : std::nullopt;
    std::optional<int64_t> graphCount =
        header ? header->getInteger("graph_count") : std::nullopt;
    std::optional<int64_t> schemaCount =
        header ? header->getInteger("schema_count") : std::nullopt;
    std::optional<StringRef> expectedHash =
        header ? header->getString("stable_hash") : std::nullopt;
    if (!header || !record || *record != "header" || !version ||
        *version != kSchemaVersion || !family || !semantics || !maximum ||
        *maximum < 0 || !quotient || !graphCount || *graphCount < 0 ||
        static_cast<uint64_t>(*graphCount) > maxGraphs ||
        !schemaCount || *schemaCount < 0 || !expectedHash) {
      error = "enumeration prefix JSON Lines header is invalid";
      return std::nullopt;
    }

    OfflineEnumerationPrefixCache cache;
    cache.partitionFamilyHash = family->str();
    cache.semanticsHash = semantics->str();
    cache.maximumOperatorSize = static_cast<unsigned>(*maximum);
    cache.quotientEquivalentOperands = *quotient;
    cache.realizedCandidateSchemas.reserve(
        static_cast<size_t>(*schemaCount));
    for (int64_t index = 0; index < *schemaCount; ++index) {
      std::optional<StringRef> line = takeLine();
      if (!line) {
        error = "enumeration prefix schema records are truncated";
        return std::nullopt;
      }
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*line);
      if (!parsed) {
        error = "invalid enumeration prefix schema record: " +
                llvm::toString(parsed.takeError());
        return std::nullopt;
      }
      const llvm::json::Object *object = parsed->getAsObject();
      std::optional<StringRef> kind =
          object ? object->getString("record") : std::nullopt;
      std::optional<StringRef> value =
          object ? object->getString("value") : std::nullopt;
      if (!object || !kind || *kind != "schema" || !value) {
        error = "enumeration prefix schema record is invalid";
        return std::nullopt;
      }
      cache.realizedCandidateSchemas.push_back(value->str());
    }
    if (!std::is_sorted(cache.realizedCandidateSchemas.begin(),
                        cache.realizedCandidateSchemas.end()) ||
        std::adjacent_find(cache.realizedCandidateSchemas.begin(),
                           cache.realizedCandidateSchemas.end()) !=
            cache.realizedCandidateSchemas.end()) {
      error = "enumeration prefix schema list must be sorted and unique";
      return std::nullopt;
    }

    cache.graphs.reserve(static_cast<size_t>(*graphCount));
    for (int64_t index = 0; index < *graphCount; ++index) {
      std::optional<StringRef> line = takeLine();
      if (!line) {
        error = "enumeration prefix graph records are truncated";
        return std::nullopt;
      }
      llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(*line);
      if (!parsed) {
        error = "invalid enumeration prefix graph record: " +
                llvm::toString(parsed.takeError());
        return std::nullopt;
      }
      const llvm::json::Object *object = parsed->getAsObject();
      if (!object ||
          !appendPrefixCacheGraphRecord(*object, cache, context, error))
        return std::nullopt;
    }
    if (!serialized.trim().empty()) {
      error = "enumeration prefix JSON Lines cache has trailing records";
      return std::nullopt;
    }
    const std::string actualHash = cache.getStableHash();
    if (actualHash != *expectedHash) {
      error = "enumeration prefix stable hash mismatch";
      return std::nullopt;
    }
    cache.validatedStableHash = actualHash;
    return cache;
  }

  // Legacy caches are accepted only while they are small enough for the
  // single-object parser.  A production-sized legacy file would recreate the
  // exact DOM peak that the record stream is designed to eliminate.
  constexpr size_t maxLegacyPrefixCacheBytes = 64 * 1024 * 1024;
  if (serialized.size() > maxLegacyPrefixCacheBytes) {
    error = "legacy enumeration prefix JSON exceeds 64 MiB; regenerate it "
            "with the JSON Lines writer";
    return std::nullopt;
  }

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(serialized);
  if (!parsed) {
    error = "invalid enumeration prefix JSON: " +
            llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "enumeration prefix root must be an object";
    return std::nullopt;
  }
  std::optional<int64_t> version = root->getInteger("schema_version");
  std::optional<StringRef> family =
      root->getString("partition_family_hash");
  std::optional<StringRef> semantics = root->getString("semantics_hash");
  std::optional<int64_t> maximum =
      root->getInteger("maximum_operator_size");
  std::optional<bool> quotient =
      root->getBoolean("quotient_equivalent_operands");
  std::optional<int64_t> graphCount = root->getInteger("graph_count");
  std::optional<StringRef> expectedHash = root->getString("stable_hash");
  const llvm::json::Array *schemaArray =
      root->getArray("realized_candidate_schemas");
  const llvm::json::Array *graphArray = root->getArray("graphs");
  if (!version || *version != kSchemaVersion || !family || !semantics ||
      !maximum || *maximum < 0 || !quotient || !graphCount ||
      *graphCount < 0 || static_cast<uint64_t>(*graphCount) > maxGraphs ||
      !expectedHash || !schemaArray || !graphArray) {
    error = "enumeration prefix is missing a required field or version";
    return std::nullopt;
  }

  OfflineEnumerationPrefixCache cache;
  cache.partitionFamilyHash = family->str();
  cache.semanticsHash = semantics->str();
  cache.maximumOperatorSize = static_cast<unsigned>(*maximum);
  cache.quotientEquivalentOperands = *quotient;
  for (const llvm::json::Value &value : *schemaArray) {
    std::optional<StringRef> schema = value.getAsString();
    if (!schema) {
      error = "enumeration prefix schema list must contain strings";
      return std::nullopt;
    }
    cache.realizedCandidateSchemas.push_back(schema->str());
  }
  if (!std::is_sorted(cache.realizedCandidateSchemas.begin(),
                      cache.realizedCandidateSchemas.end()) ||
      std::adjacent_find(cache.realizedCandidateSchemas.begin(),
                         cache.realizedCandidateSchemas.end()) !=
          cache.realizedCandidateSchemas.end()) {
    error = "enumeration prefix schema list must be sorted and unique";
    return std::nullopt;
  }

  cache.graphs.reserve(graphArray->size());
  for (const llvm::json::Value &value : *graphArray) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object) {
      error = "enumeration prefix graph must be an object";
      return std::nullopt;
    }
    std::optional<int64_t> id = object->getInteger("id");
    std::optional<int64_t> operatorSize =
        object->getInteger("operator_size");
    std::optional<StringRef> operation = object->getString("operation_name");
    std::optional<int64_t> effect = object->getInteger("effect_kind");
    std::optional<StringRef> typeText = object->getString("result_type");
    std::optional<StringRef> attributeText = object->getString("attributes");
    const llvm::json::Array *operands = object->getArray("operands");
    const llvm::json::Array *reachable =
        object->getArray("reachable_operator_ids");
    std::optional<StringRef> canonical = object->getString("canonical_key");
    std::optional<bool> operandFrontier =
        object->getBoolean("operand_frontier");
    if (!id || *id < 0 || !operatorSize || *operatorSize < 0 || !operation ||
        !effect || *effect < 0 ||
        *effect > static_cast<int64_t>(TTIRCandidateEffectKind::BufferWrite) ||
        !typeText || !attributeText || !operands || !reachable || !canonical ||
        !operandFrontier) {
      error = "enumeration prefix graph is missing a required field";
      return std::nullopt;
    }
    if (static_cast<size_t>(*id) != cache.graphs.size()) {
      error = "enumeration prefix graph IDs must be contiguous";
      return std::nullopt;
    }
    Type type = mlir::parseType(*typeText, &context);
    Attribute attribute = mlir::parseAttribute(*attributeText, &context);
    DictionaryAttr dictionary = dyn_cast_or_null<DictionaryAttr>(attribute);
    if (!type || !dictionary) {
      error = "enumeration prefix graph has an invalid type or attribute";
      return std::nullopt;
    }
    OfflineTypedGraph graph;
    graph.id = static_cast<unsigned>(*id);
    graph.operatorSize = static_cast<unsigned>(*operatorSize);
    graph.operationName = operation->str();
    graph.effectKind = static_cast<TTIRCandidateEffectKind>(*effect);
    graph.resultType = type;
    graph.attributes = dictionary;
    graph.canonicalKey = canonical->str();
    for (const llvm::json::Value &operandValue : *operands) {
      std::optional<int64_t> operand = operandValue.getAsInteger();
      if (!operand || *operand < 0 ||
          static_cast<uint64_t>(*operand) >= cache.graphs.size()) {
        error = "enumeration prefix operand is not topological";
        return std::nullopt;
      }
      graph.operands.push_back(static_cast<unsigned>(*operand));
    }
    for (const llvm::json::Value &reachableValue : *reachable) {
      std::optional<int64_t> reachableId = reachableValue.getAsInteger();
      if (!reachableId || *reachableId < 0 ||
          static_cast<uint64_t>(*reachableId) > cache.graphs.size()) {
        error = "enumeration prefix reachable ID is invalid";
        return std::nullopt;
      }
      graph.reachableOperatorIds.push_back(
          static_cast<unsigned>(*reachableId));
    }
    if (graph.operatorSize > cache.maximumOperatorSize ||
        graph.reachableOperatorIds.size() != graph.operatorSize) {
      error = "enumeration prefix graph violates its operator-size bound";
      return std::nullopt;
    }
    cache.graphs.push_back(std::move(graph));
    cache.operandFrontier.push_back(*operandFrontier ? 1 : 0);
  }
  if (static_cast<int64_t>(cache.graphs.size()) != *graphCount) {
    error = "enumeration prefix graph count mismatch";
    return std::nullopt;
  }
  const std::string actualHash = cache.getStableHash();
  if (actualHash != *expectedHash) {
    error = "enumeration prefix stable hash mismatch";
    return std::nullopt;
  }
  cache.validatedStableHash = actualHash;
  return cache;
}

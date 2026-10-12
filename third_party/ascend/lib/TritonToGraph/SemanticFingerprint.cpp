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

#include "TritonToGraph/SemanticFingerprint.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <thread>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static uint64_t chooseTwo(uint64_t count) {
  return count < 2 ? 0 : count * (count - 1) / 2;
}

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

static std::string printType(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static uint64_t stableHash64(llvm::StringRef value) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static std::string stableHash(llvm::StringRef value) {
  uint64_t hash = stableHash64(value);
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

static bool denotationsStructurallyEqual(const TensorDenotation &lhs,
                                         const TensorDenotation &rhs) {
  if (lhs.shape != rhs.shape || lhs.shapeSymbols != rhs.shapeSymbols ||
      lhs.layoutIdentity != rhs.layoutIdentity ||
      lhs.resultIndices.size() != rhs.resultIndices.size() ||
      lhs.isMemoryStateToken != rhs.isMemoryStateToken ||
      lhs.isLaunchStateToken != rhs.isLaunchStateToken ||
      lhs.isBufferStateToken != rhs.isBufferStateToken)
    return false;
  for (auto [lhsIndex, rhsIndex] :
       llvm::zip_equal(lhs.resultIndices, rhs.resultIndices))
    if (!lhsIndex.structurallyEquals(rhsIndex))
      return false;
  return lhs.value.structurallyEquals(rhs.value) &&
         lhs.defined.structurallyEquals(rhs.defined) &&
         lhs.effect.structurallyEquals(rhs.effect) &&
         lhs.memory.structurallyEquals(rhs.memory) &&
         lhs.memoryDefined.structurallyEquals(rhs.memoryDefined) &&
         lhs.launch.structurallyEquals(rhs.launch) &&
         lhs.launchDefined.structurallyEquals(rhs.launchDefined) &&
         lhs.buffer.structurallyEquals(rhs.buffer) &&
         lhs.bufferDefined.structurallyEquals(rhs.bufferDefined);
}

// SplitMix64 finalizer. In particular, every sampled output bit depends on
// high input bits; simple affine sampling would make i1/i8 fingerprints retain
// correlations between symbol ordinals and create avoidable false positives.
static uint64_t mix64(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

static std::optional<unsigned> getScalarBitWidth(Type type) {
  if (auto integer = dyn_cast_or_null<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast_or_null<FloatType>(type))
    return floating.getWidth();
  if (isa_and_nonnull<IndexType>(type))
    return 64;
  if (isa_and_nonnull<triton::PointerType>(type))
    return 64;
  return std::nullopt;
}

static llvm::APInt resizeUnsigned(const llvm::APInt &value, unsigned width) {
  return value.getBitWidth() == width ? value
         : value.getBitWidth() < width ? value.zext(width)
                                       : value.trunc(width);
}

// Samples model fixed-width bit-vectors, so an arbitrary deterministic host
// value must be reduced modulo 2^width.  APInt's uint64_t constructor asserts
// instead of truncating when the value does not fit a narrow width.
static llvm::APInt wrapUnsigned(unsigned width, uint64_t value) {
  return resizeUnsigned(llvm::APInt(64, value), width);
}

static llvm::APInt resizeSigned(const llvm::APInt &value, unsigned width) {
  return value.getBitWidth() == width ? value
         : value.getBitWidth() < width ? value.sext(width)
                                       : value.trunc(width);
}

static unsigned getIntegerWidthClass(unsigned width) {
  switch (width) {
  case 1:
    return 0;
  case 8:
    return 1;
  case 16:
    return 2;
  case 32:
    return 3;
  case 64:
    return 4;
  default:
    return width % 8;
  }
}

// The first twenty-five adaptive rows over GF(5) form a strength-2 covering
// array for six symbolic columns: every pair observes all 5x5 value pairs.
// Later rows add deterministic small-domain hash assignments, which break
// higher-arity correlations without naming any operation.  Together they let
// adaptive counterexample search cover mixed OP/type combinations while the
// compact primary fingerprint remains cheap.
static unsigned pairwiseCoveringArrayValue(uint64_t row,
                                           uint64_t symbolOrdinal) {
  if (row >= 25)
    return static_cast<unsigned>(
        mix64((row + 1) * UINT64_C(0x9e3779b97f4a7c15) ^
              (symbolOrdinal + 1) * UINT64_C(0xd1b54a32d192ed03)) %
        5);
  unsigned a = static_cast<unsigned>(row / 5);
  unsigned b = static_cast<unsigned>(row % 5);
  unsigned column = static_cast<unsigned>(symbolOrdinal % 6);
  return column == 5 ? a : (a * column + b) % 5;
}

// Sixteen deterministic rows form a small covering array for the first three
// symbolic inputs. Rows 1..3 make exactly one input +1, rows 4..6 make exactly
// one input -1, and the second half adds inactive masks and signed boundary
// values beyond {-1, 0, 1}. This separates common equality, order and aggregate
// collisions without naming any source operation.
static llvm::APInt integerStructuredSample(unsigned width, uint64_t row,
                                           uint64_t symbolOrdinal,
                                           uint64_t laneOrdinal = 0) {
  llvm::APInt zero(width, 0);
  llvm::APInt one(width, 1);
  if (width == 1) {
    // Rows 1..7 enumerate every non-zero truth assignment of the first three
    // Boolean symbols. Row zero is all true so a simultaneous integer-zero
    // model exercises active masks; the odd typed model contributes all false.
    unsigned truthRow = static_cast<unsigned>(row % 16);
    if (truthRow == 0)
      return one;
    if (truthRow == 8)
      return zero;
    if (truthRow > 8)
      truthRow -= 8;
    return llvm::APInt(
        1, (truthRow >> ((symbolOrdinal + laneOrdinal) % 3)) & 1u);
  }
  if (row >= 16) {
    switch (pairwiseCoveringArrayValue(row - 16,
                                       symbolOrdinal + laneOrdinal)) {
    case 0:
      return zero;
    case 1:
      return one;
    case 2:
      return llvm::APInt::getAllOnes(width);
    case 3:
      return wrapUnsigned(width, static_cast<uint64_t>(-2));
    default:
      // Make equalities against small constants type-sensitive. Signed-min is
      // already covered by structured row 10; repeating it here correlates
      // independent i8/i16/i32/i64 leaves at predicates such as x == 4.
      return wrapUnsigned(width, 3 + getIntegerWidthClass(width));
    }
  }
  switch (row % 16) {
  case 0:
    return zero;
  case 1:
    return symbolOrdinal % 3 == 2 ? one : zero;
  case 2:
    return symbolOrdinal % 3 == 1 ? one : zero;
  case 3:
    return symbolOrdinal % 3 == 0 ? one : zero;
  case 4:
    return (symbolOrdinal + laneOrdinal) % 3 == 2
               ? llvm::APInt::getAllOnes(width)
               : zero;
  case 5:
    return (symbolOrdinal + laneOrdinal) % 3 == 1
               ? llvm::APInt::getAllOnes(width)
               : zero;
  case 6:
    return (symbolOrdinal + laneOrdinal) % 3 == 0
               ? llvm::APInt::getAllOnes(width)
               : zero;
  case 7:
    // Model 14 tracks the sampled rank-1 histogram bin. With the coordinate
    // schedule below, tensor input symbol zero at lane zero evaluates to bin
    // three, while adjacent symbols/lanes remain distinct.
    return wrapUnsigned(width, 2 + symbolOrdinal + laneOrdinal);
  case 8:
    return one;
  case 9:
    // x == -1 and x < 0 agree on {-1, 0, +1}; -2 separates them.
    return wrapUnsigned(width, static_cast<uint64_t>(-2));
  case 10:
    return llvm::APInt::getSignedMinValue(width);
  case 11:
    return wrapUnsigned(width, 2);
  case 12:
    return wrapUnsigned(width, 3);
  case 13:
    return wrapUnsigned(width, 4);
  case 14:
    return wrapUnsigned(width, static_cast<uint64_t>(-3));
  case 15:
    return wrapUnsigned(width, static_cast<uint64_t>(-4));
  default:
    return wrapUnsigned(width,
                        3 + symbolOrdinal + laneOrdinal + 7 * (row - 11));
  }
}

// Tensor Boolean sampling must cover two independent dimensions.  Rows 0..7
// are a complete pointwise truth table for any three consecutive tensor
// identities.  This prevents select/and/or graphs from aliasing merely because
// their operands received correlated lane patterns.  Rows 8..15 exercise
// aggregate boundaries with all-false, isolated-true and isolated-false
// tensors.  A random 16-lane tensor alone makes product/forall false almost
// always and cannot distinguish full reductions, prefixes and suffixes.
static llvm::APInt booleanTensorStructuredSample(uint64_t row,
                                                 uint64_t tensorOrdinal,
                                                 uint64_t laneOrdinal) {
  if (row >= 16)
    return llvm::APInt(
        1, pairwiseCoveringArrayValue(row - 16,
                                      tensorOrdinal + 4 * laneOrdinal) & 1u);
  unsigned sampleRow = static_cast<unsigned>(row % 16);
  if (sampleRow < 8)
    return llvm::APInt(1, (sampleRow >> (tensorOrdinal % 3)) & 1u);
  switch (sampleRow) {
  case 8:
    return llvm::APInt(1, 0);
  case 9:
    return llvm::APInt(1, laneOrdinal == 0);
  case 10:
    return llvm::APInt(1, laneOrdinal == 4);
  case 11:
    return llvm::APInt(1, laneOrdinal == 15);
  case 12:
    return llvm::APInt(1, (laneOrdinal & 1u) == 0);
  case 13:
    return llvm::APInt(1, laneOrdinal != 0);
  case 14:
    return llvm::APInt(1, laneOrdinal != 4);
  default:
    return llvm::APInt(1, laneOrdinal != 15);
  }
}

static llvm::APInt integerTensorAggregateSample(unsigned width, uint64_t row,
                                                uint64_t tensorOrdinal,
                                                uint64_t laneOrdinal) {
  llvm::APInt zero(width, 0);
  llvm::APInt one(width, 1);
  llvm::APInt two = wrapUnsigned(width, 2);
  llvm::APInt minusTwo = wrapUnsigned(width, static_cast<uint64_t>(-2));
  if (row >= 16) {
    switch (pairwiseCoveringArrayValue(row - 16,
                                       tensorOrdinal + 4 * laneOrdinal)) {
    case 0:
      return zero;
    case 1:
      return one;
    case 2:
      return llvm::APInt::getAllOnes(width);
    case 3:
      return minusTwo;
    default:
      return wrapUnsigned(width, 3 + getIntegerWidthClass(width));
    }
  }
  switch (row % 16) {
  case 0:
    return one;
  case 1:
  case 2:
  case 3: {
    uint64_t focus = (row - 1) % 3;
    if (tensorOrdinal % 3 == focus)
      return zero;
    return laneOrdinal == 0 ? two : one;
  }
  case 4:
    if (tensorOrdinal % 3 == 0)
      return llvm::APInt::getAllOnes(width);
    return laneOrdinal == 0 ? two : one;
  case 5:
    if (tensorOrdinal % 3 == 1)
      return llvm::APInt::getAllOnes(width);
    return laneOrdinal == 4 ? two : one;
  case 6:
    if (tensorOrdinal % 3 == 2)
      return llvm::APInt::getAllOnes(width);
    return laneOrdinal == 15 ? two : one;
  case 7:
    return integerStructuredSample(width, row, tensorOrdinal, laneOrdinal);
  case 8:
    // Boolean row 8 is all false and this row is all one. At output bin one,
    // masked and unmasked aggregate behavior therefore cannot alias.
    return one;
  case 9:
    return minusTwo;
  case 10:
    return llvm::APInt::getSignedMinValue(width);
  case 11:
    return two;
  case 12:
    return wrapUnsigned(width, 3);
  case 13:
    return wrapUnsigned(width, 4);
  case 14:
    return wrapUnsigned(width, static_cast<uint64_t>(-3));
  default:
    return wrapUnsigned(width, static_cast<uint64_t>(-4));
  }
}

static unsigned getIntegerTypeClass(Type type) {
  auto integer = dyn_cast_or_null<IntegerType>(type);
  if (!integer)
    return 0;
  return getIntegerWidthClass(integer.getWidth());
}

// Fifteen odd-numbered models deliberately assign a distinct three-bit
// equality code to every common integer type. Five groups use -1, +1, 0, 2
// and 4 as the active value. This catches cross-width comparisons with the
// complete small-constant domain used by the production depth-one universe:
// independent random values would almost always make both sides false even
// though the typed inputs are unrelated.
static llvm::APInt integerTypedEqualitySample(unsigned width, unsigned bit,
                                              unsigned typeClass,
                                               uint64_t symbolOrdinal,
                                               uint64_t laneOrdinal = 0) {
  if (width == 1) {
    // Typed equality models deliberately pin an integer family to -1/+1/0.
    // Give Boolean inputs an independent three-bit identity code instead of
    // only parity: symbols two positions apart otherwise agree in every odd
    // model.  Mixing the lane keeps tensor inputs non-constant while retaining
    // pairwise separation of the first eight symbolic identities.
    uint64_t identity = symbolOrdinal + 5 * laneOrdinal;
    return llvm::APInt(1, (identity >> (bit % 3)) & 1u);
  }
  unsigned codeBit = bit % 3;
  llvm::APInt active = bit < 3   ? llvm::APInt::getAllOnes(width)
                       : bit < 6 ? llvm::APInt(width, 1)
                       : bit < 9 ? llvm::APInt(width, 0)
                       : bit < 12 ? wrapUnsigned(width, 2)
                                  : wrapUnsigned(width, 4);
  if ((typeClass & (1u << codeBit)) != 0)
    return active;
  // The inactive value is adjacent but guaranteed distinct for every common
  // integer width. Symbol/lane identity is covered by the even covering-array
  // models; equality-code models isolate type families on purpose.
  return active + llvm::APInt(width, 1);
}

static std::optional<unsigned> getTypedEqualityBit(unsigned model) {
  if ((model & 1) == 0)
    return std::nullopt;
  if (model < 12)
    return model / 2;
  // Keep models 13 and 15 pseudo-random. Larger offline audits add groups for
  // equality against zero, two and four at models 17..33.
  if (model >= 17 && model <= 33)
    return 6 + (model - 17) / 2;
  return std::nullopt;
}

struct EvaluationContext {
  unsigned model = 0;
  SmallVector<std::pair<unsigned, llvm::APInt>, 8> boundIndices;
  llvm::StringMap<uint64_t> extentValues;
};

static std::optional<unsigned>
getPayloadOrdinal(const TensorSemanticExpr &expression) {
  auto attribute = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
  if (!attribute)
    return std::nullopt;
  return static_cast<unsigned>(attribute.getValue().getLimitedValue());
}

static std::optional<uint64_t>
getPayloadSymbolKey(const TensorSemanticExpr &expression) {
  std::optional<uint64_t> payloadKey;
  if (std::optional<unsigned> ordinal = getPayloadOrdinal(expression))
    payloadKey = static_cast<uint64_t>(*ordinal) + 1;
  if (auto name = dyn_cast_or_null<StringAttr>(expression.getPayload())) {
    payloadKey = stableHash64(name.getValue());
  }
  if (!payloadKey) {
    auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
    auto marker = payload
                      ? dyn_cast_or_null<BoolAttr>(
                            payload.get("private_buffer"))
                      : BoolAttr{};
    auto symbol = payload
                      ? dyn_cast_or_null<IntegerAttr>(payload.get("symbol_id"))
                      : IntegerAttr{};
    if (marker && marker.getValue() && symbol)
      payloadKey = mix64(UINT64_C(0x7072697661746500) ^
                         symbol.getValue().getLimitedValue());
  }
  if (!payloadKey)
    return std::nullopt;

  // A symbolic input is identified by payload, scalar type and semantic role.
  // Reusing only the numeric symbol ordinal makes unrelated boundary values
  // such as input#1:i32 and input#1:i64 share every low-bit sample.  Casts keep
  // the original typed leaf, so including this identity still gives identical
  // samples to two genuinely equivalent expressions over the same input.
  uint64_t typeKey = stableHash64(printType(expression.getType()));
  uint64_t kindKey = static_cast<uint64_t>(expression.getKind()) + 1;
  return mix64(*payloadKey ^ mix64(typeKey) ^
               (UINT64_C(0x9e3779b97f4a7c15) * kindKey));
}

static std::optional<uint64_t>
getPrivateBufferOrdinal(const TensorSemanticExpr &expression) {
  auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
  auto marker = payload
                    ? dyn_cast_or_null<BoolAttr>(
                          payload.get("private_buffer"))
                    : BoolAttr{};
  auto symbol = payload
                    ? dyn_cast_or_null<IntegerAttr>(payload.get("symbol_id"))
                    : IntegerAttr{};
  if (!marker || !marker.getValue() || !symbol)
    return std::nullopt;
  return symbol.getValue().getLimitedValue();
}

static uint64_t getSamplingOrdinal(const TensorSemanticExpr &expression,
                                   uint64_t symbolKey) {
  uint64_t ordinal = symbolKey;
  if (std::optional<unsigned> payload = getPayloadOrdinal(expression))
    ordinal = *payload;
  else if (std::optional<uint64_t> privateBuffer =
               getPrivateBufferOrdinal(expression))
    // Keep allocation identities consecutive for the pairwise covering
    // array. Hashing first and taking a small column modulo can alias two
    // private buffers in every valid-shift sample. The offset also separates
    // private allocation #N from boundary input #N.
    ordinal = *privateBuffer + 17;
  if (isa<IntegerType>(expression.getType())) {
    unsigned typeClass = getIntegerTypeClass(expression.getType());
    // Common type classes 0 and 3 (i1/i32) would otherwise have the same
    // modulo-three structured row, correlating a histogram mask with its data
    // merely because both use the same user-facing symbol number.
    ordinal += typeClass + (typeClass >= 3 ? 1 : 0);
  }
  // A scalar and a tensor element with the same user-facing symbol number are
  // independent inputs. Give Element a separate structured-sampling lane;
  // otherwise every non-random model aliases scalar#N with tensor#N[...].
  if (expression.getKind() == TensorSemanticKind::Element)
    ++ordinal;
  return ordinal;
}

static std::optional<llvm::APInt>
evaluateExpression(const TensorSemanticExpr &expression,
                   EvaluationContext &context) {
  if (!expression)
    return std::nullopt;
  std::optional<unsigned> width = getScalarBitWidth(expression.getType());
  switch (expression.getKind()) {
  case TensorSemanticKind::Symbol:
    // Descriptor, pointer-like and state handles are first-class boundary
    // values in the semantic language even when MLIR does not expose an
    // integer bit width for their type. A deterministic 64-bit carrier is a
    // finite interpretation of that opaque sort; exact proof remains with the
    // normalizer/oracle.
    if (!width)
      width = 64;
    break;
  case TensorSemanticKind::MemoryWrite:
  case TensorSemanticKind::MemoryAtomic:
  case TensorSemanticKind::MemoryObservable:
  case TensorSemanticKind::PartitionableMemory:
  case TensorSemanticKind::BufferWrite:
  case TensorSemanticKind::BufferFill:
  case TensorSemanticKind::BufferView:
  case TensorSemanticKind::BufferAllocate:
  case TensorSemanticKind::LaunchBound:
  case TensorSemanticKind::LaunchRemap:
  case TensorSemanticKind::KernelEffect:
  case TensorSemanticKind::KernelPartition:
    // State is encoded in the proof language with an i1 carrier type, but a
    // fingerprint must retain the full transition hash.  Truncating it to one
    // bit creates thousands of false collisions between distinct stores,
    // buffers, and launch states.
    width = 64;
    break;
  default:
    break;
  }
  if (!width && expression.getKind() == TensorSemanticKind::ApplyNary)
    width = 64;
  if (!width)
    return std::nullopt;

  auto payloadInteger = [&]() -> std::optional<llvm::APInt> {
    auto attribute = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
    if (!attribute)
      return std::nullopt;
    return resizeUnsigned(attribute.getValue(), *width);
  };
  auto evaluateOperands = [&]() {
    SmallVector<llvm::APInt, 4> values;
    for (const TensorSemanticExpr &operand : expression.getOperands()) {
      std::optional<llvm::APInt> value =
          evaluateExpression(operand, context);
      if (!value)
        return SmallVector<llvm::APInt, 4>{};
      values.push_back(std::move(*value));
    }
    return values;
  };

  switch (expression.getKind()) {
  case TensorSemanticKind::Integer:
  case TensorSemanticKind::Boolean:
    return payloadInteger();
  case TensorSemanticKind::Unobserved:
    // Fingerprints totalize the single semantic bottom value uniformly.  The
    // corresponding Value/Memory/Launch/Buffer validity predicate decides
    // whether this sample is observable; choosing zero here merely lets guard
    // formulas and inactive masked transitions be evaluated without assigning
    // operation-specific meaning to an undefined value.
    return llvm::APInt(*width, 0);
  case TensorSemanticKind::Symbol: {
    if (auto name = dyn_cast_or_null<StringAttr>(expression.getPayload());
        name && name.getValue().starts_with("extent-")) {
      auto extent = context.extentValues.find(name.getValue());
      if (extent != context.extentValues.end())
        return resizeUnsigned(llvm::APInt(64, extent->second), *width);
    }
    std::optional<uint64_t> symbolKey = getPayloadSymbolKey(expression);
    if (!symbolKey)
      return std::nullopt;
    uint64_t sample = mix64(
        UINT64_C(0x9e3779b97f4a7c15) * *symbolKey ^
        (UINT64_C(0xd1b54a32d192ed03) * (context.model + 1)));
    // Extent symbols are positive, bounded samples in the executable model.
    // Their symbolic identity remains in symbolKey, so the exact proof stage
    // still distinguishes unrelated dynamic dimensions.
    if (auto name = dyn_cast_or_null<StringAttr>(expression.getPayload());
        name && name.getValue().starts_with("extent-"))
      sample = 1 + (sample % 8);
    // Index-typed symbols commonly model dynamic tensor coordinates, offsets,
    // and strides.  Sampling them over the full machine word makes virtually
    // every bounded tensor access invalid; fingerprints then observe only
    // undefinedness and collapse unrelated dynamic graphs into one bucket.
    // Exercise a small coordinate domain instead.  This remains a prefilter:
    // the exact oracle still proves every surviving collision.
    else if (isa<IndexType>(expression.getType()))
      sample %= 8;
    // Interleave full-width pseudo-random models with a deterministic integer
    // boundary domain. Equality, range checks, histograms and masks often
    // distinguish graphs only at values such as 0 or 1; drawing every i32/i64
    // sample from the full bit space would almost never exercise those points.
    // Odd models retain full-width coverage for high-bit behavior.
    else if (isa<IntegerType>(expression.getType()) &&
             (context.model & 1) == 0) {
      uint64_t symbolOrdinal = getSamplingOrdinal(expression, *symbolKey);
      return integerStructuredSample(*width, context.model / 2,
                                     symbolOrdinal);
    }
    else if (isa<IntegerType>(expression.getType()) &&
             getTypedEqualityBit(context.model)) {
      uint64_t symbolOrdinal = getSamplingOrdinal(expression, *symbolKey);
      return integerTypedEqualitySample(
          *width, *getTypedEqualityBit(context.model),
          getIntegerTypeClass(expression.getType()),
          symbolOrdinal);
    }
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::BoundIndex: {
    std::optional<unsigned> binder = getPayloadOrdinal(expression);
    if (!binder)
      return std::nullopt;
    for (auto binding = context.boundIndices.rbegin();
         binding != context.boundIndices.rend(); ++binding)
      if (binding->first == *binder)
        return resizeUnsigned(binding->second, *width);
    return std::nullopt;
  }
  case TensorSemanticKind::Element:
  case TensorSemanticKind::Defined: {
    std::optional<uint64_t> symbolKey = getPayloadSymbolKey(expression);
    if (!symbolKey)
      return std::nullopt;
    SmallVector<llvm::APInt, 4> indices = evaluateOperands();
    if (indices.size() != expression.getOperands().size())
      return std::nullopt;
    uint64_t sample = mix64(
        UINT64_C(0xa0761d6478bd642f) * *symbolKey ^
        (UINT64_C(0xe7037ed1a0b428db) * (context.model + 1)));
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      sample ^= index.getLimitedValue() +
                UINT64_C(0x9e3779b97f4a7c15) * (dimension + 1);
      sample = mix64(sample);
    }
    if (expression.getKind() == TensorSemanticKind::Defined)
      return llvm::APInt(1, (sample & 7) != 0);
    if (isa<IntegerType>(expression.getType()) &&
        (context.model & 1) == 0) {
      uint64_t laneOrdinal = 0;
      for (auto [dimension, index] : llvm::enumerate(indices)) {
        // Every axis receives an odd coefficient.  Even coefficients make an
        // i1 tensor constant along that axis after truncation and can hide
        // scan/reduction counterexamples on interior lanes.
        laneOrdinal += (2 * dimension + 1) * index.getLimitedValue();
      }
      uint64_t symbolOrdinal = getSamplingOrdinal(expression, *symbolKey);
      // Operand count is tensor rank for Element.  Include it in the sampled
      // identity so unrelated rank-1/rank-2 inputs do not receive the same
      // aggregate pattern merely because their symbol ids match.
      uint64_t tensorOrdinal =
          symbolOrdinal + (indices.empty() ? 0 : indices.size() - 1);
      if (*width == 1) {
        return booleanTensorStructuredSample(
            context.model / 2, tensorOrdinal, laneOrdinal);
      }
      return integerTensorAggregateSample(
          *width, context.model / 2, tensorOrdinal, laneOrdinal);
    }
    if (isa<IntegerType>(expression.getType()) &&
        getTypedEqualityBit(context.model)) {
      uint64_t laneOrdinal = 0;
      for (auto [dimension, index] : llvm::enumerate(indices))
        laneOrdinal += (2 * dimension + 1) * index.getLimitedValue();
      uint64_t symbolOrdinal = getSamplingOrdinal(expression, *symbolKey);
      return integerTypedEqualitySample(
          *width, *getTypedEqualityBit(context.model),
          getIntegerTypeClass(expression.getType()),
          symbolOrdinal, laneOrdinal);
    }
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::Sum:
  case TensorSemanticKind::Product:
  case TensorSemanticKind::ForAll: {
    ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
    if (operands.size() != 4 ||
        operands[0].getKind() != TensorSemanticKind::BoundIndex)
      return std::nullopt;
    std::optional<unsigned> binder = getPayloadOrdinal(operands[0]);
    std::optional<unsigned> binderWidth =
        getScalarBitWidth(operands[0].getType());
    std::optional<llvm::APInt> lower =
        evaluateExpression(operands[1], context);
    std::optional<llvm::APInt> upper =
        evaluateExpression(operands[2], context);
    if (!binder || !binderWidth || !lower || !upper ||
        lower->getBitWidth() > 64 || upper->getBitWidth() > 64)
      return std::nullopt;
    int64_t begin = lower->getSExtValue();
    int64_t end = upper->getSExtValue();
    if (end < begin || end - begin > 64)
      return std::nullopt;
    llvm::APInt accumulator(
        *width, expression.getKind() == TensorSemanticKind::Sum ? 0 : 1);
    for (int64_t index = begin; index < end; ++index) {
      context.boundIndices.emplace_back(
          *binder, wrapUnsigned(*binderWidth, static_cast<uint64_t>(index)));
      std::optional<llvm::APInt> body =
          evaluateExpression(operands[3], context);
      context.boundIndices.pop_back();
      if (!body)
        return std::nullopt;
      if (expression.getKind() == TensorSemanticKind::ForAll)
        accumulator = llvm::APInt(1, !accumulator.isZero() && !body->isZero());
      else if (expression.getKind() == TensorSemanticKind::Product)
        accumulator *= resizeUnsigned(*body, *width);
      else
        accumulator += resizeUnsigned(*body, *width);
    }
    return accumulator;
  }
  case TensorSemanticKind::Not:
  case TensorSemanticKind::And:
  case TensorSemanticKind::Or:
  case TensorSemanticKind::Xor:
  case TensorSemanticKind::BitwiseAnd:
  case TensorSemanticKind::BitwiseOr:
  case TensorSemanticKind::BitwiseXor:
  case TensorSemanticKind::Equal:
  case TensorSemanticKind::Add:
  case TensorSemanticKind::Subtract:
  case TensorSemanticKind::Multiply:
  case TensorSemanticKind::SignedMultiplyHigh:
  case TensorSemanticKind::UnsignedMultiplyHigh:
  case TensorSemanticKind::IntegerDivide:
  case TensorSemanticKind::Select:
  case TensorSemanticKind::ZeroExtend:
  case TensorSemanticKind::SignExtend:
  case TensorSemanticKind::Truncate:
  case TensorSemanticKind::ShiftLeft:
  case TensorSemanticKind::ShiftRightLogical:
  case TensorSemanticKind::ShiftRightArithmetic:
  case TensorSemanticKind::SignedRemainder:
  case TensorSemanticKind::SignedLessThan:
  case TensorSemanticKind::UnsignedLessThan:
  case TensorSemanticKind::PointerAdd:
  case TensorSemanticKind::MemoryRead:
  case TensorSemanticKind::MemoryValid:
  case TensorSemanticKind::MemoryWrite:
  case TensorSemanticKind::MemoryAtomic:
  case TensorSemanticKind::MemoryObservable:
  case TensorSemanticKind::PartitionableMemory:
  case TensorSemanticKind::BufferRead:
  case TensorSemanticKind::BufferValid:
  case TensorSemanticKind::BufferWrite:
  case TensorSemanticKind::BufferFill:
  case TensorSemanticKind::BufferView:
  case TensorSemanticKind::BufferAllocate:
  case TensorSemanticKind::ProgramId:
  case TensorSemanticKind::NumPrograms:
  case TensorSemanticKind::LaunchBound:
  case TensorSemanticKind::LaunchRemap:
  case TensorSemanticKind::KernelEffect:
  case TensorSemanticKind::KernelPartition:
  case TensorSemanticKind::ApplyUnary:
  case TensorSemanticKind::ApplyBinary:
  case TensorSemanticKind::ApplyTernary:
  case TensorSemanticKind::ApplyNary:
  case TensorSemanticKind::MicroscaleDecode:
    break;
  default:
    return std::nullopt;
  }

  // Floating-point inputs/elements are sampled by deterministic bit pattern.
  // Arithmetic over floats is not interpreted as integer arithmetic; future
  // floating semantic kinds must add their own IEEE/fast-math evaluator.
  if (isa<FloatType>(expression.getType()) &&
      expression.getKind() != TensorSemanticKind::Select &&
      expression.getKind() != TensorSemanticKind::MemoryRead &&
      expression.getKind() != TensorSemanticKind::BufferRead &&
      expression.getKind() != TensorSemanticKind::ApplyUnary &&
      expression.getKind() != TensorSemanticKind::ApplyBinary &&
      expression.getKind() != TensorSemanticKind::ApplyTernary &&
      expression.getKind() != TensorSemanticKind::ApplyNary &&
      expression.getKind() != TensorSemanticKind::MicroscaleDecode)
    return std::nullopt;

  // Select is intentionally lazy. In particular, an inactive masked load has
  // an Unobserved value arm; evaluating that arm would turn an otherwise
  // concrete fingerprint into a wildcard even though Definedness has already
  // proved the arm irrelevant.
  if (expression.getKind() == TensorSemanticKind::Select) {
    ArrayRef<TensorSemanticExpr> selected = expression.getOperands();
    if (selected.size() != 3)
      return std::nullopt;
    std::optional<llvm::APInt> condition =
        evaluateExpression(selected[0], context);
    if (!condition)
      return std::nullopt;
    std::optional<llvm::APInt> chosen = evaluateExpression(
        condition->isZero() ? selected[2] : selected[1], context);
    return chosen ? std::optional<llvm::APInt>(
                        resizeUnsigned(*chosen, *width))
                  : std::nullopt;
  }

  if (expression.getKind() == TensorSemanticKind::ProgramId) {
    auto axis = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
    if (!axis || expression.getOperands().size() != 1)
      return std::nullopt;
    std::optional<llvm::APInt> launch =
        evaluateExpression(expression.getOperands().front(), context);
    if (!launch)
      return std::nullopt;
    uint64_t sample = mix64(
        UINT64_C(0x8ebc6af09c88c6e3) *
        (static_cast<uint64_t>(axis.getInt()) + 1) ^
        launch->getLimitedValue() ^ (context.model + 1));
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  if (expression.getKind() == TensorSemanticKind::NumPrograms) {
    auto axis = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
    if (!axis || expression.getOperands().size() != 1)
      return std::nullopt;
    std::optional<llvm::APInt> launch =
        evaluateExpression(expression.getOperands().front(), context);
    if (!launch)
      return std::nullopt;
    uint64_t sample = mix64(
        UINT64_C(0xc2b2ae3d27d4eb4f) *
            (static_cast<uint64_t>(axis.getInt()) + 1) ^
        launch->getLimitedValue() ^ (context.model + 1));
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }

  if (expression.getKind() == TensorSemanticKind::BufferRead &&
      expression.getOperands().size() == 2) {
    // Interpret the functional buffer algebra before hashing operands.  A
    // buffer handle is an imported tensor symbol plus concrete coordinates,
    // not the sampled data stored at that coordinate.  This makes the
    // fingerprint a sound prefilter for finite read-after-write proofs.
    std::function<std::optional<std::string>(const TensorSemanticExpr &)>
        bufferHandleKey = [&](const TensorSemanticExpr &handle)
        -> std::optional<std::string> {
      if (handle.getKind() != TensorSemanticKind::Element)
        return std::nullopt;
      std::string key;
      llvm::raw_string_ostream stream(key);
      stream << handle.getPayload() << '[';
      for (const TensorSemanticExpr &index : handle.getOperands()) {
        std::optional<llvm::APInt> value =
            evaluateExpression(index, context);
        if (!value)
          return std::nullopt;
        value->print(stream, /*isSigned=*/false);
        stream << ',';
      }
      stream << ']';
      return stream.str();
    };
    const TensorSemanticExpr &readHandle = expression.getOperands()[1];
    std::optional<std::string> readKey = bufferHandleKey(readHandle);
    TensorSemanticExpr state = expression.getOperands()[0];
    if (readKey) {
      while ((state.getKind() == TensorSemanticKind::BufferWrite ||
              state.getKind() == TensorSemanticKind::BufferFill) &&
             state.getOperands().size() == 3) {
        ArrayRef<TensorSemanticExpr> update = state.getOperands();
        if (state.getKind() == TensorSemanticKind::BufferFill) {
          if (expression.getPayload() == state.getPayload() &&
              readHandle.structurallyEquals(update[1]))
            return evaluateExpression(update[2], context);
          break;
        }
        std::optional<std::string> writeKey = bufferHandleKey(update[1]);
        if (!writeKey)
          break;
        if (*writeKey == *readKey)
          return evaluateExpression(update[2], context);
        state = update[0];
      }
    }
  }

  if (expression.getKind() == TensorSemanticKind::BufferValid &&
      expression.getOperands().size() == 2) {
    TensorSemanticExpr state = expression.getOperands()[0];
    while ((state.getKind() == TensorSemanticKind::BufferWrite ||
            state.getKind() == TensorSemanticKind::BufferFill) &&
           state.getOperands().size() == 3)
      state = state.getOperands()[0];
    if (!state.structurallyEquals(expression.getOperands()[0]))
      return evaluateExpression(
          TensorSemanticExpr::operation(TensorSemanticKind::BufferValid,
                                        expression.getType(),
                                        {state, expression.getOperands()[1]}),
          context);
  }

  if (expression.getKind() == TensorSemanticKind::BufferFill &&
      expression.getOperands().size() == 3) {
    // The middle operand is a coordinate template whose BoundIndex values are
    // quantified by BufferFill.  Do not evaluate those binders as free
    // variables.  Hash the canonical template/region descriptor and evaluate
    // only the incoming state and scalar fill value.  A fingerprint remains a
    // prefilter; the exact normalizer/oracle still supplies the proof.
    std::optional<llvm::APInt> state =
        evaluateExpression(expression.getOperands()[0], context);
    std::optional<llvm::APInt> value =
        evaluateExpression(expression.getOperands()[2], context);
    if (!state || !value)
      return std::nullopt;
    uint64_t sample = mix64(
        state->getLimitedValue() ^
        (UINT64_C(0x9e3779b97f4a7c15) * value->getLimitedValue()) ^
        (UINT64_C(0xd6e8feb86659fd93) * (context.model + 1)));
    std::string region;
    llvm::raw_string_ostream stream(region);
    expression.getOperands()[1].print(stream);
    stream << expression.getPayload();
    stream.flush();
    for (unsigned char byte : region)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }

  SmallVector<llvm::APInt, 4> operands = evaluateOperands();
  if (operands.size() != expression.getOperands().size())
    return std::nullopt;
  auto unary = [&]() { return operands.size() == 1; };
  auto binary = [&]() { return operands.size() == 2; };
  switch (expression.getKind()) {
  case TensorSemanticKind::Not:
    if (!unary())
      return std::nullopt;
    return ~resizeUnsigned(operands[0], *width);
  case TensorSemanticKind::And:
  case TensorSemanticKind::BitwiseAnd:
    if (operands.empty())
      return std::nullopt;
    {
      llvm::APInt value = llvm::APInt::getAllOnes(*width);
      for (const llvm::APInt &operand : operands)
        value &= resizeUnsigned(operand, *width);
      return value;
    }
  case TensorSemanticKind::Or:
  case TensorSemanticKind::BitwiseOr:
    if (operands.empty())
      return std::nullopt;
    {
      llvm::APInt value(*width, 0);
      for (const llvm::APInt &operand : operands)
        value |= resizeUnsigned(operand, *width);
      return value;
    }
  case TensorSemanticKind::Xor:
  case TensorSemanticKind::BitwiseXor:
    if (operands.empty())
      return std::nullopt;
    {
      llvm::APInt value(*width, 0);
      for (const llvm::APInt &operand : operands)
        value ^= resizeUnsigned(operand, *width);
      return value;
    }
  case TensorSemanticKind::Equal:
    if (!binary())
      return std::nullopt;
    return llvm::APInt(1, operands[0] == operands[1]);
  case TensorSemanticKind::Add:
  case TensorSemanticKind::PointerAdd:
    if (!binary())
      return std::nullopt;
    return resizeUnsigned(operands[0], *width) +
           resizeUnsigned(operands[1], *width);
  case TensorSemanticKind::Subtract:
    if (!binary())
      return std::nullopt;
    return resizeUnsigned(operands[0], *width) -
           resizeUnsigned(operands[1], *width);
  case TensorSemanticKind::Multiply:
    if (!binary())
      return std::nullopt;
    return resizeUnsigned(operands[0], *width) *
           resizeUnsigned(operands[1], *width);
  case TensorSemanticKind::SignedMultiplyHigh:
  case TensorSemanticKind::UnsignedMultiplyHigh: {
    if (!binary())
      return std::nullopt;
    llvm::APInt lhs = resizeUnsigned(operands[0], *width);
    llvm::APInt rhs = resizeUnsigned(operands[1], *width);
    const unsigned wideWidth = *width * 2;
    llvm::APInt wideLhs =
        expression.getKind() == TensorSemanticKind::SignedMultiplyHigh
            ? lhs.sext(wideWidth)
            : lhs.zext(wideWidth);
    llvm::APInt wideRhs =
        expression.getKind() == TensorSemanticKind::SignedMultiplyHigh
            ? rhs.sext(wideWidth)
            : rhs.zext(wideWidth);
    return (wideLhs * wideRhs).lshr(*width).trunc(*width);
  }
  case TensorSemanticKind::IntegerDivide:
    if (!binary())
      return std::nullopt;
    // Value and Valid are separate semantic domains. The value of an invalid
    // lane is unobservable, but nested expressions still need a deterministic
    // total interpretation so their enclosing validity formula can evaluate.
    // Definedness carries the divide-by-zero fact; zero is only the chosen
    // representative for the otherwise unconstrained Value.
    if (operands[1].isZero())
      return llvm::APInt(*width, 0);
    return resizeUnsigned(operands[0], *width)
        .udiv(resizeUnsigned(operands[1], *width));
  case TensorSemanticKind::Select:
    llvm_unreachable("select was evaluated lazily above");
  case TensorSemanticKind::ZeroExtend:
    if (!unary())
      return std::nullopt;
    return resizeUnsigned(operands[0], *width);
  case TensorSemanticKind::SignExtend:
    if (!unary())
      return std::nullopt;
    return resizeSigned(operands[0], *width);
  case TensorSemanticKind::Truncate:
    if (!unary())
      return std::nullopt;
    return operands[0].trunc(*width);
  case TensorSemanticKind::ShiftLeft:
  case TensorSemanticKind::ShiftRightLogical:
  case TensorSemanticKind::ShiftRightArithmetic: {
    if (!binary())
      return std::nullopt;
    llvm::APInt lhs = resizeUnsigned(operands[0], *width);
    uint64_t amount = operands[1].getLimitedValue();
    // As with division, an out-of-range shift has Valid=false. Totalize its
    // unobservable Value so a parent validity expression can still be sampled
    // without turning a supported graph into a wildcard.
    if (amount >= *width)
      return llvm::APInt(*width, 0);
    if (expression.getKind() == TensorSemanticKind::ShiftLeft)
      return lhs.shl(amount);
    return expression.getKind() == TensorSemanticKind::ShiftRightLogical
               ? lhs.lshr(amount)
               : lhs.ashr(amount);
  }
  case TensorSemanticKind::SignedRemainder: {
    if (!binary())
      return std::nullopt;
    llvm::APInt lhs = resizeUnsigned(operands[0], *width);
    llvm::APInt rhs = resizeUnsigned(operands[1], *width);
    return rhs.isZero() ? llvm::APInt(*width, 0) : lhs.srem(rhs);
  }
  case TensorSemanticKind::SignedLessThan:
    if (!binary())
      return std::nullopt;
    return llvm::APInt(1, operands[0].slt(operands[1]));
  case TensorSemanticKind::UnsignedLessThan:
    if (!binary())
      return std::nullopt;
    return llvm::APInt(1, operands[0].ult(operands[1]));
  case TensorSemanticKind::MemoryRead: {
    if (operands.size() != 2)
      return std::nullopt;
    uint64_t sample = mix64(
        operands[0].getLimitedValue() ^
        (UINT64_C(0x9e3779b97f4a7c15) *
         operands[1].getLimitedValue()) ^
        (UINT64_C(0xd6e8feb86659fd93) * (context.model + 1)));
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::MemoryValid: {
    if (!binary())
      return std::nullopt;
    uint64_t sample = mix64(
        operands[0].getLimitedValue() ^
        (UINT64_C(0x9e3779b97f4a7c15) *
         operands[1].getLimitedValue()) ^
        (UINT64_C(0xa5a3564e27f8862f) * (context.model + 1)));
    return llvm::APInt(1, (sample & 7) != 0);
  }
  case TensorSemanticKind::MemoryWrite:
  case TensorSemanticKind::MemoryAtomic: {
    if (operands.size() != 3 && operands.size() != 4)
      return std::nullopt;
    bool masked = expression.getKind() == TensorSemanticKind::MemoryWrite;
    if (expression.getKind() == TensorSemanticKind::MemoryAtomic)
      if (auto payload =
              dyn_cast_or_null<DictionaryAttr>(expression.getPayload()))
        masked = static_cast<bool>(payload.get("masked"));
    if (masked && operands.size() == 4 && operands[3].isZero())
      return resizeUnsigned(operands[0], *width);
    uint64_t sample = operands[0].getLimitedValue();
    for (auto [index, operand] :
         llvm::enumerate(ArrayRef<llvm::APInt>(operands).drop_front()))
      sample = mix64(sample ^ operand.getLimitedValue() ^
                     (UINT64_C(0x94d049bb133111eb) * (index + 1)));
    // A concrete memory transition includes access width/address space and,
    // for atomics, the operation/ordering payload.  Two typed stores can have
    // identical sampled bit values while updating a different byte range;
    // omitting this metadata makes the prefilter claim a false collision.
    auto mixPrinted = [&](auto value) {
      std::string printed;
      llvm::raw_string_ostream stream(printed);
      stream << value;
      stream.flush();
      for (unsigned char byte : printed)
        sample = mix64(sample ^ byte);
    };
    for (const TensorSemanticExpr &operandExpr : expression.getOperands())
      mixPrinted(operandExpr.getType());
    mixPrinted(expression.getPayload());
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::MemoryObservable: {
    if (operands.empty())
      return std::nullopt;
    uint64_t sample = UINT64_C(0x6f627365727665) ^
                      operands.front().getLimitedValue();
    for (auto [index, operand] :
         llvm::enumerate(ArrayRef<llvm::APInt>(operands).drop_front()))
      sample = mix64(sample ^ operand.getLimitedValue() ^
                     (UINT64_C(0x9e3779b97f4a7c15) * (index + 1)));
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::PartitionableMemory: {
    if (!operands.empty())
      return std::nullopt;
    uint64_t sample = UINT64_C(0x706172746d656d);
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::BufferRead:
  case TensorSemanticKind::BufferValid:
  case TensorSemanticKind::BufferWrite:
  case TensorSemanticKind::BufferFill:
  case TensorSemanticKind::BufferView:
  case TensorSemanticKind::BufferAllocate: {
    if (operands.empty())
      return std::nullopt;
    uint64_t sample = UINT64_C(0x62ff0000) ^
                      static_cast<uint64_t>(expression.getKind());
    for (auto [index, operand] : llvm::enumerate(operands))
      sample = mix64(sample ^ operand.getLimitedValue() ^
                     (UINT64_C(0x9e3779b97f4a7c15) * (index + 1)));
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::LaunchBound:
  case TensorSemanticKind::LaunchRemap: {
    const unsigned expectedOperands =
        expression.getKind() == TensorSemanticKind::LaunchBound ? 2 : 1;
    if (operands.size() != expectedOperands)
      return std::nullopt;
    uint64_t sample = UINT64_C(0x6c61756e6368) ^
                      operands.front().getLimitedValue();
    if (operands.size() == 2)
      sample = mix64(sample ^ operands[1].getLimitedValue());
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::KernelEffect: {
    if (operands.size() != 2)
      return std::nullopt;
    uint64_t sample = mix64(UINT64_C(0x6b65726e656c) ^
                            operands[0].getLimitedValue());
    sample = mix64(sample ^ operands[1].getLimitedValue());
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::KernelPartition: {
    if (operands.size() != 2)
      return std::nullopt;
    uint64_t sample = mix64(UINT64_C(0x706172746974696f) ^
                            operands[0].getLimitedValue());
    sample = mix64(sample ^ operands[1].getLimitedValue());
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload)
      sample = mix64(sample ^ byte);
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  case TensorSemanticKind::ApplyUnary:
  case TensorSemanticKind::ApplyBinary:
  case TensorSemanticKind::ApplyTernary:
  case TensorSemanticKind::ApplyNary:
  case TensorSemanticKind::MicroscaleDecode: {
    const unsigned expectedArity =
        expression.getKind() == TensorSemanticKind::ApplyUnary
            ? 1
            : (expression.getKind() == TensorSemanticKind::ApplyBinary
                   ? 2
                   : (expression.getKind() == TensorSemanticKind::ApplyTernary
                          ? 3
                          : operands.size()));
    if (operands.size() != expectedArity)
      return std::nullopt;
    if (expression.getKind() == TensorSemanticKind::ApplyUnary) {
      auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
      auto role = payload
                      ? dyn_cast_or_null<StringAttr>(
                            payload.get("semantic_role"))
                      : StringAttr{};
      // mixed_slice_value uses this node only to coerce an index-typed
      // dynamic coordinate to the canonical result-index type.  Treating it
      // as an opaque unary operation hashes a small legal offset into an
      // arbitrary i32, making practically every sampled slice out of bounds.
      if (role && role.getValue() == "index_coordinate")
        return resizeSigned(operands.front(), *width);
      // This is a proof-language primitive, not an operation-specific
      // special case. MLIR shifts are defined only when the unsigned shift
      // amount is smaller than the element bit width. Interpreting the guard
      // exactly also prevents the value evaluator from trying an invalid
      // shift on samples for which Definedness should be false.
      if (role && role.getValue() == "shift_amount_valid") {
        std::optional<unsigned> operandWidth =
            getScalarBitWidth(expression.getOperands().front().getType());
        if (!operandWidth)
          return std::nullopt;
        return llvm::APInt(1,
                           operands.front().getLimitedValue() < *operandWidth);
      }
    }
    uint64_t sample = UINT64_C(14695981039346656037);
    std::string payload;
    llvm::raw_string_ostream payloadStream(payload);
    payloadStream << expression.getPayload();
    payloadStream.flush();
    for (unsigned char byte : payload) {
      sample ^= byte;
      sample *= UINT64_C(1099511628211);
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      std::string operandType;
      llvm::raw_string_ostream typeStream(operandType);
      typeStream << expression.getOperands()[index].getType();
      typeStream.flush();
      sample = mix64(sample ^ stableHash64(operandType) ^
                     operand.getLimitedValue() ^
                     (UINT64_C(0x9e3779b97f4a7c15) * (index + 1)));
    }
    // Apply* denotes a typed uninterpreted function. Its pseudo-value is
    // narrowed to the MLIR result width below, so a fixed hash would make an
    // accidental i1/i8 collision persist in every model. Independent model
    // salts form a deterministic universal-hash family: equal typed terms
    // still agree in every model, while distinct opaque terms are not trapped
    // by one low-bit collision. No operation name is interpreted here.
    sample = mix64(sample ^
                   (UINT64_C(0xd6e8feb86659fd93) * (context.model + 1)));
    return resizeUnsigned(llvm::APInt(64, sample), *width);
  }
  default:
    return std::nullopt;
  }
}

static std::optional<std::string>
fingerprintDenotation(const TensorDenotation &denotation, unsigned firstModel,
                       unsigned modelCount, std::string &reason) {
  std::string signature;
  llvm::raw_string_ostream stream(signature);
  stream << "rank=" << denotation.shape.size() << '|'
         << denotation.value.getType() << "|layout="
         << denotation.layoutIdentity << '|';
  for (unsigned offset = 0; offset < modelCount; ++offset) {
    unsigned model = firstModel + offset;
    EvaluationContext context{model, {}, {}};
    SmallVector<int64_t, 4> concreteShape;
    SmallVector<unsigned, 4> resultIndexWidths;
    stream << "shape[";
    for (auto [dimension, size] : llvm::enumerate(denotation.shape)) {
      int64_t concreteSize = size;
      if (ShapedType::isDynamic(size)) {
        StringRef shapeSymbol =
            dimension < denotation.shapeSymbols.size()
                ? StringRef(denotation.shapeSymbols[dimension])
                : StringRef();
        const bool hasStableSymbol = !shapeSymbol.empty() &&
                                     shapeSymbol != "?" &&
                                     shapeSymbol != "!conflict";
        if (hasStableSymbol) {
          std::string extentName = (llvm::Twine("extent-") + shapeSymbol).str();
          auto known = context.extentValues.find(extentName);
          if (known == context.extentValues.end()) {
            uint64_t symbolKey = stableHash64(shapeSymbol);
            static constexpr int64_t extentSchedule[] = {4, 1, 2, 3,
                                                         5, 8, 16};
            concreteSize =
                model % 8 < std::size(extentSchedule)
                    ? extentSchedule[model % 8]
                    : 2 + (mix64(symbolKey ^ model) % 15);
            context.extentValues.try_emplace(extentName, concreteSize);
          } else {
            concreteSize = known->second;
          }
        } else {
          static constexpr int64_t extentSchedule[] = {4, 1, 2, 3,
                                                       5, 8, 16};
          concreteSize =
              model % 8 < std::size(extentSchedule)
                  ? extentSchedule[model % 8]
                  : 2 + ((model + dimension) % 15);
        }
      }
      if (concreteSize <= 0) {
        reason = "invalid sampled tensor shape";
        return std::nullopt;
      }
      concreteShape.push_back(concreteSize);
      stream << concreteSize << ',';
      if (dimension >= denotation.resultIndices.size() ||
          denotation.resultIndices[dimension].getKind() !=
              TensorSemanticKind::BoundIndex) {
        reason = "unsupported result index expression";
        return std::nullopt;
      }
      std::optional<unsigned> binder =
          getPayloadOrdinal(denotation.resultIndices[dimension]);
      std::optional<unsigned> binderWidth =
          getScalarBitWidth(denotation.resultIndices[dimension].getType());
      if (!binder || !binderWidth) {
        reason = "unsupported result index binder";
        return std::nullopt;
      }
      resultIndexWidths.push_back(*binderWidth);
      // Pair each structured even model with the following typed/random model
      // at one output coordinate.  The first eight pairs keep that coordinate
      // fixed so every aggregate input row is exercised at the same bin.  The
      // next eight use a four-bit reversal permutation: row zero observes lane
      // one, and the remaining rows spread over interior and final lanes of a
      // length-16 tensor.  This preserves truth-table coverage for reductions
      // while exposing prefix/suffix differences without OP-specific logic.
      uint64_t observationOrdinal = model / 2;
      uint64_t lane = observationOrdinal & 15;
      uint64_t bitReversed = ((lane & 1) << 3) | ((lane & 2) << 1) |
                             ((lane & 4) >> 1) | ((lane & 8) >> 3);
      uint64_t coordinateSeed =
          observationOrdinal < 8
              ? 0
              : bitReversed + 2 * (observationOrdinal / 16);
      // Keep the aggregate truth-table phase at the origin in every axis.
      // For later spatial phases alternate two dimension-dependent salts.
      // Either fixed parity makes one of axis-0/axis-1 prefix or suffix pairs
      // preserve parity on square tensors; alternating phases covers both.
      uint64_t spatialPhase = (observationOrdinal / 8) & 1u;
      uint64_t coordinate = observationOrdinal < 8
                                ? 0
                                : (coordinateSeed * (2 * dimension + 1) +
                                   dimension * (dimension + spatialPhase)) %
                                      concreteSize;
      context.boundIndices.emplace_back(
          *binder, wrapUnsigned(*binderWidth, coordinate));
      stream << '@' << coordinate << ',';
    }
    stream << "]";
    std::optional<llvm::APInt> defined =
        evaluateExpression(denotation.defined, context);
    if (!defined) {
      reason = "unsupported definedness expression";
      return std::nullopt;
    }
    if (defined->isZero()) {
      stream << "U;";
    } else {
      std::optional<llvm::APInt> value =
          evaluateExpression(denotation.value, context);
      if (!value) {
        reason = "unsupported value expression";
        return std::nullopt;
      }
      stream << "D:";
      value->print(stream, /*isSigned=*/false);
      stream << ';';
    }

    // Sparse and prefix-like tensor functions can agree at one sampled output
    // coordinate even under rich input assignments.  On a small bounded
    // result, append a full extensional digest for a few structured models.
    // This is a generic finite-function observation: it depends only on shape
    // and Value/Definedness, not on the operation that produced the tensor.
    // Primary fingerprints use the bounded structured rows 16..30. Adaptive
    // counterexample search continues with later even structured rows; keep
    // the same generic full-output observation there as well.
    const bool digestModel = model == 0 || (model >= 16 && (model & 1) == 0);
    int64_t digestElementCount = 1;
    bool digestIsBounded = !concreteShape.empty();
    for (int64_t extent : concreteShape) {
      if (extent <= 0 || digestElementCount > 16 / extent) {
        digestIsBounded = false;
        break;
      }
      digestElementCount *= extent;
    }
    if (digestModel && digestIsBounded && digestElementCount > 1) {
      stream << "tensor_digest[";
      for (int64_t linear = 0; linear < digestElementCount; ++linear) {
        int64_t remainder = linear;
        for (int64_t dimension =
                 static_cast<int64_t>(concreteShape.size()) - 1;
             dimension >= 0; --dimension) {
          int64_t coordinate = remainder % concreteShape[dimension];
          remainder /= concreteShape[dimension];
          context.boundIndices[dimension].second = wrapUnsigned(
              resultIndexWidths[dimension], coordinate);
        }
        std::optional<llvm::APInt> probeDefined =
            evaluateExpression(denotation.defined, context);
        if (!probeDefined) {
          reason = "unsupported tensor-digest definedness expression";
          return std::nullopt;
        }
        if (probeDefined->isZero()) {
          stream << "U,";
          continue;
        }
        std::optional<llvm::APInt> probeValue =
            evaluateExpression(denotation.value, context);
        if (!probeValue) {
          reason = "unsupported tensor-digest value expression";
          return std::nullopt;
        }
        stream << 'D';
        probeValue->print(stream, /*isSigned=*/false);
        stream << ',';
      }
      stream << "];";
    }

    // Value + Definedness alone is insufficient for effectful graphs: two
    // stores can return the same token while producing different memory
    // states.  Value expressions are sampled because the exact oracle can
    // prove non-trivial algebraic equalities there.  State/effect domains are
    // different: the exact oracle accepts them only when their normalized
    // forms are structurally equal.  Preserve that complete normalized form
    // in the fingerprint.  Numerically evaluating an Effect `and` as i1 would
    // otherwise truncate nested 64-bit MemoryAtomic transition hashes and put
    // observably different atomic histories in the same proof bucket.
    auto appendStructuralDomain =
        [&](StringRef label, const TensorSemanticExpr &domain) {
          stream << label << ':';
          if (!domain) {
            stream << "-;";
            return;
          }
          domain.print(stream);
          stream << ';';
        };
    // Effects of operand graphs remain observable even when the root result is
    // undefined (for example, an inner atomic feeding a masked-off outer
    // atomic), so effect is intentionally not gated by result definedness.
    appendStructuralDomain("effect", denotation.effect);
    appendStructuralDomain("memory", denotation.memory);
    appendStructuralDomain("memory_defined", denotation.memoryDefined);
    appendStructuralDomain("launch", denotation.launch);
    appendStructuralDomain("launch_defined", denotation.launchDefined);
    // BufferState is a private, eliminable proof domain for ordinary value
    // roots.  The exact oracle compares it only when either side is a
    // BufferState token; including it for a read value would separate
    // read(copy(src, dst), dst) from read(src) before extensional proof gets a
    // chance to establish their value equality.
    if (denotation.isBufferStateToken) {
      appendStructuralDomain("buffer", denotation.buffer);
      appendStructuralDomain("buffer_defined", denotation.bufferDefined);
    } else {
      stream << "buffer:-;buffer_defined:-;";
    }
    stream << "tokens:" << denotation.isMemoryStateToken
           << denotation.isLaunchStateToken << denotation.isBufferStateToken
           << ';';
  }
  return stableHash(signature);
}

} // namespace

std::optional<std::string>
OfflineSemanticFingerprinter::fingerprintDenotationRange(
    const TensorDenotation &denotation, unsigned firstModel,
    unsigned modelCount, std::string &reason) {
  return fingerprintDenotation(denotation, firstModel, modelCount, reason);
}

bool OfflineSemanticFingerprinter::hasConcreteCounterexample(
    const TensorDenotation &lhs, const TensorDenotation &rhs,
    unsigned firstModel, unsigned additionalModelCount, std::string &reason) {
  if (additionalModelCount == 0)
    return false;
  std::string lhsReason;
  std::string rhsReason;
  std::optional<std::string> lhsFingerprint = fingerprintDenotation(
      lhs, firstModel, additionalModelCount, lhsReason);
  std::optional<std::string> rhsFingerprint = fingerprintDenotation(
      rhs, firstModel, additionalModelCount, rhsReason);
  if (!lhsFingerprint || !rhsFingerprint) {
    reason = !lhsFingerprint ? std::move(lhsReason) : std::move(rhsReason);
    return false;
  }
  if (*lhsFingerprint == *rhsFingerprint)
    return false;
  reason = (llvm::Twine("concrete semantic counterexample in model range [") +
            llvm::Twine(firstModel) + ", " +
            llvm::Twine(firstModel + additionalModelCount) + ")")
               .str();
  return true;
}

uint64_t OfflineSemanticFingerprintResult::getFingerprintedGraphCount() const {
  uint64_t count = 0;
  for (const OfflineSemanticFingerprintBucket &bucket : buckets)
    count += bucket.graphIds.size();
  return count;
}

void OfflineSemanticFingerprintResult::writeJSON(
    llvm::raw_ostream &output, llvm::StringRef universeHash,
    bool enumerationComplete) const {
  // The production all-OP universe has hundreds of thousands of buckets.
  // Stream them directly: materializing json::Array/json::Object and then a
  // second serialized string can add gigabytes to the offline RSS peak.
  llvm::json::OStream json(output, /*IndentSize=*/2);
  json.object([&] {
    json.attribute("universe_hash", universeHash);
    json.attribute("enumeration_complete", enumerationComplete);
    json.attribute("fingerprint_complete", isComplete());
    json.attribute("model_count", static_cast<int64_t>(modelCount));
    json.attribute("fingerprinted_graph_count",
                   static_cast<int64_t>(getFingerprintedGraphCount()));
    json.attribute("wildcard_graph_count",
                   static_cast<int64_t>(wildcardGraphIds.size()));
    json.attribute("wildcard_graph_detail_count",
                   static_cast<int64_t>(wildcardGraphs.size()));
    json.attribute("wildcard_graph_details_truncated",
                   wildcardGraphs.size() < wildcardGraphIds.size());
    json.attribute("bucket_count", static_cast<int64_t>(buckets.size()));
    json.attribute("comparable_pairs_before",
                   static_cast<int64_t>(comparablePairsBefore));
    json.attribute("proof_candidate_pairs_after",
                   static_cast<int64_t>(proofCandidatePairsAfter));
    json.attribute("unique_denotations_fingerprinted",
                   static_cast<int64_t>(uniqueDenotationsFingerprinted));
    json.attribute("denotation_fingerprint_cache_hits",
                   static_cast<int64_t>(denotationFingerprintCacheHits));
    json.attribute("denotation_model_evaluations",
                   static_cast<int64_t>(denotationModelEvaluations));
    json.attribute("denotation_model_evaluations_avoided",
                   static_cast<int64_t>(denotationModelEvaluationsAvoided));
    json.attribute("parallel_jobs", static_cast<int64_t>(parallelJobs));
    json.attributeArray("unsupported_reasons", [&] {
      for (const std::string &reason : unsupportedReasons)
        json.value(reason);
    });
    json.attributeArray("wildcard_graph_ids", [&] {
      for (unsigned graphId : wildcardGraphIds)
        json.value(static_cast<int64_t>(graphId));
    });
    json.attributeArray("wildcard_graphs", [&] {
      for (const OfflineSemanticFingerprintGap &gap : wildcardGraphs) {
        json.object([&] {
          json.attribute("graph_id", static_cast<int64_t>(gap.graphId));
          json.attribute("graph_key", gap.graphKey);
          json.attribute("reason", gap.reason);
          json.attribute("semantic_key", gap.semanticKey);
        });
      }
    });
    json.attributeArray("buckets", [&] {
      for (const OfflineSemanticFingerprintBucket &bucket : buckets) {
        json.object([&] {
          json.attribute("fingerprint", bucket.fingerprint);
          json.attributeArray("graph_ids", [&] {
            for (unsigned graphId : bucket.graphIds)
              json.value(static_cast<int64_t>(graphId));
          });
        });
      }
    });
  });
  output << '\n';
}

std::string OfflineSemanticFingerprintResult::toJSON(
    llvm::StringRef universeHash, bool enumerationComplete) const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream, universeHash, enumerationComplete);
  return storage;
}

OfflineSemanticFingerprintResult OfflineSemanticFingerprinter::fingerprint(
    llvm::ArrayRef<OfflineTypedGraph> graphs,
    const TTOpSemanticsRegistry &semantics,
    const TensorSemanticNormalizer &normalizer) const {
  OfflineSemanticFingerprintResult result;
  result.denotationsNormalized = true;
  result.modelCount = modelCount;
  result.parallelJobs = parallelJobs;
  result.denotations.resize(graphs.size());
  result.exactDenotationGroupIds.assign(
      graphs.size(), std::numeric_limits<unsigned>::max());
  if (modelCount == 0) {
    result.unsupportedReasons.push_back("model count must be positive");
    for (const OfflineTypedGraph &graph : graphs)
      result.wildcardGraphIds.push_back(graph.id);
    return result;
  }

  llvm::StringMap<std::vector<unsigned>> graphsByType;
  llvm::StringMap<std::vector<unsigned>> wildcardsByType;
  llvm::StringSet<> reasons;
  auto recordWildcard = [&](const OfflineTypedGraph &graph,
                            llvm::StringRef typeKey, llvm::StringRef reason,
                            llvm::StringRef semanticKey = {}) {
    reasons.insert(reason);
    result.wildcardGraphIds.push_back(graph.id);
    if (result.wildcardGraphs.size() <
        OfflineSemanticFingerprintResult::MaxWildcardGraphDetails)
      result.wildcardGraphs.push_back({graph.id, graph.canonicalKey,
                                       reason.str(), semanticKey.str()});
    wildcardsByType[typeKey].push_back(graph.id);
  };
  struct DenotationGroup {
    std::string typeKey;
    unsigned representativeGraphId = 0;
    std::vector<unsigned> graphIds;
  };
  std::vector<DenotationGroup> denotationGroups;
  // A hash is only an index. Every matching hash is confirmed by an exact
  // structural comparison, so collisions cannot merge different semantics.
  // This avoids retaining a second full printed denotation for every unique
  // graph in large production universes.
  llvm::DenseMap<uint64_t, std::vector<unsigned>> groupsByDenotationHash;
  llvm::StringMap<std::vector<unsigned>> groupsByType;
  struct DenotationWork {
    std::string typeKey;
    std::string semanticKey;
    std::string reason;
  };
  unsigned maximumOperatorSize = 0;
  for (const OfflineTypedGraph &graph : graphs)
    maximumOperatorSize = std::max(maximumOperatorSize, graph.operatorSize);

  // Operands always come from a smaller operator-size layer. Compute each
  // layer concurrently, then update maps and stable group ids sequentially.
  // This preserves byte-for-byte deterministic output while allowing large
  // offline universes to use the machine allocated to rule generation.
  for (unsigned operatorSize = 0; operatorSize <= maximumOperatorSize;
       ++operatorSize) {
    std::vector<unsigned> layer;
    for (auto [graphIndex, graph] : llvm::enumerate(graphs))
      if (graph.operatorSize == operatorSize)
        layer.push_back(graphIndex);

    // Keep parallel denotation construction, but do not retain the printed
    // keys for an entire production layer. A depth-one layer can contain
    // hundreds of thousands of graphs; fixed-size batches bound temporary
    // string memory without changing stable graph/group order.
    constexpr unsigned denotationBatchSize = 8192;
    for (unsigned batchBegin = 0; batchBegin < layer.size();
         batchBegin += denotationBatchSize) {
      const unsigned batchEnd =
          std::min<unsigned>(layer.size(), batchBegin + denotationBatchSize);
      const unsigned batchCount = batchEnd - batchBegin;
      std::vector<DenotationWork> denotationWork(batchCount);

      parallelForWork(batchCount, parallelJobs, [&](unsigned batchIndex) {
        const OfflineTypedGraph &graph =
            graphs[layer[batchBegin + batchIndex]];
        DenotationWork &work = denotationWork[batchIndex];
        work.typeKey = printType(graph.resultType);
        const TTOpSemantics *definition =
            semantics.lookup(graph.operationName);
        if (!definition) {
          work.reason = "missing semantics for " + graph.operationName;
          return;
        }

        SmallVector<TensorDenotation, 3> operands;
        for (unsigned operand : graph.operands) {
          if (operand >= result.denotations.size() ||
              !result.denotations[operand]) {
            work.reason = "operand denotation unavailable";
            return;
          }
          operands.push_back(*result.denotations[operand]);
        }

        Builder builder(graph.resultType.getContext());
        SmallVector<NamedAttribute, 8> attributes;
        if (graph.attributes)
          attributes.append(graph.attributes.begin(), graph.attributes.end());
        attributes.push_back(builder.getNamedAttr(
            "ags.result_type", TypeAttr::get(graph.resultType)));
        GraphExprNode node = GraphExprNode::create(
            graph.resultType.getContext(), graph.operationName, graph.operands,
            attributes);
        FailureOr<TensorDenotation> denotation =
            definition->denote(node, operands);
        if (failed(denotation)) {
          work.reason = "denotation failed for " + graph.operationName;
          return;
        }
        result.denotations[graph.id] =
            normalizer.normalize(*denotation).denotation;
        work.semanticKey =
            getStableTensorDenotationKey(*result.denotations[graph.id]);
      });

      for (unsigned batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        const OfflineTypedGraph &graph =
            graphs[layer[batchBegin + batchIndex]];
        DenotationWork &work = denotationWork[batchIndex];
        graphsByType[work.typeKey].push_back(graph.id);
        if (!work.reason.empty()) {
          recordWildcard(graph, work.typeKey, work.reason);
          continue;
        }

        uint64_t denotationHash =
            mix64(stableHash64(work.typeKey) ^
                  mix64(stableHash64(work.semanticKey)));
        std::vector<unsigned> &hashGroups =
            groupsByDenotationHash[denotationHash];
        std::optional<unsigned> existingGroup;
        for (unsigned groupId : hashGroups) {
          const DenotationGroup &group = denotationGroups[groupId];
          if (group.typeKey == work.typeKey &&
              denotationsStructurallyEqual(
                  *result.denotations[group.representativeGraphId],
                  *result.denotations[graph.id])) {
            existingGroup = groupId;
            break;
          }
        }
        if (!existingGroup) {
          ++result.uniqueDenotationsFingerprinted;
          DenotationGroup group;
          group.typeKey = work.typeKey;
          group.representativeGraphId = graph.id;
          group.graphIds.push_back(graph.id);
          const unsigned groupId = denotationGroups.size();
          result.exactDenotationGroupIds[graph.id] = groupId;
          groupsByType[work.typeKey].push_back(groupId);
          hashGroups.push_back(groupId);
          denotationGroups.push_back(std::move(group));
        } else {
          ++result.denotationFingerprintCacheHits;
          result.exactDenotationGroupIds[graph.id] = *existingGroup;
          denotationGroups[*existingGroup].graphIds.push_back(graph.id);
        }
      }
    }
  }

  // The exact-lookup index can otherwise remain live during concrete-model
  // evaluation. Batch-local printed keys are already gone here.
  llvm::DenseMap<uint64_t, std::vector<unsigned>>().swap(
      groupsByDenotationHash);

  struct AdaptiveBucket {
    std::string fingerprint;
    std::vector<unsigned> denotationGroupIds;
  };
  std::vector<AdaptiveBucket> completedBuckets;

  // Evaluate model zero for every unique denotation so unsupported semantic
  // kinds still become wildcards. After that, a bucket containing only one
  // semantic normal form cannot possibly be split or joined by later models:
  // its model-prefix already differs from every other normal form of the same
  // result type. Only colliding buckets therefore need the next model.
  for (const auto &typeEntry : groupsByType) {
    AdaptiveBucket initial;
    initial.fingerprint = stableHash(
        (llvm::Twine("type:") + typeEntry.getKey()).str());
    initial.denotationGroupIds = typeEntry.second;
    std::vector<AdaptiveBucket> active;
    active.push_back(std::move(initial));

    for (unsigned model = 0; model < modelCount && !active.empty(); ++model) {
      std::vector<AdaptiveBucket> next;
      struct ModelEvaluation {
        std::optional<std::string> fingerprint;
        std::string reason;
      };
      SmallVector<unsigned, 64> evaluationGroupIds;
      SmallVector<unsigned, 64> evaluationOffsets;
      SmallVector<bool, 64> evaluateBucket;
      evaluationOffsets.reserve(active.size() + 1);
      evaluateBucket.reserve(active.size());
      evaluationOffsets.push_back(0);

      // Evaluate one model across all still-colliding buckets in one parallel
      // batch. Parallelizing each bucket separately loses nearly all
      // concurrency after early models split a production universe into many
      // small buckets, and repeatedly creates worker threads. Flattening does
      // not change bucket or group order, so the deterministic partition and
      // serialized result remain byte-for-byte identical.
      for (AdaptiveBucket &bucket : active) {
        if (model > 0 && bucket.denotationGroupIds.size() == 1) {
          completedBuckets.push_back(std::move(bucket));
          evaluateBucket.push_back(false);
          evaluationOffsets.push_back(evaluationGroupIds.size());
          continue;
        }
        evaluateBucket.push_back(true);
        evaluationGroupIds.append(bucket.denotationGroupIds.begin(),
                                  bucket.denotationGroupIds.end());
        evaluationOffsets.push_back(evaluationGroupIds.size());
      }

      std::vector<ModelEvaluation> evaluations(evaluationGroupIds.size());
      parallelForWork(evaluationGroupIds.size(), parallelJobs,
                      [&](unsigned index) {
                        unsigned groupId = evaluationGroupIds[index];
                        DenotationGroup &group = denotationGroups[groupId];
                        evaluations[index].fingerprint = fingerprintDenotation(
                            *result.denotations[group.representativeGraphId],
                            model, /*modelCount=*/1,
                            evaluations[index].reason);
                      });
      result.denotationModelEvaluations += evaluationGroupIds.size();

      for (auto [bucketIndex, bucket] : llvm::enumerate(active)) {
        if (!evaluateBucket[bucketIndex])
          continue;
        llvm::StringMap<std::vector<unsigned>> partitions;
        for (unsigned index = evaluationOffsets[bucketIndex];
             index < evaluationOffsets[bucketIndex + 1]; ++index) {
          unsigned groupId = evaluationGroupIds[index];
          DenotationGroup &group = denotationGroups[groupId];
          ModelEvaluation &evaluation = evaluations[index];
          if (!evaluation.fingerprint) {
            std::string diagnosticSemanticKey;
            if (result.wildcardGraphs.size() <
                OfflineSemanticFingerprintResult::MaxWildcardGraphDetails)
              diagnosticSemanticKey = getStableTensorDenotationKey(
                  *result.denotations[group.representativeGraphId]);
            for (unsigned graphId : group.graphIds) {
              recordWildcard(graphs[graphId], group.typeKey,
                             evaluation.reason, diagnosticSemanticKey);
            }
            continue;
          }
          partitions[*evaluation.fingerprint].push_back(groupId);
        }

        for (auto &partition : partitions) {
          AdaptiveBucket child;
          child.fingerprint = stableHash(
              (llvm::Twine(bucket.fingerprint) + ":" + partition.getKey())
                  .str());
          child.denotationGroupIds = std::move(partition.second);
          next.push_back(std::move(child));
        }
      }
      active = std::move(next);
    }
    for (AdaptiveBucket &bucket : active)
      completedBuckets.push_back(std::move(bucket));
  }

  const uint64_t eagerModelEvaluations =
      result.uniqueDenotationsFingerprinted * modelCount;
  result.denotationModelEvaluationsAvoided =
      eagerModelEvaluations >= result.denotationModelEvaluations
          ? eagerModelEvaluations - result.denotationModelEvaluations
          : 0;

  llvm::StringMap<std::vector<unsigned>> buckets;
  for (AdaptiveBucket &bucket : completedBuckets) {
    std::vector<unsigned> &graphIds = buckets[bucket.fingerprint];
    for (unsigned groupId : bucket.denotationGroupIds) {
      const DenotationGroup &group = denotationGroups[groupId];
      graphIds.insert(graphIds.end(), group.graphIds.begin(),
                      group.graphIds.end());
    }
  }

  for (const auto &entry : graphsByType)
    result.comparablePairsBefore += chooseTwo(entry.second.size());
  for (const auto &entry : buckets) {
    OfflineSemanticFingerprintBucket bucket;
    bucket.fingerprint = entry.getKey().str();
    bucket.graphIds = entry.second;
    llvm::sort(bucket.graphIds);
    result.proofCandidatePairsAfter += chooseTwo(bucket.graphIds.size());
    result.buckets.push_back(std::move(bucket));
  }
  for (const auto &entry : wildcardsByType) {
    uint64_t wildcardCount = entry.second.size();
    uint64_t typeCount = graphsByType.lookup(entry.getKey()).size();
    result.proofCandidatePairsAfter +=
        chooseTwo(wildcardCount) + wildcardCount * (typeCount - wildcardCount);
  }
  llvm::sort(result.buckets,
             [](const OfflineSemanticFingerprintBucket &lhs,
                const OfflineSemanticFingerprintBucket &rhs) {
               return lhs.fingerprint < rhs.fingerprint;
             });
  llvm::sort(result.wildcardGraphIds);
  for (const auto &reason : reasons)
    result.unsupportedReasons.push_back(reason.getKey().str());
  llvm::sort(result.unsupportedReasons);
  return result;
}

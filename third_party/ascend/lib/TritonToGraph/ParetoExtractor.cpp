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

#include "TritonToGraph/ParetoExtractor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>

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

static unsigned getElementBitWidth(llvm::StringRef type) {
  unsigned lastWidth = 0;
  for (size_t index = 0; index + 1 < type.size(); ++index) {
    if (type[index] != 'i' && type[index] != 'f')
      continue;
    size_t digit = index + 1;
    if (!std::isdigit(static_cast<unsigned char>(type[digit])))
      continue;
    unsigned width = 0;
    while (digit < type.size() &&
           std::isdigit(static_cast<unsigned char>(type[digit]))) {
      unsigned value = static_cast<unsigned>(type[digit] - '0');
      if (width > (std::numeric_limits<unsigned>::max() - value) / 10)
        return 0;
      width = width * 10 + value;
      ++digit;
    }
    lastWidth = width;
  }
  return lastWidth ? lastWidth : 32;
}

static bool isConversion(llvm::StringRef operation) {
  return operation == "arith.extui" || operation == "arith.extsi" ||
         operation == "arith.trunci" || operation == "arith.index_cast" ||
         operation == "arith.index_castui" ||
         operation == "arith.bitcast" || operation == "arith.fptosi" ||
         operation == "arith.fptoui" || operation == "arith.sitofp" ||
         operation == "arith.uitofp" || operation == "arith.extf" ||
         operation == "arith.truncf" || operation == "ttg.convert_layout";
}

static bool isMemoryOperation(llvm::StringRef operation) {
  return operation == "tt.load" || operation == "tt.store" ||
         operation.starts_with("tt.atomic_") ||
         operation == "ascend.indirect_load" ||
         operation == "ascend.indirect_store" ||
         operation == "ascend.stride_load" ||
         operation == "ascend.stride_store";
}

static bool isBoundaryTerminal(llvm::StringRef operation) {
  return operation == "ags.input" || operation == "ags.private_buffer" ||
         operation == "ags.memory" ||
         operation == "ags.partitionable_memory" ||
         operation == "ags.launch" ||
         operation == "ags.buffer";
}

static uint64_t ownStructuralWeight(const PersistentGraphRecord &graph) {
  llvm::StringRef operation = graph.operationName;
  if (isBoundaryTerminal(operation))
    return 0;
  // Proof-only launch nodes are not target instructions.  ags.kernel carries
  // a small explicit per-physical-program launch/setup cost; launch remapping
  // amortizes this and all reachable body work over its certified factor.
  if (operation == "ags.kernel" || operation == "ags.kernel_partition")
    return 4;
  if (operation == "ags.launch_bound" || operation == "ags.launch_remap")
    return 0;
  if (operation == "arith.constant")
    return 1;
  if (isConversion(operation))
    return 1;
  // Stores and atomics produce the proof-language `none` state type.  Its
  // fallback bit width must not make one physical memory instruction look
  // like two 16-bit arithmetic instructions; memory traffic is represented
  // separately by memoryOperationCount and memoryStridePenalty.
  if (isMemoryOperation(operation))
    return 1;

  uint64_t widthMultiplier =
      std::max<uint64_t>(1, (getElementBitWidth(graph.resultType) + 15) / 16);
  if (operation == "arith.divsi" || operation == "arith.divui" ||
      operation == "arith.remsi" || operation == "arith.remui" ||
      operation == "arith.divf" || operation == "arith.remf")
    return 5 * widthMultiplier;
  if (operation == "math.exp" || operation == "math.exp2" ||
      operation == "math.log" || operation == "math.log2" ||
      operation == "math.sqrt" || operation == "math.rsqrt" ||
      operation == "math.tanh")
    return 4 * widthMultiplier;
  return widthMultiplier;
}

static std::optional<uint64_t>
getPositiveIntegerAttribute(llvm::StringRef attributes,
                            llvm::StringRef name) {
  std::string needle = (name + " = ").str();
  size_t position = attributes.find(needle);
  if (position == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef suffix = attributes.drop_front(position + needle.size());
  size_t digitCount = 0;
  while (digitCount < suffix.size() &&
         std::isdigit(static_cast<unsigned char>(suffix[digitCount])))
    ++digitCount;
  if (digitCount == 0)
    return std::nullopt;
  uint64_t value = 0;
  if (suffix.take_front(digitCount).getAsInteger(10, value) || value == 0)
    return std::nullopt;
  return value;
}

static std::optional<int64_t>
getSignedIntegerAttribute(llvm::StringRef attributes, llvm::StringRef name) {
  std::string needle = (name + " = ").str();
  size_t position = attributes.find(needle);
  if (position == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef suffix = attributes.drop_front(position + needle.size());
  if (suffix.starts_with("dense<"))
    suffix = suffix.drop_front(6);
  size_t length = 0;
  if (length < suffix.size() &&
      (suffix[length] == '-' || suffix[length] == '+'))
    ++length;
  const size_t digitsBegin = length;
  while (length < suffix.size() &&
         std::isdigit(static_cast<unsigned char>(suffix[length])))
    ++length;
  if (length == digitsBegin)
    return std::nullopt;
  int64_t value = 0;
  if (suffix.take_front(length).getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<std::vector<int64_t>>
getStaticTensorShape(llvm::StringRef type) {
  if (!type.consume_front("tensor<"))
    return std::nullopt;
  std::vector<int64_t> shape;
  while (!type.empty() &&
         std::isdigit(static_cast<unsigned char>(type.front()))) {
    size_t digits = 0;
    while (digits < type.size() &&
           std::isdigit(static_cast<unsigned char>(type[digits])))
      ++digits;
    if (digits == type.size() || type[digits] != 'x')
      return std::nullopt;
    int64_t dimension = 0;
    if (type.take_front(digits).getAsInteger(10, dimension) ||
        dimension <= 0)
      return std::nullopt;
    shape.push_back(dimension);
    type = type.drop_front(digits + 1);
  }
  return shape.empty() ? std::nullopt
                       : std::optional<std::vector<int64_t>>(shape);
}

static std::optional<std::vector<int64_t>>
getIntegerArrayAttribute(llvm::StringRef attributes, llvm::StringRef name) {
  std::string needle = (name + " = array<i32:").str();
  size_t position = attributes.find(needle);
  if (position == llvm::StringRef::npos)
    return std::nullopt;
  llvm::StringRef suffix = attributes.drop_front(position + needle.size());
  size_t end = suffix.find('>');
  if (end == llvm::StringRef::npos)
    return std::nullopt;
  suffix = suffix.take_front(end);
  std::vector<int64_t> values;
  while (!suffix.empty()) {
    suffix = suffix.ltrim();
    size_t comma = suffix.find(',');
    llvm::StringRef token = suffix.take_front(comma).trim();
    int64_t value = 0;
    if (token.empty() || token.getAsInteger(10, value))
      return std::nullopt;
    values.push_back(value);
    if (comma == llvm::StringRef::npos)
      break;
    suffix = suffix.drop_front(comma + 1);
  }
  return values;
}

struct AffineLaneForm {
  std::vector<int64_t> shape;
  std::vector<int64_t> coefficients;
  bool constantKnown = false;
  int64_t constant = 0;
};

static std::optional<int64_t> checkedAdd(int64_t lhs, int64_t rhs) {
  __int128 result = static_cast<__int128>(lhs) + rhs;
  if (result < std::numeric_limits<int64_t>::min() ||
      result > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(result);
}

static std::optional<int64_t> checkedMul(int64_t lhs, int64_t rhs) {
  __int128 result = static_cast<__int128>(lhs) * rhs;
  if (result < std::numeric_limits<int64_t>::min() ||
      result > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(result);
}

static bool isLaneInvariant(const AffineLaneForm &form) {
  return llvm::all_of(form.coefficients,
                      [](int64_t coefficient) { return coefficient == 0; });
}

static std::optional<AffineLaneForm> inferAffineLaneForm(
    llvm::StringRef graphHash,
    llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)> lookup,
    llvm::StringMap<std::optional<AffineLaneForm>> &memo,
    llvm::StringSet<> &active) {
  if (auto known = memo.find(graphHash); known != memo.end())
    return known->second;
  if (!active.insert(graphHash).second)
    return std::nullopt;
  const PersistentGraphRecord *graph = lookup(graphHash);
  if (!graph) {
    active.erase(graphHash);
    return std::nullopt;
  }
  std::optional<std::vector<int64_t>> shape =
      getStaticTensorShape(graph->resultType);
  auto operand = [&](unsigned index) -> std::optional<AffineLaneForm> {
    if (index >= graph->operandHashes.size())
      return std::nullopt;
    return inferAffineLaneForm(graph->operandHashes[index], lookup, memo,
                               active);
  };
  auto finish = [&](std::optional<AffineLaneForm> result) {
    active.erase(graphHash);
    memo[graphHash] = result;
    return result;
  };

  if (!shape)
    return finish(std::nullopt);

  if (graph->operationName == "arith.constant") {
    std::optional<int64_t> value =
        getSignedIntegerAttribute(graph->attributes, "value");
    if (!value)
      return finish(std::nullopt);
    return finish(AffineLaneForm{*shape,
                                 std::vector<int64_t>(shape->size(), 0), true,
                                 *value});
  }
  if (graph->operationName == "tt.make_range") {
    if (shape->size() != 1)
      return finish(std::nullopt);
    int64_t start =
        getSignedIntegerAttribute(graph->attributes, "start").value_or(0);
    return finish(AffineLaneForm{*shape, {1}, true, start});
  }
  if (graph->operationName == "tt.splat") {
    AffineLaneForm result{*shape, std::vector<int64_t>(shape->size(), 0),
                          false, 0};
    if (!graph->operandHashes.empty()) {
      const PersistentGraphRecord *source = lookup(graph->operandHashes[0]);
      if (source && source->operationName == "arith.constant") {
        std::optional<int64_t> value =
            getSignedIntegerAttribute(source->attributes, "value");
        if (value) {
          result.constantKnown = true;
          result.constant = *value;
        }
      }
    }
    return finish(result);
  }
  if (graph->operationName == "tt.expand_dims") {
    std::optional<AffineLaneForm> input = operand(0);
    std::optional<int64_t> axis =
        getSignedIntegerAttribute(graph->attributes, "axis");
    if (!input || !axis || *axis < 0 ||
        static_cast<size_t>(*axis) >= shape->size() ||
        input->shape.size() + 1 != shape->size())
      return finish(std::nullopt);
    AffineLaneForm result{*shape, {}, input->constantKnown, input->constant};
    result.coefficients.reserve(shape->size());
    for (size_t dimension = 0, source = 0; dimension < shape->size();
         ++dimension) {
      if (dimension == static_cast<size_t>(*axis))
        result.coefficients.push_back(0);
      else
        result.coefficients.push_back(input->coefficients[source++]);
    }
    return finish(result);
  }
  if (graph->operationName == "tt.broadcast") {
    std::optional<AffineLaneForm> input = operand(0);
    if (!input || input->shape.size() != shape->size())
      return finish(std::nullopt);
    AffineLaneForm result{*shape, input->coefficients, input->constantKnown,
                          input->constant};
    for (size_t dimension = 0; dimension < shape->size(); ++dimension) {
      if (input->shape[dimension] == (*shape)[dimension])
        continue;
      if (input->shape[dimension] != 1)
        return finish(std::nullopt);
      result.coefficients[dimension] = 0;
    }
    return finish(result);
  }
  if (graph->operationName == "tt.trans") {
    std::optional<AffineLaneForm> input = operand(0);
    std::optional<std::vector<int64_t>> order =
        getIntegerArrayAttribute(graph->attributes, "order");
    if (!input || !order || order->size() != shape->size() ||
        input->shape.size() != shape->size())
      return finish(std::nullopt);
    AffineLaneForm result{*shape, {}, input->constantKnown, input->constant};
    for (int64_t source : *order) {
      if (source < 0 || static_cast<size_t>(source) >= shape->size())
        return finish(std::nullopt);
      result.coefficients.push_back(input->coefficients[source]);
    }
    return finish(result);
  }
  if (graph->operationName == "arith.extsi" ||
      graph->operationName == "arith.extui" ||
      graph->operationName == "arith.trunci" ||
      graph->operationName == "arith.index_cast" ||
      graph->operationName == "arith.index_castui") {
    std::optional<AffineLaneForm> input = operand(0);
    if (!input || input->shape != *shape)
      return finish(std::nullopt);
    input->shape = *shape;
    return finish(*input);
  }
  if (graph->operationName == "arith.addi" ||
      graph->operationName == "arith.subi" ||
      graph->operationName == "tt.addptr") {
    std::optional<AffineLaneForm> lhs = operand(0);
    std::optional<AffineLaneForm> rhs = operand(1);
    if (!lhs || !rhs || lhs->shape != *shape || rhs->shape != *shape)
      return finish(std::nullopt);
    const int64_t sign = graph->operationName == "arith.subi" ? -1 : 1;
    AffineLaneForm result{*shape, {},
                          lhs->constantKnown && rhs->constantKnown, 0};
    for (size_t dimension = 0; dimension < shape->size(); ++dimension) {
      std::optional<int64_t> scaled =
          checkedMul(sign, rhs->coefficients[dimension]);
      std::optional<int64_t> sum =
          scaled ? checkedAdd(lhs->coefficients[dimension], *scaled)
                 : std::nullopt;
      if (!sum)
        return finish(std::nullopt);
      result.coefficients.push_back(*sum);
    }
    if (result.constantKnown) {
      std::optional<int64_t> scaled = checkedMul(sign, rhs->constant);
      std::optional<int64_t> sum =
          scaled ? checkedAdd(lhs->constant, *scaled) : std::nullopt;
      if (!sum)
        return finish(std::nullopt);
      result.constant = *sum;
    }
    return finish(result);
  }
  if (graph->operationName == "arith.muli") {
    std::optional<AffineLaneForm> lhs = operand(0);
    std::optional<AffineLaneForm> rhs = operand(1);
    if (!lhs || !rhs || lhs->shape != *shape || rhs->shape != *shape)
      return finish(std::nullopt);
    const AffineLaneForm *varying = nullptr;
    int64_t factor = 0;
    if (isLaneInvariant(*lhs) && lhs->constantKnown) {
      varying = &*rhs;
      factor = lhs->constant;
    } else if (isLaneInvariant(*rhs) && rhs->constantKnown) {
      varying = &*lhs;
      factor = rhs->constant;
    } else if (isLaneInvariant(*lhs) && !isLaneInvariant(*rhs)) {
      // A runtime scalar splatted across the tile is a symbolic physical
      // stride.  The structural fallback cannot assign it a cycle count, but
      // it can still distinguish it from a proven unit stride.  Use the
      // smallest non-unit representative; StageCostModel remains the higher
      // fidelity authority whenever it can score the complete graph.
      varying = &*rhs;
      factor = 2;
    } else if (isLaneInvariant(*rhs) && !isLaneInvariant(*lhs)) {
      varying = &*lhs;
      factor = 2;
    } else {
      return finish(std::nullopt);
    }
    AffineLaneForm result{*shape, {}, varying->constantKnown, 0};
    for (int64_t coefficient : varying->coefficients) {
      std::optional<int64_t> scaled = checkedMul(coefficient, factor);
      if (!scaled)
        return finish(std::nullopt);
      result.coefficients.push_back(*scaled);
    }
    if (result.constantKnown) {
      std::optional<int64_t> scaled =
          checkedMul(varying->constant, factor);
      if (!scaled)
        return finish(std::nullopt);
      result.constant = *scaled;
    }
    return finish(result);
  }
  return finish(std::nullopt);
}

static uint64_t computeMemoryStridePenalty(
    llvm::StringRef rootGraphHash,
    llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)> lookup) {
  llvm::StringSet<> visited;
  std::vector<std::string> pending = {rootGraphHash.str()};
  llvm::StringMap<std::optional<AffineLaneForm>> affineMemo;
  llvm::StringSet<> affineActive;
  uint64_t penalty = 0;
  while (!pending.empty()) {
    std::string hash = std::move(pending.back());
    pending.pop_back();
    if (!visited.insert(hash).second)
      continue;
    const PersistentGraphRecord *graph = lookup(hash);
    if (!graph)
      return std::numeric_limits<uint64_t>::max();
    pending.insert(pending.end(), graph->operandHashes.begin(),
                   graph->operandHashes.end());
    if ((graph->operationName != "tt.load" &&
         graph->operationName != "tt.store") ||
        graph->operandHashes.empty())
      continue;
    std::optional<AffineLaneForm> address = inferAffineLaneForm(
        graph->operandHashes.front(), lookup, affineMemo, affineActive);
    if (!address || address->coefficients.empty())
      continue;
    int64_t stride = address->coefficients.back();
    uint64_t magnitude =
        stride == std::numeric_limits<int64_t>::min()
            ? uint64_t{1} << 63
            : static_cast<uint64_t>(std::abs(stride));
    uint64_t accessPenalty = magnitude == 1 ? 0 : (magnitude == 0 ? 1
                                                                  : magnitude - 1);
    if (penalty > std::numeric_limits<uint64_t>::max() - accessPenalty)
      return std::numeric_limits<uint64_t>::max();
    penalty += accessPenalty;
  }
  return penalty;
}

static std::optional<uint64_t>
computeDepth(llvm::StringRef graphHash,
             llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)>
                 lookup,
             llvm::StringMap<uint64_t> &memo,
             llvm::StringSet<> &active) {
  auto known = memo.find(graphHash);
  if (known != memo.end())
    return known->second;
  const PersistentGraphRecord *graph = lookup(graphHash);
  if (!graph || !active.insert(graphHash).second)
    return std::nullopt;
  uint64_t depth = isBoundaryTerminal(graph->operationName) ? 0 : 1;
  uint64_t childDepth = 0;
  for (llvm::StringRef operand : graph->operandHashes) {
    std::optional<uint64_t> nested =
        computeDepth(operand, lookup, memo, active);
    if (!nested) {
      active.erase(graphHash);
      return std::nullopt;
    }
    childDepth = std::max(childDepth, *nested);
  }
  if (!graph->operandHashes.empty())
    depth = 1 + childDepth;
  active.erase(graphHash);
  memo[graphHash] = depth;
  return depth;
}

static std::optional<StructuralCostFeatures>
computeStructuralFeatures(
    llvm::StringRef rootHash,
    llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)> lookup,
    llvm::StringMap<uint64_t> &depthMemo) {
  llvm::StringSet<> active;
  std::optional<uint64_t> depth =
      computeDepth(rootHash, lookup, depthMemo, active);
  if (!depth)
    return std::nullopt;

  StructuralCostFeatures features;
  features.maxDepth = *depth;
  uint64_t launchFactor = 1;
  llvm::StringSet<> visited;
  std::vector<std::string> pending = {rootHash.str()};
  while (!pending.empty()) {
    std::string graphHash = std::move(pending.back());
    pending.pop_back();
    if (!visited.insert(graphHash).second)
      continue;
    const PersistentGraphRecord *found = lookup(graphHash);
    if (!found)
      return std::nullopt;
    const PersistentGraphRecord &graph = *found;
    if (!isBoundaryTerminal(graph.operationName))
      ++features.operatorCount;
    features.weightedOperatorCost += ownStructuralWeight(graph);
    if (graph.operationName == "ags.launch_remap" ||
        graph.operationName == "ags.kernel_partition") {
      if (std::optional<uint64_t> factor =
              getPositiveIntegerAttribute(graph.attributes, "factor")) {
        if (launchFactor > std::numeric_limits<uint64_t>::max() / *factor)
          return std::nullopt;
        launchFactor *= *factor;
      }
    }
    if (isConversion(graph.operationName))
      ++features.conversionCount;
    if (isMemoryOperation(graph.operationName))
      ++features.memoryOperationCount;
    pending.insert(pending.end(), graph.operandHashes.begin(),
                   graph.operandHashes.end());
  }
  // A launch-remapped kernel performs this graph once for `launchFactor`
  // logical programs.  Structural fallback therefore compares amortized work
  // per logical program.  Raw operator/memory counts and depth remain intact
  // as Pareto dimensions, so this cannot make a wider graph dominate merely
  // by hiding its resource footprint.  StageModel cycles take precedence when
  // they are available for the complete equivalence class.
  if (launchFactor > 1) {
    features.weightedOperatorCost =
        features.weightedOperatorCost / launchFactor +
        (features.weightedOperatorCost % launchFactor != 0);
  }
  features.memoryStridePenalty =
      computeMemoryStridePenalty(rootHash, lookup);
  return features;
}

static bool dominates(const ParetoCandidateCost &lhs,
                      const ParetoCandidateCost &rhs, bool useStageModel) {
  if (!lhs.structural.dominates(rhs.structural) &&
      !lhs.structural.equals(rhs.structural))
    return false;

  bool strict = lhs.structural.dominates(rhs.structural);
  if (useStageModel) {
    if (!lhs.stageCycles || !rhs.stageCycles ||
        *lhs.stageCycles > *rhs.stageCycles)
      return false;
    strict |= *lhs.stageCycles < *rhs.stageCycles;
  }
  return strict;
}

static std::string getCostIdentity(const ParetoCandidateCost &cost,
                                   bool useStageModel) {
  const StructuralCostFeatures &structural = cost.structural;
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << structural.operatorCount << ':'
         << structural.weightedOperatorCost << ':' << structural.maxDepth
         << ':' << structural.conversionCount << ':'
         << structural.memoryOperationCount << ':'
         << structural.memoryStridePenalty;
  if (useStageModel)
    stream << ':' << *cost.stageCycles;
  return storage;
}

static llvm::json::Object
costToJSON(const ParetoCandidateCost &cost, bool structuralFallback) {
  llvm::json::Object structural;
  structural["operator_count"] =
      static_cast<int64_t>(cost.structural.operatorCount);
  structural["weighted_operator_cost"] =
      static_cast<int64_t>(cost.structural.weightedOperatorCost);
  structural["max_depth"] = static_cast<int64_t>(cost.structural.maxDepth);
  structural["conversion_count"] =
      static_cast<int64_t>(cost.structural.conversionCount);
  structural["memory_operation_count"] =
      static_cast<int64_t>(cost.structural.memoryOperationCount);
  structural["memory_stride_penalty"] =
      static_cast<int64_t>(cost.structural.memoryStridePenalty);
  llvm::json::Object object;
  object["provenance"] =
      cost.stageCycles      ? "stage-model"
      : structuralFallback ? "structural-fallback"
                           : "singleton-no-choice";
  if (cost.stageCycles)
    object["stage_cycles"] = static_cast<int64_t>(*cost.stageCycles);
  object["structural"] = std::move(structural);
  return object;
}

} // namespace

std::optional<StructuralCostFeatures>
mlir::triton::cfg::estimateStructuralCost(
    llvm::StringRef rootGraphHash,
    llvm::function_ref<const PersistentGraphRecord *(llvm::StringRef)> lookup) {
  llvm::StringMap<uint64_t> depthMemo;
  return computeStructuralFeatures(rootGraphHash, lookup, depthMemo);
}

std::string GraphCostContext::getStableKey() const {
  std::string canonical = target + "\n" + shapeClass + "\n" + layoutClass +
                          "\n" + dtypeClass;
  return "costctx-v1:" + stableHash(canonical);
}

bool StructuralCostFeatures::dominates(
    const StructuralCostFeatures &other) const {
  bool lessOrEqual = operatorCount <= other.operatorCount &&
                     weightedOperatorCost <= other.weightedOperatorCost &&
                     maxDepth <= other.maxDepth &&
                     conversionCount <= other.conversionCount &&
                     memoryOperationCount <= other.memoryOperationCount &&
                     memoryStridePenalty <= other.memoryStridePenalty;
  return lessOrEqual && !equals(other);
}

bool StructuralCostFeatures::equals(
    const StructuralCostFeatures &other) const {
  return operatorCount == other.operatorCount &&
         weightedOperatorCost == other.weightedOperatorCost &&
         maxDepth == other.maxDepth &&
         conversionCount == other.conversionCount &&
         memoryOperationCount == other.memoryOperationCount &&
         memoryStridePenalty == other.memoryStridePenalty;
}

std::string StructuralFallbackCostModel::getVersionHash() const {
  return stableHash(getName());
}

std::optional<uint64_t> StructuralFallbackCostModel::estimateStageCycles(
    const PersistentEquivalenceClassStore &,
    const PersistentGraphRecord &, const GraphCostContext &,
    std::string &unknownReason) const {
  unknownReason =
      "whole-graph StageModel estimate unavailable; structural fallback used";
  return std::nullopt;
}

StageModelCostAdapter::StageModelCostAdapter(Evaluator evaluator,
                                             std::string evaluatorVersion)
    : evaluator(std::move(evaluator)),
      evaluatorVersion(std::move(evaluatorVersion)) {}

std::string StageModelCostAdapter::getVersionHash() const {
  return stableHash((llvm::Twine(getName()) + ":" + evaluatorVersion).str());
}

std::optional<uint64_t> StageModelCostAdapter::estimateStageCycles(
    const PersistentEquivalenceClassStore &store,
    const PersistentGraphRecord &root, const GraphCostContext &context,
    std::string &unknownReason) const {
  if (!evaluator) {
    unknownReason = "StageModel evaluator is not configured";
    return std::nullopt;
  }
  std::optional<uint64_t> cycles =
      evaluator(store, root, context, unknownReason);
  if (!cycles && unknownReason.empty())
    unknownReason = "StageModel returned unknown without a reason";
  return cycles;
}

ParetoExtractionResult ParetoExtractor::extract(
    const PersistentEquivalenceClassStore &store,
    const OfflineGraphCostModel &costModel,
    const GraphCostContext &context,
    llvm::StringRef precomputedStoreHash) const {
  ParetoExtractionResult result;
  result.eclassStoreHash = precomputedStoreHash.empty()
                               ? store.getStableHash()
                               : precomputedStoreHash.str();
  result.costContextKey = context.getStableKey();
  result.costModelName = costModel.getName().str();
  result.costModelHash = costModel.getVersionHash();

  llvm::StringMap<const PersistentGraphRecord *> byHash;
  for (const PersistentGraphRecord &graph : store.graphs)
    byHash.try_emplace(graph.graphHash, &graph);
  llvm::StringMap<uint64_t> depthMemo;
  llvm::StringSet<> unknownReasons;

  for (const PersistentEquivalenceClass &eclass : store.classes) {
    ParetoEquivalenceClass extracted;
    extracted.classHash = eclass.classHash;
    result.candidateCountBefore += eclass.memberGraphHashes.size();
    const bool hasCostChoice = eclass.memberGraphHashes.size() > 1;
    const bool evaluateStageCycles =
        costModel.providesStageCycles() && hasCostChoice;
    bool allStageCyclesKnown = evaluateStageCycles;

    for (llvm::StringRef graphHash : eclass.memberGraphHashes) {
      auto graph = byHash.find(graphHash);
      if (graph == byHash.end())
        continue;
      ParetoCandidateRecord candidate;
      candidate.graphHash = graphHash.str();
      std::optional<StructuralCostFeatures> structural =
          computeStructuralFeatures(
              graphHash,
              [&](llvm::StringRef hash) -> const PersistentGraphRecord * {
                auto found = byHash.find(hash);
                return found == byHash.end() ? nullptr : found->second;
              },
              depthMemo);
      if (!structural) {
        ++result.invalidStructuralGraphCount;
        structural = StructuralCostFeatures{
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max(),
            std::numeric_limits<uint64_t>::max()};
        unknownReasons.insert("invalid or cyclic persistent graph: " +
                              graphHash.str());
      }
      candidate.cost.structural = *structural;
      // Stage cycles are an all-or-nothing cost domain for one e-class. Once
      // any member is unknown the whole class must use structural fallback,
      // so evaluating the remaining members cannot affect the result.
      if (evaluateStageCycles && allStageCyclesKnown) {
        std::string reason;
        candidate.cost.stageCycles = costModel.estimateStageCycles(
            store, *graph->second, context, reason);
        if (!candidate.cost.stageCycles) {
          allStageCyclesKnown = false;
          if (!reason.empty())
            unknownReasons.insert(reason);
        }
      }
      extracted.candidates.push_back(std::move(candidate));
    }

    // A target StageModel does not need to compile a singleton e-class: there
    // is no alternative to rank.  It is complete for selection purposes even
    // though the sole candidate intentionally has no stage_cycles field.
    extracted.usedStructuralFallback =
        !costModel.providesStageCycles() ||
        (hasCostChoice && !allStageCyclesKnown);
    if (extracted.usedStructuralFallback) {
      ++result.structuralFallbackClassCount;
      for (ParetoCandidateRecord &candidate : extracted.candidates)
        candidate.cost.stageCycles.reset();
    }

    extracted.allCandidates = extracted.candidates;

    // Equal cost vectors are interchangeable for this fixed offline cost
    // context. Retain the lexicographically first graph as their stable
    // representative before dominance testing. This changes a potentially
    // quadratic N-by-N comparison (one production e-class has >52K members)
    // into N-by-U plus U-by-U, where U is the number of distinct cost vectors.
    // allCandidates deliberately remains complete for proof-backed template
    // generation and online re-costing.
    llvm::StringSet<> seenCostIdentities;
    std::vector<ParetoCandidateRecord> uniqueCostCandidates;
    for (ParetoCandidateRecord &candidate : extracted.candidates) {
      if (seenCostIdentities
              .insert(getCostIdentity(candidate.cost,
                                      allStageCyclesKnown))
              .second)
        uniqueCostCandidates.push_back(std::move(candidate));
    }

    std::vector<ParetoCandidateRecord> frontier;
    for (auto [candidateIndex, candidate] :
         llvm::enumerate(uniqueCostCandidates)) {
      bool isDominated = false;
      for (auto [otherIndex, other] :
           llvm::enumerate(uniqueCostCandidates)) {
        if (candidateIndex == otherIndex)
          continue;
        if (dominates(other.cost, candidate.cost, allStageCyclesKnown)) {
          isDominated = true;
          break;
        }
      }
      if (!isDominated)
        frontier.push_back(std::move(candidate));
    }
    llvm::sort(frontier,
               [](const ParetoCandidateRecord &lhs,
                  const ParetoCandidateRecord &rhs) {
                 return lhs.graphHash < rhs.graphHash;
               });
    extracted.candidates = std::move(frontier);
    result.candidateCountAfter += extracted.candidates.size();
    result.classes.push_back(std::move(extracted));
  }

  llvm::sort(result.classes,
             [](const ParetoEquivalenceClass &lhs,
                const ParetoEquivalenceClass &rhs) {
               return lhs.classHash < rhs.classHash;
             });
  for (const auto &entry : unknownReasons)
    result.stageModelUnknownReasons.push_back(entry.getKey().str());
  llvm::sort(result.stageModelUnknownReasons);
  return result;
}

void ParetoExtractionResult::writeJSON(llvm::raw_ostream &output) const {
  auto writeString = [&](StringRef value) {
    output << llvm::formatv("{0}", llvm::json::Value(value));
  };
  auto classToJSON = [](const ParetoEquivalenceClass &eclass) {
    llvm::json::Array candidates;
    for (const ParetoCandidateRecord &candidate : eclass.candidates) {
      llvm::json::Object entry;
      entry["graph_hash"] = candidate.graphHash;
      entry["cost"] =
          costToJSON(candidate.cost, eclass.usedStructuralFallback);
      candidates.push_back(std::move(entry));
    }
    llvm::json::Array evaluatedCandidates;
    for (const ParetoCandidateRecord &candidate : eclass.allCandidates) {
      llvm::json::Object entry;
      entry["graph_hash"] = candidate.graphHash;
      entry["cost"] =
          costToJSON(candidate.cost, eclass.usedStructuralFallback);
      evaluatedCandidates.push_back(std::move(entry));
    }
    llvm::json::Object entry;
    entry["class_hash"] = eclass.classHash;
    entry["used_structural_fallback"] = eclass.usedStructuralFallback;
    entry["evaluated_candidates"] = std::move(evaluatedCandidates);
    entry["candidates"] = std::move(candidates);
    return entry;
  };

  output << "{\n  \"schema_version\": 1,\n  \"eclass_store_hash\": ";
  writeString(eclassStoreHash);
  output << ",\n  \"cost_context_key\": ";
  writeString(costContextKey);
  output << ",\n  \"cost_model_name\": ";
  writeString(costModelName);
  output << ",\n  \"cost_model_hash\": ";
  writeString(costModelHash);
  output << ",\n  \"candidate_count_before\": " << candidateCountBefore
         << ",\n  \"candidate_count_after\": " << candidateCountAfter
         << ",\n  \"eclass_count\": " << classes.size()
         << ",\n  \"structural_fallback_class_count\": "
         << structuralFallbackClassCount
         << ",\n  \"invalid_structural_graph_count\": "
         << invalidStructuralGraphCount
         << ",\n  \"structural_cost_complete\": "
         << (hasStructuralCostForEveryGraph() ? "true" : "false")
         << ",\n  \"stage_model_complete\": "
         << (usedStageModelForEveryClass() ? "true" : "false")
         << ",\n  \"stage_model_unknown_reasons\": [";
  bool first = true;
  for (StringRef reason : stageModelUnknownReasons) {
    output << (first ? "" : ", ");
    first = false;
    writeString(reason);
  }
  output << "],\n  \"eclasses\": [";
  first = true;
  for (const ParetoEquivalenceClass &eclass : classes) {
    output << (first ? "\n    " : ",\n    ");
    first = false;
    output << llvm::formatv("{0}",
                            llvm::json::Value(classToJSON(eclass)));
  }
  output << (first ? "]\n}\n" : "\n  ]\n}\n");
}

std::string ParetoExtractionResult::toJSON() const {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  writeJSON(stream);
  return storage;
}

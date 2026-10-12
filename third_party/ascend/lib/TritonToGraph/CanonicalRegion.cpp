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

#include "TritonToGraph/CanonicalRegion.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static ArrayAttr typeArray(Builder &builder, TypeRange types) {
  SmallVector<Attribute, 4> attributes;
  for (Type type : types)
    attributes.push_back(TypeAttr::get(type));
  return builder.getArrayAttr(attributes);
}

static std::optional<SmallVector<Type, 4>> parseTypes(Attribute attribute) {
  auto array = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!array)
    return std::nullopt;
  SmallVector<Type, 4> types;
  for (Attribute element : array) {
    auto type = dyn_cast<TypeAttr>(element);
    if (!type)
      return std::nullopt;
    types.push_back(type.getValue());
  }
  return types;
}

static std::optional<SmallVector<int32_t, 4>> parseReferences(
    Attribute attribute) {
  auto references = dyn_cast_or_null<DenseI32ArrayAttr>(attribute);
  if (!references)
    return std::nullopt;
  return SmallVector<int32_t, 4>(references.asArrayRef().begin(),
                                 references.asArrayRef().end());
}

static bool resolveReference(int32_t reference, Block *block,
                             ArrayRef<Value> results, Value &value) {
  if (reference < 0) {
    const int64_t argument = -static_cast<int64_t>(reference) - 1;
    if (argument < 0 ||
        static_cast<uint64_t>(argument) >= block->getNumArguments())
      return false;
    value = block->getArgument(static_cast<unsigned>(argument));
    return true;
  }
  if (static_cast<uint32_t>(reference) >= results.size())
    return false;
  value = results[static_cast<unsigned>(reference)];
  return true;
}

static DictionaryAttr operationRecipe(
    Builder &builder, StringRef name, DictionaryAttr attributes,
    ArrayRef<int32_t> operands, TypeRange resultTypes,
    ArrayRef<int32_t> successors = {}, ArrayAttr nestedRegions = {}) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("name", builder.getStringAttr(name)),
       builder.getNamedAttr("attributes",
                            attributes ? attributes
                                       : builder.getDictionaryAttr({})),
       builder.getNamedAttr("operands",
                            builder.getDenseI32ArrayAttr(operands)),
       builder.getNamedAttr("result_types",
                            typeArray(builder, resultTypes)),
       builder.getNamedAttr("successors",
                            builder.getDenseI32ArrayAttr(successors)),
       builder.getNamedAttr("regions",
                            nestedRegions ? nestedRegions
                                          : builder.getArrayAttr({}))});
}

static LogicalResult validateOperationRecipe(DictionaryAttr operation,
                                             unsigned argumentCount,
                                             unsigned firstLocalResult,
                                             unsigned availableResults,
                                             unsigned blockCount,
                                             unsigned &newResults,
                                             std::string &error) {
  auto name = dyn_cast_or_null<StringAttr>(operation.get("name"));
  auto attributes =
      dyn_cast_or_null<DictionaryAttr>(operation.get("attributes"));
  std::optional<SmallVector<int32_t, 4>> operands =
      parseReferences(operation.get("operands"));
  std::optional<SmallVector<Type, 4>> resultTypes =
      parseTypes(operation.get("result_types"));
  std::optional<SmallVector<int32_t, 4>> successors =
      parseReferences(operation.get("successors"));
  auto nestedRegions =
      dyn_cast_or_null<ArrayAttr>(operation.get("regions"));
  if (!name || name.getValue().empty() || !attributes || !operands ||
      !resultTypes || !successors || !nestedRegions) {
    error = "malformed canonical region operation recipe";
    return failure();
  }
  for (int32_t reference : *operands) {
    if (reference < 0) {
      if (-static_cast<int64_t>(reference) - 1 >= argumentCount) {
        error = "canonical region operand references an unknown block argument";
        return failure();
      }
    } else if (static_cast<unsigned>(reference) < firstLocalResult ||
               static_cast<unsigned>(reference) >= availableResults) {
      error = "canonical region operand references a non-dominating or "
              "cross-block result";
      return failure();
    }
  }
  for (int32_t successor : *successors) {
    if (successor < 0 || static_cast<unsigned>(successor) >= blockCount) {
      error = "canonical region operation references an unknown successor";
      return failure();
    }
  }
  if (failed(validateCanonicalRegions(nestedRegions, error)))
    return failure();
  newResults = resultTypes->size();
  return success();
}

} // namespace

std::optional<ArrayAttr>
mlir::triton::cfg::captureCanonicalRegions(Operation *operation,
                                           std::string &error) {
  if (!operation) {
    error = "cannot capture regions from a null operation";
    return std::nullopt;
  }
  Builder builder(operation->getContext());
  SmallVector<Attribute, 2> regions;
  for (Region &region : operation->getRegions()) {
    if (region.empty()) {
      error = "canonical region recipe requires at least one block";
      return std::nullopt;
    }
    DenseMap<Block *, int32_t> blockOrdinals;
    for (auto [index, block] : llvm::enumerate(region))
      blockOrdinals[&block] = static_cast<int32_t>(index);
    int32_t nextResult = 0;
    SmallVector<Attribute, 4> blocks;
    for (Block &block : region) {
      if (!block.mightHaveTerminator()) {
        error = "canonical region recipe requires an explicit terminator";
        return std::nullopt;
      }
      DenseMap<Value, int32_t> references;
      for (auto [index, argument] : llvm::enumerate(block.getArguments()))
        references[argument] = -static_cast<int32_t>(index) - 1;

      auto encode = [&](Operation &nested) -> std::optional<DictionaryAttr> {
        ArrayAttr nestedRegions;
        if (nested.getNumRegions() != 0) {
          std::optional<ArrayAttr> captured =
              captureCanonicalRegions(&nested, error);
          if (!captured)
            return std::nullopt;
          nestedRegions = *captured;
        }
        SmallVector<int32_t, 4> operands;
        for (Value operand : nested.getOperands()) {
          auto found = references.find(operand);
          if (found == references.end()) {
            error = "canonical region contains a non-dominating SSA operand";
            return std::nullopt;
          }
          operands.push_back(found->second);
        }
        SmallVector<int32_t, 2> successors;
        for (Block *successor : nested.getSuccessors()) {
          auto found = blockOrdinals.find(successor);
          if (found == blockOrdinals.end()) {
            error = "canonical region successor leaves its parent region";
            return std::nullopt;
          }
          successors.push_back(found->second);
        }
        DictionaryAttr encoded = operationRecipe(
            builder, nested.getName().getStringRef(),
            builder.getDictionaryAttr(nested.getAttrs()), operands,
            nested.getResultTypes(), successors, nestedRegions);
        for (Value result : nested.getResults()) {
          references[result] = nextResult;
          ++nextResult;
        }
        return encoded;
      };

      SmallVector<Attribute, 8> operations;
      Operation *terminatorOperation = block.getTerminator();
      for (Operation &nested : block) {
        if (&nested == terminatorOperation)
          break;
        std::optional<DictionaryAttr> encoded = encode(nested);
        if (!encoded)
          return std::nullopt;
        operations.push_back(*encoded);
      }
      std::optional<DictionaryAttr> terminator = encode(*terminatorOperation);
      if (!terminator)
        return std::nullopt;
      blocks.push_back(builder.getDictionaryAttr(
          {builder.getNamedAttr("argument_types",
                                typeArray(builder, block.getArgumentTypes())),
           builder.getNamedAttr("operations", builder.getArrayAttr(operations)),
           builder.getNamedAttr("terminator", *terminator)}));
    }
    regions.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("blocks", builder.getArrayAttr(blocks))}));
  }
  ArrayAttr result = builder.getArrayAttr(regions);
  return succeeded(validateCanonicalRegions(result, error))
             ? std::optional<ArrayAttr>(result)
             : std::nullopt;
}

LogicalResult mlir::triton::cfg::validateCanonicalRegions(ArrayAttr regions,
                                                          std::string &error) {
  if (!regions) {
    error = "missing canonical region array";
    return failure();
  }
  for (Attribute regionAttribute : regions) {
    auto region = dyn_cast<DictionaryAttr>(regionAttribute);
    auto blocks = region ? dyn_cast_or_null<ArrayAttr>(region.get("blocks"))
                         : ArrayAttr{};
    if (!region || !blocks || blocks.empty()) {
      error = "malformed canonical region recipe";
      return failure();
    }
    unsigned availableResults = 0;
    for (Attribute blockAttribute : blocks) {
      auto block = dyn_cast<DictionaryAttr>(blockAttribute);
      auto operations =
          block ? dyn_cast_or_null<ArrayAttr>(block.get("operations"))
                : ArrayAttr{};
      std::optional<SmallVector<Type, 4>> arguments =
          block ? parseTypes(block.get("argument_types")) : std::nullopt;
      auto terminator =
          block ? dyn_cast_or_null<DictionaryAttr>(block.get("terminator"))
                : DictionaryAttr{};
      if (!block || !operations || !arguments || !terminator) {
        error = "malformed canonical region block recipe";
        return failure();
      }
      const unsigned firstLocalResult = availableResults;
      for (Attribute operationAttribute : operations) {
        auto operation = dyn_cast<DictionaryAttr>(operationAttribute);
        unsigned newResults = 0;
        if (!operation || failed(validateOperationRecipe(
                              operation, arguments->size(), firstLocalResult,
                              availableResults, blocks.size(), newResults,
                              error)))
          return failure();
        availableResults += newResults;
      }
      unsigned terminatorResults = 0;
      if (failed(validateOperationRecipe(
              terminator, arguments->size(), firstLocalResult,
              availableResults, blocks.size(), terminatorResults, error)) ||
          terminatorResults != 0) {
        if (error.empty())
          error = "canonical region terminator must have zero results";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult mlir::triton::cfg::materializeCanonicalRegions(
    Operation *operation, ArrayAttr regions, RewriterBase &rewriter,
    std::string &error) {
  if (!operation || operation->getNumRegions() != regions.size() ||
      failed(validateCanonicalRegions(regions, error))) {
    if (error.empty())
      error = "operation/recipe region count mismatch";
    return failure();
  }
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  for (auto [regionIndex, regionAttribute] : llvm::enumerate(regions)) {
    auto recipe = cast<DictionaryAttr>(regionAttribute);
    auto blockRecipes = cast<ArrayAttr>(recipe.get("blocks"));
    Region &region = operation->getRegion(regionIndex);
    SmallVector<Block *, 4> blocks;
    for (Attribute blockAttribute : blockRecipes) {
      auto blockRecipe = cast<DictionaryAttr>(blockAttribute);
      auto argumentTypes = *parseTypes(blockRecipe.get("argument_types"));
      SmallVector<Location, 4> argumentLocations(argumentTypes.size(),
                                                  operation->getLoc());
      blocks.push_back(rewriter.createBlock(&region, region.end(),
                                            argumentTypes,
                                            argumentLocations));
    }
    SmallVector<Value, 8> results;
    auto emit = [&](DictionaryAttr encoded, Block *block) -> LogicalResult {
      auto name = cast<StringAttr>(encoded.get("name"));
      auto attributes = cast<DictionaryAttr>(encoded.get("attributes"));
      auto references = *parseReferences(encoded.get("operands"));
      auto resultTypes = *parseTypes(encoded.get("result_types"));
      auto successorOrdinals = *parseReferences(encoded.get("successors"));
      auto nestedRegions = cast<ArrayAttr>(encoded.get("regions"));
      SmallVector<Value, 4> operands;
      for (int32_t reference : references) {
        Value value;
        if (!resolveReference(reference, block, results, value))
          return failure();
        operands.push_back(value);
      }
      OperationState state(operation->getLoc(), name.getValue());
      state.addOperands(operands);
      state.addTypes(resultTypes);
      state.addAttributes(attributes.getValue());
      SmallVector<Block *, 2> successors;
      for (int32_t ordinal : successorOrdinals)
        successors.push_back(blocks[static_cast<unsigned>(ordinal)]);
      state.addSuccessors(successors);
      for (size_t index = 0; index < nestedRegions.size(); ++index)
        state.addRegion();
      rewriter.setInsertionPointToEnd(block);
      Operation *nested = rewriter.create(state);
      if (!nestedRegions.empty() &&
          failed(materializeCanonicalRegions(nested, nestedRegions, rewriter,
                                             error))) {
        rewriter.eraseOp(nested);
        return failure();
      }
      results.append(nested->result_begin(), nested->result_end());
      return success();
    };
    for (auto [blockIndex, blockAttribute] : llvm::enumerate(blockRecipes)) {
      auto blockRecipe = cast<DictionaryAttr>(blockAttribute);
      Block *block = blocks[blockIndex];
      for (Attribute nested : cast<ArrayAttr>(blockRecipe.get("operations")))
        if (failed(emit(cast<DictionaryAttr>(nested), block)))
          return failure();
      if (failed(emit(cast<DictionaryAttr>(blockRecipe.get("terminator")),
                      block)))
        return failure();
    }
  }
  return success();
}

ArrayAttr mlir::triton::cfg::buildCanonicalBinaryRegion(
    MLIRContext *context, Type elementType, StringRef combinerOperation,
    StringRef terminatorOperation) {
  Builder builder(context);
  SmallVector<int32_t, 2> combinerOperands{-1, -2};
  SmallVector<Type, 1> combinerResultTypes{elementType};
  DictionaryAttr combiner = operationRecipe(
      builder, combinerOperation, builder.getDictionaryAttr({}),
      combinerOperands, combinerResultTypes);
  SmallVector<int32_t, 1> terminatorOperands{0};
  SmallVector<Type, 0> terminatorResultTypes;
  DictionaryAttr terminator = operationRecipe(
      builder, terminatorOperation, builder.getDictionaryAttr({}),
      terminatorOperands, terminatorResultTypes);
  SmallVector<Type, 2> argumentTypes{elementType, elementType};
  SmallVector<Attribute, 1> operations{combiner};
  DictionaryAttr block = builder.getDictionaryAttr(
      {builder.getNamedAttr("argument_types",
                            typeArray(builder, argumentTypes)),
       builder.getNamedAttr("operations", builder.getArrayAttr(operations)),
       builder.getNamedAttr("terminator", terminator)});
  SmallVector<Attribute, 1> blocks{block};
  DictionaryAttr region = builder.getDictionaryAttr(
      {builder.getNamedAttr("blocks", builder.getArrayAttr(blocks))});
  SmallVector<Attribute, 1> regions{region};
  return builder.getArrayAttr(regions);
}

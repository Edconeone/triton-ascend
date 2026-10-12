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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static std::string printAttribute(Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << attribute;
  return text;
}

TEST(CanonicalRegionTest, BinaryRecipeRoundTripsThroughGenericMaterializer) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  IRRewriter rewriter(&context);
  Location location = rewriter.getUnknownLoc();
  ModuleOp module = ModuleOp::create(location);
  rewriter.setInsertionPointToEnd(module.getBody());

  OperationState parentState(location, "test.parent");
  parentState.addRegion();
  Operation *parent = rewriter.create(parentState);
  Type i32 = rewriter.getI32Type();
  ArrayAttr recipe = buildCanonicalBinaryRegion(
      &context, i32, "test.combine", "test.yield");
  std::string error;
  ASSERT_TRUE(succeeded(validateCanonicalRegions(recipe, error))) << error;
  ASSERT_TRUE(succeeded(
      materializeCanonicalRegions(parent, recipe, rewriter, error)))
      << error;

  std::optional<ArrayAttr> captured =
      captureCanonicalRegions(parent, error);
  ASSERT_TRUE(captured.has_value()) << error;
  EXPECT_EQ(*captured, recipe)
      << "captured: " << printAttribute(*captured)
      << "\nrecipe: " << printAttribute(recipe);
  ASSERT_EQ(parent->getNumRegions(), 1u);
  ASSERT_TRUE(llvm::hasSingleElement(parent->getRegion(0)));
  Block &body = parent->getRegion(0).front();
  EXPECT_EQ(body.getNumArguments(), 2u);
  EXPECT_EQ(body.front().getName().getStringRef(), "test.combine");
  EXPECT_EQ(body.getTerminator()->getName().getStringRef(), "test.yield");
}

TEST(CanonicalRegionTest, RejectsForwardSSAReference) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  DictionaryAttr badOperation = builder.getDictionaryAttr(
      {builder.getNamedAttr("name", builder.getStringAttr("test.combine")),
       builder.getNamedAttr("attributes", builder.getDictionaryAttr({})),
       builder.getNamedAttr("operands",
                            builder.getDenseI32ArrayAttr({0})),
       builder.getNamedAttr("result_types",
                            builder.getArrayAttr({TypeAttr::get(i32)})),
       builder.getNamedAttr("successors",
                            builder.getDenseI32ArrayAttr({})),
       builder.getNamedAttr("regions", builder.getArrayAttr({}))});
  DictionaryAttr terminator = builder.getDictionaryAttr(
      {builder.getNamedAttr("name", builder.getStringAttr("test.yield")),
       builder.getNamedAttr("attributes", builder.getDictionaryAttr({})),
       builder.getNamedAttr("operands",
                            builder.getDenseI32ArrayAttr({0})),
       builder.getNamedAttr("result_types", builder.getArrayAttr({})),
       builder.getNamedAttr("successors",
                            builder.getDenseI32ArrayAttr({})),
       builder.getNamedAttr("regions", builder.getArrayAttr({}))});
  DictionaryAttr block = builder.getDictionaryAttr(
      {builder.getNamedAttr(
           "argument_types",
           builder.getArrayAttr({TypeAttr::get(i32), TypeAttr::get(i32)})),
       builder.getNamedAttr("operations",
                            builder.getArrayAttr({badOperation})),
       builder.getNamedAttr("terminator", terminator)});
  DictionaryAttr region = builder.getDictionaryAttr(
      {builder.getNamedAttr("blocks", builder.getArrayAttr({block}))});
  ArrayAttr recipe = builder.getArrayAttr({region});
  std::string error;
  EXPECT_TRUE(failed(validateCanonicalRegions(recipe, error)));
  EXPECT_NE(error.find("non-dominating"), std::string::npos) << error;
}

TEST(CanonicalRegionTest, RejectsImplicitCrossBlockSSAReference) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  auto operation = [&](StringRef name, ArrayRef<int32_t> operands,
                       ArrayRef<Type> results,
                       ArrayRef<int32_t> successors) {
    SmallVector<Attribute, 2> resultTypes;
    for (Type result : results)
      resultTypes.push_back(TypeAttr::get(result));
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("name", builder.getStringAttr(name)),
         builder.getNamedAttr("attributes", builder.getDictionaryAttr({})),
         builder.getNamedAttr("operands",
                              builder.getDenseI32ArrayAttr(operands)),
         builder.getNamedAttr("result_types",
                              builder.getArrayAttr(resultTypes)),
         builder.getNamedAttr("successors",
                              builder.getDenseI32ArrayAttr(successors)),
         builder.getNamedAttr("regions", builder.getArrayAttr({}))});
  };
  DictionaryAttr producer = operation("test.make", {}, {i32}, {});
  DictionaryAttr branch = operation("test.br", {}, {}, {1});
  DictionaryAttr illegalYield = operation("test.yield", {0}, {}, {});
  DictionaryAttr first = builder.getDictionaryAttr(
      {builder.getNamedAttr("argument_types", builder.getArrayAttr({})),
       builder.getNamedAttr("operations", builder.getArrayAttr({producer})),
       builder.getNamedAttr("terminator", branch)});
  DictionaryAttr second = builder.getDictionaryAttr(
      {builder.getNamedAttr("argument_types", builder.getArrayAttr({})),
       builder.getNamedAttr("operations", builder.getArrayAttr({})),
       builder.getNamedAttr("terminator", illegalYield)});
  DictionaryAttr region = builder.getDictionaryAttr(
      {builder.getNamedAttr("blocks", builder.getArrayAttr({first, second}))});
  std::string error;
  EXPECT_TRUE(failed(validateCanonicalRegions(builder.getArrayAttr({region}),
                                                error)));
  EXPECT_NE(error.find("cross-block"), std::string::npos) << error;
}

TEST(CanonicalRegionTest, MultiBlockCFGRecipeRoundTrips) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  IRRewriter rewriter(&context);
  Location location = rewriter.getUnknownLoc();
  ModuleOp module = ModuleOp::create(location);
  rewriter.setInsertionPointToEnd(module.getBody());

  OperationState sourceState(location, "test.cfg_parent");
  sourceState.addRegion();
  Operation *source = rewriter.create(sourceState);
  Type i32 = rewriter.getI32Type();
  SmallVector<Type, 1> argumentTypes{i32};
  SmallVector<Location, 1> argumentLocations{location};
  Region &sourceRegion = source->getRegion(0);
  Block *entry = rewriter.createBlock(&sourceRegion, sourceRegion.end(),
                                      argumentTypes, argumentLocations);
  Block *exit = rewriter.createBlock(&sourceRegion, sourceRegion.end(),
                                     argumentTypes, argumentLocations);
  rewriter.setInsertionPointToEnd(entry);
  OperationState branchState(location, "test.br");
  branchState.addOperands(entry->getArgument(0));
  branchState.addSuccessors(exit);
  rewriter.create(branchState);
  rewriter.setInsertionPointToEnd(exit);
  OperationState yieldState(location, "test.yield");
  yieldState.addOperands(exit->getArgument(0));
  rewriter.create(yieldState);

  std::string error;
  std::optional<ArrayAttr> recipe = captureCanonicalRegions(source, error);
  ASSERT_TRUE(recipe.has_value()) << error;
  ASSERT_TRUE(succeeded(validateCanonicalRegions(*recipe, error))) << error;

  rewriter.setInsertionPointAfter(source);
  OperationState targetState(location, "test.cfg_parent");
  targetState.addRegion();
  Operation *target = rewriter.create(targetState);
  ASSERT_TRUE(succeeded(
      materializeCanonicalRegions(target, *recipe, rewriter, error)))
      << error;
  std::optional<ArrayAttr> captured = captureCanonicalRegions(target, error);
  ASSERT_TRUE(captured.has_value()) << error;
  EXPECT_EQ(*captured, *recipe)
      << "captured: " << printAttribute(*captured)
      << "\nrecipe: " << printAttribute(*recipe);
  ASSERT_EQ(target->getRegion(0).getBlocks().size(), 2u);
  EXPECT_EQ(target->getRegion(0).front().getTerminator()->getSuccessor(0),
            &target->getRegion(0).back());
}

TEST(CanonicalRegionTest, NestedRegionRecipeRoundTripsRecursively) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  IRRewriter rewriter(&context);
  Location location = rewriter.getUnknownLoc();
  ModuleOp module = ModuleOp::create(location);
  rewriter.setInsertionPointToEnd(module.getBody());

  OperationState parentState(location, "test.parent");
  parentState.addRegion();
  Operation *parent = rewriter.create(parentState);
  Region &parentRegion = parent->getRegion(0);
  Block *parentBlock =
      rewriter.createBlock(&parentRegion, parentRegion.end());
  rewriter.setInsertionPointToEnd(parentBlock);
  OperationState nestedState(location, "test.nested");
  nestedState.addRegion();
  Operation *nested = rewriter.create(nestedState);
  Region &nestedRegion = nested->getRegion(0);
  Block *nestedBlock =
      rewriter.createBlock(&nestedRegion, nestedRegion.end());
  rewriter.setInsertionPointToEnd(nestedBlock);
  OperationState nestedYieldState(location, "test.nested_yield");
  rewriter.create(nestedYieldState);
  rewriter.setInsertionPointToEnd(parentBlock);
  OperationState parentYieldState(location, "test.parent_yield");
  rewriter.create(parentYieldState);

  std::string error;
  std::optional<ArrayAttr> recipe = captureCanonicalRegions(parent, error);
  ASSERT_TRUE(recipe.has_value()) << error;
  rewriter.setInsertionPointAfter(parent);
  OperationState targetState(location, "test.parent");
  targetState.addRegion();
  Operation *target = rewriter.create(targetState);
  ASSERT_TRUE(succeeded(
      materializeCanonicalRegions(target, *recipe, rewriter, error)))
      << error;
  std::optional<ArrayAttr> captured = captureCanonicalRegions(target, error);
  ASSERT_TRUE(captured.has_value()) << error;
  EXPECT_EQ(*captured, *recipe)
      << "captured: " << printAttribute(*captured)
      << "\nrecipe: " << printAttribute(*recipe);
  Operation &materializedNested =
      target->getRegion(0).front().front();
  EXPECT_EQ(materializedNested.getNumRegions(), 1u);
  EXPECT_TRUE(llvm::hasSingleElement(materializedNested.getRegion(0)));
}

} // namespace

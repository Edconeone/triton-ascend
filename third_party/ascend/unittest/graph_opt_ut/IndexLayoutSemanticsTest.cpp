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

#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static GraphExprNode makeNode(MLIRContext &context, StringRef operation,
                              Type resultType,
                              ArrayRef<NamedAttribute> extra = {}) {
  Builder builder(&context);
  SmallVector<NamedAttribute, 4> attributes(extra);
  attributes.push_back(builder.getNamedAttr("ags.result_type",
                                             TypeAttr::get(resultType)));
  return GraphExprNode::create(&context, operation, {}, attributes);
}

TEST(IndexLayoutSemanticsTest, DoubleTransposeProvesOriginalCoordinates) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType inputType = RankedTensorType::get({2, 3}, i32);
  RankedTensorType transposedType = RankedTensorType::get({3, 2}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  GraphExprNode input = makeNode(
      context, "ags.input", inputType,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")->denote(input, {});
  ASSERT_TRUE(succeeded(source));

  auto order = builder.getDenseI32ArrayAttr({1, 0});
  GraphExprNode first = makeNode(
      context, "tt.trans", transposedType,
      {builder.getNamedAttr("order", order)});
  FailureOr<TensorDenotation> transposed =
      semantics.lookup("tt.trans")->denote(first, {*source});
  ASSERT_TRUE(succeeded(transposed));
  EXPECT_EQ(transposed->shape, (SmallVector<int64_t, 4>{3, 2}));

  GraphExprNode second = makeNode(
      context, "tt.trans", inputType,
      {builder.getNamedAttr("order", order)});
  FailureOr<TensorDenotation> restored =
      semantics.lookup("tt.trans")->denote(second, {*transposed});
  ASSERT_TRUE(succeeded(restored));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*source, *restored);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(IndexLayoutSemanticsTest, InvalidPermutationHasNoDenotation) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType type = RankedTensorType::get({2, 2}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  GraphExprNode input = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")->denote(input, {});
  ASSERT_TRUE(succeeded(source));

  GraphExprNode invalid = makeNode(
      context, "tt.trans", type,
      {builder.getNamedAttr("order", builder.getDenseI32ArrayAttr({0, 0}))});
  EXPECT_TRUE(failed(semantics.lookup("tt.trans")->denote(invalid, {*source})));
}

TEST(IndexLayoutSemanticsTest, DynamicShapeIdentitySurvivesDoubleTranspose) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType type = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  GraphExprNode input = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(7))});
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")->denote(input, {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_EQ(source->shapeSymbols.size(), 2u);
  EXPECT_NE(source->shapeSymbols[0], source->shapeSymbols[1]);

  auto order = builder.getDenseI32ArrayAttr({1, 0});
  GraphExprNode trans = makeNode(
      context, "tt.trans", type,
      {builder.getNamedAttr("order", order)});
  FailureOr<TensorDenotation> first =
      semantics.lookup("tt.trans")->denote(trans, {*source});
  ASSERT_TRUE(succeeded(first));
  EXPECT_EQ(first->shapeSymbols[0], source->shapeSymbols[1]);
  EXPECT_EQ(first->shapeSymbols[1], source->shapeSymbols[0]);
  FailureOr<TensorDenotation> second =
      semantics.lookup("tt.trans")->denote(trans, {*first});
  ASSERT_TRUE(succeeded(second));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*source, *second);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(IndexLayoutSemanticsTest, UnrelatedDynamicDimensionsDoNotMerge) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType type =
      RankedTensorType::get({ShapedType::kDynamic}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  GraphExprNode lhsNode = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  GraphExprNode rhsNode = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(1))});
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")->denote(lhsNode, {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")->denote(rhsNode, {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  // Isolate the shape-domain decision from the distinct input values.
  rhs->value = lhs->value;
  rhs->defined = lhs->defined;
  rhs->resultIndices = lhs->resultIndices;
  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*lhs, *rhs);
  EXPECT_EQ(proof.status, SemanticProofStatus::Unknown);
  EXPECT_EQ(proof.reason, "symbolic shape relation not proved");
}

TEST(IndexLayoutSemanticsTest,
     SameShapeRelationsAreCanonicalAcrossCommutedOperands) {
  MLIRContext context;
  Builder builder(&context);
  RankedTensorType type =
      RankedTensorType::get({ShapedType::kDynamic}, builder.getI32Type());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  GraphExprNode lhsNode = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  GraphExprNode rhsNode = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(1))});
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")->denote(lhsNode, {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")->denote(rhsNode, {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  GraphExprNode add = makeNode(context, "arith.addi", type);
  FailureOr<TensorDenotation> lhsRhs =
      semantics.lookup("arith.addi")->denote(add, {*lhs, *rhs});
  FailureOr<TensorDenotation> rhsLhs =
      semantics.lookup("arith.addi")->denote(add, {*rhs, *lhs});
  ASSERT_TRUE(succeeded(lhsRhs));
  ASSERT_TRUE(succeeded(rhsLhs));
  ASSERT_EQ(lhsRhs->shapeSymbols.size(), 1u);
  EXPECT_EQ(lhsRhs->shapeSymbols.front(),
            "eq{input-0-dim-0,input-1-dim-0}");
  EXPECT_EQ(lhsRhs->shapeSymbols, rhsLhs->shapeSymbols);

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*lhsRhs, *rhsLhs);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(IndexLayoutSemanticsTest,
     UnconstrainedDynamicResultUsesStableContextShape) {
  MLIRContext context;
  Builder builder(&context);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  RankedTensorType sourceType =
      RankedTensorType::get({1}, builder.getI1Type());
  GraphExprNode sourceNode = makeNode(
      context, "ags.input", sourceType,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")->denote(sourceNode, {});
  ASSERT_TRUE(succeeded(source));

  RankedTensorType resultType = RankedTensorType::get(
      {ShapedType::kDynamic}, builder.getI1Type());
  GraphExprNode broadcast = makeNode(context, "tt.broadcast", resultType);
  FailureOr<TensorDenotation> first =
      semantics.lookup("tt.broadcast")->denote(broadcast, {*source});
  FailureOr<TensorDenotation> second =
      semantics.lookup("tt.broadcast")->denote(broadcast, {*source});
  ASSERT_TRUE(succeeded(first));
  ASSERT_TRUE(succeeded(second));
  ASSERT_EQ(first->shapeSymbols.size(), 1u);
  EXPECT_EQ(first->shapeSymbols.front(), "result-dim-0");

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*first, *second);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(IndexLayoutSemanticsTest, DifferentPhysicalLayoutsDoNotMerge) {
  MLIRContext context;
  Builder builder(&context);
  RankedTensorType type = RankedTensorType::get({8}, builder.getI32Type());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  GraphExprNode input = makeNode(
      context, "ags.input", type,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(0))});
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")->denote(input, {});
  ASSERT_TRUE(succeeded(lhs));
  TensorDenotation rhs = *lhs;
  rhs.layoutIdentity = "#blocked-layout<8>";

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*lhs, rhs);
  EXPECT_EQ(proof.status, SemanticProofStatus::Refuted);
  EXPECT_EQ(proof.reason, "physical layout/encoding mismatch");
}

TEST(IndexLayoutSemanticsTest, DynamicShapeRelationsFlowThroughViewOps) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  RankedTensorType sourceType =
      RankedTensorType::get({ShapedType::kDynamic, 1}, i32);
  GraphExprNode input = makeNode(
      context, "ags.input", sourceType,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(3))});
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")->denote(input, {});
  ASSERT_TRUE(succeeded(source));

  RankedTensorType expandedType =
      RankedTensorType::get({ShapedType::kDynamic, 1, 1}, i32);
  GraphExprNode expandedNode = makeNode(
      context, "tt.expand_dims", expandedType,
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(2))});
  FailureOr<TensorDenotation> expanded =
      semantics.lookup("tt.expand_dims")->denote(expandedNode, {*source});
  ASSERT_TRUE(succeeded(expanded));
  EXPECT_EQ(expanded->shapeSymbols[0], source->shapeSymbols[0]);

  RankedTensorType broadcastType =
      RankedTensorType::get({ShapedType::kDynamic, 4}, i32);
  GraphExprNode broadcastNode =
      makeNode(context, "tt.broadcast", broadcastType);
  FailureOr<TensorDenotation> broadcast =
      semantics.lookup("tt.broadcast")->denote(broadcastNode, {*source});
  ASSERT_TRUE(succeeded(broadcast));
  EXPECT_EQ(broadcast->shapeSymbols[0], source->shapeSymbols[0]);
}

TEST(IndexLayoutSemanticsTest, SplitJoinRoundTripIsProvedFromCoordinateMaps) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType inputType = RankedTensorType::get({4}, i32);
  RankedTensorType joinedType = RankedTensorType::get({4, 2}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  GraphExprNode lhsNode = makeNode(
      context, "ags.input", inputType,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(20))});
  GraphExprNode rhsNode = makeNode(
      context, "ags.input", inputType,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(21))});
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")->denote(lhsNode, {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")->denote(rhsNode, {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  GraphExprNode joinNode = makeNode(context, "tt.join", joinedType);
  FailureOr<TensorDenotation> joined =
      semantics.lookup("tt.join")->denote(joinNode, {*lhs, *rhs});
  ASSERT_TRUE(succeeded(joined));

  GraphExprNode leftSplit = makeNode(
      context, "tt.split", inputType,
      {builder.getNamedAttr("ags.result_index",
                            builder.getI64IntegerAttr(0))});
  GraphExprNode rightSplit = makeNode(
      context, "tt.split", inputType,
      {builder.getNamedAttr("ags.result_index",
                            builder.getI64IntegerAttr(1))});
  FailureOr<TensorDenotation> restoredLhs =
      semantics.lookup("tt.split")->denote(leftSplit, {*joined});
  FailureOr<TensorDenotation> restoredRhs =
      semantics.lookup("tt.split")->denote(rightSplit, {*joined});
  ASSERT_TRUE(succeeded(restoredLhs));
  ASSERT_TRUE(succeeded(restoredRhs));

  TensorSemanticNormalizer normalizer;
  SemanticEquivalenceOracle oracle(normalizer);
  SemanticProofResult lhsProof = oracle.compare(*lhs, *restoredLhs);
  SemanticProofResult rhsProof = oracle.compare(*rhs, *restoredRhs);
  EXPECT_EQ(lhsProof.status, SemanticProofStatus::Proved) << lhsProof.reason;
  EXPECT_EQ(rhsProof.status, SemanticProofStatus::Proved) << rhsProof.reason;
}

TEST(IndexLayoutSemanticsTest, UnsplatSplatRoundTripIsProved) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType tensorType = RankedTensorType::get({1}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  GraphExprNode scalarNode = makeNode(
      context, "ags.input", i32,
      {builder.getNamedAttr("ags.symbol_id", builder.getI64IntegerAttr(30))});
  FailureOr<TensorDenotation> scalar =
      semantics.lookup("ags.input")->denote(scalarNode, {});
  ASSERT_TRUE(succeeded(scalar));
  GraphExprNode splatNode = makeNode(context, "tt.splat", tensorType);
  FailureOr<TensorDenotation> splat =
      semantics.lookup("tt.splat")->denote(splatNode, {*scalar});
  ASSERT_TRUE(succeeded(splat));
  GraphExprNode unsplatNode = makeNode(context, "tt.unsplat", i32);
  FailureOr<TensorDenotation> restored =
      semantics.lookup("tt.unsplat")->denote(unsplatNode, {*splat});
  ASSERT_TRUE(succeeded(restored));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*scalar, *restored);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

} // namespace

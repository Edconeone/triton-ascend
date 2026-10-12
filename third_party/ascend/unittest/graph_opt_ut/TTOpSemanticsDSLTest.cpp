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
#include "TritonToGraph/GraphOperationCatalog.h"
#include "TritonToGraph/GraphUniverseSpec.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <functional>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static bool hasDomain(ArrayRef<GraphSemanticDomain> domains,
                      GraphSemanticDomain expected) {
  return llvm::is_contained(domains, expected);
}

TEST(TTOpSemanticsDSLTest, ReportsExecutableDomainsPerOperation) {
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  ASSERT_GT(registry.size(), 0u);

  for (StringRef operation : registry.getOperationNames()) {
    ArrayRef<GraphSemanticDomain> domains =
        registry.getImplementedDomains(operation);
    EXPECT_EQ(domains.size(), 10u) << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Signature))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Value))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Definedness))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Shape))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Index))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Layout))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Effect))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Memory))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Launch))
        << operation.str();
    EXPECT_TRUE(hasDomain(domains, GraphSemanticDomain::Buffer))
        << operation.str();
  }
}

TEST(TTOpSemanticsDSLTest, EveryRegisteredTritonOperationIsClassified) {
  DialectRegistry dialects;
  dialects.insert<triton::TritonDialect>();
  MLIRContext context(dialects);
  ASSERT_NE(context.getOrLoadDialect<triton::TritonDialect>(), nullptr);
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  unsigned registered = 0;
  for (RegisteredOperationName operation : context.getRegisteredOperations()) {
    if (!operation.getStringRef().starts_with("tt."))
      continue;
    ++registered;
    EXPECT_NE(registry.lookup(operation.getStringRef()), nullptr)
        << operation.getStringRef().str();
  }
  EXPECT_GE(registered, 50u);
}

TEST(TTOpSemanticsDSLTest, DescriptorReduceUsesFiniteEnumAttributeDomain) {
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options, /*maxCandidateOperands=*/4);
  auto schema = llvm::find_if(schemas, [](const TTIRCandidateOpSchema &entry) {
    return entry.operationName == "tt.descriptor_reduce" &&
           entry.operandCount == 3;
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->constructible);
  EXPECT_TRUE(schema->offlineOnly);
  ASSERT_EQ(schema->attributeBindings.size(), 1u);
  EXPECT_EQ(schema->attributeBindings.front().attributeName, "kind");
  EXPECT_EQ(schema->attributeBindings.front().domainName,
            "descriptor_reduce_kind");
  EXPECT_EQ(schema->attributeBindings.front().storageKind,
            TTIRCandidateAttributeBinding::StorageKind::I32);
}

TEST(TTOpSemanticsDSLTest, IndirectMemorySchemasAreDeclarativeCandidates) {
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options, /*maxCandidateOperands=*/4);
  auto findSchema = [&](StringRef operation, unsigned operands) {
    return llvm::find_if(schemas, [&](const TTIRCandidateOpSchema &entry) {
      return entry.operationName == operation &&
             entry.operandCount == operands;
    });
  };

  for (unsigned operands : {2u, 3u, 4u}) {
    auto schema = findSchema("ascend.indirect_load", operands);
    ASSERT_NE(schema, schemas.end());
    EXPECT_TRUE(schema->constructible);
    EXPECT_TRUE(schema->offlineOnly);
    EXPECT_EQ(schema->effectKind, TTIRCandidateEffectKind::Read);
  }
  for (unsigned operands : {3u, 4u}) {
    auto schema = findSchema("ascend.indirect_store", operands);
    ASSERT_NE(schema, schemas.end());
    EXPECT_TRUE(schema->constructible);
    EXPECT_TRUE(schema->offlineOnly);
    EXPECT_EQ(schema->effectKind, TTIRCandidateEffectKind::Write);
  }
}

TEST(TTOpSemanticsDSLTest, StridedMemorySchemasCoverRanksOneThroughFive) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  Type i64 = builder.getI64Type();
  Type pointer = triton::PointerType::get(f32, /*addressSpace=*/1);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options,
                                          /*maxCandidateOperands=*/13);

  for (unsigned rank = 1; rank <= 5; ++rank) {
    unsigned operandCount = 3 + 2 * rank;
    SmallVector<int64_t, 5> shape(rank, 2);
    RankedTensorType tensor = RankedTensorType::get(shape, f32);
    SmallVector<Type, 13> loadOperands = {pointer, i64, f32};
    SmallVector<Type, 13> storeOperands = {pointer, tensor, i64};
    loadOperands.append(rank, i64);
    loadOperands.append(rank, i64);
    storeOperands.append(rank, i64);
    storeOperands.append(rank, i64);

    auto load = llvm::find_if(
        schemas, [&](const TTIRCandidateOpSchema &schema) {
          return schema.operationName == "ascend.stride_load" &&
                 schema.operandCount == operandCount;
        });
    ASSERT_NE(load, schemas.end()) << rank;
    EXPECT_TRUE(load->constructible);
    EXPECT_TRUE(load->offlineOnly);
    EXPECT_EQ(load->effectKind, TTIRCandidateEffectKind::Read);
    EXPECT_TRUE(load->isWellTyped(tensor, loadOperands,
                                  builder.getDictionaryAttr({})))
        << rank;

    auto store = llvm::find_if(
        schemas, [&](const TTIRCandidateOpSchema &schema) {
          return schema.operationName == "ascend.stride_store" &&
                 schema.operandCount == operandCount;
        });
    ASSERT_NE(store, schemas.end()) << rank;
    EXPECT_TRUE(store->constructible);
    EXPECT_TRUE(store->offlineOnly);
    EXPECT_EQ(store->effectKind, TTIRCandidateEffectKind::Write);
    EXPECT_TRUE(store->isWellTyped(none, storeOperands,
                                   builder.getDictionaryAttr({})))
        << rank;
  }
}

static GraphExprNode makeSemanticNode(MLIRContext &context,
                                      StringRef operation, Type resultType,
                                      int64_t symbol = -1) {
  Builder builder(&context);
  SmallVector<NamedAttribute, 2> attributes;
  attributes.push_back(builder.getNamedAttr("ags.result_type",
                                             TypeAttr::get(resultType)));
  if (symbol >= 0)
    attributes.push_back(builder.getNamedAttr(
        "ags.symbol_id", builder.getI64IntegerAttr(symbol)));
  return GraphExprNode::create(&context, operation, {}, attributes);
}

TEST(TTOpSemanticsDSLTest, ProvesIndirectLoadEqualsTTIRAddressGraph) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  Type pointer = triton::PointerType::get(f32, /*addressSpace=*/1);
  RankedTensorType pointerTensor = RankedTensorType::get({4}, pointer);
  RankedTensorType offsetTensor =
      RankedTensorType::get({4}, builder.getI64Type());
  RankedTensorType maskTensor =
      RankedTensorType::get({4}, builder.getI1Type());
  RankedTensorType valueTensor = RankedTensorType::get({4}, f32);
  Type none = NoneType::get(&context);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> base = input(pointer, 0);
  FailureOr<TensorDenotation> offsets = input(offsetTensor, 1);
  FailureOr<TensorDenotation> mask = input(maskTensor, 2);
  FailureOr<TensorDenotation> other = input(valueTensor, 3);
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory", none), {});
  ASSERT_TRUE(succeeded(base));
  ASSERT_TRUE(succeeded(offsets));
  ASSERT_TRUE(succeeded(mask));
  ASSERT_TRUE(succeeded(other));
  ASSERT_TRUE(succeeded(memory));

  FailureOr<TensorDenotation> splat =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", pointerTensor),
                   {*base});
  ASSERT_TRUE(succeeded(splat));
  FailureOr<TensorDenotation> address =
      semantics.lookup("tt.addptr")
          ->denote(makeSemanticNode(context, "tt.addptr", pointerTensor),
                   {*splat, *offsets});
  ASSERT_TRUE(succeeded(address));

  GraphExprNode load = GraphExprNode::create(
      &context, "tt.load", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(valueTensor)),
       builder.getNamedAttr("operandSegmentSizes",
                            builder.getDenseI32ArrayAttr({1, 1, 1}))});
  FailureOr<TensorDenotation> expanded =
      semantics.lookup("tt.load")
          ->denote(load, {*address, *mask, *other, *memory});
  ASSERT_TRUE(succeeded(expanded));

  GraphExprNode indirect = GraphExprNode::create(
      &context, "ascend.indirect_load", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(valueTensor)),
       builder.getNamedAttr("operandSegmentSizes",
                            builder.getDenseI32ArrayAttr({1, 1, 1, 1})),
       builder.getNamedAttr("isVolatile", builder.getBoolAttr(false))});
  FailureOr<TensorDenotation> compact =
      semantics.lookup("ascend.indirect_load")
          ->denote(indirect, {*base, *offsets, *mask, *other, *memory});
  ASSERT_TRUE(succeeded(compact));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*expanded, *compact);
  EXPECT_TRUE(proof.isProved()) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, ProvesRankOneStrideLoadFromAddressArithmetic) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  Type i64 = builder.getI64Type();
  Type pointer = triton::PointerType::get(f32, /*addressSpace=*/1);
  Type none = builder.getNoneType();
  RankedTensorType i32Tensor =
      RankedTensorType::get({4}, builder.getI32Type());
  RankedTensorType i64Tensor = RankedTensorType::get({4}, i64);
  RankedTensorType pointerTensor = RankedTensorType::get({4}, pointer);
  RankedTensorType maskTensor =
      RankedTensorType::get({4}, builder.getI1Type());
  RankedTensorType valueTensor = RankedTensorType::get({4}, f32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> base = input(pointer, 0);
  FailureOr<TensorDenotation> offset = input(i64, 1);
  FailureOr<TensorDenotation> other = input(f32, 2);
  FailureOr<TensorDenotation> stride = input(i64, 3);
  FailureOr<TensorDenotation> numel = input(i64, 4);
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory", none), {});
  ASSERT_TRUE(succeeded(base));
  ASSERT_TRUE(succeeded(offset));
  ASSERT_TRUE(succeeded(other));
  ASSERT_TRUE(succeeded(stride));
  ASSERT_TRUE(succeeded(numel));
  ASSERT_TRUE(succeeded(memory));

  GraphExprNode rangeNode = GraphExprNode::create(
      &context, "tt.make_range", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i32Tensor)),
       builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("end", builder.getI32IntegerAttr(4))});
  FailureOr<TensorDenotation> range =
      semantics.lookup("tt.make_range")->denote(rangeNode, {});
  ASSERT_TRUE(succeeded(range));
  FailureOr<TensorDenotation> wideRange =
      semantics.lookup("arith.extsi")
          ->denote(makeSemanticNode(context, "arith.extsi", i64Tensor),
                   {*range});
  FailureOr<TensorDenotation> strideTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*stride});
  FailureOr<TensorDenotation> offsetTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*offset});
  FailureOr<TensorDenotation> numelTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*numel});
  FailureOr<TensorDenotation> otherTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", valueTensor),
                   {*other});
  ASSERT_TRUE(succeeded(wideRange));
  ASSERT_TRUE(succeeded(strideTensor));
  ASSERT_TRUE(succeeded(offsetTensor));
  ASSERT_TRUE(succeeded(numelTensor));
  ASSERT_TRUE(succeeded(otherTensor));

  FailureOr<TensorDenotation> scaled =
      semantics.lookup("arith.muli")
          ->denote(makeSemanticNode(context, "arith.muli", i64Tensor),
                   {*wideRange, *strideTensor});
  ASSERT_TRUE(succeeded(scaled));
  FailureOr<TensorDenotation> offsets =
      semantics.lookup("arith.addi")
          ->denote(makeSemanticNode(context, "arith.addi", i64Tensor),
                   {*offsetTensor, *scaled});
  ASSERT_TRUE(succeeded(offsets));
  FailureOr<TensorDenotation> baseTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", pointerTensor),
                   {*base});
  ASSERT_TRUE(succeeded(baseTensor));
  FailureOr<TensorDenotation> addresses =
      semantics.lookup("tt.addptr")
          ->denote(makeSemanticNode(context, "tt.addptr", pointerTensor),
                   {*baseTensor, *offsets});
  ASSERT_TRUE(succeeded(addresses));

  GraphExprNode compareNode = GraphExprNode::create(
      &context, "arith.cmpi", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(maskTensor)),
       builder.getNamedAttr(
           "predicate",
           builder.getI64IntegerAttr(
               static_cast<int64_t>(arith::CmpIPredicate::slt)))});
  FailureOr<TensorDenotation> mask =
      semantics.lookup("arith.cmpi")
          ->denote(compareNode, {*wideRange, *numelTensor});
  ASSERT_TRUE(succeeded(mask));
  GraphExprNode loadNode = GraphExprNode::create(
      &context, "tt.load", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(valueTensor)),
       builder.getNamedAttr("operandSegmentSizes",
                            builder.getDenseI32ArrayAttr({1, 1, 1}))});
  FailureOr<TensorDenotation> expanded =
      semantics.lookup("tt.load")
          ->denote(loadNode,
                   {*addresses, *mask, *otherTensor, *memory});
  ASSERT_TRUE(succeeded(expanded));

  FailureOr<TensorDenotation> compact =
      semantics.lookup("ascend.stride_load")
          ->denote(makeSemanticNode(context, "ascend.stride_load",
                                    valueTensor),
                   {*base, *offset, *other, *stride, *numel, *memory});
  ASSERT_TRUE(succeeded(compact));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*expanded, *compact);
  EXPECT_TRUE(proof.isProved()) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, ProvesRankOneStrideStoreFromAddressArithmetic) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  Type i64 = builder.getI64Type();
  Type pointer = triton::PointerType::get(f32, /*addressSpace=*/1);
  Type none = builder.getNoneType();
  RankedTensorType i32Tensor =
      RankedTensorType::get({4}, builder.getI32Type());
  RankedTensorType i64Tensor = RankedTensorType::get({4}, i64);
  RankedTensorType pointerTensor = RankedTensorType::get({4}, pointer);
  RankedTensorType maskTensor =
      RankedTensorType::get({4}, builder.getI1Type());
  RankedTensorType valueTensor = RankedTensorType::get({4}, f32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> base = input(pointer, 0);
  FailureOr<TensorDenotation> source = input(valueTensor, 1);
  FailureOr<TensorDenotation> offset = input(i64, 2);
  FailureOr<TensorDenotation> stride = input(i64, 3);
  FailureOr<TensorDenotation> numel = input(i64, 4);
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory", none), {});
  ASSERT_TRUE(succeeded(base));
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(offset));
  ASSERT_TRUE(succeeded(stride));
  ASSERT_TRUE(succeeded(numel));
  ASSERT_TRUE(succeeded(memory));

  GraphExprNode rangeNode = GraphExprNode::create(
      &context, "tt.make_range", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i32Tensor)),
       builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("end", builder.getI32IntegerAttr(4))});
  FailureOr<TensorDenotation> range =
      semantics.lookup("tt.make_range")->denote(rangeNode, {});
  ASSERT_TRUE(succeeded(range));
  FailureOr<TensorDenotation> wideRange =
      semantics.lookup("arith.extsi")
          ->denote(makeSemanticNode(context, "arith.extsi", i64Tensor),
                   {*range});
  FailureOr<TensorDenotation> strideTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*stride});
  FailureOr<TensorDenotation> offsetTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*offset});
  FailureOr<TensorDenotation> numelTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", i64Tensor),
                   {*numel});
  ASSERT_TRUE(succeeded(wideRange));
  ASSERT_TRUE(succeeded(strideTensor));
  ASSERT_TRUE(succeeded(offsetTensor));
  ASSERT_TRUE(succeeded(numelTensor));
  FailureOr<TensorDenotation> scaled =
      semantics.lookup("arith.muli")
          ->denote(makeSemanticNode(context, "arith.muli", i64Tensor),
                   {*wideRange, *strideTensor});
  ASSERT_TRUE(succeeded(scaled));
  FailureOr<TensorDenotation> offsets =
      semantics.lookup("arith.addi")
          ->denote(makeSemanticNode(context, "arith.addi", i64Tensor),
                   {*offsetTensor, *scaled});
  ASSERT_TRUE(succeeded(offsets));
  FailureOr<TensorDenotation> baseTensor =
      semantics.lookup("tt.splat")
          ->denote(makeSemanticNode(context, "tt.splat", pointerTensor),
                   {*base});
  ASSERT_TRUE(succeeded(baseTensor));
  FailureOr<TensorDenotation> addresses =
      semantics.lookup("tt.addptr")
          ->denote(makeSemanticNode(context, "tt.addptr", pointerTensor),
                   {*baseTensor, *offsets});
  ASSERT_TRUE(succeeded(addresses));
  GraphExprNode compareNode = GraphExprNode::create(
      &context, "arith.cmpi", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(maskTensor)),
       builder.getNamedAttr(
           "predicate",
           builder.getI64IntegerAttr(
               static_cast<int64_t>(arith::CmpIPredicate::slt)))});
  FailureOr<TensorDenotation> mask =
      semantics.lookup("arith.cmpi")
          ->denote(compareNode, {*wideRange, *numelTensor});
  ASSERT_TRUE(succeeded(mask));

  FailureOr<TensorDenotation> expanded =
      semantics.lookup("tt.store")
          ->denote(makeSemanticNode(context, "tt.store", none),
                   {*addresses, *source, *mask, *memory});
  FailureOr<TensorDenotation> compact =
      semantics.lookup("ascend.stride_store")
          ->denote(makeSemanticNode(context, "ascend.stride_store", none),
                   {*base, *source, *offset, *stride, *numel, *memory});
  ASSERT_TRUE(succeeded(expanded));
  ASSERT_TRUE(succeeded(compact));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*expanded, *compact);
  EXPECT_TRUE(proof.isProved()) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, ExecutesImportedMapElementwiseScalarRegion) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  RankedTensorType tensorType = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", tensorType, 0), {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", tensorType, 1), {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  FailureOr<TensorDenotation> direct =
      semantics.lookup("arith.addi")
          ->denote(makeSemanticNode(context, "arith.addi", tensorType),
                   {*lhs, *rhs});
  ASSERT_TRUE(succeeded(direct));

  ArrayAttr region = buildCanonicalBinaryRegion(
      &context, i32, "arith.addi", "tt.map_elementwise.return");
  GraphExprNode map = GraphExprNode::create(
      &context, "tt.map_elementwise", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(tensorType)),
       builder.getNamedAttr("ags.canonical_regions", region),
       builder.getNamedAttr("pack", builder.getI32IntegerAttr(1))});
  FailureOr<TensorDenotation> mapped =
      semantics.lookup("tt.map_elementwise")->denote(map, {*lhs, *rhs});
  ASSERT_TRUE(succeeded(mapped));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*direct, *mapped);
  EXPECT_TRUE(proof.isProved()) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, ExecutesPureCallSummaryRegion) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 1), {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));
  FailureOr<TensorDenotation> direct =
      semantics.lookup("arith.addi")
          ->denote(makeSemanticNode(context, "arith.addi", i32),
                   {*lhs, *rhs});
  ASSERT_TRUE(succeeded(direct));

  ArrayAttr summary = buildCanonicalBinaryRegion(
      &context, i32, "arith.addi", "tt.return");
  GraphExprNode call = GraphExprNode::create(
      &context, "tt.call", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i32)),
       builder.getNamedAttr("ags.call_summary_regions", summary),
       builder.getNamedAttr("callee",
                            FlatSymbolRefAttr::get(&context, "add"))});
  FailureOr<TensorDenotation> summarized =
      semantics.lookup("tt.call")->denote(call, {*lhs, *rhs});
  ASSERT_TRUE(succeeded(summarized));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*direct, *summarized);
  EXPECT_TRUE(proof.isProved()) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, DotScaledExpandsToMicroscaleReduction) {
  MLIRContext context;
  Builder builder(&context);
  Type f16 = builder.getF16Type();
  Type f32 = builder.getF32Type();
  RankedTensorType lhsType = RankedTensorType::get({2, 4}, f16);
  RankedTensorType rhsType = RankedTensorType::get({4, 2}, f16);
  RankedTensorType resultType = RankedTensorType::get({2, 2}, f32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> lhs = input(lhsType, 0);
  FailureOr<TensorDenotation> rhs = input(rhsType, 1);
  FailureOr<TensorDenotation> acc = input(resultType, 2);
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));
  ASSERT_TRUE(succeeded(acc));

  GraphExprNode scaled = GraphExprNode::create(
      &context, "tt.dot_scaled", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(resultType)),
       builder.getNamedAttr("operandSegmentSizes",
                            builder.getDenseI32ArrayAttr({1, 1, 1, 0, 0})),
       builder.getNamedAttr("a_elem_type", builder.getI32IntegerAttr(6)),
       builder.getNamedAttr("b_elem_type", builder.getI32IntegerAttr(6)),
       builder.getNamedAttr("fastMath", builder.getBoolAttr(false)),
       builder.getNamedAttr("lhs_k_pack", builder.getBoolAttr(true)),
       builder.getNamedAttr("rhs_k_pack", builder.getBoolAttr(true))});
  FailureOr<TensorDenotation> denotation =
      semantics.lookup("tt.dot_scaled")->denote(scaled,
                                                  {*lhs, *rhs, *acc});
  ASSERT_TRUE(succeeded(denotation));
  EXPECT_EQ(denotation->value.getKind(), TensorSemanticKind::Add);
  std::function<bool(const TensorSemanticExpr &)> containsMicroscale =
      [&](const TensorSemanticExpr &expression) {
        if (expression.getKind() == TensorSemanticKind::MicroscaleDecode)
          return true;
        return llvm::any_of(expression.getOperands(), containsMicroscale);
      };
  EXPECT_TRUE(containsMicroscale(denotation->value));
  EXPECT_TRUE(denotation->defined);
}

TEST(TTOpSemanticsDSLTest, RejectsUnmodeledVolatileAndBoundaryMemoryOps) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  Block block;
  Value address = block.addArgument(pointer, builder.getUnknownLoc());
  Value stored = block.addArgument(i32, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);

  OperationState plainLoad(builder.getUnknownLoc(), "tt.load");
  plainLoad.addOperands(address);
  plainLoad.addTypes(i32);
  Operation *supportedLoad = opBuilder.create(plainLoad);

  OperationState volatileLoad(builder.getUnknownLoc(), "tt.load");
  volatileLoad.addOperands(address);
  volatileLoad.addTypes(i32);
  volatileLoad.addAttribute("isVolatile", builder.getBoolAttr(true));
  Operation *unsupportedVolatile = opBuilder.create(volatileLoad);

  OperationState boundaryStore(builder.getUnknownLoc(), "tt.store");
  boundaryStore.addOperands({address, stored});
  boundaryStore.addAttribute("boundaryCheck",
                             builder.getDenseI32ArrayAttr({0}));
  Operation *unsupportedBoundary = opBuilder.create(boundaryStore);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  EXPECT_TRUE(semantics.supports(supportedLoad));
  EXPECT_FALSE(semantics.supports(unsupportedVolatile));
  EXPECT_FALSE(semantics.supports(unsupportedBoundary));
}

TEST(TTOpSemanticsDSLTest, NonVolatileLoadEffectsAreNotObservable) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointerType = RankedTensorType::get({4}, pointer);
  RankedTensorType maskType =
      RankedTensorType::get({4}, builder.getI1Type());
  RankedTensorType resultType = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> pointerValue =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", pointerType, 0),
                   {});
  FailureOr<TensorDenotation> lhsMask =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", maskType, 1), {});
  FailureOr<TensorDenotation> rhsMask =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", maskType, 2), {});
  ASSERT_TRUE(succeeded(pointerValue));
  ASSERT_TRUE(succeeded(lhsMask));
  ASSERT_TRUE(succeeded(rhsMask));

  GraphExprNode load = makeSemanticNode(context, "tt.load", resultType);
  FailureOr<TensorDenotation> lhs = semantics.lookup("tt.load")
                                          ->denote(load, {*pointerValue,
                                                          *lhsMask});
  FailureOr<TensorDenotation> rhs = semantics.lookup("tt.load")
                                          ->denote(load, {*pointerValue,
                                                          *rhsMask});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));
  EXPECT_TRUE(lhs->effect);
  EXPECT_TRUE(lhs->memory);

  // Keep Value/Definedness identical so this assertion isolates Effect. A
  // non-volatile read is represented by its MemorySSA-dependent value, not by
  // an independently observable event; changing only its unused lane mask
  // must therefore not prevent an otherwise valid equivalence proof.
  rhs->value = lhs->value;
  rhs->defined = lhs->defined;
  TensorSemanticNormalizer normalizer;
  SemanticProofResult comparison =
      SemanticEquivalenceOracle(normalizer).compare(*lhs, *rhs);
  EXPECT_EQ(comparison.status, SemanticProofStatus::Proved)
      << comparison.reason;
}

TEST(TTOpSemanticsDSLTest, ProofCertificateRecordsEffectAndMemoryDomains) {
  MLIRContext context;
  Builder builder(&context);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> input =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input",
                                    RankedTensorType::get({4},
                                                          builder.getI32Type()),
                                    0),
                   {});
  ASSERT_TRUE(succeeded(input));
  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*input, *input);
  ASSERT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
  ASSERT_TRUE(proof.proof);
  EXPECT_TRUE(hasDomain(proof.proof->verifiedDomains,
                        GraphSemanticDomain::Effect));
  EXPECT_TRUE(hasDomain(proof.proof->verifiedDomains,
                        GraphSemanticDomain::Memory));
  EXPECT_TRUE(hasDomain(proof.proof->verifiedDomains,
                        GraphSemanticDomain::Layout));
}

TEST(TTOpSemanticsDSLTest, MaskedOffStoreIsMemoryIdentity) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointerType = RankedTensorType::get({4}, pointer);
  RankedTensorType valueType = RankedTensorType::get({4}, i32);
  RankedTensorType maskType =
      RankedTensorType::get({4}, builder.getI1Type());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> pointerValue =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", pointerType, 0),
                   {});
  FailureOr<TensorDenotation> storedValue =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", valueType, 1), {});
  FailureOr<TensorDenotation> mask =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", maskType, 2), {});
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(pointerValue));
  ASSERT_TRUE(succeeded(storedValue));
  ASSERT_TRUE(succeeded(mask));
  mask->value = TensorSemanticExpr::boolean(&context, false);

  FailureOr<TensorDenotation> store =
      semantics.lookup("tt.store")
          ->denote(makeSemanticNode(context, "tt.store",
                                    NoneType::get(&context)),
                   {*pointerValue, *storedValue, *mask, *memory});
  ASSERT_TRUE(succeeded(store));
  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*memory, *store);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, LargeTensorStoreUsesQuantifiedValidity) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointerType = RankedTensorType::get({8, 16}, pointer);
  RankedTensorType valueType = RankedTensorType::get({8, 16}, i32);
  RankedTensorType maskType =
      RankedTensorType::get({8, 16}, builder.getI1Type());
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> pointerValue = input(pointerType, 0);
  FailureOr<TensorDenotation> storedValue = input(valueType, 1);
  FailureOr<TensorDenotation> mask = input(maskType, 2);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(pointerValue));
  ASSERT_TRUE(succeeded(storedValue));
  ASSERT_TRUE(succeeded(mask));

  FailureOr<TensorDenotation> store =
      semantics.lookup("tt.store")
          ->denote(makeSemanticNode(context, "tt.store",
                                    NoneType::get(&context)),
                   {*pointerValue, *storedValue, *mask, *memory});
  ASSERT_TRUE(succeeded(store));
  std::function<bool(const TensorSemanticExpr &)> containsForAll =
      [&](const TensorSemanticExpr &expression) {
        if (expression.getKind() == TensorSemanticKind::ForAll)
          return true;
        return llvm::any_of(expression.getOperands(), containsForAll);
      };
  EXPECT_TRUE(containsForAll(store->defined));
  EXPECT_TRUE(containsForAll(store->memoryDefined));

  TensorSemanticNormalizer normalizer;
  TensorDenotation normalized = normalizer.normalize(*store).denotation;
  std::string key = getStableTensorDenotationKey(normalized);
  EXPECT_TRUE(StringRef(key).starts_with("sha256:"));
  EXPECT_EQ(key.size(), 71u);
}

TEST(TTOpSemanticsDSLTest, StoreOrderDoesNotMergeWithoutAliasProof) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointerType = RankedTensorType::get({4}, pointer);
  RankedTensorType valueType = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> pointerA = input(pointerType, 0);
  FailureOr<TensorDenotation> pointerB = input(pointerType, 1);
  FailureOr<TensorDenotation> value = input(valueType, 2);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(pointerA));
  ASSERT_TRUE(succeeded(pointerB));
  ASSERT_TRUE(succeeded(value));
  GraphExprNode storeNode = makeSemanticNode(
      context, "tt.store", NoneType::get(&context));
  auto write = [&](const TensorDenotation &pointerValue,
                   const TensorDenotation &state) {
    return semantics.lookup("tt.store")
        ->denote(storeNode, {pointerValue, *value, state});
  };
  FailureOr<TensorDenotation> aThen = write(*pointerA, *memory);
  FailureOr<TensorDenotation> bThen = write(*pointerB, *memory);
  ASSERT_TRUE(succeeded(aThen));
  ASSERT_TRUE(succeeded(bThen));
  FailureOr<TensorDenotation> ab = write(*pointerB, *aThen);
  FailureOr<TensorDenotation> ba = write(*pointerA, *bThen);
  ASSERT_TRUE(succeeded(ab));
  ASSERT_TRUE(succeeded(ba));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*ab, *ba);
  EXPECT_EQ(proof.status, SemanticProofStatus::Unknown);
  EXPECT_EQ(proof.reason, "normal forms differ; counterexample required");
}

TEST(TTOpSemanticsDSLTest, LaterStoreOverwritesSameAddress) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> address = input(pointer, 0);
  FailureOr<TensorDenotation> firstValue = input(i32, 1);
  FailureOr<TensorDenotation> finalValue = input(i32, 2);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(firstValue));
  ASSERT_TRUE(succeeded(finalValue));

  GraphExprNode storeNode = makeSemanticNode(
      context, "tt.store", NoneType::get(&context));
  FailureOr<TensorDenotation> first =
      semantics.lookup("tt.store")
          ->denote(storeNode, {*address, *firstValue, *memory});
  ASSERT_TRUE(succeeded(first));
  FailureOr<TensorDenotation> overwritten =
      semantics.lookup("tt.store")
          ->denote(storeNode, {*address, *finalValue, *first});
  FailureOr<TensorDenotation> direct =
      semantics.lookup("tt.store")
          ->denote(storeNode, {*address, *finalValue, *memory});
  ASSERT_TRUE(succeeded(overwritten));
  ASSERT_TRUE(succeeded(direct));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*overwritten, *direct);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, AdjacentTensorStoresEqualOneWideStore) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointer4 = RankedTensorType::get({4}, pointer);
  RankedTensorType pointer8 = RankedTensorType::get({8}, pointer);
  RankedTensorType value4 = RankedTensorType::get({4}, i32);
  RankedTensorType value8 = RankedTensorType::get({8}, i32);
  RankedTensorType index4 = RankedTensorType::get({4}, i32);
  RankedTensorType index8 = RankedTensorType::get({8}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto nodeWithIntegers = [&](StringRef operation, Type resultType,
                              ArrayRef<std::pair<StringRef, int64_t>> values) {
    SmallVector<NamedAttribute, 4> attributes = {
        builder.getNamedAttr("ags.result_type", TypeAttr::get(resultType))};
    for (auto [name, value] : values)
      attributes.push_back(
          builder.getNamedAttr(name, builder.getI32IntegerAttr(value)));
    return GraphExprNode::create(&context, operation, {}, attributes);
  };
  auto insertSliceNode = [&](int64_t offset) {
    SmallVector<NamedAttribute, 4> attributes = {
        builder.getNamedAttr("ags.result_type", TypeAttr::get(value8)),
        builder.getNamedAttr("static_offsets",
                             builder.getDenseI64ArrayAttr({offset})),
        builder.getNamedAttr("static_sizes",
                             builder.getDenseI64ArrayAttr({4})),
        builder.getNamedAttr("static_strides",
                             builder.getDenseI64ArrayAttr({1}))};
    return GraphExprNode::create(&context, "tensor.insert_slice", {},
                                 attributes);
  };
  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  auto splatBase = [&](const TensorDenotation &base, Type resultType) {
    return semantics.lookup("tt.splat")
        ->denote(makeSemanticNode(context, "tt.splat", resultType), {base});
  };
  auto range = [&](Type resultType, int64_t start, int64_t end) {
    return semantics.lookup("tt.make_range")
        ->denote(nodeWithIntegers("tt.make_range", resultType,
                                  {{"start", start}, {"end", end}}),
                 {});
  };
  auto address = [&](const TensorDenotation &base,
                     const TensorDenotation &offset, Type resultType) {
    return semantics.lookup("tt.addptr")
        ->denote(makeSemanticNode(context, "tt.addptr", resultType),
                 {base, offset});
  };

  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> base = input(pointer, 0);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(base));
  FailureOr<TensorDenotation> base4 = splatBase(*base, pointer4);
  FailureOr<TensorDenotation> base8 = splatBase(*base, pointer8);
  FailureOr<TensorDenotation> lowRange = range(index4, 0, 4);
  FailureOr<TensorDenotation> highRange = range(index4, 4, 8);
  FailureOr<TensorDenotation> wideRange = range(index8, 0, 8);
  ASSERT_TRUE(succeeded(base4));
  ASSERT_TRUE(succeeded(base8));
  ASSERT_TRUE(succeeded(lowRange));
  ASSERT_TRUE(succeeded(highRange));
  ASSERT_TRUE(succeeded(wideRange));
  FailureOr<TensorDenotation> lowAddress =
      address(*base4, *lowRange, pointer4);
  FailureOr<TensorDenotation> highAddress =
      address(*base4, *highRange, pointer4);
  FailureOr<TensorDenotation> wideAddress =
      address(*base8, *wideRange, pointer8);
  ASSERT_TRUE(succeeded(lowAddress));
  ASSERT_TRUE(succeeded(highAddress));
  ASSERT_TRUE(succeeded(wideAddress));

  FailureOr<TensorDenotation> lowValue = input(value4, 1);
  FailureOr<TensorDenotation> highValue = input(value4, 2);
  ASSERT_TRUE(succeeded(lowValue));
  ASSERT_TRUE(succeeded(highValue));
  FailureOr<TensorDenotation> empty =
      semantics.lookup("tensor.empty")
          ->denote(makeSemanticNode(context, "tensor.empty", value8), {});
  ASSERT_TRUE(succeeded(empty));
  FailureOr<TensorDenotation> lowInserted =
      semantics.lookup("tensor.insert_slice")
          ->denote(insertSliceNode(0), {*lowValue, *empty});
  ASSERT_TRUE(succeeded(lowInserted));
  FailureOr<TensorDenotation> packedValue =
      semantics.lookup("tensor.insert_slice")
          ->denote(insertSliceNode(4), {*highValue, *lowInserted});
  ASSERT_TRUE(succeeded(packedValue));

  GraphExprNode store = makeSemanticNode(
      context, "tt.store", NoneType::get(&context));
  FailureOr<TensorDenotation> high =
      semantics.lookup("tt.store")
          ->denote(store, {*highAddress, *highValue, *memory});
  ASSERT_TRUE(succeeded(high));
  FailureOr<TensorDenotation> split =
      semantics.lookup("tt.store")
          ->denote(store, {*lowAddress, *lowValue, *high});
  FailureOr<TensorDenotation> wide =
      semantics.lookup("tt.store")
          ->denote(store, {*wideAddress, *packedValue, *memory});
  ASSERT_TRUE(succeeded(split));
  ASSERT_TRUE(succeeded(wide));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*split, *wide);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, MaskedOffAtomicProjectsIdentityMemoryState) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type i1 = builder.getI1Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  RankedTensorType pointerType = RankedTensorType::get({4}, pointer);
  RankedTensorType valueType = RankedTensorType::get({4}, i32);
  RankedTensorType maskType = RankedTensorType::get({4}, i1);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> pointerValue = input(pointerType, 0);
  FailureOr<TensorDenotation> update = input(valueType, 1);
  FailureOr<TensorDenotation> mask = input(maskType, 2);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(pointerValue));
  ASSERT_TRUE(succeeded(update));
  ASSERT_TRUE(succeeded(mask));
  mask->value = TensorSemanticExpr::boolean(&context, false);

  FailureOr<TensorDenotation> atomic =
      semantics.lookup("tt.atomic_rmw")
          ->denote(makeSemanticNode(context, "tt.atomic_rmw", valueType),
                   {*pointerValue, *update, *mask, *memory});
  ASSERT_TRUE(succeeded(atomic));
  FailureOr<TensorDenotation> projected =
      semantics.lookup("ags.memory_after")
          ->denote(makeSemanticNode(context, "ags.memory_after",
                                    NoneType::get(&context)),
                   {*atomic});
  ASSERT_TRUE(succeeded(projected));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*memory, *projected);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, TrueMaskedAtomicEqualsUnmaskedAtomic) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type i1 = builder.getI1Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> address = input(pointer, 0);
  FailureOr<TensorDenotation> update = input(i32, 1);
  FailureOr<TensorDenotation> mask = input(i1, 2);
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(address));
  ASSERT_TRUE(succeeded(update));
  ASSERT_TRUE(succeeded(mask));
  mask->value = TensorSemanticExpr::boolean(&context, true);
  mask->defined = TensorSemanticExpr::boolean(&context, true);

  GraphExprNode atomic =
      makeSemanticNode(context, "tt.atomic_rmw", i32);
  FailureOr<TensorDenotation> masked =
      semantics.lookup("tt.atomic_rmw")
          ->denote(atomic, {*address, *update, *mask, *memory});
  FailureOr<TensorDenotation> unmasked =
      semantics.lookup("tt.atomic_rmw")
          ->denote(atomic, {*address, *update, *memory});
  ASSERT_TRUE(succeeded(masked));
  ASSERT_TRUE(succeeded(unmasked));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*masked, *unmasked);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, ProgramIdDependsOnExplicitLaunchState) {
  MLIRContext context;
  Builder builder(&context);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> launch =
      semantics.lookup("ags.launch")
          ->denote(makeSemanticNode(context, "ags.launch",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(launch));
  EXPECT_TRUE(launch->isLaunchStateToken);
  EXPECT_FALSE(launch->isMemoryStateToken);

  GraphExprNode programId = GraphExprNode::create(
      &context, "tt.get_program_id", {},
      {builder.getNamedAttr("ags.result_type",
                            TypeAttr::get(builder.getI32Type())),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(1))});
  FailureOr<TensorDenotation> result =
      semantics.lookup("tt.get_program_id")->denote(programId, {*launch});
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->value.getKind(), TensorSemanticKind::ProgramId);
  ASSERT_EQ(result->value.getOperands().size(), 1u);
  EXPECT_TRUE(result->value.getOperands().front().structurallyEquals(
      launch->launch));
}

TEST(TTOpSemanticsDSLTest, NumProgramsDependsOnExplicitLaunchState) {
  MLIRContext context;
  Builder builder(&context);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> launch =
      semantics.lookup("ags.launch")
          ->denote(makeSemanticNode(context, "ags.launch",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(launch));

  GraphExprNode numPrograms = GraphExprNode::create(
      &context, "tt.get_num_programs", {},
      {builder.getNamedAttr("ags.result_type",
                            TypeAttr::get(builder.getI32Type())),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(2))});
  FailureOr<TensorDenotation> result =
      semantics.lookup("tt.get_num_programs")
          ->denote(numPrograms, {*launch});
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->value.getKind(), TensorSemanticKind::NumPrograms);
  ASSERT_EQ(result->value.getOperands().size(), 1u);
  EXPECT_TRUE(result->value.getOperands().front().structurallyEquals(
      launch->launch));
}

TEST(TTOpSemanticsDSLTest, LaunchBoundIsDeclarativeProofOnlyState) {
  MLIRContext context;
  Builder builder(&context);
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  auto schema = llvm::find_if(schemas, [](const auto &candidate) {
    return candidate.operationName == "ags.launch_bound";
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->proofOnlyMaterialization);
  ASSERT_EQ(schema->implicitStateDomains.size(), 1u);
  EXPECT_EQ(schema->implicitStateDomains.front(),
            TTIRCandidateStateDomain::Launch);
  ASSERT_TRUE(schema->producedStateDomain.has_value());
  EXPECT_EQ(*schema->producedStateDomain,
            TTIRCandidateStateDomain::Launch);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> launch =
      semantics.lookup("ags.launch")
          ->denote(makeSemanticNode(context, "ags.launch",
                                    NoneType::get(&context)), {});
  FailureOr<TensorDenotation> bound =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input",
                                    builder.getI32Type(), 0), {});
  ASSERT_TRUE(succeeded(launch));
  ASSERT_TRUE(succeeded(bound));
  GraphExprNode node = GraphExprNode::create(
      &context, "ags.launch_bound", {0},
      {builder.getNamedAttr("ags.result_type",
                            TypeAttr::get(NoneType::get(&context))),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))});
  FailureOr<TensorDenotation> result =
      semantics.lookup("ags.launch_bound")->denote(node, {*bound, *launch});
  ASSERT_TRUE(succeeded(result));
  EXPECT_TRUE(result->isLaunchStateToken);
  EXPECT_EQ(result->launch.getKind(), TensorSemanticKind::LaunchBound);
}

TEST(TTOpSemanticsDSLTest, BufferStateIsDistinctFromMemoryAndLaunch) {
  MLIRContext context;
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(buffer));
  EXPECT_TRUE(buffer->isBufferStateToken);
  EXPECT_FALSE(buffer->isMemoryStateToken);
  EXPECT_FALSE(buffer->isLaunchStateToken);
  EXPECT_TRUE(static_cast<bool>(buffer->buffer));
  EXPECT_TRUE(static_cast<bool>(buffer->bufferDefined));
}

TEST(TTOpSemanticsDSLTest, StageBufferReadAndWriteConsumeBufferState) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type memrefType = MemRefType::get({4}, i32);
  Type tensorType = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", memrefType, 0),
                   {});
  FailureOr<TensorDenotation> target =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", memrefType, 1),
                   {});
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(target));
  ASSERT_TRUE(succeeded(buffer));

  FailureOr<TensorDenotation> copy =
      semantics.lookup("memref.copy")
          ->denote(makeSemanticNode(context, "memref.copy",
                                    NoneType::get(&context)),
                   {*source, *target, *buffer});
  ASSERT_TRUE(succeeded(copy));
  EXPECT_TRUE(copy->isBufferStateToken);
  EXPECT_EQ(copy->buffer.getKind(), TensorSemanticKind::BufferWrite);
  EXPECT_FALSE(copy->isMemoryStateToken);

  FailureOr<TensorDenotation> tensor =
      semantics.lookup("bufferization.to_tensor")
          ->denote(makeSemanticNode(context, "bufferization.to_tensor",
                                    tensorType),
                   {*target, *copy});
  ASSERT_TRUE(succeeded(tensor));
  EXPECT_EQ(tensor->value.getKind(), TensorSemanticKind::BufferRead);
  EXPECT_TRUE(tensor->buffer.structurallyEquals(copy->buffer));
}

TEST(TTOpSemanticsDSLTest,
     StaticMemrefSubviewReadEqualsTensorExtractSliceRead) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type memref8 = MemRefType::get({8}, i32);
  Type memref4 = MemRefType::get(
      {4}, i32, StridedLayoutAttr::get(&context, 2, {1}));
  Type tensor8 = RankedTensorType::get({8}, i32);
  Type tensor4 = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto sliceNode = [&](StringRef operation, Type resultType) {
    SmallVector<NamedAttribute, 5> attributes = {
        builder.getNamedAttr("ags.result_type", TypeAttr::get(resultType)),
        builder.getNamedAttr("static_offsets",
                             builder.getDenseI64ArrayAttr({2})),
        builder.getNamedAttr("static_sizes",
                             builder.getDenseI64ArrayAttr({4})),
        builder.getNamedAttr("static_strides",
                             builder.getDenseI64ArrayAttr({1}))};
    return GraphExprNode::create(&context, operation, {}, attributes);
  };

  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", memref8, 0), {});
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(buffer));
  FailureOr<TensorDenotation> view =
      semantics.lookup("memref.subview")
          ->denote(sliceNode("memref.subview", memref4), {*source});
  ASSERT_TRUE(succeeded(view));
  FailureOr<TensorDenotation> viewRead =
      semantics.lookup("bufferization.to_tensor")
          ->denote(makeSemanticNode(context, "bufferization.to_tensor",
                                    tensor4),
                   {*view, *buffer});
  FailureOr<TensorDenotation> wholeRead =
      semantics.lookup("bufferization.to_tensor")
          ->denote(makeSemanticNode(context, "bufferization.to_tensor",
                                    tensor8),
                   {*source, *buffer});
  ASSERT_TRUE(succeeded(viewRead));
  ASSERT_TRUE(succeeded(wholeRead));
  FailureOr<TensorDenotation> tensorSlice =
      semantics.lookup("tensor.extract_slice")
          ->denote(sliceNode("tensor.extract_slice", tensor4), {*wholeRead});
  ASSERT_TRUE(succeeded(tensorSlice));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*viewRead, *tensorSlice);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved)
      << proof.reason << "\nview: "
      << getStableTensorDenotationKey(normalizer.normalize(*viewRead).denotation)
      << "\nslice: "
      << getStableTensorDenotationKey(
             normalizer.normalize(*tensorSlice).denotation);
}

TEST(TTOpSemanticsDSLTest,
     VariadicMixedSliceSchemasAndDenotationUseSegmentOwnership) {
  MLIRContext context;
  Builder builder(&context);
  Type indexType = builder.getIndexType();
  Type tensor8 = RankedTensorType::get({8}, builder.getI32Type());
  Type tensor4 = RankedTensorType::get({4}, builder.getI32Type());
  Type dynamicTensor = RankedTensorType::get(
      {ShapedType::kDynamic}, builder.getI32Type());

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options,
                                          /*maxCandidateOperands=*/4);
  auto dynamicExtract = llvm::find_if(
      schemas, [](const TTIRCandidateOpSchema &schema) {
        return schema.operationName == "tensor.extract_slice" &&
               schema.operandCount == 2;
      });
  ASSERT_NE(dynamicExtract, schemas.end());
  EXPECT_TRUE(llvm::any_of(schemas, [](const auto &schema) {
    return schema.operationName == "tensor.extract_slice" &&
           schema.operandCount == 4;
  }));

  auto attributes = [&](ArrayRef<int32_t> segments,
                        ArrayRef<int64_t> offsets,
                        ArrayRef<int64_t> sizes) {
    return builder.getDictionaryAttr({
        builder.getNamedAttr("operandSegmentSizes",
                             builder.getDenseI32ArrayAttr(segments)),
        builder.getNamedAttr("static_offsets",
                             builder.getDenseI64ArrayAttr(offsets)),
        builder.getNamedAttr("static_sizes",
                             builder.getDenseI64ArrayAttr(sizes)),
        builder.getNamedAttr("static_strides",
                             builder.getDenseI64ArrayAttr({1}))});
  };
  DictionaryAttr dynamicOffsetAttributes = attributes(
      {1, 1, 0, 0}, {ShapedType::kDynamic}, {4});
  ASSERT_TRUE(dynamicExtract->isWellTyped(
      tensor4, {tensor8, indexType}, dynamicOffsetAttributes));

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> source =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", tensor8, 0), {});
  FailureOr<TensorDenotation> offset =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", indexType, 1), {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(offset));
  SmallVector<NamedAttribute, 6> nodeAttributes(
      dynamicOffsetAttributes.begin(), dynamicOffsetAttributes.end());
  nodeAttributes.push_back(builder.getNamedAttr(
      "ags.result_type", TypeAttr::get(tensor4)));
  FailureOr<TensorDenotation> dynamicOffset =
      semantics.lookup("tensor.extract_slice")
          ->denote(GraphExprNode::create(&context, "tensor.extract_slice", {},
                                         nodeAttributes),
                   {*source, *offset});
  ASSERT_TRUE(succeeded(dynamicOffset));
  EXPECT_TRUE(dynamicOffset->value);
  EXPECT_TRUE(dynamicOffset->defined);

  DictionaryAttr dynamicSizeAttributes = attributes(
      {1, 0, 1, 0}, {0}, {ShapedType::kDynamic});
  ASSERT_TRUE(dynamicExtract->isWellTyped(
      dynamicTensor, {tensor8, indexType}, dynamicSizeAttributes));
  FailureOr<TensorDenotation> size =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", indexType, 2), {});
  ASSERT_TRUE(succeeded(size));
  nodeAttributes.assign(dynamicSizeAttributes.begin(),
                        dynamicSizeAttributes.end());
  nodeAttributes.push_back(builder.getNamedAttr(
      "ags.result_type", TypeAttr::get(dynamicTensor)));
  FailureOr<TensorDenotation> dynamicSize =
      semantics.lookup("tensor.extract_slice")
          ->denote(GraphExprNode::create(&context, "tensor.extract_slice", {},
                                         nodeAttributes),
                   {*source, *size});
  ASSERT_TRUE(succeeded(dynamicSize));
  ASSERT_EQ(dynamicSize->shapeSymbols.size(), 1u);
  EXPECT_NE(dynamicSize->shapeSymbols.front(), "?");
}

TEST(TTOpSemanticsDSLTest, FullCopyMakesEarlierBufferFillUnobservable) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type memref4 = MemRefType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  FailureOr<TensorDenotation> source = input(memref4, 0);
  FailureOr<TensorDenotation> target =
      semantics.lookup("ags.private_buffer")
          ->denote(makeSemanticNode(context, "ags.private_buffer", memref4, 1),
                   {});
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  GraphExprNode zeroNode = GraphExprNode::create(
      &context, "arith.constant", {},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i32)),
       builder.getNamedAttr("value", builder.getI32IntegerAttr(0))});
  FailureOr<TensorDenotation> zero =
      semantics.lookup("arith.constant")->denote(zeroNode, {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(target));
  ASSERT_TRUE(succeeded(buffer));
  ASSERT_TRUE(succeeded(zero));

  GraphExprNode fillNode = makeSemanticNode(
      context, "linalg.fill", NoneType::get(&context));
  FailureOr<TensorDenotation> fill =
      semantics.lookup("linalg.fill")
          ->denote(fillNode, {*zero, *target, *buffer});
  ASSERT_TRUE(succeeded(fill));
  GraphExprNode copyNode = makeSemanticNode(
      context, "memref.copy", NoneType::get(&context));
  FailureOr<TensorDenotation> afterFill =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source, *target, *fill});
  FailureOr<TensorDenotation> direct =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source, *target, *buffer});
  ASSERT_TRUE(succeeded(afterFill));
  ASSERT_TRUE(succeeded(direct));
  Type tensor4 = RankedTensorType::get({4}, i32);
  GraphExprNode readNode = makeSemanticNode(
      context, "bufferization.to_tensor", tensor4);
  FailureOr<TensorDenotation> readAfterFill =
      semantics.lookup("bufferization.to_tensor")
          ->denote(readNode, {*target, *afterFill});
  FailureOr<TensorDenotation> readDirect =
      semantics.lookup("bufferization.to_tensor")
          ->denote(readNode, {*target, *direct});
  ASSERT_TRUE(succeeded(readAfterFill));
  ASSERT_TRUE(succeeded(readDirect));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*readAfterFill,
                                                    *readDirect);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved)
      << proof.reason << "\nafter-fill: "
      << getStableTensorDenotationKey(
             normalizer.normalize(*readAfterFill).denotation)
      << "\ndirect: "
      << getStableTensorDenotationKey(
             normalizer.normalize(*readDirect).denotation);
}

TEST(TTOpSemanticsDSLTest, DoesNotAssumeArbitraryMemrefInputsAreDisjoint) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type memref4 = MemRefType::get({4}, i32);
  Type tensor4 = RankedTensorType::get({4}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", memref4, symbol), {});
  };
  FailureOr<TensorDenotation> source = input(0);
  FailureOr<TensorDenotation> possiblyAliasingTarget = input(1);
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(source));
  ASSERT_TRUE(succeeded(possiblyAliasingTarget));
  ASSERT_TRUE(succeeded(buffer));

  GraphExprNode copyNode = makeSemanticNode(
      context, "memref.copy", NoneType::get(&context));
  FailureOr<TensorDenotation> state =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source, *possiblyAliasingTarget, *buffer});
  ASSERT_TRUE(succeeded(state));
  GraphExprNode readNode = makeSemanticNode(
      context, "bufferization.to_tensor", tensor4);
  FailureOr<TensorDenotation> before =
      semantics.lookup("bufferization.to_tensor")
          ->denote(readNode, {*source, *buffer});
  FailureOr<TensorDenotation> after =
      semantics.lookup("bufferization.to_tensor")
          ->denote(readNode, {*source, *state});
  ASSERT_TRUE(succeeded(before));
  ASSERT_TRUE(succeeded(after));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*before, *after);
  EXPECT_NE(proof.status, SemanticProofStatus::Proved);
}

TEST(TTOpSemanticsDSLTest, DynamicBufferFillHasExecutableStateSemantics) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type dynamicMemref = MemRefType::get({ShapedType::kDynamic}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  FailureOr<TensorDenotation> fillValue =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  FailureOr<TensorDenotation> target =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", dynamicMemref, 1),
                   {});
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)),
                   {});
  ASSERT_TRUE(succeeded(fillValue));
  ASSERT_TRUE(succeeded(target));
  ASSERT_TRUE(succeeded(buffer));

  FailureOr<TensorDenotation> filled =
      semantics.lookup("linalg.fill")
          ->denote(makeSemanticNode(context, "linalg.fill",
                                    NoneType::get(&context)),
                   {*fillValue, *target, *buffer});
  ASSERT_TRUE(succeeded(filled));
  EXPECT_TRUE(filled->isBufferStateToken);
  EXPECT_EQ(filled->buffer.getKind(), TensorSemanticKind::BufferFill);

  Type dynamicTensor =
      RankedTensorType::get({ShapedType::kDynamic}, i32);
  FailureOr<TensorDenotation> read =
      semantics.lookup("bufferization.to_tensor")
          ->denote(makeSemanticNode(context, "bufferization.to_tensor",
                                    dynamicTensor),
                   {*target, *filled});
  ASSERT_TRUE(succeeded(read));
  SemanticNormalizationResult normalized =
      TensorSemanticNormalizer().normalize(*read);
  EXPECT_TRUE(normalized.denotation.value.structurallyEquals(fillValue->value));
}

TEST(TTOpSemanticsDSLTest,
     ProvesTwoPrivateBuffersEqualOneDisjointlyWrittenBuffer) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type memref8 = MemRefType::get({8}, i32);
  Type memrefLow = MemRefType::get(
      {4}, i32, StridedLayoutAttr::get(&context, 0, {1}));
  Type memrefHigh = MemRefType::get(
      {4}, i32, StridedLayoutAttr::get(&context, 4, {1}));
  Type tensor4 = RankedTensorType::get({4}, i32);
  Type tensor8 = RankedTensorType::get({8}, i32);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();

  auto input = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.input")
        ->denote(makeSemanticNode(context, "ags.input", type, symbol), {});
  };
  auto privateBuffer = [&](Type type, int64_t symbol) {
    return semantics.lookup("ags.private_buffer")
        ->denote(
            makeSemanticNode(context, "ags.private_buffer", type, symbol),
            {});
  };
  auto sliceNode = [&](StringRef operation, Type resultType, int64_t offset) {
    SmallVector<NamedAttribute, 5> attributes = {
        builder.getNamedAttr("ags.result_type", TypeAttr::get(resultType)),
        builder.getNamedAttr("static_offsets",
                             builder.getDenseI64ArrayAttr({offset})),
        builder.getNamedAttr("static_sizes",
                             builder.getDenseI64ArrayAttr({4})),
        builder.getNamedAttr("static_strides",
                             builder.getDenseI64ArrayAttr({1}))};
    return GraphExprNode::create(&context, operation, {}, attributes);
  };
  auto subview = [&](const TensorDenotation &source, Type type,
                     int64_t offset) {
    return semantics.lookup("memref.subview")
        ->denote(sliceNode("memref.subview", type, offset), {source});
  };
  auto read = [&](const TensorDenotation &source,
                  const TensorDenotation &state) {
    return semantics.lookup("bufferization.to_tensor")
        ->denote(makeSemanticNode(context, "bufferization.to_tensor",
                                  tensor8),
                 {source, state});
  };

  FailureOr<TensorDenotation> source0 = input(memref8, 0);
  FailureOr<TensorDenotation> source1 = input(memref8, 1);
  FailureOr<TensorDenotation> target0 = privateBuffer(memref8, 2);
  FailureOr<TensorDenotation> target1 = privateBuffer(memref8, 3);
  FailureOr<TensorDenotation> buffer =
      semantics.lookup("ags.buffer")
          ->denote(makeSemanticNode(context, "ags.buffer",
                                    NoneType::get(&context)), {});
  ASSERT_TRUE(succeeded(source0));
  ASSERT_TRUE(succeeded(source1));
  ASSERT_TRUE(succeeded(target0));
  ASSERT_TRUE(succeeded(target1));
  ASSERT_TRUE(succeeded(buffer));

  FailureOr<TensorDenotation> source0Low = subview(*source0, memrefLow, 0);
  FailureOr<TensorDenotation> source1High = subview(*source1, memrefHigh, 4);
  FailureOr<TensorDenotation> target0Low = subview(*target0, memrefLow, 0);
  FailureOr<TensorDenotation> target1Low = subview(*target1, memrefLow, 0);
  FailureOr<TensorDenotation> target1High =
      subview(*target1, memrefHigh, 4);
  ASSERT_TRUE(succeeded(source0Low));
  ASSERT_TRUE(succeeded(source1High));
  ASSERT_TRUE(succeeded(target0Low));
  ASSERT_TRUE(succeeded(target1Low));
  ASSERT_TRUE(succeeded(target1High));

  GraphExprNode copyNode = makeSemanticNode(
      context, "memref.copy", NoneType::get(&context));
  FailureOr<TensorDenotation> firstCopy =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source0Low, *target0Low, *buffer});
  FailureOr<TensorDenotation> splitState =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source1High, *target1High, *firstCopy});
  FailureOr<TensorDenotation> mergedFirstState =
      semantics.lookup("memref.copy")
          ->denote(copyNode, {*source0Low, *target1Low, *buffer});
  FailureOr<TensorDenotation> mergedState =
      semantics.lookup("memref.copy")
          ->denote(copyNode,
                   {*source1High, *target1High, *mergedFirstState});
  ASSERT_TRUE(succeeded(firstCopy));
  ASSERT_TRUE(succeeded(splitState));
  ASSERT_TRUE(succeeded(mergedFirstState));
  ASSERT_TRUE(succeeded(mergedState));

  FailureOr<TensorDenotation> splitRead0 = read(*target0, *splitState);
  FailureOr<TensorDenotation> splitRead1 = read(*target1, *splitState);
  FailureOr<TensorDenotation> mergedRead = read(*target1, *mergedState);
  ASSERT_TRUE(succeeded(splitRead0));
  ASSERT_TRUE(succeeded(splitRead1));
  ASSERT_TRUE(succeeded(mergedRead));
  FailureOr<TensorDenotation> low =
      semantics.lookup("tensor.extract_slice")
          ->denote(sliceNode("tensor.extract_slice", tensor4, 0),
                   {*splitRead0});
  ASSERT_TRUE(succeeded(low));
  FailureOr<TensorDenotation> splitResult =
      semantics.lookup("tensor.insert_slice")
          ->denote(sliceNode("tensor.insert_slice", tensor8, 0),
                   {*low, *splitRead1});
  ASSERT_TRUE(succeeded(splitResult));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*splitResult,
                                                    *mergedRead);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved)
      << proof.reason << "\nsplit: "
      << getStableTensorDenotationKey(
             normalizer.normalize(*splitResult).denotation)
      << "\nmerged: "
      << getStableTensorDenotationKey(
             normalizer.normalize(*mergedRead).denotation);
}

TEST(TTOpSemanticsDSLTest, CatalogDoesNotClaimAnUnimplementedDomain) {
  static constexpr StringLiteral source = R"json(
  {
    "schema_version": 2,
    "name": "semantic-domain-test",
    "target": "generic",
    "bounds": {
      "max_nodes": 2,
      "max_rank": 1,
      "max_operands": 2,
      "max_results": 1,
      "max_shape_symbols": 0,
      "allow_dynamic_shapes": false,
      "allow_control_flow": false,
      "allow_effects": false
    },
    "catalog_dialects": ["arith"],
    "operations": ["arith.subi"],
    "integer_bits": [32],
    "floating_bits": [],
    "integer_constants": [0],
    "tensor_shapes": [[4]],
    "attribute_domains": {},
    "semantic_domains": ["signature", "value", "definedness", "shape", "index"]
  }
  )json";

  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;

  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  GraphOperationCatalog catalog =
      GraphOperationCatalog::build(context, *spec, registry);

  auto selected = llvm::find_if(
      catalog.entries, [](const GraphOperationCatalogEntry &entry) {
        return entry.operationName == "arith.subi";
      });
  ASSERT_NE(selected, catalog.entries.end());
  EXPECT_EQ(selected->coverage, OperationSemanticCoverage::Complete);
  EXPECT_TRUE(hasDomain(selected->implementedDomains,
                        GraphSemanticDomain::Signature));
  EXPECT_TRUE(
      hasDomain(selected->implementedDomains, GraphSemanticDomain::Value));
  EXPECT_TRUE(hasDomain(selected->implementedDomains,
                        GraphSemanticDomain::Definedness));
  EXPECT_TRUE(
      hasDomain(selected->implementedDomains, GraphSemanticDomain::Shape));
  EXPECT_TRUE(
      hasDomain(selected->implementedDomains, GraphSemanticDomain::Index));
  EXPECT_TRUE(selected->missingDomains.empty());
}

TEST(TTOpSemanticsDSLTest, MultiResultSchemasPreserveEveryProjection) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type i1 = builder.getI1Type();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  SmallVector<const TTIRCandidateOpSchema *, 2> extended;
  for (const TTIRCandidateOpSchema &schema : schemas)
    if (schema.operationName == "arith.addui_extended")
      extended.push_back(&schema);
  ASSERT_EQ(extended.size(), 2u);
  llvm::sort(extended, [](const auto *lhs, const auto *rhs) {
    return lhs->resultIndex < rhs->resultIndex;
  });
  for (unsigned index = 0; index < extended.size(); ++index) {
    EXPECT_EQ(extended[index]->resultIndex, index);
    EXPECT_EQ(extended[index]->operationResultCount, 2u);
    Type selectedType = index == 0 ? i32 : i1;
    ASSERT_TRUE(extended[index]->isWellTyped(
        selectedType, {i32, i32}, builder.getDictionaryAttr({})));
    SmallVector<Type, 2> resultTypes =
        extended[index]->buildOperationResultTypes(
            selectedType, {i32, i32}, builder.getDictionaryAttr({}));
    ASSERT_EQ(resultTypes.size(), 2u);
    EXPECT_EQ(resultTypes[0], i32);
    EXPECT_EQ(resultTypes[1], i1);
  }
}

TEST(TTOpSemanticsDSLTest, MultiResultDenotationDoesNotAliasSiblingResult) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 1), {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  SmallVector<Attribute, 2> resultTypes = {TypeAttr::get(i32),
                                           TypeAttr::get(i32)};
  auto makeProjection = [&](unsigned index) {
    SmallVector<NamedAttribute, 4> attributes = {
        builder.getNamedAttr("ags.result_type", TypeAttr::get(i32)),
        builder.getNamedAttr("ags.result_index",
                             builder.getI64IntegerAttr(index)),
        builder.getNamedAttr("ags.operation_result_types",
                             builder.getArrayAttr(resultTypes))};
    return GraphExprNode::create(&context, "arith.mului_extended", {0, 1},
                                 attributes);
  };
  FailureOr<TensorDenotation> low =
      semantics.lookup("arith.mului_extended")
          ->denote(makeProjection(0), {*lhs, *rhs});
  FailureOr<TensorDenotation> high =
      semantics.lookup("arith.mului_extended")
          ->denote(makeProjection(1), {*lhs, *rhs});
  ASSERT_TRUE(succeeded(low));
  ASSERT_TRUE(succeeded(high));
  TensorSemanticNormalizer normalizer;
  EXPECT_NE(SemanticEquivalenceOracle(normalizer).compare(*low, *high).status,
            SemanticProofStatus::Proved);
}

TEST(TTOpSemanticsDSLTest, AddUIExtendedCarryUsesBitVectorSemantics) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type i1 = builder.getI1Type();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 1), {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  SmallVector<Attribute, 2> resultTypes = {TypeAttr::get(i32),
                                           TypeAttr::get(i1)};
  GraphExprNode projection = GraphExprNode::create(
      &context, "arith.addui_extended", {0, 1},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i1)),
       builder.getNamedAttr("ags.result_index",
                            builder.getI64IntegerAttr(1)),
       builder.getNamedAttr("ags.operation_result_types",
                            builder.getArrayAttr(resultTypes))});
  FailureOr<TensorDenotation> carry =
      semantics.lookup("arith.addui_extended")
          ->denote(projection, {*lhs, *rhs});
  ASSERT_TRUE(succeeded(carry));

  TensorDenotation expected = *carry;
  TensorSemanticExpr sum = TensorSemanticExpr::operation(
      TensorSemanticKind::Add, i32, {lhs->value, rhs->value});
  expected.value = TensorSemanticExpr::operation(
      TensorSemanticKind::UnsignedLessThan, i1, {sum, lhs->value});

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*carry, expected);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, IndexCastsUseSignedAndUnsignedBitVectorSemantics) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type index = builder.getIndexType();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> input =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  ASSERT_TRUE(succeeded(input));

  for (StringRef operation : {StringRef("arith.index_cast"),
                              StringRef("arith.index_castui")}) {
    FailureOr<TensorDenotation> widened =
        semantics.lookup(operation)
            ->denote(makeSemanticNode(context, operation, index), {*input});
    ASSERT_TRUE(succeeded(widened)) << operation.str();
    FailureOr<TensorDenotation> roundTrip =
        semantics.lookup(operation)
            ->denote(makeSemanticNode(context, operation, i32), {*widened});
    ASSERT_TRUE(succeeded(roundTrip)) << operation.str();

    TensorSemanticNormalizer normalizer;
    SemanticProofResult proof =
        SemanticEquivalenceOracle(normalizer).compare(*input, *roundTrip);
    EXPECT_EQ(proof.status, SemanticProofStatus::Proved)
        << operation.str() << ": " << proof.reason;
  }
}

TEST(TTOpSemanticsDSLTest, UnsignedMultiplyHighSharesOneBitVectorMeaning) {
  MLIRContext context;
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> lhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 0), {});
  FailureOr<TensorDenotation> rhs =
      semantics.lookup("ags.input")
          ->denote(makeSemanticNode(context, "ags.input", i32, 1), {});
  ASSERT_TRUE(succeeded(lhs));
  ASSERT_TRUE(succeeded(rhs));

  FailureOr<TensorDenotation> tritonHigh =
      semantics.lookup("tt.mulhiui")
          ->denote(makeSemanticNode(context, "tt.mulhiui", i32),
                   {*lhs, *rhs});
  ASSERT_TRUE(succeeded(tritonHigh));

  SmallVector<Attribute, 2> resultTypes = {TypeAttr::get(i32),
                                           TypeAttr::get(i32)};
  GraphExprNode projection = GraphExprNode::create(
      &context, "arith.mului_extended", {0, 1},
      {builder.getNamedAttr("ags.result_type", TypeAttr::get(i32)),
       builder.getNamedAttr("ags.result_index",
                            builder.getI64IntegerAttr(1)),
       builder.getNamedAttr("ags.operation_result_types",
                            builder.getArrayAttr(resultTypes))});
  FailureOr<TensorDenotation> arithHigh =
      semantics.lookup("arith.mului_extended")
          ->denote(projection, {*lhs, *rhs});
  ASSERT_TRUE(succeeded(arithHigh));

  TensorSemanticNormalizer normalizer;
  SemanticProofResult proof =
      SemanticEquivalenceOracle(normalizer).compare(*tritonHigh, *arithHigh);
  EXPECT_EQ(proof.status, SemanticProofStatus::Proved) << proof.reason;
}

TEST(TTOpSemanticsDSLTest, RegionOverloadsAreDataDriven) {
  MLIRContext context;
  Builder builder(&context);
  Type tensor = RankedTensorType::get({4}, builder.getI32Type());
  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("reverse", builder.getBoolAttr(false))});
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);

  bool sawAddCombiner = false;
  bool sawMultiplyCombiner = false;
  unsigned scanOverloadCount = 0;
  for (const TTIRCandidateOpSchema &schema : schemas) {
    if (schema.operationName != "tt.scan" || !schema.buildRegions)
      continue;
    ASSERT_TRUE(schema.isWellTyped(tensor, {tensor}, attributes));
    ArrayAttr regions = schema.buildRegions(tensor, {tensor}, attributes);
    ASSERT_TRUE(regions);
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    stream << regions;
    stream.flush();
    sawAddCombiner |= StringRef(printed).contains("arith.addi");
    sawMultiplyCombiner |= StringRef(printed).contains("arith.muli");
    ++scanOverloadCount;
  }
  EXPECT_EQ(scanOverloadCount, 2u);
  EXPECT_TRUE(sawAddCombiner);
  EXPECT_TRUE(sawMultiplyCombiner);
}

TEST(TTOpSemanticsDSLTest, LaunchActionIsDeclarativeAndProofCarrying) {
  MLIRContext context;
  Builder builder(&context);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  auto schema = llvm::find_if(schemas, [](const auto &candidate) {
    return candidate.operationName == "ags.launch_remap";
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->proofOnlyMaterialization);
  ASSERT_TRUE(schema->producedStateDomain.has_value());
  EXPECT_EQ(*schema->producedStateDomain,
            TTIRCandidateStateDomain::Launch);
  ASSERT_TRUE(schema->buildModuleAttributePatch);
  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("factor", builder.getI32IntegerAttr(1)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(0))});
  ASSERT_TRUE(schema->isWellTyped(none, {}, attributes));
  DictionaryAttr patch = schema->buildModuleAttributePatch(attributes);
  ASSERT_TRUE(patch);
  EXPECT_EQ(cast<IntegerAttr>(patch.get("hacc.coalesce_axis")).getInt(), 0);
  EXPECT_EQ(cast<IntegerAttr>(patch.get("hacc.coalesce_factor")).getInt(), 1);
  EXPECT_EQ(
      cast<IntegerAttr>(patch.get("hacc.coalesce_grid_ceil_div")).getInt(),
      0);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> launch =
      semantics.lookup("ags.launch")
          ->denote(makeSemanticNode(context, "ags.launch", none), {});
  ASSERT_TRUE(succeeded(launch));
  SmallVector<NamedAttribute, 4> remapAttributes(attributes.begin(),
                                                  attributes.end());
  remapAttributes.push_back(builder.getNamedAttr("ags.result_type",
                                                  TypeAttr::get(none)));
  GraphExprNode remap = GraphExprNode::create(
      &context, "ags.launch_remap", {0}, remapAttributes);
  FailureOr<TensorDenotation> identityRemap =
      semantics.lookup("ags.launch_remap")->denote(remap, {*launch});
  ASSERT_TRUE(succeeded(identityRemap));
  TensorSemanticNormalizer normalizer;
  EXPECT_EQ(SemanticEquivalenceOracle(normalizer)
                .compare(*launch, *identityRemap)
                .status,
            SemanticProofStatus::Proved);
}

TEST(TTOpSemanticsDSLTest, KernelRootBindsMemoryAndLaunchDeclaratively) {
  MLIRContext context;
  Builder builder(&context);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  auto schema = llvm::find_if(schemas, [](const auto &candidate) {
    return candidate.operationName == "ags.kernel";
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->proofOnlyMaterialization);
  ASSERT_EQ(schema->implicitStateDomains.size(), 2u);
  EXPECT_EQ(schema->implicitStateDomains[0],
            TTIRCandidateStateDomain::Memory);
  EXPECT_EQ(schema->implicitStateDomains[1],
            TTIRCandidateStateDomain::Launch);
  ASSERT_TRUE(schema->producedStateDomain.has_value());
  EXPECT_EQ(*schema->producedStateDomain,
            TTIRCandidateStateDomain::Memory);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> memory =
      semantics.lookup("ags.memory")
          ->denote(makeSemanticNode(context, "ags.memory", none), {});
  FailureOr<TensorDenotation> launch =
      semantics.lookup("ags.launch")
          ->denote(makeSemanticNode(context, "ags.launch", none), {});
  ASSERT_TRUE(succeeded(memory));
  ASSERT_TRUE(succeeded(launch));

  FailureOr<TensorDenotation> kernel =
      semantics.lookup("ags.kernel")
          ->denote(makeSemanticNode(context, "ags.kernel", none),
                   {*memory, *launch});
  ASSERT_TRUE(succeeded(kernel));
  EXPECT_TRUE(kernel->isMemoryStateToken);
  EXPECT_FALSE(kernel->isLaunchStateToken);
  EXPECT_FALSE(kernel->isBufferStateToken);
  EXPECT_EQ(kernel->memory.getKind(), TensorSemanticKind::KernelEffect);
  ASSERT_EQ(kernel->memory.getOperands().size(), 2u);
  EXPECT_TRUE(
      kernel->memory.getOperands()[0].structurallyEquals(memory->memory));
  EXPECT_TRUE(
      kernel->memory.getOperands()[1].structurallyEquals(launch->launch));
  EXPECT_TRUE(kernel->launch.structurallyEquals(kernel->memory));
  EXPECT_EQ(kernel->effect.getKind(), TensorSemanticKind::KernelEffect);
}

TEST(TTOpSemanticsDSLTest,
     ParameterizedKernelPartitionDeclaresProofAndCompilerAction) {
  MLIRContext context;
  Builder builder(&context);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  auto schema = llvm::find_if(schemas, [](const auto &candidate) {
    return candidate.operationName == "ags.kernel_partition";
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->proofOnlyMaterialization);
  ASSERT_EQ(schema->implicitStateDomains.size(), 2u);
  EXPECT_EQ(schema->implicitStateDomains[0],
            TTIRCandidateStateDomain::Memory);
  EXPECT_EQ(schema->implicitStateDomains[1],
            TTIRCandidateStateDomain::Launch);
  ASSERT_TRUE(schema->producedStateDomain.has_value());
  EXPECT_EQ(*schema->producedStateDomain,
            TTIRCandidateStateDomain::Memory);
  ASSERT_TRUE(schema->buildCompilerAction);
  ASSERT_TRUE(schema->buildModuleAttributePatch);

  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("factor", builder.getI32IntegerAttr(16)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(0))});
  ASSERT_TRUE(schema->isWellTyped(none, {}, attributes));
  DictionaryAttr action = schema->buildCompilerAction(attributes);
  ASSERT_TRUE(action);
  EXPECT_EQ(cast<StringAttr>(action.get("kind")).getValue(),
            "launch_partition");
  EXPECT_EQ(cast<IntegerAttr>(action.get("factor")).getInt(), 16);
  DictionaryAttr patch = schema->buildModuleAttributePatch(attributes);
  ASSERT_TRUE(patch);
  EXPECT_EQ(cast<IntegerAttr>(patch.get("hacc.coalesce_factor")).getInt(),
            16);
}

TEST(TTOpSemanticsDSLTest,
     PartitionableMemoryCarriesTheQuantifiedTheoremPremise) {
  MLIRContext context;
  Builder builder(&context);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  auto schema = llvm::find_if(schemas, [](const auto &candidate) {
    return candidate.operationName == "ags.partitionable_memory";
  });
  ASSERT_NE(schema, schemas.end());
  EXPECT_TRUE(schema->offlineOnly);
  EXPECT_TRUE(schema->proofOnlyMaterialization);
  ASSERT_TRUE(schema->producedStateDomain.has_value());
  EXPECT_EQ(*schema->producedStateDomain,
            TTIRCandidateStateDomain::Memory);
  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0)),
       builder.getNamedAttr("ceil_div", builder.getI32IntegerAttr(1))});
  ASSERT_TRUE(schema->isWellTyped(none, {}, attributes));

  SmallVector<NamedAttribute, 4> nodeAttributes(attributes.begin(),
                                                 attributes.end());
  nodeAttributes.push_back(builder.getNamedAttr("ags.result_type",
                                                 TypeAttr::get(none)));
  GraphExprNode node = GraphExprNode::create(
      &context, "ags.partitionable_memory", {}, nodeAttributes);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  FailureOr<TensorDenotation> premise =
      semantics.lookup("ags.partitionable_memory")->denote(node, {});
  ASSERT_TRUE(succeeded(premise));
  EXPECT_TRUE(premise->isMemoryStateToken);
  EXPECT_EQ(premise->memory.getKind(),
            TensorSemanticKind::PartitionableMemory);
  EXPECT_EQ(premise->effect.getKind(),
            TensorSemanticKind::PartitionableMemory);
}

TEST(TTOpSemanticsDSLTest, CoreTritonSchemasComeOnlyFromDeclarativeSemantics) {
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  for (StringRef operation :
       {"tt.int_to_ptr", "tt.ptr_to_int", "tt.bitcast", "tt.fp_to_fp",
        "tt.clampf", "tt.precise_divf", "tt.mulhiui", "tt.unsplat",
        "tt.cat", "tt.join", "tt.gather", "tt.dot",
        "tt.make_tensor_ptr", "tt.advance"}) {
    auto schema = llvm::find_if(schemas, [&](const auto &candidate) {
      return candidate.operationName == operation && candidate.constructible;
    });
    EXPECT_NE(schema, schemas.end()) << operation.str();
  }

  unsigned splitProjections = llvm::count_if(
      schemas, [](const TTIRCandidateOpSchema &candidate) {
        return candidate.operationName == "tt.split" &&
               candidate.constructible && candidate.operationResultCount == 2;
      });
  EXPECT_EQ(splitProjections, 2u);
}

TEST(TTOpSemanticsDSLTest, RegionTerminatorsAreNotSearchCandidates) {
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);

  for (StringRef operation : {"tt.reduce.return", "tt.scan.return",
                              "tt.map_elementwise.return", "tt.return",
                              "tt.func"}) {
    EXPECT_EQ(registry.getSearchPolicy(operation),
              TTOpSearchPolicy::RegionOnly)
        << operation.str();
    auto schema = llvm::find_if(schemas, [&](const auto &candidate) {
      return candidate.operationName == operation;
    });
    ASSERT_NE(schema, schemas.end()) << operation.str();
    EXPECT_EQ(schema->operandCount, 0u) << operation.str();
    EXPECT_FALSE(schema->constructible) << operation.str();
    EXPECT_TRUE(schema->structuralRegionOnly) << operation.str();
    EXPECT_FALSE(schema->intentionalImportOnly) << operation.str();
  }
}

TEST(TTOpSemanticsDSLTest, SeedTerminalsAreNotGenericSearchCandidates) {
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);

  // The offline enumerator creates these leaves from finite universe domains.
  // They must not also participate in the generic OP x operand enumeration.
  for (StringRef operation :
       {"ags.input", "ags.private_buffer", "ags.memory", "ags.launch",
        "ags.buffer", "arith.constant"}) {
    EXPECT_EQ(registry.getSearchPolicy(operation),
              TTOpSearchPolicy::ImportOnly)
        << operation.str();
    auto schema = llvm::find_if(schemas, [&](const auto &candidate) {
      return candidate.operationName == operation;
    });
    ASSERT_NE(schema, schemas.end()) << operation.str();
    EXPECT_EQ(schema->operandCount, 0u) << operation.str();
    EXPECT_FALSE(schema->constructible) << operation.str();
    EXPECT_TRUE(schema->intentionalImportOnly) << operation.str();
  }
}

TEST(TTOpSemanticsDSLTest, ObservableAndExternalOpsAreImportOnly) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);

  for (StringRef operation :
       {"tt.print", "tt.assert", "tt.extern_elementwise"})
    EXPECT_EQ(registry.getSearchPolicy(operation),
              TTOpSearchPolicy::ImportOnly)
        << operation.str();

  auto zeroArgumentPrint = llvm::find_if(
      schemas, [](const TTIRCandidateOpSchema &schema) {
        return schema.operationName == "tt.print" &&
               schema.operandCount == 0;
      });
  ASSERT_NE(zeroArgumentPrint, schemas.end());
  EXPECT_EQ(zeroArgumentPrint->effectKind,
            TTIRCandidateEffectKind::Observable);
  EXPECT_FALSE(zeroArgumentPrint->constructible);
  EXPECT_TRUE(zeroArgumentPrint->intentionalImportOnly);

  Block block;
  Value input = block.addArgument(builder.getI32Type(),
                                  builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  auto makeExternal = [&](bool pure) {
    OperationState state(builder.getUnknownLoc(), "tt.extern_elementwise");
    state.addOperands(input);
    state.addTypes(builder.getI32Type());
    state.addAttribute("libname", builder.getStringAttr("lib"));
    state.addAttribute("libpath", builder.getStringAttr("/tmp/lib.so"));
    state.addAttribute("symbol", builder.getStringAttr("f"));
    state.addAttribute("pure", builder.getBoolAttr(pure));
    return opBuilder.create(state);
  };
  Operation *pure = makeExternal(true);
  Operation *impure = makeExternal(false);

  auto selectedEffect = [&](Operation *operation) {
    SmallVector<TTIRCandidateEffectKind, 2> effects;
    for (const TTIRCandidateOpSchema &schema : schemas)
      if (schema.operationName == "tt.extern_elementwise" &&
          schema.operandCount == operation->getNumOperands() &&
          schema.matchesOperation && schema.matchesOperation(operation))
        effects.push_back(schema.effectKind);
    EXPECT_EQ(effects.size(), 1u);
    return effects.empty() ? TTIRCandidateEffectKind::Pure : effects.front();
  };
  EXPECT_EQ(selectedEffect(pure), TTIRCandidateEffectKind::Pure);
  EXPECT_EQ(selectedEffect(impure), TTIRCandidateEffectKind::Observable);
  EXPECT_TRUE(registry.supports(pure));
  EXPECT_TRUE(registry.supports(impure));

  OperationState inlineState(builder.getUnknownLoc(),
                             "tt.elementwise_inline_asm");
  inlineState.addOperands(input);
  inlineState.addTypes({builder.getI32Type(), builder.getF32Type()});
  inlineState.addAttribute("asm_string", builder.getStringAttr("opaque"));
  inlineState.addAttribute("constraints", builder.getStringAttr("=r,r"));
  inlineState.addAttribute("pure", builder.getBoolAttr(true));
  inlineState.addAttribute("packed_element", builder.getI32IntegerAttr(1));
  Operation *inlineAsm = opBuilder.create(inlineState);
  EXPECT_TRUE(registry.supports(inlineAsm));
  EXPECT_EQ(registry.getSearchPolicy("tt.elementwise_inline_asm"),
            TTOpSearchPolicy::ImportOnly);
  EXPECT_EQ(registry.getSearchPolicy("tt.map_elementwise"),
            TTOpSearchPolicy::ImportOnly);
  EXPECT_EQ(registry.getSearchPolicy("tt.dot_scaled"),
            TTOpSearchPolicy::ImportOnly);

  auto dynamicInline = llvm::find_if(
      schemas, [](const TTIRCandidateOpSchema &schema) {
        return schema.operationName == "tt.elementwise_inline_asm" &&
               schema.operandCount == 1;
      });
  ASSERT_NE(dynamicInline, schemas.end());
  EXPECT_EQ(dynamicInline->operationResultCount, 0u);
  EXPECT_FALSE(dynamicInline->constructible);
}

TEST(TTOpSemanticsDSLTest, DescriptorSchemasUseSharedTypeAndAddressLanguage) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  Type pointer = triton::PointerType::get(i32, 1);
  RankedTensorType block = RankedTensorType::get({4}, i32);
  Type descriptor = triton::TensorDescType::get(&context, block);
  Type none = builder.getNoneType();
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);

  auto findSchema = [&](StringRef operation, unsigned operands) {
    return llvm::find_if(schemas, [&](const TTIRCandidateOpSchema &schema) {
      return schema.operationName == operation &&
             schema.operandCount == operands && schema.constructible;
    });
  };
  auto makeDescriptor = findSchema("tt.make_tensor_descriptor", 3);
  ASSERT_NE(makeDescriptor, schemas.end());
  EXPECT_TRUE(makeDescriptor->isWellTyped(
      descriptor, {pointer, i32, i64}, builder.getDictionaryAttr({})));

  auto load = findSchema("tt.descriptor_load", 2);
  ASSERT_NE(load, schemas.end());
  EXPECT_EQ(load->effectKind, TTIRCandidateEffectKind::Read);
  EXPECT_TRUE(load->isWellTyped(block, {descriptor, i32},
                                builder.getDictionaryAttr({})));

  auto store = findSchema("tt.descriptor_store", 3);
  ASSERT_NE(store, schemas.end());
  EXPECT_EQ(store->effectKind, TTIRCandidateEffectKind::Write);
  EXPECT_TRUE(store->isWellTyped(none, {descriptor, block, i32},
                                 builder.getDictionaryAttr({})));

  TTOpSemanticsRegistry registry = createBuiltinTTOpSemanticsRegistry();
  EXPECT_EQ(registry.getSearchPolicy("tt.descriptor_reduce"),
            TTOpSearchPolicy::Searchable);
}

} // namespace

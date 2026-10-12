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
#include "TritonToGraph/OfflineGraphEnumerator.h"
#include "TritonToGraph/ParetoExtractor.h"
#include "TritonToGraph/RuleDatabase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

TEST(CanonicalGraphKeyTest, CooperativelyStopsCanonicalConstruction) {
  MLIRContext context;
  Builder builder(&context);
  Block block;
  Value root =
      block.addArgument(builder.getI32Type(), builder.getUnknownLoc());
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  unsigned checks = 0;
  std::string error;

  std::optional<CanonicalGraphBuildResult> graph =
      CanonicalTTIRGraphBuilder(semantics, schemas, [&]() {
        ++checks;
        return true;
      }).build(root, error);

  EXPECT_FALSE(graph);
  EXPECT_EQ(checks, 1u);
  EXPECT_EQ(error, "canonical graph construction interrupted");
}

TEST(CanonicalGraphKeyTest, CapturesPureCalleeBodyAsCallSummary) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  std::string source = R"mlir(
    module {
      tt.func @add(%lhs: i32, %rhs: i32) -> i32 {
        %sum = arith.addi %lhs, %rhs : i32
        tt.return %sum : i32
      }
      tt.func @caller(%lhs: i32, %rhs: i32) -> i32 {
        %result = tt.call @add(%lhs, %rhs) : (i32, i32) -> i32
        tt.return %result : i32
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  triton::FuncOp caller = module->lookupSymbol<triton::FuncOp>("caller");
  ASSERT_TRUE(caller);
  triton::CallOp call;
  caller.walk([&](triton::CallOp operation) { call = operation; });
  ASSERT_TRUE(call);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  ASSERT_TRUE(semantics.supports(call));
  std::string error;
  std::optional<CanonicalGraphBuildResult> graph =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(call.getResult(0), error);
  ASSERT_TRUE(graph) << error;
  auto root = llvm::find_if(graph->graphs, [&](const auto &record) {
    return record.graphHash == graph->rootGraphHash;
  });
  ASSERT_NE(root, graph->graphs.end());
  EXPECT_EQ(root->operationName, "tt.call");
  EXPECT_NE(root->attributes.find("ags.call_summary_regions"),
            std::string::npos);
  EXPECT_NE(root->attributes.find("arith.addi"), std::string::npos);
}

TEST(CanonicalGraphKeyTest,
     OnlineImportMatchesOfflineEnumerationWithoutPublishingNoOp) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();

  GraphUniverseSpec spec;
  spec.name = "canonical-key-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 1;
  spec.bounds.maxRank = 0;
  spec.bounds.maxOperands = 2;
  spec.bounds.maxResults = 1;
  spec.operations = {"ags.input", "arith.constant", "arith.subi"};
  spec.integerBits = {32};
  spec.integerConstants = {0};
  spec.semanticDomains = {GraphSemanticDomain::Signature,
                          GraphSemanticDomain::Shape,
                          GraphSemanticDomain::Index,
                          GraphSemanticDomain::Value,
                          GraphSemanticDomain::Definedness};
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  OfflineEnumerationResult enumeration =
      OfflineTypedGraphEnumerator().enumerate(spec, schemas, {i32});
  ASSERT_TRUE(enumeration.isComplete());
  OfflineEquivalenceProofResult noProofs;
  std::string error;
  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          enumeration.graphs, noProofs, spec.getStableHash(), "semantics",
          true, true, nullptr, error);
  ASSERT_TRUE(store) << error;

  Block block;
  Value argument = block.addArgument(i32, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  Value zero = opBuilder.create<arith::ConstantIntOp>(
      builder.getUnknownLoc(), 0, 32);
  Value root = opBuilder.create<arith::SubIOp>(builder.getUnknownLoc(),
                                               argument, zero);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::optional<CanonicalGraphBuildResult> online =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(online) << error;
  ASSERT_EQ(online->boundaryInputs.size(), 1u);
  EXPECT_EQ(online->boundaryInputs.front(), argument);
  ASSERT_FALSE(online->normalizedSemanticKey.empty());
  std::optional<CanonicalGraphBuildResult> direct =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(argument, error);
  ASSERT_TRUE(direct) << error;
  EXPECT_NE(direct->rootGraphHash, online->rootGraphHash);
  EXPECT_EQ(direct->normalizedSemanticKey,
            online->normalizedSemanticKey);

  auto offline = llvm::find_if(store->graphs, [&](const auto &graph) {
    return graph.graphHash == online->rootGraphHash;
  });
  ASSERT_NE(offline, store->graphs.end());
  EXPECT_EQ(offline->operationName, "arith.subi");
  EXPECT_EQ(offline->operatorSize, 1u);

  GraphCostContext costContext;
  costContext.target = "generic";
  ParetoExtractionResult pareto = ParetoExtractor().extract(
      *store, StructuralFallbackCostModel(), costContext);
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(*store, pareto, error);
  ASSERT_TRUE(database) << error;
  OnlineRuleLookupResult lookup = OnlineRuleMatcher(*database).lookup(
      online->rootGraphHash, spec.getStableHash(), "semantics",
      costContext.getStableKey());
  // Canonical identity still matches the offline graph exactly, but a class
  // with no proved alternative is audit data rather than an executable rule.
  EXPECT_EQ(lookup.status, OnlineRuleLookupStatus::Miss);
}

TEST(CanonicalGraphKeyTest, CountsSharedImportedOperationOnce) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  Builder builder(&context);
  Block block;
  Value argument = block.addArgument(builder.getI32Type(),
                                     builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  Value one = opBuilder.create<arith::ConstantIntOp>(
      builder.getUnknownLoc(), 1, 32);
  Value shared = opBuilder.create<arith::AddIOp>(
      builder.getUnknownLoc(), argument, one);
  Value root = opBuilder.create<arith::SubIOp>(
      builder.getUnknownLoc(), shared, shared);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> graph =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(graph) << error;
  auto rootRecord = llvm::find_if(graph->graphs, [&](const auto &record) {
    return record.graphHash == graph->rootGraphHash;
  });
  ASSERT_NE(rootRecord, graph->graphs.end());
  ASSERT_EQ(rootRecord->operandHashes.size(), 2u);
  EXPECT_EQ(rootRecord->operandHashes[0], rootRecord->operandHashes[1]);
  EXPECT_EQ(rootRecord->operatorSize, 2u);
}

TEST(CanonicalGraphKeyTest, LoadKeyIncludesPrecedingStoreMemoryState) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);

  GraphUniverseSpec spec;
  spec.name = "canonical-memory-state-test";
  spec.target = "generic";
  spec.bounds.maxNodes = 2;
  spec.bounds.maxRank = 0;
  // Cover every declared tt.load/tt.store overload selected by the operation
  // names.  The masked forms add an i1 data operand and the implicit memory
  // state makes the three-operand load/store schemas have semantic arity four.
  spec.bounds.maxOperands = 4;
  spec.bounds.maxResults = 1;
  spec.bounds.allowEffects = true;
  spec.operations = {"ags.input", "ags.memory", "tt.load", "tt.store"};
  spec.integerBits = {1, 32};
  spec.pointerAddressSpaces = {1};
  auto segmentDomain = [](StringRef name, ArrayRef<int32_t> segments) {
    GraphAttributeDomain domain;
    domain.name = name.str();
    GraphAttributeValue value;
    value.kind = GraphAttributeValue::Kind::IntegerArray;
    value.integerArray.assign(segments.begin(), segments.end());
    domain.values.push_back(std::move(value));
    return domain;
  };
  spec.attributeDomains = {
      segmentDomain("load_segments_unmasked", {1, 0, 0}),
      segmentDomain("load_segments_masked", {1, 1, 0}),
      segmentDomain("load_segments_other", {1, 1, 1})};
  spec.semanticDomains = {GraphSemanticDomain::Signature,
                          GraphSemanticDomain::Shape,
                          GraphSemanticDomain::Index,
                          GraphSemanticDomain::Value,
                          GraphSemanticDomain::Definedness,
                          GraphSemanticDomain::Effect,
                          GraphSemanticDomain::Memory};
  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  OfflineEnumerationResult enumeration = OfflineTypedGraphEnumerator().enumerate(
      spec, schemas,
      {builder.getI1Type(), i32, pointer, NoneType::get(&context)});
  ASSERT_TRUE(enumeration.isComplete())
      << enumeration.toJSON(spec.getStableHash());
  OfflineEquivalenceProofResult noProofs;
  std::string error;
  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          enumeration.graphs, noProofs, spec.getStableHash(), "semantics",
          true, true, nullptr, error);
  ASSERT_TRUE(store) << error;

  Block block;
  Value address = block.addArgument(pointer, builder.getUnknownLoc());
  Value stored = block.addArgument(i32, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  OperationState storeState(builder.getUnknownLoc(), "tt.store");
  storeState.addOperands({address, stored});
  Operation *storeOperation = opBuilder.create(storeState);
  OperationState loadState(builder.getUnknownLoc(), "tt.load");
  loadState.addOperands(address);
  loadState.addTypes(i32);
  loadState.addAttribute("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr({1, 0, 0}));
  Value root = opBuilder.create(loadState)->getResult(0);

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::optional<CanonicalGraphBuildResult> onlineStore =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(storeOperation, error);
  ASSERT_TRUE(onlineStore) << error;
  auto onlineStoreRecord =
      llvm::find_if(onlineStore->graphs, [&](const auto &graph) {
        return graph.graphHash == onlineStore->rootGraphHash;
      });
  ASSERT_NE(onlineStoreRecord, onlineStore->graphs.end());
  EXPECT_TRUE(StringRef(onlineStoreRecord->attributes)
                  .contains("ags.state_domain = \"memory\""));
  auto offlineStore = llvm::find_if(store->graphs, [&](const auto &graph) {
    return graph.graphHash == onlineStore->rootGraphHash;
  });
  std::string mismatch;
  llvm::raw_string_ostream mismatchStream(mismatch);
  if (offlineStore == store->graphs.end()) {
    mismatchStream << "online store: hash=" << onlineStoreRecord->graphHash
                   << " type=" << onlineStoreRecord->resultType
                   << " attrs=" << onlineStoreRecord->attributes
                   << " size=" << onlineStoreRecord->operatorSize
                   << " operands=";
    for (const std::string &hash : onlineStoreRecord->operandHashes)
      mismatchStream << hash << ',';
    mismatchStream << "\noffline candidates with matching op/type/arity:\n";
    unsigned printed = 0;
    for (const PersistentGraphRecord &graph : store->graphs) {
      if (graph.operationName != onlineStoreRecord->operationName ||
          graph.resultType != onlineStoreRecord->resultType ||
          graph.operandHashes.size() !=
              onlineStoreRecord->operandHashes.size())
        continue;
      mismatchStream << "  hash=" << graph.graphHash
                     << " attrs=" << graph.attributes
                     << " size=" << graph.operatorSize << " operands=";
      for (const std::string &hash : graph.operandHashes)
        mismatchStream << hash << ',';
      mismatchStream << '\n';
      if (++printed == 8)
        break;
    }
  }
  ASSERT_NE(offlineStore, store->graphs.end()) << mismatchStream.str();
  EXPECT_EQ(offlineStore->operationName, "tt.store");
  EXPECT_EQ(offlineStore->resultType, "none");
  EXPECT_EQ(offlineStore->operatorSize, 1u);

  std::optional<CanonicalGraphBuildResult> online =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(online) << error;
  ASSERT_EQ(online->boundaryInputs.size(), 2u);
  auto onlineRoot = llvm::find_if(online->graphs, [&](const auto &graph) {
    return graph.graphHash == online->rootGraphHash;
  });
  ASSERT_NE(onlineRoot, online->graphs.end());
  ASSERT_EQ(onlineRoot->operationName, "tt.load");
  ASSERT_EQ(onlineRoot->operandHashes.size(), 2u);
  auto state = llvm::find_if(online->graphs, [&](const auto &graph) {
    return graph.graphHash == onlineRoot->operandHashes.back();
  });
  ASSERT_NE(state, online->graphs.end());
  EXPECT_EQ(state->operationName, "tt.store");

  auto offline = llvm::find_if(store->graphs, [&](const auto &graph) {
    return graph.graphHash == online->rootGraphHash;
  });
  ASSERT_NE(offline, store->graphs.end());
  EXPECT_EQ(offline->operationName, "tt.load");
  EXPECT_EQ(offline->operatorSize, 2u);
}

TEST(CanonicalGraphKeyTest,
     GuardedModuloLoadAndMaskedLinearLoadHaveTheSameStateKey) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @wrapped(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
        %offset = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %bound = tt.splat %n : i32 -> tensor<16xi32>
        %wrapped = arith.remsi %offset, %bound : tensor<16xi32>
        %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %wrapped_ptr = tt.addptr %src_base, %wrapped : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %value = tt.load %wrapped_ptr : tensor<16x!tt.ptr<f32>>
        %guard = arith.cmpi slt, %offset, %bound : tensor<16xi32>
        %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %linear_dst = tt.addptr %dst_base, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        tt.store %linear_dst, %value, %guard : tensor<16x!tt.ptr<f32>>
        tt.return
      }
      tt.func @linear(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
        %offset = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %bound = tt.splat %n : i32 -> tensor<16xi32>
        %guard = arith.cmpi slt, %offset, %bound : tensor<16xi32>
        %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %linear_src = tt.addptr %src_base, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %value = tt.load %linear_src, %guard : tensor<16x!tt.ptr<f32>>
        %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %linear_dst = tt.addptr %dst_base, %offset : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        tt.store %linear_dst, %value, %guard : tensor<16x!tt.ptr<f32>>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  auto findStore = [&](StringRef functionName) {
    triton::FuncOp function =
        module->lookupSymbol<triton::FuncOp>(functionName);
    Operation *store = nullptr;
    function.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "tt.store")
        store = operation;
    });
    return store;
  };
  Operation *wrappedStore = findStore("wrapped");
  Operation *linearStore = findStore("linear");
  ASSERT_TRUE(wrappedStore);
  ASSERT_TRUE(linearStore);

  GraphOptimizationOptions options;
  options.compileOn91095 = true;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> wrapped =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(wrappedStore, error);
  ASSERT_TRUE(wrapped) << error;
  std::optional<CanonicalGraphBuildResult> linear =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(linearStore, error);
  ASSERT_TRUE(linear) << error;
  ASSERT_FALSE(wrapped->normalizedSemanticKey.empty());
  ASSERT_FALSE(linear->normalizedSemanticKey.empty());
  EXPECT_EQ(wrapped->normalizedSemanticKey, linear->normalizedSemanticKey);
}

TEST(CanonicalGraphKeyTest,
     GuardedRank2ModuloLoadAndMaskedLinearLoadHaveTheSameStateKey) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @wrapped(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
        %columns = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %bound_columns = tt.splat %n : i32 -> tensor<16xi32>
        %wrapped = arith.remsi %columns, %bound_columns : tensor<16xi32>
        %wrapped_row = tt.expand_dims %wrapped {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %wrapped_full = tt.broadcast %wrapped_row : tensor<1x16xi32> -> tensor<8x16xi32>
        %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
        %rows_column = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
        %bound_rows = tt.splat %n : i32 -> tensor<8x1xi32>
        %row_offset = arith.muli %rows_column, %bound_rows : tensor<8x1xi32>
        %row_full = tt.broadcast %row_offset : tensor<8x1xi32> -> tensor<8x16xi32>
        %wrapped_offset = arith.addi %row_full, %wrapped_full : tensor<8x16xi32>
        %linear_row = tt.expand_dims %columns {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %linear_full = tt.broadcast %linear_row : tensor<1x16xi32> -> tensor<8x16xi32>
        %linear_offset = arith.addi %row_full, %linear_full : tensor<8x16xi32>
        %bound_guard = tt.splat %n : i32 -> tensor<1x16xi32>
        %guard_row = arith.cmpi slt, %linear_row, %bound_guard : tensor<1x16xi32>
        %guard = tt.broadcast %guard_row : tensor<1x16xi1> -> tensor<8x16xi1>
        %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
        %wrapped_ptr = tt.addptr %src_base, %wrapped_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
        %value = tt.load %wrapped_ptr : tensor<8x16x!tt.ptr<f32>>
        %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
        %linear_ptr = tt.addptr %dst_base, %linear_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
        tt.store %linear_ptr, %value, %guard : tensor<8x16x!tt.ptr<f32>>
        tt.return
      }
      tt.func @linear(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
        %columns = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
        %rows_column = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
        %bound_rows = tt.splat %n : i32 -> tensor<8x1xi32>
        %row_offset = arith.muli %rows_column, %bound_rows : tensor<8x1xi32>
        %row_full = tt.broadcast %row_offset : tensor<8x1xi32> -> tensor<8x16xi32>
        %linear_row = tt.expand_dims %columns {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %linear_full = tt.broadcast %linear_row : tensor<1x16xi32> -> tensor<8x16xi32>
        %linear_offset = arith.addi %row_full, %linear_full : tensor<8x16xi32>
        %bound_guard = tt.splat %n : i32 -> tensor<1x16xi32>
        %guard_row = arith.cmpi slt, %linear_row, %bound_guard : tensor<1x16xi32>
        %guard = tt.broadcast %guard_row : tensor<1x16xi1> -> tensor<8x16xi1>
        %src_base = tt.splat %src : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
        %linear_src = tt.addptr %src_base, %linear_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
        %value = tt.load %linear_src, %guard : tensor<8x16x!tt.ptr<f32>>
        %dst_base = tt.splat %dst : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
        %linear_ptr = tt.addptr %dst_base, %linear_offset : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
        tt.store %linear_ptr, %value, %guard : tensor<8x16x!tt.ptr<f32>>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);

  auto findStore = [&](StringRef functionName) {
    triton::FuncOp function =
        module->lookupSymbol<triton::FuncOp>(functionName);
    Operation *store = nullptr;
    function.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "tt.store")
        store = operation;
    });
    return store;
  };
  Operation *wrappedStore = findStore("wrapped");
  Operation *linearStore = findStore("linear");
  ASSERT_TRUE(wrappedStore);
  ASSERT_TRUE(linearStore);

  GraphOptimizationOptions options;
  options.compileOn91095 = true;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> wrapped =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(wrappedStore, error);
  ASSERT_TRUE(wrapped) << error;
  std::optional<CanonicalGraphBuildResult> linear =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(linearStore, error);
  ASSERT_TRUE(linear) << error;
  ASSERT_FALSE(wrapped->normalizedSemanticKey.empty());
  ASSERT_FALSE(linear->normalizedSemanticKey.empty());
  EXPECT_EQ(wrapped->normalizedSemanticKey, linear->normalizedSemanticKey);
}

TEST(CanonicalGraphKeyTest, TensorReadKeyIncludesPrecedingBufferWriteState) {
  MLIRContext context;
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  Builder builder(&context);
  Type memrefType = MemRefType::get({4}, builder.getI32Type());
  Type tensorType = RankedTensorType::get({4}, builder.getI32Type());

  Block block;
  Value source = block.addArgument(memrefType, builder.getUnknownLoc());
  Value target = block.addArgument(memrefType, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  OperationState copyState(builder.getUnknownLoc(), "memref.copy");
  copyState.addOperands({source, target});
  Operation *copy = opBuilder.create(copyState);
  OperationState tensorState(builder.getUnknownLoc(),
                             "bufferization.to_tensor");
  tensorState.addOperands(target);
  tensorState.addTypes(tensorType);
  Value root = opBuilder.create(tensorState)->getResult(0);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> write =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(copy, error);
  ASSERT_TRUE(write) << error;
  auto writeRoot = llvm::find_if(write->graphs, [&](const auto &graph) {
    return graph.graphHash == write->rootGraphHash;
  });
  ASSERT_NE(writeRoot, write->graphs.end());
  EXPECT_EQ(writeRoot->operationName, "memref.copy");
  EXPECT_EQ(writeRoot->resultType, "none");

  std::optional<CanonicalGraphBuildResult> read =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(read) << error;
  auto readRoot = llvm::find_if(read->graphs, [&](const auto &graph) {
    return graph.graphHash == read->rootGraphHash;
  });
  ASSERT_NE(readRoot, read->graphs.end());
  ASSERT_EQ(readRoot->operationName, "bufferization.to_tensor");
  ASSERT_EQ(readRoot->operandHashes.size(), 2u);
  auto bufferState = llvm::find_if(read->graphs, [&](const auto &graph) {
    return graph.graphHash == readRoot->operandHashes.back();
  });
  ASSERT_NE(bufferState, read->graphs.end());
  EXPECT_EQ(bufferState->operationName, "memref.copy");
  EXPECT_TRUE(StringRef(bufferState->attributes)
                  .contains("ags.state_domain = \"buffer\""));
}

TEST(CanonicalGraphKeyTest, WholeKernelRootBindsFinalMemoryAndLaunch) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<triton::TritonDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type pointer = triton::PointerType::get(i32, /*addressSpace=*/1);

  OperationState scopeState(builder.getUnknownLoc(), "test.kernel");
  scopeState.addRegion();
  Operation *scope = Operation::create(scopeState);
  Block *body = new Block();
  scope->getRegion(0).push_back(body);
  Value address = body->addArgument(pointer, builder.getUnknownLoc());
  Value value = body->addArgument(i32, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(body);
  OperationState storeState(builder.getUnknownLoc(), "tt.store");
  storeState.addOperands({address, value});
  Operation *store = opBuilder.create(storeState);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> result =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .buildKernel(scope, error);
  ASSERT_TRUE(result) << error;
  // Whole-kernel RuleDB lookup uses exact/template topology.  It must not
  // expand the complete shared tensor DAG into an unused semantic expression.
  EXPECT_TRUE(result->normalizedSemanticKey.empty());
  auto kernel = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == result->rootGraphHash;
  });
  ASSERT_NE(kernel, result->graphs.end());
  EXPECT_EQ(kernel->operationName, "ags.kernel");
  EXPECT_TRUE(StringRef(kernel->attributes)
                  .contains("ags.proof_only = true"));
  ASSERT_EQ(kernel->operandHashes.size(), 2u);
  auto memory = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == kernel->operandHashes[0];
  });
  auto launch = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == kernel->operandHashes[1];
  });
  ASSERT_NE(memory, result->graphs.end());
  ASSERT_NE(launch, result->graphs.end());
  EXPECT_EQ(memory->operationName, "tt.store");
  EXPECT_EQ(launch->operationName, "ags.launch");
  ASSERT_EQ(result->stateOperations.size(), 1u);
  EXPECT_EQ(result->stateOperations.front().second, store);
  scope->destroy();
}

TEST(CanonicalGraphKeyTest, WholeKernelRootIncludesExplicitLaunchBound) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @kernel(%ptr: !tt.ptr<i32>, %value: i32, %n: i32)
          attributes {ags.launch_grid_bound_axis = 0 : i32,
                      ags.launch_grid_bound_argument = 2 : i32} {
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<triton::FuncOp>().begin();

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> result =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .buildKernel(function.getOperation(), error);
  ASSERT_TRUE(result) << error;
  auto kernel = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == result->rootGraphHash;
  });
  ASSERT_NE(kernel, result->graphs.end());
  ASSERT_EQ(kernel->operandHashes.size(), 2u);
  auto launch = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == kernel->operandHashes[1];
  });
  ASSERT_NE(launch, result->graphs.end());
  EXPECT_EQ(launch->operationName, "ags.launch_bound");
  EXPECT_TRUE(StringRef(launch->attributes).contains("axis = 0 : i32"));
  ASSERT_EQ(launch->operandHashes.size(), 2u);
  auto terminal = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == launch->operandHashes[1];
  });
  ASSERT_NE(terminal, result->graphs.end());
  EXPECT_EQ(terminal->operationName, "ags.launch");
}

TEST(CanonicalGraphKeyTest, WholeKernelRootIncludesMultipleLaunchBounds) {
  MLIRContext context;
  context.getOrLoadDialect<triton::TritonDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @kernel(%ptr: !tt.ptr<i32>, %rows: i32, %columns: i32)
          attributes {ags.launch_grid_bound_axes = array<i32: 0, 1>,
                      ags.launch_grid_bound_arguments = array<i32: 1, 2>} {
        tt.store %ptr, %rows : !tt.ptr<i32>
        tt.return
      }
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<triton::FuncOp>().begin();

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> result =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .buildKernel(function.getOperation(), error);
  ASSERT_TRUE(result) << error;
  auto kernel = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == result->rootGraphHash;
  });
  ASSERT_NE(kernel, result->graphs.end());
  auto outer = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == kernel->operandHashes[1];
  });
  ASSERT_NE(outer, result->graphs.end());
  EXPECT_EQ(outer->operationName, "ags.launch_bound");
  EXPECT_TRUE(StringRef(outer->attributes).contains("axis = 1 : i32"));
  auto inner = llvm::find_if(result->graphs, [&](const auto &graph) {
    return graph.graphHash == outer->operandHashes[1];
  });
  ASSERT_NE(inner, result->graphs.end());
  EXPECT_EQ(inner->operationName, "ags.launch_bound");
  EXPECT_TRUE(StringRef(inner->attributes).contains("axis = 0 : i32"));
}

TEST(CanonicalGraphKeyTest, MarksPrivateAllocationBoundaryWithoutAssumingArgs) {
  MLIRContext context;
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  Builder builder(&context);
  Type memrefType = MemRefType::get({4}, builder.getI32Type());
  Type tensorType = RankedTensorType::get({4}, builder.getI32Type());

  Block block;
  Value source = block.addArgument(memrefType, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  OperationState allocState(builder.getUnknownLoc(), "memref.alloc");
  allocState.addTypes(memrefType);
  Value target = opBuilder.create(allocState)->getResult(0);
  OperationState copyState(builder.getUnknownLoc(), "memref.copy");
  copyState.addOperands({source, target});
  opBuilder.create(copyState);
  OperationState tensorState(builder.getUnknownLoc(),
                             "bufferization.to_tensor");
  tensorState.addOperands(target);
  tensorState.addTypes(tensorType);
  Value root = opBuilder.create(tensorState)->getResult(0);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  std::string error;
  std::optional<CanonicalGraphBuildResult> result =
      CanonicalTTIRGraphBuilder(semantics, schemas).build(root, error);
  ASSERT_TRUE(result) << error;
  EXPECT_EQ(result->boundaryInputs.size(), 2u);
  EXPECT_EQ(llvm::count_if(result->graphs, [](const auto &graph) {
              return graph.operationName == "ags.private_buffer";
            }),
            1u);
  EXPECT_EQ(llvm::count_if(result->graphs, [](const auto &graph) {
              return graph.operationName == "ags.input";
            }),
            1u);
}

TEST(CanonicalGraphKeyTest, MultiResultSiblingsHaveDistinctCanonicalRoots) {
  MLIRContext context;
  context.getOrLoadDialect<arith::ArithDialect>();
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Block block;
  Value lhs = block.addArgument(i32, builder.getUnknownLoc());
  Value rhs = block.addArgument(i32, builder.getUnknownLoc());
  OpBuilder opBuilder(&context);
  opBuilder.setInsertionPointToStart(&block);
  OperationState state(builder.getUnknownLoc(), "arith.mului_extended");
  state.addOperands({lhs, rhs});
  state.addTypes({i32, i32});
  Operation *extended = opBuilder.create(state);

  GraphOptimizationOptions options;
  SmallVector<TTIRCandidateOpSchema, 4> schemas =
      createBuiltinTTIRCandidateOpSchemas(options);
  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  ASSERT_TRUE(semantics.supports(extended));
  std::string error;
  std::optional<CanonicalGraphBuildResult> low =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(extended->getResult(0), error);
  ASSERT_TRUE(low) << error;
  error.clear();
  std::optional<CanonicalGraphBuildResult> high =
      CanonicalTTIRGraphBuilder(semantics, schemas)
          .build(extended->getResult(1), error);
  ASSERT_TRUE(high) << error;
  EXPECT_NE(low->rootGraphHash, high->rootGraphHash);
  auto highRoot = llvm::find_if(high->graphs, [&](const auto &graph) {
    return graph.graphHash == high->rootGraphHash;
  });
  ASSERT_NE(highRoot, high->graphs.end());
  EXPECT_NE(highRoot->attributes.find("ags.result_index = 1"),
            std::string::npos);
  EXPECT_NE(highRoot->attributes.find("ags.operation_result_types"),
            std::string::npos);
}

} // namespace

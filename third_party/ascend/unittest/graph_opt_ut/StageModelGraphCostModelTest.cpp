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

#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

static PersistentGraphRecord makeGraph(
    std::string operation, std::string type, std::string attributes,
    unsigned operatorSize, std::vector<std::string> operands = {}) {
  PersistentGraphRecord graph;
  graph.operationName = std::move(operation);
  graph.resultType = std::move(type);
  graph.attributes = std::move(attributes);
  graph.operatorSize = operatorSize;
  graph.operandHashes = std::move(operands);
  graph.graphHash = getPersistentGraphHash(graph);
  return graph;
}

static PersistentEquivalenceClassStore makeTensorAddStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", "tensor<128xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", "tensor<128xi32>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord add =
      makeGraph("arith.addi", "tensor<128xi32>", "{}", 1,
                {lhs.graphHash, rhs.graphHash});
  store.graphs = {lhs, rhs, add};
  return store;
}

static PersistentEquivalenceClassStore makeScalarStoreState() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord pointer = makeGraph(
      "ags.input", "!tt.ptr<i32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord value = makeGraph(
      "ags.input", "i32", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord memory =
      makeGraph("ags.memory", "none", "{}", 0);
  PersistentGraphRecord write = makeGraph(
      "tt.store", "none", "{ags.state_domain = \"memory\"}", 1,
      {pointer.graphHash, value.graphHash, memory.graphHash});
  store.graphs = {pointer, value, memory, write};
  return store;
}

static PersistentEquivalenceClassStore makeMaskedHistogramStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord source = makeGraph(
      "ags.input", "tensor<256xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord mask = makeGraph(
      "ags.input", "tensor<256xi1>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord histogram = makeGraph(
      "tt.histogram", "tensor<16xi32>", "{}", 1,
      {source.graphHash, mask.graphHash});
  store.graphs = {source, mask, histogram};
  return store;
}

static PersistentEquivalenceClassStore makeParkedHistogramStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord source = makeGraph(
      "ags.input", "tensor<256xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord mask = makeGraph(
      "ags.input", "tensor<256xi1>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord parking = makeGraph(
      "arith.constant", "tensor<256xi32>",
      "{value = dense<-1> : tensor<256xi32>}", 0);
  PersistentGraphRecord select =
      makeGraph("arith.select", "tensor<256xi32>", "{}", 1,
                {mask.graphHash, source.graphHash, parking.graphHash});
  PersistentGraphRecord histogram =
      makeGraph("tt.histogram", "tensor<16xi32>", "{}", 1,
                {select.graphHash});
  store.graphs = {source, mask, parking, select, histogram};
  return store;
}

static PersistentEquivalenceClassStore
makeCanonicalScanStore(llvm::StringRef combiner = "arith.addi",
                       int64_t axis = 0) {
  MLIRContext context;
  Builder builder(&context);
  ArrayAttr regions = buildCanonicalBinaryRegion(
      &context, builder.getI32Type(), combiner, "tt.scan.return");
  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("ags.canonical_regions", regions),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis)),
       builder.getNamedAttr("reverse", builder.getBoolAttr(false))});
  std::string attributeText;
  llvm::raw_string_ostream stream(attributeText);
  attributes.print(stream);
  stream.flush();

  PersistentEquivalenceClassStore store;
  PersistentGraphRecord input = makeGraph(
      "ags.input", "tensor<16xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord scan =
      makeGraph("tt.scan", "tensor<16xi32>", attributeText, 1,
                {input.graphHash});
  store.graphs = {input, scan};
  return store;
}

static PersistentEquivalenceClassStore
makeCanonicalReduceStore(llvm::StringRef combiner = "arith.addi",
                         int64_t axis = 0) {
  MLIRContext context;
  Builder builder(&context);
  ArrayAttr regions = buildCanonicalBinaryRegion(
      &context, builder.getI32Type(), combiner, "tt.reduce.return");
  DictionaryAttr attributes = builder.getDictionaryAttr(
      {builder.getNamedAttr("ags.canonical_regions", regions),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))});
  std::string attributeText;
  llvm::raw_string_ostream stream(attributeText);
  attributes.print(stream);
  stream.flush();

  PersistentEquivalenceClassStore store;
  PersistentGraphRecord input = makeGraph(
      "ags.input", "tensor<16xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord reduce =
      makeGraph("tt.reduce", "i32", attributeText, 1, {input.graphHash});
  store.graphs = {input, reduce};
  return store;
}

static PersistentEquivalenceClassStore makeTransposeStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord input = makeGraph(
      "ags.input", "tensor<4x8xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord transpose = makeGraph(
      "tt.trans", "tensor<8x4xi32>",
      "{order = array<i32: 1, 0>}", 1, {input.graphHash});
  store.graphs = {input, transpose};
  return store;
}

static PersistentEquivalenceClassStore makeTensorXorStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", "tensor<128xi1>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", "tensor<128xi1>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord xori =
      makeGraph("arith.xori", "tensor<128xi1>", "{}", 1,
                {lhs.graphHash, rhs.graphHash});
  store.graphs = {lhs, rhs, xori};
  return store;
}

static PersistentEquivalenceClassStore makeIntegerDotStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", "tensor<16x16xi8>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", "tensor<16x16xi8>", "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord accumulator = makeGraph(
      "arith.constant", "tensor<16x16xi32>",
      "{value = dense<0> : tensor<16x16xi32>}", 0);
  PersistentGraphRecord dot =
      makeGraph("tt.dot", "tensor<16x16xi32>", "{}", 1,
                {lhs.graphHash, rhs.graphHash, accumulator.graphHash});
  store.graphs = {lhs, rhs, accumulator, dot};
  return store;
}

static PersistentEquivalenceClassStore makeEncodedTensorAddStore() {
  constexpr llvm::StringLiteral type =
      "tensor<8xi32, #ttg.blocked<{sizePerThread = [4], "
      "threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>";
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", type.str(), "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", type.str(), "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord add = makeGraph(
      "arith.addi", type.str(), "{}", 1, {lhs.graphHash, rhs.graphHash});
  store.graphs = {lhs, rhs, add};
  return store;
}

static PersistentEquivalenceClassStore makeDynamicEncodedTensorAddStore() {
  constexpr llvm::StringLiteral type =
      "tensor<?xi32, #ttg.blocked<{sizePerThread = [4], "
      "threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>";
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord lhs = makeGraph(
      "ags.input", type.str(), "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord rhs = makeGraph(
      "ags.input", type.str(), "{ags.symbol_id = 1 : i64}", 0);
  PersistentGraphRecord add = makeGraph(
      "arith.addi", type.str(), "{}", 1, {lhs.graphHash, rhs.graphHash});
  store.graphs = {lhs, rhs, add};
  return store;
}

static PersistentEquivalenceClassStore makeTensorPointerStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord input = makeGraph(
      "ags.input", "!tt.ptr<tensor<8xi32>>",
      "{ags.symbol_id = 0 : i64}", 0);
  store.graphs = {input};
  return store;
}

static PersistentEquivalenceClassStore makeMemrefSubviewStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord source = makeGraph(
      "ags.input", "memref<8xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord subview = makeGraph(
      "memref.subview", "memref<4xi32, strided<[1], offset: 0>>",
      "{static_offsets = array<i64: 0>, static_sizes = array<i64: 4>, "
      "static_strides = array<i64: 1>}",
      1, {source.graphHash});
  store.graphs = {source, subview};
  return store;
}

static PersistentEquivalenceClassStore makeTritonAscendLoadStore() {
  PersistentEquivalenceClassStore store;
  PersistentGraphRecord source = makeGraph(
      "ags.input", "tensor<8xi32>", "{ags.symbol_id = 0 : i64}", 0);
  PersistentGraphRecord load = makeGraph(
      "ascend.indirect_load", "tensor<8xi32>", "{}", 1,
      {source.graphHash});
  store.graphs = {source, load};
  return store;
}

TEST(StageModelGraphCostModelTest, ScoresMaterializedTensorDag) {
  PersistentEquivalenceClassStore store = makeTensorAddStore();
  const PersistentGraphRecord &root = store.graphs.back();
  GraphCostContext context;
  context.target = "ascend910b";
  AscendStageModelGraphCostModel model;
  std::string reason;
  std::optional<uint64_t> cycles =
      model.estimateStageCycles(store, root, context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, ScoresMaterializedMemoryStateDag) {
  PersistentEquivalenceClassStore store = makeScalarStoreState();
  const PersistentGraphRecord &root = store.graphs.back();
  GraphCostContext context;
  context.target = "ascend910b";
  AscendStageModelGraphCostModel model;
  std::string reason;
  std::optional<uint64_t> cycles =
      model.estimateStageCycles(store, root, context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, RejectsImplicitConfigForAnotherTarget) {
  PersistentEquivalenceClassStore store = makeTensorAddStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  AscendStageModelGraphCostModel model;
  std::string reason;

  EXPECT_FALSE(model
                   .estimateStageCycles(store, store.graphs.back(), context,
                                        reason)
                   .has_value());
  EXPECT_NE(reason.find("provide an explicit hardware config"),
            std::string::npos);
}

TEST(StageModelGraphCostModelTest, Scores950TargetWithExplicitDavidConfig) {
  PersistentEquivalenceClassStore store = makeTensorAddStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, ScoresMaskedHistogramOn950Target) {
  PersistentEquivalenceClassStore store = makeMaskedHistogramStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, PrefersOutOfRangeParkingToMaskedHistogram) {
  PersistentEquivalenceClassStore maskedStore = makeMaskedHistogramStore();
  PersistentEquivalenceClassStore parkedStore = makeParkedHistogramStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string maskedReason;
  std::string parkedReason;
  std::optional<uint64_t> maskedCycles = model.estimateStageCycles(
      maskedStore, maskedStore.graphs.back(), context, maskedReason);
  std::optional<uint64_t> parkedCycles = model.estimateStageCycles(
      parkedStore, parkedStore.graphs.back(), context, parkedReason);

  ASSERT_TRUE(maskedCycles.has_value()) << maskedReason;
  ASSERT_TRUE(parkedCycles.has_value()) << parkedReason;
  EXPECT_LT(*parkedCycles, *maskedCycles);
}

TEST(StageModelGraphCostModelTest, ScoresCanonicalScanRegionOn950Target) {
  PersistentEquivalenceClassStore store = makeCanonicalScanStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, RejectsUnsupportedScanCombiner) {
  PersistentEquivalenceClassStore store =
      makeCanonicalScanStore("arith.subi");
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("tt.scan"), std::string::npos);
}

TEST(StageModelGraphCostModelTest, RejectsOutOfRangeScanAxis) {
  PersistentEquivalenceClassStore store =
      makeCanonicalScanStore("arith.addi", 1);
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("tt.scan"), std::string::npos);
}

TEST(StageModelGraphCostModelTest, ScoresCanonicalReduceRegionOn950Target) {
  PersistentEquivalenceClassStore store = makeCanonicalReduceStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, RejectsUnsupportedReduceCombiner) {
  PersistentEquivalenceClassStore store =
      makeCanonicalReduceStore("arith.subi");
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("tt.reduce"), std::string::npos);
}

TEST(StageModelGraphCostModelTest, PreservesZeroCostTransposeResultType) {
  PersistentEquivalenceClassStore store = makeTransposeStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_EQ(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, ScoresTensorBitwiseDag) {
  PersistentEquivalenceClassStore store = makeTensorXorStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, ScoresIntegerDotWithDenseZeroAccumulator) {
  PersistentEquivalenceClassStore store = makeIntegerDotStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest, ScoresTensorWithRegisteredGpuEncoding) {
  PersistentEquivalenceClassStore store = makeEncodedTensorAddStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  ASSERT_TRUE(cycles.has_value()) << reason;
  EXPECT_GT(*cycles, 0u);
  EXPECT_TRUE(reason.empty());
}

TEST(StageModelGraphCostModelTest,
     RejectsDynamicGpuEncodingBeforeUnsafeLayoutVerification) {
  PersistentEquivalenceClassStore store = makeDynamicEncodedTensorAddStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("dynamic tensor shapes"), std::string::npos);
}

TEST(StageModelGraphCostModelTest,
     RejectsTensorPointerBeforeUnsafeOperationVerification) {
  PersistentEquivalenceClassStore store = makeTensorPointerStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("tensor-pointer decomposition"), std::string::npos);
}

TEST(StageModelGraphCostModelTest,
     RegisteredMemrefSubviewFailsClosedWithoutDialectAbort) {
  PersistentEquivalenceClassStore store = makeMemrefSubviewStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_FALSE(reason.empty());
}

TEST(StageModelGraphCostModelTest,
     RejectsTritonAscendSourceOpBeforeDialectNamespaceCollision) {
  PersistentEquivalenceClassStore store = makeTritonAscendLoadStore();
  GraphCostContext context;
  context.target = "ascend950dt";
  std::string config = std::string(AGS_TEST_SOURCE_DIR) +
      "/third_party/ascend/costmodel/configs/ascend_davidv100.json";
  AscendStageModelGraphCostModel model(config);
  std::string reason;
  std::optional<uint64_t> cycles = model.estimateStageCycles(
      store, store.graphs.back(), context, reason);

  EXPECT_FALSE(cycles.has_value());
  EXPECT_NE(reason.find("no conversion for source operation"),
            std::string::npos);
  EXPECT_NE(reason.find("ascend.indirect_load"), std::string::npos);
}

} // namespace

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

#include "TritonToGraph/GraphUniverseSpec.h"

#include "gtest/gtest.h"

using namespace mlir::triton::cfg;

namespace {

static constexpr const char kSpecA[] = R"json(
{
  "schema_version": 2,
  "name": "test-universe",
  "target": "generic",
  "bounds": {
    "max_nodes": 4,
    "max_rank": 2,
    "max_operands": 2,
    "max_results": 1,
    "max_shape_symbols": 1,
    "allow_dynamic_shapes": true,
    "allow_control_flow": false,
    "allow_effects": false
  },
  "catalog_dialects": ["tt", "arith"],
  "operations": ["tt.scan", "arith.subi"],
  "integer_bits": [64, 32],
  "floating_bits": [],
  "pointer_address_spaces": [3, 1],
  "integer_constants": [1, -1, 0],
  "tensor_shapes": [[16, 1], [16], [1, 16]],
  "attribute_domains": {
    "reverse": [true, false],
    "axis": [1, 0],
    "order": [[1, 0], [0, 1]]
  },
  "semantic_domains": ["value", "definedness", "signature"]
}
)json";

static constexpr const char kSpecB[] = R"json(
{
  "semantic_domains": ["definedness", "signature", "value"],
  "integer_constants": [-1, 0, 1],
  "tensor_shapes": [[1, 16], [16], [16, 1]],
  "attribute_domains": {
    "axis": [0, 1],
    "order": [[0, 1], [1, 0]],
    "reverse": [false, true]
  },
  "floating_bits": [],
  "integer_bits": [32, 64],
  "pointer_address_spaces": [1, 3],
  "catalog_dialects": ["arith", "tt"],
  "operations": ["arith.subi", "tt.scan"],
  "bounds": {
    "allow_effects": false,
    "allow_control_flow": false,
    "allow_dynamic_shapes": true,
    "max_shape_symbols": 1,
    "max_results": 1,
    "max_operands": 2,
    "max_rank": 2,
    "max_nodes": 4
  },
  "target": "generic",
  "name": "test-universe",
  "schema_version": 2
}
)json";

TEST(GraphUniverseSpecTest, CanonicalFormAndHashIgnoreJSONOrder) {
  std::string error;
  std::optional<GraphUniverseSpec> lhs = GraphUniverseSpec::parse(kSpecA, error);
  ASSERT_TRUE(lhs) << error;
  std::optional<GraphUniverseSpec> rhs = GraphUniverseSpec::parse(kSpecB, error);
  ASSERT_TRUE(rhs) << error;
  EXPECT_EQ(lhs->toCanonicalJSON(), rhs->toCanonicalJSON());
  EXPECT_EQ(lhs->getStableHash(), rhs->getStableHash());
  EXPECT_EQ(lhs->bounds.getMaxInputSymbols(), lhs->bounds.maxOperands);
}

TEST(GraphUniverseSpecTest, InputSymbolBoundIsIndependentFromOperationArity) {
  std::string source = kSpecA;
  size_t begin = source.find("\"max_operands\": 2,");
  ASSERT_NE(begin, std::string::npos);
  begin += std::string("\"max_operands\": 2,").size();
  source.insert(begin, "\n    \"max_input_symbols\": 1,");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->bounds.maxOperands, 2u);
  EXPECT_EQ(spec->bounds.getMaxInputSymbols(), 1u);
  EXPECT_NE(spec->toCanonicalJSON(),
            GraphUniverseSpec::parse(kSpecA, error)->toCanonicalJSON());
}

TEST(GraphUniverseSpecTest, OperationOperandCountsAreFiniteAndCanonical) {
  std::string source = kSpecA;
  size_t begin = source.find("\"integer_bits\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin,
                "\"operation_operand_counts\": {\"tt.scan\": [2, 1], "
                "\"arith.subi\": [2]},\n  ");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_TRUE(spec->allowsOperationOperandCount("tt.scan", 1));
  EXPECT_TRUE(spec->allowsOperationOperandCount("tt.scan", 2));
  EXPECT_FALSE(spec->allowsOperationOperandCount("tt.scan", 3));
  EXPECT_TRUE(spec->allowsOperationOperandCount("arith.subi", 2));
  EXPECT_TRUE(spec->allowsOperationOperandCount("ags.input", 0));
  EXPECT_NE(spec->toCanonicalJSON().find(
                "\"operation_operand_counts\":{\"arith.subi\":[2],"
                "\"tt.scan\":[1,2]}"),
            std::string::npos);
}

TEST(GraphUniverseSpecTest, OperationOperandCountsRejectUnknownOperation) {
  std::string source = kSpecA;
  size_t begin = source.find("\"integer_bits\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin,
                "\"operation_operand_counts\": {\"tt.histogram\": [1]},\n  ");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("is not an exact member of operations"),
            std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, EffectfulNodeBoundIsFiniteAndHashed) {
  std::string source = kSpecA;
  size_t begin = source.find("\"max_operands\": 2,");
  ASSERT_NE(begin, std::string::npos);
  begin += std::string("\"max_operands\": 2,").size();
  source.insert(begin, "\n    \"max_effectful_nodes\": 1,");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->bounds.getMaxEffectfulNodes(), 1u);
  EXPECT_NE(spec->toCanonicalJSON().find(
                "\"max_effectful_nodes\":1"),
            std::string::npos);

  size_t bound = source.find("\"max_effectful_nodes\": 1");
  ASSERT_NE(bound, std::string::npos);
  source.replace(bound, std::string("\"max_effectful_nodes\": 1").size(),
                 "\"max_effectful_nodes\": 5");
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("must not exceed max_nodes"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, ExplicitTypesCanDefineTheWholeFiniteTypeDomain) {
  std::string source = kSpecA;
  size_t widths = source.find("\"integer_bits\": [64, 32]");
  ASSERT_NE(widths, std::string::npos);
  source.replace(widths, std::string("\"integer_bits\": [64, 32]").size(),
                 "\"integer_bits\": []");
  size_t floating = source.find("\"floating_bits\": [],");
  ASSERT_NE(floating, std::string::npos);
  floating += std::string("\"floating_bits\": [],").size();
  source.insert(floating, "\n  \"explicit_types\": [\"i1\", \"i32\"],");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_TRUE(spec->integerBits.empty());
  EXPECT_EQ(spec->explicitTypes.size(), 2u);
}

TEST(GraphUniverseSpecTest, AttributeLookupAcceptsProgrammaticInsertionOrder) {
  GraphUniverseSpec spec;
  GraphAttributeDomain later;
  later.name = "z_domain";
  GraphAttributeDomain earlier;
  earlier.name = "a_domain";
  spec.attributeDomains = {std::move(later), std::move(earlier)};

  ASSERT_NE(spec.findAttributeDomain("a_domain"), nullptr);
  EXPECT_EQ(spec.findAttributeDomain("a_domain")->name, "a_domain");
  ASSERT_NE(spec.findAttributeDomain("z_domain"), nullptr);
  EXPECT_EQ(spec.findAttributeDomain("z_domain")->name, "z_domain");
  EXPECT_EQ(spec.findAttributeDomain("missing"), nullptr);
}

TEST(GraphUniverseSpecTest, FinalLayerPartitionIsFiniteAndHashed) {
  std::string source = kSpecA;
  size_t begin = source.find("\"max_shape_symbols\": 1,");
  ASSERT_NE(begin, std::string::npos);
  begin += std::string("\"max_shape_symbols\": 1,").size();
  source.insert(begin,
                "\n    \"final_layer_partition_count\": 4,"
                "\n    \"final_layer_partition_index\": 2,");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->bounds.finalLayerPartitionCount, 4u);
  EXPECT_EQ(spec->bounds.finalLayerPartitionIndex, 2u);
  EXPECT_NE(spec->toCanonicalJSON().find(
                "\"final_layer_partition_count\":4"),
            std::string::npos);

  std::string invalid = source;
  size_t index = invalid.find("\"final_layer_partition_index\": 2");
  ASSERT_NE(index, std::string::npos);
  invalid.replace(
      index, std::string("\"final_layer_partition_index\": 2").size(),
      "\"final_layer_partition_index\": 4");
  EXPECT_FALSE(GraphUniverseSpec::parse(invalid, error));
  EXPECT_NE(error.find("index must be smaller than count"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, PointerDomainIsExplicitAndCanonical) {
  std::string error;
  std::optional<GraphUniverseSpec> spec = GraphUniverseSpec::parse(kSpecA, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->pointerAddressSpaces, std::vector<unsigned>({1, 3}));
  EXPECT_NE(spec->toCanonicalJSON().find("\"pointer_address_spaces\":[1,3]"),
            std::string::npos);
}

TEST(GraphUniverseSpecTest, IndexTypeDomainIsExplicitAndCanonical) {
  std::string source = kSpecA;
  size_t begin = source.find("\"integer_constants\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin, "\"include_index_type\": true,\n  ");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_TRUE(spec->includeIndexType);
  EXPECT_NE(spec->toCanonicalJSON().find("\"include_index_type\":true"),
            std::string::npos);
}

TEST(GraphUniverseSpecTest, ExplicitTypesAreFiniteAndCanonical) {
  std::string source = kSpecA;
  size_t begin = source.find("\"integer_constants\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin,
                "\"explicit_types\": [\"tuple<i32, i1>\", "
                "\"memref<16xi32>\"],\n  ");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->explicitTypes,
            std::vector<std::string>({"memref<16xi32>",
                                      "tuple<i32, i1>"}));
  EXPECT_NE(spec->toCanonicalJSON().find(
                "\"explicit_types\":[\"memref<16xi32>\","
                "\"tuple<i32, i1>\"]"),
            std::string::npos);
}

TEST(GraphUniverseSpecTest, InputSymbolTypesPreserveBoundaryOrder) {
  std::string source = kSpecA;
  size_t begin = source.find("\"integer_constants\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin,
                "\"input_symbol_types\": [\"i64\", \"i32\"],\n  ");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_EQ(spec->inputSymbolTypes,
            std::vector<std::string>({"i64", "i32"}));
  EXPECT_NE(spec->toCanonicalJSON().find(
                "\"input_symbol_types\":[\"i64\",\"i32\"]"),
            std::string::npos);
}

TEST(GraphUniverseSpecTest, RejectsAnImplicitInfiniteOperationSet) {
  std::string source = kSpecA;
  size_t begin = source.find("\"tt.scan\", \"arith.subi\"");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"tt.scan\", \"arith.subi\"").size(),
                 "");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("non-empty finite set"), std::string::npos) << error;
}

TEST(GraphUniverseSpecTest, ResolvesDialectPatternsToFiniteSemanticSet) {
  std::string source = kSpecA;
  size_t begin = source.find("\"tt.scan\", \"arith.subi\"");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"tt.scan\", \"arith.subi\"").size(),
                 "\"arith.*\", \"tt.*\"");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  ASSERT_TRUE(spec->resolveOperationPatterns(
      {"tt.scan", "tt.histogram", "arith.subi", "math.exp"}, error))
      << error;
  EXPECT_EQ(spec->operations,
            std::vector<std::string>({"arith.subi", "tt.histogram",
                                      "tt.scan"}));
  EXPECT_EQ(spec->toCanonicalJSON().find(".*"), std::string::npos);
}

TEST(GraphUniverseSpecTest, SharedPrefixFamilyHashUsesResolvedOperations) {
  std::string source = kSpecA;
  size_t begin = source.find("\"tt.scan\", \"arith.subi\"");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"tt.scan\", \"arith.subi\"").size(),
                 "\"arith.*\", \"tt.*\"");
  std::string error;
  std::optional<GraphUniverseSpec> shard =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(shard) << error;
  std::optional<GraphUniverseSpec> prefixFamily =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(prefixFamily) << error;
  const std::vector<std::string> registered = {
      "tt.scan", "tt.histogram", "arith.subi", "math.exp"};
  std::vector<llvm::StringRef> registeredRefs;
  registeredRefs.reserve(registered.size());
  for (const std::string &operation : registered)
    registeredRefs.emplace_back(operation);
  ASSERT_TRUE(shard->resolveOperationPatterns(registeredRefs, error)) << error;
  EXPECT_NE(shard->getStableHash(), prefixFamily->getStableHash());
  ASSERT_TRUE(prefixFamily->resolveOperationPatterns(registeredRefs, error))
      << error;
  EXPECT_EQ(shard->getStableHash(), prefixFamily->getStableHash());
}

TEST(GraphUniverseSpecTest, RejectsUnmatchedDialectPattern) {
  std::string source = kSpecA;
  size_t begin = source.find("\"tt.scan\", \"arith.subi\"");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"tt.scan\", \"arith.subi\"").size(),
                 "\"arith.*\", \"tt.*\"");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_FALSE(spec->resolveOperationPatterns({"arith.subi"}, error));
  EXPECT_NE(error.find("tt.*"), std::string::npos) << error;
}

TEST(GraphUniverseSpecTest, EffectsRequireEffectSemantics) {
  std::string source = kSpecA;
  size_t begin = source.find("\"allow_effects\": false");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"allow_effects\": false").size(),
                 "\"allow_effects\": true");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("effect semantic domain"), std::string::npos) << error;
}

TEST(GraphUniverseSpecTest, BufferRequiresEffectButNotMemorySemantics) {
  std::string source = kSpecA;
  size_t effects = source.find("\"allow_effects\": false");
  ASSERT_NE(effects, std::string::npos);
  source.replace(effects, std::string("\"allow_effects\": false").size(),
                 "\"allow_effects\": true");
  size_t domains =
      source.find("\"value\", \"definedness\", \"signature\"");
  ASSERT_NE(domains, std::string::npos);
  source.replace(
      domains, std::string("\"value\", \"definedness\", \"signature\"").size(),
      "\"value\", \"definedness\", \"effect\", \"buffer\", \"signature\"");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  EXPECT_TRUE(spec->hasDomain(GraphSemanticDomain::Buffer));
  EXPECT_FALSE(spec->hasDomain(GraphSemanticDomain::Memory));
}

TEST(GraphUniverseSpecTest, LaunchRequiresIndexSemantics) {
  std::string source = kSpecA;
  size_t begin = source.find("\"value\", \"definedness\", \"signature\"");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin,
                 std::string("\"value\", \"definedness\", \"signature\"")
                     .size(),
                 "\"value\", \"definedness\", \"signature\", \"launch\"");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("launch semantic domain requires the index domain"),
            std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RejectsMissingMandatorySignatureDomain) {
  std::string source = kSpecA;
  size_t begin = source.find(", \"signature\"");
  ASSERT_NE(begin, std::string::npos);
  source.erase(begin, std::string(", \"signature\"").size());
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("signature, value and definedness"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RejectsUnknownKeys) {
  std::string source = kSpecA;
  size_t begin = source.find("\"schema_version\": 2,");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin, "\"max_nodez\": 4,\n");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("unknown key 'max_nodez'"), std::string::npos) << error;
}

TEST(GraphUniverseSpecTest, TensorEncodingDomainRequiresLayoutSemantics) {
  std::string source = kSpecA;
  size_t begin = source.find("\"tensor_shapes\":");
  ASSERT_NE(begin, std::string::npos);
  source.insert(begin, "\"tensor_encodings\": [\"layout-a\"],\n  ");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("require the layout semantic domain"),
            std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RequiresFiniteShapesForTensorUniverse) {
  std::string source = kSpecA;
  size_t begin = source.find("[[16, 1], [16], [1, 16]]");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("[[16, 1], [16], [1, 16]]").size(),
                 "[]");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("explicit finite tensor_shapes"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RejectsShapeAboveMaxRank) {
  std::string source = kSpecA;
  size_t begin = source.find("[[16, 1], [16], [1, 16]]");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("[[16, 1], [16], [1, 16]]").size(),
                 "[[1, 2, 3]]");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("rank in [1, max_rank]"), std::string::npos) << error;
}

TEST(GraphUniverseSpecTest, AcceptsExplicitFiniteDynamicShapeTemplates) {
  std::string source = kSpecA;
  size_t begin = source.find("[[16, 1], [16], [1, 16]]");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("[[16, 1], [16], [1, 16]]").size(),
                 "[[-1, 16], [16]]");
  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::parse(source, error);
  ASSERT_TRUE(spec) << error;
  ASSERT_EQ(spec->tensorShapes.size(), 2u);
  EXPECT_EQ(spec->tensorShapes.front(),
            std::vector<int64_t>({GraphUniverseSpec::kDynamicDimension, 16}));
}

TEST(GraphUniverseSpecTest, RejectsDynamicShapeWhenDisabled) {
  std::string source = kSpecA;
  size_t shape = source.find("[[16, 1], [16], [1, 16]]");
  ASSERT_NE(shape, std::string::npos);
  source.replace(shape, std::string("[[16, 1], [16], [1, 16]]").size(),
                 "[[-1]]");
  size_t enabled = source.find("\"allow_dynamic_shapes\": true");
  ASSERT_NE(enabled, std::string::npos);
  source.replace(enabled, std::string("\"allow_dynamic_shapes\": true").size(),
                 "\"allow_dynamic_shapes\": false");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("allow_dynamic_shapes=true"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RejectsTooManyDynamicShapeSymbols) {
  std::string source = kSpecA;
  size_t begin = source.find("[[16, 1], [16], [1, 16]]");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("[[16, 1], [16], [1, 16]]").size(),
                 "[[-1, -1]]");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("more dynamic dimensions"), std::string::npos)
      << error;
}

TEST(GraphUniverseSpecTest, RejectsMixedAttributeDomainKinds) {
  std::string source = kSpecA;
  size_t begin = source.find("\"axis\": [1, 0]");
  ASSERT_NE(begin, std::string::npos);
  source.replace(begin, std::string("\"axis\": [1, 0]").size(),
                 "\"axis\": [1, true]");
  std::string error;
  EXPECT_FALSE(GraphUniverseSpec::parse(source, error));
  EXPECT_NE(error.find("same kind"), std::string::npos) << error;
}

} // namespace

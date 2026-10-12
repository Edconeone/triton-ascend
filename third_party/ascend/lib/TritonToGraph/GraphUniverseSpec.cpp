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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace mlir::triton::cfg;

namespace {

template <typename T>
static bool isSortedAndUnique(llvm::ArrayRef<T> values) {
  return llvm::is_sorted(values) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

static bool isIdentifier(llvm::StringRef value, bool allowDot) {
  if (value.empty())
    return false;
  return llvm::all_of(value, [allowDot](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return std::isalnum(byte) || c == '_' || c == '-' ||
           (allowDot && c == '.');
  });
}

static bool rejectUnknownKeys(const llvm::json::Object &object,
                              llvm::ArrayRef<llvm::StringRef> allowed,
                              llvm::StringRef context, std::string &error) {
  for (const auto &entry : object) {
    llvm::StringRef key = entry.first;
    if (!llvm::is_contained(allowed, key)) {
      error = ("unknown key '" + key + "' in " + context).str();
      return false;
    }
  }
  return true;
}

static bool readRequiredUnsigned(const llvm::json::Object &object,
                                 llvm::StringRef key, unsigned &result,
                                 std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value || *value < 0 ||
      static_cast<uint64_t>(*value) > std::numeric_limits<unsigned>::max()) {
    error = ("'" + key + "' must be a non-negative integer").str();
    return false;
  }
  result = static_cast<unsigned>(*value);
  return true;
}

static bool readRequiredBool(const llvm::json::Object &object,
                             llvm::StringRef key, bool &result,
                             std::string &error) {
  std::optional<bool> value = object.getBoolean(key);
  if (!value) {
    error = ("'" + key + "' must be a boolean").str();
    return false;
  }
  result = *value;
  return true;
}

static bool readStringArray(const llvm::json::Object &object,
                            llvm::StringRef key,
                            std::vector<std::string> &result,
                            std::string &error) {
  const llvm::json::Array *array = object.getArray(key);
  if (!array) {
    error = ("'" + key + "' must be an array").str();
    return false;
  }
  for (const llvm::json::Value &element : *array) {
    std::optional<llvm::StringRef> value = element.getAsString();
    if (!value) {
      error = ("every element of '" + key + "' must be a string").str();
      return false;
    }
    result.push_back(value->str());
  }
  llvm::sort(result);
  return true;
}

static bool readOrderedStringArray(const llvm::json::Object &object,
                                   llvm::StringRef key,
                                   std::vector<std::string> &result,
                                   std::string &error) {
  const llvm::json::Array *array = object.getArray(key);
  if (!array) {
    error = ("'" + key + "' must be an array").str();
    return false;
  }
  for (const llvm::json::Value &element : *array) {
    std::optional<llvm::StringRef> value = element.getAsString();
    if (!value) {
      error = ("every element of '" + key + "' must be a string").str();
      return false;
    }
    result.push_back(value->str());
  }
  return true;
}

static bool readUnsignedArray(const llvm::json::Object &object,
                              llvm::StringRef key,
                              std::vector<unsigned> &result,
                              std::string &error) {
  const llvm::json::Array *array = object.getArray(key);
  if (!array) {
    error = ("'" + key + "' must be an array").str();
    return false;
  }
  for (const llvm::json::Value &element : *array) {
    std::optional<int64_t> value = element.getAsInteger();
    if (!value || *value <= 0 ||
        static_cast<uint64_t>(*value) > std::numeric_limits<unsigned>::max()) {
      error = ("every element of '" + key +
               "' must be a positive integer")
                  .str();
      return false;
    }
    result.push_back(static_cast<unsigned>(*value));
  }
  llvm::sort(result);
  return true;
}

static bool readIntegerArray(const llvm::json::Object &object,
                             llvm::StringRef key,
                             std::vector<int64_t> &result,
                             std::string &error) {
  const llvm::json::Array *array = object.getArray(key);
  if (!array) {
    error = ("'" + key + "' must be an array").str();
    return false;
  }
  for (const llvm::json::Value &element : *array) {
    std::optional<int64_t> value = element.getAsInteger();
    if (!value) {
      error = ("every element of '" + key + "' must be an integer").str();
      return false;
    }
    result.push_back(*value);
  }
  llvm::sort(result);
  return true;
}

static bool readTensorShapes(const llvm::json::Object &object,
                             llvm::StringRef key,
                             std::vector<std::vector<int64_t>> &result,
                             std::string &error) {
  const llvm::json::Array *shapes = object.getArray(key);
  if (!shapes) {
    error = ("'" + key + "' must be an array").str();
    return false;
  }
  for (const llvm::json::Value &shapeValue : *shapes) {
    const llvm::json::Array *shape = shapeValue.getAsArray();
    if (!shape || shape->empty()) {
      error = ("every element of '" + key +
               "' must be a non-empty integer array")
                  .str();
      return false;
    }
    std::vector<int64_t> dimensions;
    for (const llvm::json::Value &dimensionValue : *shape) {
      std::optional<int64_t> dimension = dimensionValue.getAsInteger();
      if (!dimension || *dimension == 0 || *dimension < -1) {
        error = ("every dimension in '" + key +
                 "' must be a positive integer or -1")
                    .str();
        return false;
      }
      dimensions.push_back(*dimension);
    }
    result.push_back(std::move(dimensions));
  }
  llvm::sort(result);
  return true;
}

static bool readAttributeDomains(
    const llvm::json::Object &object, llvm::StringRef key,
    std::vector<GraphAttributeDomain> &result, std::string &error) {
  const llvm::json::Object *domains = object.getObject(key);
  if (!domains) {
    error = ("'" + key + "' must be an object").str();
    return false;
  }
  for (const auto &entry : *domains) {
    const std::string domainName = entry.first.str();
    const llvm::json::Array *values = entry.second.getAsArray();
    if (!values) {
      error = "attribute domain '" + domainName + "' must be an array";
      return false;
    }
    GraphAttributeDomain domain;
    domain.name = domainName;
    for (const llvm::json::Value &jsonValue : *values) {
      GraphAttributeValue value;
      if (std::optional<int64_t> integer = jsonValue.getAsInteger()) {
        value.kind = GraphAttributeValue::Kind::Integer;
        value.integer = *integer;
      } else if (std::optional<bool> boolean = jsonValue.getAsBoolean()) {
        value.kind = GraphAttributeValue::Kind::Boolean;
        value.boolean = *boolean;
      } else if (const llvm::json::Array *array = jsonValue.getAsArray()) {
        if (array->empty()) {
          error = ("integer-array value in attribute domain '" +
                   domainName + "' must be non-empty");
          return false;
        }
        value.kind = GraphAttributeValue::Kind::IntegerArray;
        for (const llvm::json::Value &element : *array) {
          std::optional<int64_t> integer = element.getAsInteger();
          if (!integer || *integer < std::numeric_limits<int32_t>::min() ||
              *integer > std::numeric_limits<int32_t>::max()) {
            error = ("every integer-array element in attribute domain '" +
                     domainName + "' must fit i32");
            return false;
          }
          value.integerArray.push_back(static_cast<int32_t>(*integer));
        }
      } else {
        error = "unsupported value in attribute domain '" + domainName +
                "'; expected integer, boolean or integer array";
        return false;
      }
      domain.values.push_back(std::move(value));
    }
    llvm::sort(domain.values, [](const GraphAttributeValue &lhs,
                                 const GraphAttributeValue &rhs) {
      return lhs.toCanonicalJSON() < rhs.toCanonicalJSON();
    });
    result.push_back(std::move(domain));
  }
  llvm::sort(result, [](const GraphAttributeDomain &lhs,
                        const GraphAttributeDomain &rhs) {
    return lhs.name < rhs.name;
  });
  return true;
}

static bool readOperationOperandCountDomains(
    const llvm::json::Object &object, llvm::StringRef key,
    std::vector<GraphOperationOperandCountDomain> &result,
    std::string &error) {
  const llvm::json::Object *domains = object.getObject(key);
  if (!domains) {
    error = ("'" + key + "' must be an object").str();
    return false;
  }
  for (const auto &entry : *domains) {
    GraphOperationOperandCountDomain domain;
    domain.operation = entry.first.str();
    const llvm::json::Array *counts = entry.second.getAsArray();
    if (!counts || counts->empty()) {
      error = "operation operand-count domain '" + domain.operation +
              "' must be a non-empty array";
      return false;
    }
    for (const llvm::json::Value &element : *counts) {
      std::optional<int64_t> count = element.getAsInteger();
      if (!count || *count < 0 ||
          static_cast<uint64_t>(*count) >
              std::numeric_limits<unsigned>::max()) {
        error = "every operand count for '" + domain.operation +
                "' must be a non-negative integer";
        return false;
      }
      domain.operandCounts.push_back(static_cast<unsigned>(*count));
    }
    llvm::sort(domain.operandCounts);
    result.push_back(std::move(domain));
  }
  llvm::sort(result, [](const GraphOperationOperandCountDomain &lhs,
                        const GraphOperationOperandCountDomain &rhs) {
    return lhs.operation < rhs.operation;
  });
  return true;
}

static void appendJSONString(llvm::raw_ostream &os, llvm::StringRef value) {
  os << llvm::json::Value(value);
}

template <typename T>
static void appendNumberArray(llvm::raw_ostream &os, llvm::ArrayRef<T> values) {
  os << '[';
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index)
      os << ',';
    os << value;
  }
  os << ']';
}

static void appendStringArray(llvm::raw_ostream &os,
                              llvm::ArrayRef<std::string> values) {
  os << '[';
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index)
      os << ',';
    appendJSONString(os, value);
  }
  os << ']';
}

static void appendTensorShapes(
    llvm::raw_ostream &os,
    llvm::ArrayRef<std::vector<int64_t>> tensorShapes) {
  os << '[';
  for (auto [index, shape] : llvm::enumerate(tensorShapes)) {
    if (index)
      os << ',';
    appendNumberArray<int64_t>(os, shape);
  }
  os << ']';
}

static void appendAttributeDomains(
    llvm::raw_ostream &os,
    llvm::ArrayRef<GraphAttributeDomain> attributeDomains) {
  os << '{';
  for (auto [domainIndex, domain] : llvm::enumerate(attributeDomains)) {
    if (domainIndex)
      os << ',';
    appendJSONString(os, domain.name);
    os << ":[";
    for (auto [valueIndex, value] : llvm::enumerate(domain.values)) {
      if (valueIndex)
        os << ',';
      os << value.toCanonicalJSON();
    }
    os << ']';
  }
  os << '}';
}

static void appendOperationOperandCountDomains(
    llvm::raw_ostream &os,
    llvm::ArrayRef<GraphOperationOperandCountDomain> domains) {
  os << '{';
  for (auto [index, domain] : llvm::enumerate(domains)) {
    if (index)
      os << ',';
    appendJSONString(os, domain.operation);
    os << ':';
    appendNumberArray<unsigned>(os, domain.operandCounts);
  }
  os << '}';
}

} // namespace

std::string GraphAttributeValue::toCanonicalJSON() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  switch (kind) {
  case Kind::Integer:
    os << integer;
    break;
  case Kind::Boolean:
    os << (boolean ? "true" : "false");
    break;
  case Kind::IntegerArray:
    appendNumberArray<int32_t>(os, integerArray);
    break;
  }
  return result;
}

llvm::StringRef mlir::triton::cfg::stringifyGraphSemanticDomain(
    GraphSemanticDomain domain) {
  switch (domain) {
  case GraphSemanticDomain::Signature:
    return "signature";
  case GraphSemanticDomain::Value:
    return "value";
  case GraphSemanticDomain::Definedness:
    return "definedness";
  case GraphSemanticDomain::Shape:
    return "shape";
  case GraphSemanticDomain::Index:
    return "index";
  case GraphSemanticDomain::Layout:
    return "layout";
  case GraphSemanticDomain::Effect:
    return "effect";
  case GraphSemanticDomain::Memory:
    return "memory";
  case GraphSemanticDomain::Launch:
    return "launch";
  case GraphSemanticDomain::Buffer:
    return "buffer";
  }
  llvm_unreachable("unknown GraphSemanticDomain");
}

std::optional<GraphSemanticDomain>
mlir::triton::cfg::parseGraphSemanticDomain(llvm::StringRef spelling) {
  return llvm::StringSwitch<std::optional<GraphSemanticDomain>>(spelling)
      .Case("signature", GraphSemanticDomain::Signature)
      .Case("value", GraphSemanticDomain::Value)
      .Case("definedness", GraphSemanticDomain::Definedness)
      .Case("shape", GraphSemanticDomain::Shape)
      .Case("index", GraphSemanticDomain::Index)
      .Case("layout", GraphSemanticDomain::Layout)
      .Case("effect", GraphSemanticDomain::Effect)
      .Case("memory", GraphSemanticDomain::Memory)
      .Case("launch", GraphSemanticDomain::Launch)
      .Case("buffer", GraphSemanticDomain::Buffer)
      .Default(std::nullopt);
}

std::optional<GraphUniverseSpec>
GraphUniverseSpec::parse(llvm::StringRef source, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(source);
  if (!parsed) {
    error = "invalid JSON: " + llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "universe specification must be a JSON object";
    return std::nullopt;
  }
  if (!rejectUnknownKeys(
          *root,
          {"schema_version", "name", "target", "bounds",
           "catalog_dialects", "operations", "integer_bits",
            "operation_operand_counts",
           "floating_bits", "include_index_type", "pointer_address_spaces",
           "integer_constants",
           "tensor_shapes", "tensor_encodings", "explicit_types",
           "input_symbol_types",
           "attribute_domains", "semantic_domains"},
          "root", error))
    return std::nullopt;

  GraphUniverseSpec spec;
  if (!readRequiredUnsigned(*root, "schema_version", spec.schemaVersion,
                            error))
    return std::nullopt;
  std::optional<llvm::StringRef> name = root->getString("name");
  std::optional<llvm::StringRef> target = root->getString("target");
  if (!name || !target) {
    error = "'name' and 'target' must be strings";
    return std::nullopt;
  }
  spec.name = name->str();
  spec.target = target->str();

  const llvm::json::Object *bounds = root->getObject("bounds");
  if (!bounds) {
    error = "'bounds' must be an object";
    return std::nullopt;
  }
  if (!rejectUnknownKeys(
          *bounds,
          {"max_nodes", "max_rank", "max_operands", "max_input_symbols",
           "max_effectful_nodes",
           "max_results",
           "max_shape_symbols", "allow_dynamic_shapes",
           "allow_control_flow", "allow_effects",
           "final_layer_partition_count",
           "final_layer_partition_index"},
          "bounds", error) ||
      !readRequiredUnsigned(*bounds, "max_nodes", spec.bounds.maxNodes,
                            error) ||
      !readRequiredUnsigned(*bounds, "max_rank", spec.bounds.maxRank, error) ||
      !readRequiredUnsigned(*bounds, "max_operands", spec.bounds.maxOperands,
                            error) ||
      !readRequiredUnsigned(*bounds, "max_results", spec.bounds.maxResults,
                            error) ||
      !readRequiredUnsigned(*bounds, "max_shape_symbols",
                            spec.bounds.maxShapeSymbols, error) ||
      !readRequiredBool(*bounds, "allow_dynamic_shapes",
                        spec.bounds.allowDynamicShapes, error) ||
      !readRequiredBool(*bounds, "allow_control_flow",
                        spec.bounds.allowControlFlow, error) ||
      !readRequiredBool(*bounds, "allow_effects", spec.bounds.allowEffects,
                        error))
    return std::nullopt;
  if (bounds->get("max_input_symbols") &&
      !readRequiredUnsigned(*bounds, "max_input_symbols",
                            spec.bounds.maxInputSymbols, error))
    return std::nullopt;
  if (bounds->get("max_effectful_nodes") &&
      !readRequiredUnsigned(*bounds, "max_effectful_nodes",
                            spec.bounds.maxEffectfulNodes, error))
    return std::nullopt;
  if (bounds->get("final_layer_partition_count") &&
      !readRequiredUnsigned(*bounds, "final_layer_partition_count",
                            spec.bounds.finalLayerPartitionCount, error))
    return std::nullopt;
  if (bounds->get("final_layer_partition_index") &&
      !readRequiredUnsigned(*bounds, "final_layer_partition_index",
                            spec.bounds.finalLayerPartitionIndex, error))
    return std::nullopt;

  if (!readStringArray(*root, "catalog_dialects", spec.catalogDialects,
                       error) ||
      !readStringArray(*root, "operations", spec.operations, error) ||
      !readUnsignedArray(*root, "integer_bits", spec.integerBits, error) ||
      !readUnsignedArray(*root, "floating_bits", spec.floatingBits, error) ||
      !readIntegerArray(*root, "integer_constants", spec.integerConstants,
                        error) ||
      !readTensorShapes(*root, "tensor_shapes", spec.tensorShapes, error) ||
      !readAttributeDomains(*root, "attribute_domains",
                            spec.attributeDomains, error))
    return std::nullopt;

  if (root->get("operation_operand_counts") &&
      !readOperationOperandCountDomains(*root, "operation_operand_counts",
                                        spec.operationOperandCounts, error))
    return std::nullopt;

  if (root->get("tensor_encodings") &&
      !readStringArray(*root, "tensor_encodings", spec.tensorEncodings,
                       error))
    return std::nullopt;

  if (root->get("explicit_types") &&
      !readStringArray(*root, "explicit_types", spec.explicitTypes, error))
    return std::nullopt;

  if (root->get("input_symbol_types") &&
      !readOrderedStringArray(*root, "input_symbol_types",
                              spec.inputSymbolTypes, error))
    return std::nullopt;

  if (root->get("include_index_type")) {
    std::optional<bool> includeIndex = root->getBoolean("include_index_type");
    if (!includeIndex) {
      error = "'include_index_type' must be a boolean";
      return std::nullopt;
    }
    spec.includeIndexType = *includeIndex;
  }

  if (root->get("pointer_address_spaces") &&
      !readUnsignedArray(*root, "pointer_address_spaces",
                         spec.pointerAddressSpaces, error))
    return std::nullopt;

  std::vector<std::string> domains;
  if (!readStringArray(*root, "semantic_domains", domains, error))
    return std::nullopt;
  for (const std::string &domain : domains) {
    std::optional<GraphSemanticDomain> parsedDomain =
        parseGraphSemanticDomain(domain);
    if (!parsedDomain) {
      error = "unknown semantic domain '" + domain + "'";
      return std::nullopt;
    }
    spec.semanticDomains.push_back(*parsedDomain);
  }
  llvm::sort(spec.semanticDomains);

  if (!spec.verify(error))
    return std::nullopt;
  return spec;
}

std::optional<GraphUniverseSpec>
GraphUniverseSpec::loadFromFile(llvm::StringRef path, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read universe specification '" + path.str() + "': " +
            buffer.getError().message();
    return std::nullopt;
  }
  return parse(buffer.get()->getBuffer(), error);
}

bool GraphUniverseSpec::verify(std::string &error) const {
  if (schemaVersion != kCurrentSchemaVersion) {
    error = "unsupported schema_version " + std::to_string(schemaVersion);
    return false;
  }
  if (!isIdentifier(name, false) || !isIdentifier(target, false)) {
    error = "'name' and 'target' must use [A-Za-z0-9_-]+";
    return false;
  }
  if (bounds.maxNodes == 0 || bounds.maxOperands == 0 ||
      bounds.getMaxInputSymbols() == 0 ||
      bounds.maxResults == 0) {
    error = "max_nodes, max_operands, max_input_symbols and max_results must "
            "be positive";
    return false;
  }
  if (bounds.finalLayerPartitionCount == 0 ||
      bounds.finalLayerPartitionIndex >= bounds.finalLayerPartitionCount) {
    error = "final-layer partition count must be positive and index must be "
            "smaller than count";
    return false;
  }
  if (bounds.finalLayerPartitionCount > 1 && bounds.maxNodes < 2) {
    error = "final-layer partitioning requires max_nodes >= 2 so every "
            "partition contains the complete operand prefix";
    return false;
  }
  if (bounds.getMaxEffectfulNodes() > bounds.maxNodes) {
    error = "max_effectful_nodes must not exceed max_nodes";
    return false;
  }
  if (bounds.allowDynamicShapes && bounds.maxShapeSymbols == 0) {
    error = "dynamic shapes require max_shape_symbols > 0";
    return false;
  }
  if (bounds.maxRank == 0 && !tensorShapes.empty()) {
    error = "tensor_shapes must be empty when max_rank is zero";
    return false;
  }
  if (bounds.maxRank != 0 && tensorShapes.empty()) {
    error = "max_rank > 0 requires an explicit finite tensor_shapes set";
    return false;
  }
  if (!isSortedAndUnique<std::vector<int64_t>>(tensorShapes)) {
    error = "tensor_shapes must be unique";
    return false;
  }
  if (!isSortedAndUnique<std::string>(tensorEncodings)) {
    error = "tensor_encodings must be unique";
    return false;
  }
  if (!isSortedAndUnique<std::string>(explicitTypes)) {
    error = "explicit_types must be unique";
    return false;
  }
  for (const std::string &type : explicitTypes)
    if (type.empty()) {
      error = "explicit type spellings must be non-empty";
      return false;
    }
  if (inputSymbolTypes.size() > bounds.getMaxInputSymbols()) {
    error = "input_symbol_types exceeds max_input_symbols";
    return false;
  }
  for (const std::string &type : inputSymbolTypes)
    if (type.empty()) {
      error = "input symbol type spellings must be non-empty";
      return false;
    }
  if (!tensorEncodings.empty() && !hasDomain(GraphSemanticDomain::Layout)) {
    error = "tensor_encodings require the layout semantic domain";
    return false;
  }
  for (const std::string &encoding : tensorEncodings)
    if (encoding.empty()) {
      error = "tensor encoding spellings must be non-empty";
      return false;
    }
  for (const std::vector<int64_t> &shape : tensorShapes) {
    if (shape.empty() || shape.size() > bounds.maxRank) {
      error = "every tensor shape must have rank in [1, max_rank]";
      return false;
    }
    if (llvm::any_of(shape, [](int64_t dimension) {
          return dimension == 0 || dimension < kDynamicDimension;
        })) {
      error = "tensor shape dimensions must be positive or -1";
      return false;
    }
    unsigned dynamicDimensions = llvm::count(shape, kDynamicDimension);
    if (dynamicDimensions && !bounds.allowDynamicShapes) {
      error = "dynamic tensor shapes require allow_dynamic_shapes=true";
      return false;
    }
    if (dynamicDimensions > bounds.maxShapeSymbols) {
      error = "a tensor shape uses more dynamic dimensions than "
              "max_shape_symbols";
      return false;
    }
  }
  if (!llvm::is_sorted(attributeDomains,
                       [](const GraphAttributeDomain &lhs,
                          const GraphAttributeDomain &rhs) {
                         return lhs.name < rhs.name;
                       })) {
    error = "attribute_domains must be sorted";
    return false;
  }
  for (auto [index, domain] : llvm::enumerate(attributeDomains)) {
    if (!isIdentifier(domain.name, false) || domain.values.empty()) {
      error = "attribute domain names must be identifiers and values must be "
              "non-empty";
      return false;
    }
    if (index && attributeDomains[index - 1].name == domain.name) {
      error = "attribute domain names must be unique";
      return false;
    }
    GraphAttributeValue::Kind kind = domain.values.front().kind;
    std::string previous;
    for (const GraphAttributeValue &value : domain.values) {
      if (value.kind != kind) {
        error = "all values in an attribute domain must have the same kind";
        return false;
      }
      std::string canonical = value.toCanonicalJSON();
      if (!previous.empty() && previous == canonical) {
        error = "attribute domain values must be unique";
        return false;
      }
      previous = std::move(canonical);
    }
  }
  if (operations.empty()) {
    error = "operations must explicitly resolve to a non-empty finite set";
    return false;
  }
  if (catalogDialects.empty() ||
      !isSortedAndUnique<std::string>(catalogDialects)) {
    error = "catalog_dialects must be a non-empty unique set";
    return false;
  }
  for (const std::string &dialect : catalogDialects) {
    if (!isIdentifier(dialect, false)) {
      error = "invalid dialect namespace '" + dialect + "'";
      return false;
    }
  }
  if (!isSortedAndUnique<std::string>(operations)) {
    error = "operations must be unique";
    return false;
  }
  if (!llvm::is_sorted(operationOperandCounts,
                       [](const GraphOperationOperandCountDomain &lhs,
                          const GraphOperationOperandCountDomain &rhs) {
                         return lhs.operation < rhs.operation;
                       })) {
    error = "operation_operand_counts must be sorted";
    return false;
  }
  for (auto [index, domain] : llvm::enumerate(operationOperandCounts)) {
    if (!isIdentifier(domain.operation, true) ||
        !llvm::StringRef(domain.operation).contains('.') ||
        domain.operandCounts.empty() ||
        !isSortedAndUnique<unsigned>(domain.operandCounts)) {
      error = "operation_operand_counts requires unique fully-qualified OP "
              "names and non-empty unique count sets";
      return false;
    }
    if (index && operationOperandCounts[index - 1].operation ==
                     domain.operation) {
      error = "operation_operand_counts names must be unique";
      return false;
    }
    if (!llvm::is_contained(operations, domain.operation)) {
      error = "operation_operand_counts entry '" + domain.operation +
              "' is not an exact member of operations";
      return false;
    }
    if (llvm::any_of(domain.operandCounts, [&](unsigned count) {
          return count > bounds.maxOperands;
        })) {
      error = "operation_operand_counts entry '" + domain.operation +
              "' exceeds max_operands";
      return false;
    }
  }
  for (const std::string &operation : operations) {
    llvm::StringRef operationRef(operation);
    const bool isDialectPattern = operationRef.ends_with(".*");
    llvm::StringRef checked =
        isDialectPattern ? operationRef.drop_back(2) : operationRef;
    if (!isIdentifier(checked, !isDialectPattern) ||
        (!isDialectPattern && !operationRef.contains('.'))) {
      error = "invalid fully-qualified operation name '" + operation + "'";
      return false;
    }
    llvm::StringRef dialect = isDialectPattern
                                  ? checked
                                  : operationRef.split('.').first;
    if (dialect != "ags" && !llvm::is_contained(catalogDialects, dialect)) {
      error = "operation '" + operation +
              "' is outside catalog_dialects";
      return false;
    }
  }
  if (integerBits.empty() && floatingBits.empty() && explicitTypes.empty()) {
    error = "at least one generated scalar width or explicit type is required";
    return false;
  }
  if (!isSortedAndUnique<unsigned>(integerBits) ||
      !isSortedAndUnique<unsigned>(floatingBits) ||
      !isSortedAndUnique<unsigned>(pointerAddressSpaces) ||
      !isSortedAndUnique<int64_t>(integerConstants) ||
      !isSortedAndUnique<GraphSemanticDomain>(semanticDomains)) {
    error = "type widths, constants and semantic domains must be unique";
    return false;
  }
  if (semanticDomains.empty() ||
      !hasDomain(GraphSemanticDomain::Signature) ||
      !hasDomain(GraphSemanticDomain::Value) ||
      !hasDomain(GraphSemanticDomain::Definedness)) {
    error = "the signature, value and definedness semantic domains are "
            "mandatory";
    return false;
  }
  if (bounds.allowEffects && !hasDomain(GraphSemanticDomain::Effect)) {
    error = "allow_effects requires the effect semantic domain";
    return false;
  }
  if (hasDomain(GraphSemanticDomain::Memory) &&
      !hasDomain(GraphSemanticDomain::Effect)) {
    error = "the memory semantic domain requires the effect domain";
    return false;
  }
  if (hasDomain(GraphSemanticDomain::Buffer) &&
      !hasDomain(GraphSemanticDomain::Effect)) {
    error = "the buffer semantic domain requires the effect domain";
    return false;
  }
  if (hasDomain(GraphSemanticDomain::Launch) &&
      !hasDomain(GraphSemanticDomain::Index)) {
    error = "the launch semantic domain requires the index domain";
    return false;
  }
  return true;
}

bool GraphUniverseSpec::resolveOperationPatterns(
    llvm::ArrayRef<llvm::StringRef> semanticOps, std::string &error) {
  std::vector<std::string> resolved;
  for (const std::string &selector : operations) {
    llvm::StringRef selectorRef(selector);
    if (!selectorRef.ends_with(".*")) {
      resolved.push_back(selector);
      continue;
    }
    llvm::StringRef prefix = selectorRef.drop_back(1);
    size_t before = resolved.size();
    for (llvm::StringRef operation : semanticOps)
      if (operation.starts_with(prefix))
        resolved.push_back(operation.str());
    if (resolved.size() == before) {
      error = "operation selector '" + selector +
              "' matches no registered semantic operation";
      return false;
    }
  }
  llvm::sort(resolved);
  resolved.erase(std::unique(resolved.begin(), resolved.end()),
                 resolved.end());
  operations = std::move(resolved);
  return verify(error);
}

std::string GraphUniverseSpec::toCanonicalJSON() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "{\n  \"schema_version\":" << schemaVersion << ",\n  \"name\":";
  appendJSONString(os, name);
  os << ",\n  \"target\":";
  appendJSONString(os, target);
  os << ",\n  \"bounds\":{\n"
        "    \"max_nodes\":"
     << bounds.maxNodes << ",\n    \"max_rank\":" << bounds.maxRank
     << ",\n    \"max_operands\":" << bounds.maxOperands
     << ",\n    \"max_input_symbols\":" << bounds.getMaxInputSymbols()
     << ",\n    \"max_effectful_nodes\":"
     << bounds.getMaxEffectfulNodes()
     << ",\n    \"max_results\":" << bounds.maxResults
     << ",\n    \"max_shape_symbols\":" << bounds.maxShapeSymbols
     << ",\n    \"final_layer_partition_count\":"
     << bounds.finalLayerPartitionCount
     << ",\n    \"final_layer_partition_index\":"
     << bounds.finalLayerPartitionIndex
     << ",\n    \"allow_dynamic_shapes\":"
     << (bounds.allowDynamicShapes ? "true" : "false")
     << ",\n    \"allow_control_flow\":"
     << (bounds.allowControlFlow ? "true" : "false")
     << ",\n    \"allow_effects\":"
     << (bounds.allowEffects ? "true" : "false") << "\n  },\n";
  os << "  \"catalog_dialects\":";
  appendStringArray(os, catalogDialects);
  os << ",\n  \"operations\":";
  appendStringArray(os, operations);
  os << ",\n  \"operation_operand_counts\":";
  appendOperationOperandCountDomains(os, operationOperandCounts);
  os << ",\n  \"integer_bits\":";
  appendNumberArray<unsigned>(os, integerBits);
  os << ",\n  \"floating_bits\":";
  appendNumberArray<unsigned>(os, floatingBits);
  os << ",\n  \"include_index_type\":"
     << (includeIndexType ? "true" : "false");
  os << ",\n  \"pointer_address_spaces\":";
  appendNumberArray<unsigned>(os, pointerAddressSpaces);
  os << ",\n  \"integer_constants\":";
  appendNumberArray<int64_t>(os, integerConstants);
  os << ",\n  \"tensor_shapes\":";
  appendTensorShapes(os, tensorShapes);
  os << ",\n  \"tensor_encodings\":";
  appendStringArray(os, tensorEncodings);
  os << ",\n  \"explicit_types\":";
  appendStringArray(os, explicitTypes);
  os << ",\n  \"input_symbol_types\":";
  appendStringArray(os, inputSymbolTypes);
  os << ",\n  \"attribute_domains\":";
  appendAttributeDomains(os, attributeDomains);
  os << ",\n  \"semantic_domains\":[";
  for (auto [index, domain] : llvm::enumerate(semanticDomains)) {
    if (index)
      os << ',';
    appendJSONString(os, stringifyGraphSemanticDomain(domain));
  }
  os << "]\n}\n";
  return result;
}

std::string GraphUniverseSpec::getStableHash() const {
  // FNV-1a is sufficient here: this is a deterministic cache/version key, not
  // a cryptographic authenticity check. The canonical JSON remains the source
  // of truth and is stored alongside every generated database.
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : toCanonicalJSON()) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  std::ostringstream os;
  os << std::hex << std::setfill('0') << std::setw(16) << hash;
  return os.str();
}

bool GraphUniverseSpec::hasDomain(GraphSemanticDomain domain) const {
  return llvm::is_contained(semanticDomains, domain);
}

bool GraphUniverseSpec::allowsOperationOperandCount(
    llvm::StringRef operation, unsigned operandCount) const {
  auto found = llvm::lower_bound(
      operationOperandCounts, operation,
      [](const GraphOperationOperandCountDomain &domain,
         llvm::StringRef name) { return domain.operation < name; });
  return found == operationOperandCounts.end() ||
         found->operation != operation ||
         llvm::is_contained(found->operandCounts, operandCount);
}

const GraphAttributeDomain *
GraphUniverseSpec::findAttributeDomain(llvm::StringRef name) const {
  // Parsed specifications are canonicalized into name order, but callers can
  // also construct GraphUniverseSpec programmatically through its public data
  // model.  Lookup must not silently lose a declared finite domain merely
  // because that construction order is non-canonical.
  auto found = llvm::find_if(attributeDomains,
                             [&](const GraphAttributeDomain &domain) {
                               return llvm::StringRef(domain.name) == name;
                             });
  return found == attributeDomains.end() ? nullptr : &*found;
}

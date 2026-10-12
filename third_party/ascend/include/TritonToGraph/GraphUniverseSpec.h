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

#ifndef TRITON_TO_GRAPH_GRAPH_UNIVERSE_SPEC_H
#define TRITON_TO_GRAPH_GRAPH_UNIVERSE_SPEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton::cfg {

// A semantic domain is a separately auditable part of operation meaning. A
// graph may be merged only when every domain enabled by the universe has been
// proved equivalent.
enum class GraphSemanticDomain : uint8_t {
  Signature,
  Value,
  Definedness,
  Shape,
  Index,
  Layout,
  Effect,
  Memory,
  Launch,
  Buffer,
};

llvm::StringRef stringifyGraphSemanticDomain(GraphSemanticDomain domain);
std::optional<GraphSemanticDomain>
parseGraphSemanticDomain(llvm::StringRef spelling);

struct GraphUniverseBounds {
  unsigned maxNodes = 0;
  unsigned maxRank = 0;
  unsigned maxOperands = 0;
  // Number of distinct ags.input terminal symbols emitted per finite type.
  // This is independent from maxOperands: an operation may require four
  // operands without requiring four interchangeable symbols of every type.
  // Zero preserves the schema-v2 legacy default of maxOperands.
  unsigned maxInputSymbols = 0;
  // Maximum distinct non-pure operators in a reachable candidate DAG. Zero
  // preserves the legacy maxNodes bound. This is an effect-resource boundary,
  // independent from operation names and candidate topology.
  unsigned maxEffectfulNodes = 0;
  unsigned maxResults = 0;
  unsigned maxShapeSymbols = 0;
  // Optional exact-final-layer sharding. Every partition enumerates the full
  // prefix [0, maxNodes), then owns a deterministic subset of candidate short
  // operand prefixes at exactly maxNodes. Schemas with at least three operands
  // first map operand zero to a small shard group, then use operand one to pick
  // an owner inside that group. This avoids both a low-cardinality first
  // operand monopolizing one shard and every shard repeating the two-operand
  // prefix. The split happens before the remaining operand Cartesian product
  // is walked; the union of indices
  // [0, count) is therefore exactly the same bounded universe as an
  // unpartitioned run without repeating its full recursive search per shard.
  unsigned finalLayerPartitionCount = 1;
  unsigned finalLayerPartitionIndex = 0;
  bool allowDynamicShapes = false;
  bool allowControlFlow = false;
  bool allowEffects = false;

  unsigned getMaxInputSymbols() const {
    return maxInputSymbols == 0 ? maxOperands : maxInputSymbols;
  }

  unsigned getMaxEffectfulNodes() const {
    return maxEffectfulNodes == 0 ? maxNodes : maxEffectfulNodes;
  }
};

struct GraphAttributeValue {
  enum class Kind : uint8_t { Integer, Boolean, IntegerArray };

  Kind kind = Kind::Integer;
  int64_t integer = 0;
  bool boolean = false;
  std::vector<int32_t> integerArray;

  std::string toCanonicalJSON() const;
};

struct GraphAttributeDomain {
  std::string name;
  std::vector<GraphAttributeValue> values;
};

// Optional finite overload selector for one operation.  The counts describe
// explicit IR operands; implicit Memory/Launch/BufferState operands remain
// part of max_operands and are not repeated here.
struct GraphOperationOperandCountDomain {
  std::string operation;
  std::vector<unsigned> operandCounts;
};

// The finite search contract for an offline AGS database build. Completeness
// claims are meaningful only relative to the canonical form and stable hash of
// this specification.
struct GraphUniverseSpec {
  static constexpr unsigned kCurrentSchemaVersion = 2;
  static constexpr int64_t kDynamicDimension = -1;

  unsigned schemaVersion = kCurrentSchemaVersion;
  std::string name;
  std::string target;
  GraphUniverseBounds bounds;
  std::vector<std::string> catalogDialects;
  std::vector<std::string> operations;
  // Empty means every declarative candidate overload of every selected
  // operation.  A named entry lets independent finite shards cover different
  // overload families without changing TTOpSemantics or adding a matcher.
  std::vector<GraphOperationOperandCountDomain> operationOperandCounts;
  std::vector<unsigned> integerBits;
  std::vector<unsigned> floatingBits;
  // Index is target-sized rather than an arbitrary integer width, so it is a
  // separate finite-domain switch instead of a magic integer_bits value.
  bool includeIndexType = false;
  // Explicit finite address-space domain. For every scalar integer/float
  // element type, the offline type builder adds a pointer type in each listed
  // address space, plus tensor containers for every tensorShapes entry. An
  // empty list deliberately excludes pointer-valued graphs.
  std::vector<unsigned> pointerAddressSpaces;
  std::vector<int64_t> integerConstants;
  // Explicit finite tensor shape templates. Positive dimensions are static;
  // -1 is a symbolic/dynamic dimension when allowDynamicShapes is enabled.
  // An empty list means the universe is scalar-only. Shape enumeration is
  // never inferred from maxRank because that would leave the completeness
  // boundary ambiguous.
  std::vector<std::vector<int64_t>> tensorShapes;
  // Optional finite MLIR attribute spellings used as ranked-tensor encodings.
  // Empty means only the unencoded tensor type. When non-empty, each shape
  // and scalar element type is crossed with exactly these encodings. This is
  // an explicit completeness boundary, never an inferred target layout list.
  std::vector<std::string> tensorEncodings;
  // Extra finite MLIR type spellings that cannot be derived from the scalar
  // and ranked-tensor Cartesian product above.  This is the generic escape
  // hatch for stage types such as strided memrefs and for tuple/result types;
  // every spelling is part of the universe hash and must parse exactly.
  std::vector<std::string> explicitTypes;
  // Optional exact boundary signature for ags.input terminals. Entry N is
  // the type of symbol N. An empty list preserves the legacy Cartesian
  // product of every finite type with [0, max_input_symbols). Separating the
  // boundary signature from intermediate result types avoids enumerating
  // impossible typed inputs while keeping candidate topology unconstrained.
  std::vector<std::string> inputSymbolTypes;
  // Named finite domains consumed by `attribute(name, domain)` declarations
  // in TTOpSemantics. Values are data; the enumerator contains no OP-specific
  // attribute-generation branches.
  std::vector<GraphAttributeDomain> attributeDomains;
  std::vector<GraphSemanticDomain> semanticDomains;

  static std::optional<GraphUniverseSpec> parse(llvm::StringRef source,
                                                 std::string &error);
  static std::optional<GraphUniverseSpec> loadFromFile(llvm::StringRef path,
                                                        std::string &error);

  bool verify(std::string &error) const;

  // Expand finite dialect selectors such as `tt.*` against the operation
  // names actually defined by the loaded semantic registry. This is the
  // reachability set, not the authoritative registered-OP denominator; the
  // catalog coverage audit checks that denominator independently. Callers
  // must resolve before hashing or enumeration.
  bool resolveOperationPatterns(llvm::ArrayRef<llvm::StringRef> semanticOps,
                                std::string &error);

  // Arrays are sorted and duplicate-free in this representation. It is used
  // both for reviewable manifests and as the stable-hash input.
  std::string toCanonicalJSON() const;
  std::string getStableHash() const;

  bool hasDomain(GraphSemanticDomain domain) const;
  bool allowsOperationOperandCount(llvm::StringRef operation,
                                   unsigned operandCount) const;
  const GraphAttributeDomain *findAttributeDomain(llvm::StringRef name) const;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_GRAPH_UNIVERSE_SPEC_H

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

#ifndef TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_H
#define TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_H

#include "TritonToGraph/GraphUniverseSpec.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace mlir {

class IRRewriter;
class Operation;

namespace triton {
namespace cfg {

using GraphEClassId = unsigned;

// One executable Language e-node. Operation names and attributes are interned
// in the MLIRContext, so new operation schemas do not require extending a
// central opcode enum. Mathematical meaning lives in TensorSemanticExpr.
struct GraphExprNode {
  static GraphExprNode
  create(MLIRContext *context, StringRef operation,
         ArrayRef<GraphEClassId> operands = {},
         ArrayRef<NamedAttribute> attributes = {}, Value input = {});

  StringRef getOperationName() const { return operation.getValue(); }
  std::optional<int64_t> getIntegerAttribute(StringRef name) const;

  StringAttr operation;
  DictionaryAttr attributes;
  SmallVector<GraphEClassId, 2> operands;
  // Only source/input nodes carry an MLIR value. Semantic alternatives remain
  // independent of the source operation that happened to expose them.
  Value input;
};

// Compactly stores equivalent GraphExprNode trees using e-classes and
// hash-cons-style congruence rebuilding. Search budgets are explicit compiler
// safeguards; they do not restrict which rule families can be registered.
class EquivalenceGraph {
public:
  explicit EquivalenceGraph(unsigned maxENodes = 4096);
  ~EquivalenceGraph();

  EquivalenceGraph(EquivalenceGraph &&) noexcept;
  EquivalenceGraph &operator=(EquivalenceGraph &&) noexcept;
  EquivalenceGraph(const EquivalenceGraph &) = delete;
  EquivalenceGraph &operator=(const EquivalenceGraph &) = delete;

  std::optional<GraphEClassId> addNode(GraphExprNode node);
  GraphEClassId merge(GraphEClassId lhs, GraphEClassId rhs);
  void rebuild();

  GraphEClassId find(GraphEClassId id);
  GraphEClassId find(GraphEClassId id) const;
  SmallVector<GraphEClassId> getRootEClassIds() const;
  SmallVector<GraphExprNode, 4> getNodes(GraphEClassId id) const;

  unsigned getEClassCount() const;
  unsigned getENodeCount() const;
  unsigned getGeneration() const;
  unsigned getMaxENodes() const;
  bool reachedNodeLimit() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

// A typed, binder-aware proof language shared by all TTOP semantics. This is
// not an executable IR and is never materialized into the user's module.
enum class TensorSemanticKind : uint8_t {
  Symbol,
  Element,
  Defined,
  BoundIndex,
  Integer,
  Boolean,
  Not,
  And,
  Or,
  Xor,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  Equal,
  Add,
  Subtract,
  Multiply,
  SignedMultiplyHigh,
  UnsignedMultiplyHigh,
  IntegerDivide,
  Select,
  Sum,
  Product,
  ForAll,
  ZeroExtend,
  SignExtend,
  Truncate,
  ShiftLeft,
  ShiftRightLogical,
  ShiftRightArithmetic,
  SignedRemainder,
  SignedLessThan,
  UnsignedLessThan,
  PointerAdd,
  MemoryRead,
  MemoryValid,
  MemoryWrite,
  MemoryAtomic,
  // Ordered externally visible action (for example device print/assert or an
  // impure external call).  It advances the common observable state without
  // pretending that the action is a byte-addressable store.
  MemoryObservable,
  // Universally quantified MemoryState variable with an explicit launch
  // partition premise.  It appears only in offline theorem schemas; online
  // template instantiation must discharge the axis/tail-safety premise before
  // applying the associated compiler action.
  PartitionableMemory,
  BufferRead,
  BufferValid,
  BufferWrite,
  // A quantified functional update over every coordinate in one buffer
  // region.  Unlike BufferWrite, this operation binds the formal indices in
  // its region template and therefore remains well-defined for dynamic
  // shapes without eagerly expanding one write per lane.
  BufferFill,
  BufferView,
  BufferAllocate,
  ProgramId,
  // Number of programs visible on one launch axis. Keeping this distinct
  // from ProgramId prevents a launch remap from silently changing a kernel
  // that observes the grid extent.
  NumPrograms,
  // LaunchState refined by an explicit, externally checkable grid contract:
  // the original grid extent on `axis` equals the scalar bound operand.
  LaunchBound,
  // A target launch mapping described by declarative attributes. It is an
  // abstract state transition, never a materialized TTIR operation.
  LaunchRemap,
  // Whole-kernel observation of one per-program MemoryState transition under
  // a launch mapping. This is the proof boundary at which grid-partition
  // theorems may compare an original launch with a grouped launch.
  KernelEffect,
  // Parameterized higher-order launch partition. Unlike an explicitly
  // enumerated group of lane bodies, this node keeps one per-program body and
  // quantifies over `factor` logical lanes in its payload. The normalizer may
  // eliminate it only after the generic launch-safety theorem succeeds.
  KernelPartition,
  Unobserved,
  // Exact OCP microscaling decode of one logical dot operand element.  The
  // payload fixes element format, scale format, packing axis and fast-math
  // mode; operands carry the selected packed value, optional block scale and
  // optional fp4 nibble index.  DotScaled is expressed as ordinary
  // Sum/Multiply/Add over this primitive instead of as one opaque operation.
  MicroscaleDecode,
  // Applies one concrete, side-effect-free unary TTIR operation to an
  // element.  The payload identifies the operation and its semantic
  // attributes; keeping this in the proof language lets generic enumeration
  // commute any registered pointwise unary with index-only operations.
  ApplyUnary,
  // Binary counterpart of ApplyUnary. The payload keeps the concrete TTIR
  // operation identity, while the operands carry its per-element denotation.
  ApplyBinary,
  // Ternary counterpart used by fused/three-operand pointwise operations.
  // Arity is part of the proof node, so it cannot alias unary/binary forms.
  ApplyTernary,
  // Variadic opaque function with operation/attribute identity in the
  // payload. Arity comes from the operand list, not candidate-search code.
  ApplyNary,
};

class TensorSemanticExpr {
public:
  struct Node;

  TensorSemanticExpr() = default;

  static TensorSemanticExpr symbol(MLIRContext *context, StringRef name,
                                   Type type);
  static TensorSemanticExpr boundIndex(MLIRContext *context, unsigned depth,
                                       Type type);
  static TensorSemanticExpr integer(MLIRContext *context, int64_t value,
                                    Type type);
  static TensorSemanticExpr boolean(MLIRContext *context, bool value);
  static TensorSemanticExpr
  operation(TensorSemanticKind kind, Type type,
            ArrayRef<TensorSemanticExpr> operands = {},
            Attribute payload = {});

  explicit operator bool() const { return static_cast<bool>(node); }
  TensorSemanticKind getKind() const;
  Type getType() const;
  Attribute getPayload() const;
  ArrayRef<TensorSemanticExpr> getOperands() const;
  // Process-local structural hash. Structurally equal expressions always
  // receive the same value; callers must still compare the expressions before
  // treating a hash match as a proof of equality. The value is deliberately
  // computed on demand: adding even one cached word to every node is material
  // for production prefixes containing millions of large denotations.
  uint64_t getStructuralHash() const;
  bool structurallyEquals(const TensorSemanticExpr &other) const;
  // Process-local node identity for caches that already retain the owning
  // expression. It is never a semantic key and must not be serialized.
  const void *getAsOpaquePointer() const { return node.get(); }
  void print(llvm::raw_ostream &stream) const;

private:
  explicit TensorSemanticExpr(std::shared_ptr<const Node> node)
      : node(std::move(node)) {}
  std::shared_ptr<const Node> node;
};

struct TensorSemanticExpr::Node {
  TensorSemanticKind kind = TensorSemanticKind::Symbol;
  Type type;
  Attribute payload;
  SmallVector<TensorSemanticExpr, 3> operands;
};

// Exact logical meaning of one tensor result. The value is the element
// expression under resultIndices; defined tracks poison/undefinedness.
// Symbolic dimensions use ShapedType::kDynamic and are constrained separately
// by GraphFactAnalysis.
struct TensorDenotation {
  SmallVector<int64_t, 4> shape;
  // Static dimensions use an empty entry. Every dynamic dimension carries a
  // stable symbolic identity, or "?" when no equality relation was proved.
  // This prevents unrelated dynamic dimensions from comparing equal merely
  // because both are encoded as ShapedType::kDynamic.
  SmallVector<std::string, 4> shapeSymbols;
  // Stable textual identity of the physical tensor encoding. Logical index
  // equalities are insufficient when two values require different layouts;
  // an explicit layout conversion must be present in the graph in that case.
  std::string layoutIdentity;
  SmallVector<TensorSemanticExpr, 4> resultIndices;
  TensorSemanticExpr value;
  TensorSemanticExpr defined;
  // Effect is an unordered, idempotent read set for the current pure/read
  // fragment. Memory is the resulting abstract memory state. Write/atomic
  // semantics will extend the state transition without changing the generic
  // proof interface.
  TensorSemanticExpr effect;
  TensorSemanticExpr memory;
  TensorSemanticExpr memoryDefined;
  // True only for the synthetic offline MemoryState token. It lets the
  // generic denoter distinguish a real i1 value from the implicit state
  // operand appended to read/write/atomic candidate graphs.
  bool isMemoryStateToken = false;
  // LaunchState captures the program-id/grid interpretation. It is separate
  // from tensor layout: changing a tile representation and changing which
  // program owns a tile are different proof obligations.
  TensorSemanticExpr launch;
  TensorSemanticExpr launchDefined;
  bool isLaunchStateToken = false;
  // BufferState is the stage-local allocation/view/content contract used by
  // lowering-level graphs. It is intentionally distinct from global memory.
  TensorSemanticExpr buffer;
  TensorSemanticExpr bufferDefined;
  bool isBufferStateToken = false;

  bool isValid() const { return value && defined; }
};

// Stable textual key of a normalized denotation across every executable
// semantic domain. It is a bucket key only; callers must still obtain an
// equivalence proof before merging or pruning graphs.
std::string getStableTensorDenotationKey(const TensorDenotation &denotation);

// One TTOP definition maps an operation and already-denoted operands into the
// common TensorIndexSemantics domain. It describes what an op means, never a
// source graph to match or a replacement graph to build.
enum class TTOpSearchPolicy : uint8_t {
  Searchable,
  ImportOnly,
  RegionOnly,
  Mixed,
  Incomplete,
};

class TTOpSemantics {
public:
  virtual ~TTOpSemantics() = default;
  virtual StringRef getOperationName() const = 0;
  // Domains returned here are executable and checked by this definition. A
  // parsed-but-unimplemented field must never be advertised as coverage.
  virtual ArrayRef<GraphSemanticDomain> getImplementedDomains() const = 0;
  virtual TTOpSearchPolicy getSearchPolicy() const = 0;
  virtual bool supports(Operation *operation) const = 0;
  virtual bool allowsSharedImport(Operation *operation) const = 0;
  virtual FailureOr<TensorDenotation>
  denote(const GraphExprNode &operation,
         ArrayRef<TensorDenotation> operands) const = 0;
};

class TTOpSemanticsRegistry {
public:
  LogicalResult add(std::unique_ptr<TTOpSemantics> semantics);
  const TTOpSemantics *lookup(StringRef operationName) const;
  bool supports(Operation *operation) const;
  bool allowsSharedImport(Operation *operation) const;
  size_t size() const { return definitions.size(); }
  SmallVector<StringRef, 16> getOperationNames() const;
  ArrayRef<GraphSemanticDomain>
  getImplementedDomains(StringRef operationName) const;
  TTOpSearchPolicy getSearchPolicy(StringRef operationName) const;

private:
  SmallVector<std::unique_ptr<TTOpSemantics>, 16> definitions;
  llvm::StringMap<unsigned> indices;
};

struct SemanticProofStep {
  std::string reason;
  TensorSemanticExpr before;
  TensorSemanticExpr after;
};

struct SemanticNormalizationResult {
  TensorDenotation denotation;
  SmallVector<SemanticProofStep, 8> proof;
};

// Canonicalizes the common proof language. Implementations are algorithms
// over binders, booleans, bit-vectors and guarded sums; they are not a list of
// application-specific graph rewrites.
class TensorSemanticNormalizer {
public:
  explicit TensorSemanticNormalizer(
      std::function<bool()> shouldStop = {})
      : shouldStop(std::move(shouldStop)) {}

  SemanticNormalizationResult normalize(TensorDenotation input) const;
  bool wasInterrupted() const { return interrupted; }

private:
  bool shouldInterrupt() const;

  TensorSemanticExpr
  normalizeExpr(const TensorSemanticExpr &expression,
                SmallVectorImpl<SemanticProofStep> &proof,
                ArrayRef<int64_t> resultShape,
                ArrayRef<TensorSemanticExpr> resultIndices,
                llvm::DenseMap<const void *, TensorSemanticExpr> &memo,
                llvm::DenseMap<const void *, std::string> &orderingKeys) const;
  TensorSemanticExpr
  normalizeExprUncached(
      const TensorSemanticExpr &expression,
      SmallVectorImpl<SemanticProofStep> &proof,
      ArrayRef<int64_t> resultShape,
      ArrayRef<TensorSemanticExpr> resultIndices,
      llvm::DenseMap<const void *, TensorSemanticExpr> &memo,
      llvm::DenseMap<const void *, std::string> &orderingKeys) const;

  std::function<bool()> shouldStop;
  mutable uint64_t workItems = 0;
  mutable bool interrupted = false;
};

struct SemanticEquivalenceProof {
  // Identifies the independently replayable proof procedure.  The built-in
  // normalizer remains the default so version-2 certificates produced before
  // proof backends were introduced keep their stable hashes.
  std::string proofBackend = "normalizer-v1";
  TensorDenotation lhs;
  TensorDenotation rhs;
  // Domains checked by the oracle for this proof. Signature legality is
  // established while importing/building each operation and is therefore not
  // repeated here; equivalence itself checks the four tensor denotation
  // domains below.
  SmallVector<GraphSemanticDomain, 10> verifiedDomains;
  SmallVector<SemanticProofStep, 16> steps;

  // Stable content hash used by offline databases and replay logs. The proof
  // itself remains the source of truth; this is not a cryptographic signature.
  std::string getStableHash() const;
  std::string toJSON() const;
  static std::optional<SemanticEquivalenceProof>
  fromJSON(StringRef serialized, MLIRContext &context, std::string &error);
};

enum class SemanticProofStatus : uint8_t { Proved, Refuted, Unknown };

struct SemanticProofResult {
  SemanticProofStatus status = SemanticProofStatus::Unknown;
  std::optional<SemanticEquivalenceProof> proof;
  std::string reason;

  bool isProved() const { return status == SemanticProofStatus::Proved; }
};

// Optional exact proof procedure used only after the generic normalizer has
// returned Unknown.  Implementations live outside TritonGraphSemantics so the
// online compiler does not acquire an SMT dependency.  A backend receives
// normalized denotations and must return a replayable certificate for every
// Proved result.
class SemanticEquivalenceBackend {
public:
  virtual ~SemanticEquivalenceBackend() = default;
  virtual StringRef getProofBackendName() const = 0;
  virtual SemanticProofResult
  compareNormalized(const TensorDenotation &lhs,
                    const TensorDenotation &rhs) const = 0;
  virtual SemanticProofResult
  replay(const SemanticEquivalenceProof &proof) const = 0;
};

// One auditable e-class union. Keeping the normalized proof separate from the
// executable graph lets diagnostics explain why two real TTIR expressions are
// equivalent without introducing proof-only operations into TTIR.
struct SemanticMergeRecord {
  std::string candidateOperation;
  GraphEClassId equivalentClass = 0;
  SemanticEquivalenceProof proof;
};

// Equality is accepted only when both value and definedness normalize to the
// same typed expression. A semantic fingerprint may accelerate lookup but is
// never sufficient evidence by itself.
class SemanticEquivalenceOracle {
public:
  explicit SemanticEquivalenceOracle(
      const TensorSemanticNormalizer &normalizer,
      const SemanticEquivalenceBackend *fallbackBackend = nullptr)
      : normalizer(normalizer), fallbackBackend(fallbackBackend) {}

  std::optional<SemanticEquivalenceProof>
  prove(const TensorDenotation &lhs, const TensorDenotation &rhs) const;

  // Tri-state entry point for offline rule generation. Structural domain
  // mismatches are refuted; equal normalized denotations are proved; an
  // incomplete normal form is unknown rather than a false negative.
  SemanticProofResult compare(const TensorDenotation &lhs,
                              const TensorDenotation &rhs) const;
  // Run only the deterministic in-process normalizer.  Large offline searches
  // use this for the speculative premerge phase so an expensive exact backend
  // is reserved for candidates that survive concrete fingerprint refinement.
  SemanticProofResult comparePrimary(const TensorDenotation &lhs,
                                     const TensorDenotation &rhs) const;
  // Offline fingerprinting already stores canonical denotations. This entry
  // point preserves the same proof relation without normalizing both operands
  // again for every pair in a large collision bucket.
  SemanticProofResult compareNormalized(const TensorDenotation &lhs,
                                        const TensorDenotation &rhs) const;
  SemanticProofResult
  comparePrimaryNormalized(const TensorDenotation &lhs,
                           const TensorDenotation &rhs) const;
  bool hasFallbackBackend() const { return fallbackBackend != nullptr; }
  SemanticProofResult replay(const SemanticEquivalenceProof &proof) const;

private:
  SemanticProofResult compareImpl(const TensorDenotation &lhs,
                                  const TensorDenotation &rhs,
                                  bool inputsAlreadyNormalized,
                                  bool allowFallback) const;
  const TensorSemanticNormalizer &normalizer;
  const SemanticEquivalenceBackend *fallbackBackend;
};

// The executable search language contains real TTIR e-nodes only. The POC
// consumes explicitly registered MLIR operation schemas; the intended full
// implementation generates those registrations from ODS and keeps custom
// code restricted to legality/effect hooks.
class TTIRLanguage {
public:
  virtual ~TTIRLanguage() = default;
  virtual void
  enumerate(Type resultType, ArrayRef<GraphEClassId> availableOperands,
            const EquivalenceGraph &graph,
            SmallVectorImpl<GraphExprNode> &candidates,
            unsigned limit) const = 0;
};

struct CandidateEnumerationOptions {
  unsigned maxCandidates = 4096;
  unsigned maxRounds = 8;
};

struct CandidateEnumerationResult {
  unsigned generated = 0;
  unsigned provedEquivalent = 0;
  unsigned rejected = 0;
  bool hitBudget = false;
  SmallVector<SemanticMergeRecord, 4> merges;
};

// Type-directed enumeration computes each candidate's denotation and unions
// it with an existing e-class only after SemanticEquivalenceOracle succeeds.
class TypedCandidateEnumerator {
public:
  TypedCandidateEnumerator(const TTIRLanguage &language,
                           const TTOpSemanticsRegistry &semantics,
                           const TensorSemanticNormalizer &normalizer,
                           CandidateEnumerationOptions options = {},
                           std::optional<GraphEClassId> contextualRoot = {},
                           std::function<TensorDenotation(
                               const TensorDenotation &)> observation = {})
      : language(language), semantics(semantics), normalizer(normalizer),
        oracle(normalizer), options(options), contextualRoot(contextualRoot),
        observation(std::move(observation)) {}

  CandidateEnumerationResult
  enumerate(EquivalenceGraph &graph, ArrayRef<Type> resultTypes,
            ArrayRef<GraphEClassId> availableOperands,
            DenseMap<GraphEClassId, TensorDenotation> &denotations) const;

private:
  const TTIRLanguage &language;
  const TTOpSemanticsRegistry &semantics;
  const TensorSemanticNormalizer &normalizer;
  SemanticEquivalenceOracle oracle;
  CandidateEnumerationOptions options;
  std::optional<GraphEClassId> contextualRoot;
  std::function<TensorDenotation(const TensorDenotation &)> observation;
};

class GraphExprImporter {
public:
  virtual ~GraphExprImporter() = default;
  virtual std::optional<GraphEClassId> import(Value value) = 0;
};

// Cost is deliberately an interface: structural weights, a calibrated target
// model and later layout-aware costs can share the same extractor.
class GraphCostModel {
public:
  virtual ~GraphCostModel() = default;
  virtual StringRef getName() const = 0;
  virtual std::optional<unsigned>
  getNodeCost(const GraphExprNode &node) const = 0;
};

// P0/P1 cost provider. It proves the search/extraction path can select a
// structurally smaller executable graph, but makes no hardware-performance
// claim. StageCostModelAdapter replaces it without changing the extractor.
class StructuralGraphCostModel final : public GraphCostModel {
public:
  StringRef getName() const override { return "structural-width-cost-v2"; }
  std::optional<unsigned>
  getNodeCost(const GraphExprNode &node) const override;
};

struct GraphExtraction {
  unsigned cost = 0;
  GraphEClassId root = 0;
  DenseMap<GraphEClassId, GraphExprNode> choices;

  const GraphExprNode *lookup(GraphEClassId eclass) const;
};

class GraphExtractor {
public:
  explicit GraphExtractor(const GraphCostModel &costModel)
      : costModel(costModel) {}

  std::optional<GraphExtraction> extract(const EquivalenceGraph &graph,
                                         GraphEClassId root) const;

private:
  const GraphCostModel &costModel;
};

// Stage-specific materializers turn the selected semantic graph into MLIR.
// They are kept separate from equality search so source IR is only changed
// after extraction, revalidation and RewritePlan selection.
class GraphMaterializer {
public:
  virtual ~GraphMaterializer() = default;
  virtual FailureOr<Value>
  materialize(const EquivalenceGraph &graph,
              const GraphExtraction &extraction, GraphEClassId root,
              IRRewriter &rewriter) const = 0;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_AUTO_GRAPH_SOLVER_H

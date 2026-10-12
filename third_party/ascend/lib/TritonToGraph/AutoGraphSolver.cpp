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

#include "TritonToGraph/AutoGraphSolver.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

#define DEBUG_TYPE "auto-graph-solver"

GraphExprNode GraphExprNode::create(MLIRContext *context, StringRef operation,
                                    ArrayRef<GraphEClassId> operands,
                                    ArrayRef<NamedAttribute> attributes,
                                    Value input) {
  GraphExprNode node;
  node.operation = StringAttr::get(context, operation);
  node.attributes = DictionaryAttr::get(context, attributes);
  node.operands.append(operands.begin(), operands.end());
  node.input = input;
  return node;
}

std::optional<int64_t>
GraphExprNode::getIntegerAttribute(StringRef name) const {
  if (!attributes)
    return std::nullopt;
  auto value = dyn_cast_or_null<IntegerAttr>(attributes.get(name));
  if (!value)
    return std::nullopt;
  return value.getInt();
}

namespace {

struct EClass {
  GraphEClassId parent = 0;
  SmallVector<GraphExprNode, 4> nodes;
};

static bool sameNode(const GraphExprNode &lhs, const GraphExprNode &rhs) {
  return lhs.operation == rhs.operation && lhs.attributes == rhs.attributes &&
         lhs.operands == rhs.operands && lhs.input == rhs.input;
}

} // namespace

class EquivalenceGraph::Impl {
public:
  explicit Impl(unsigned maxENodes) : maxENodes(maxENodes) {}

  GraphEClassId find(GraphEClassId id) {
    assert(id < classes.size() && "invalid e-class ID");
    if (classes[id].parent != id)
      classes[id].parent = find(classes[id].parent);
    return classes[id].parent;
  }

  GraphEClassId find(GraphEClassId id) const {
    assert(id < classes.size() && "invalid e-class ID");
    while (classes[id].parent != id)
      id = classes[id].parent;
    return id;
  }

  void canonicalize(GraphExprNode &node) {
    for (GraphEClassId &operand : node.operands)
      operand = find(operand);
  }

  SmallVector<EClass, 16> classes;
  unsigned maxENodes;
  unsigned generation = 0;
};

EquivalenceGraph::EquivalenceGraph(unsigned maxENodes)
    : impl(std::make_unique<Impl>(maxENodes)) {}

EquivalenceGraph::~EquivalenceGraph() = default;
EquivalenceGraph::EquivalenceGraph(EquivalenceGraph &&) noexcept = default;
EquivalenceGraph &
EquivalenceGraph::operator=(EquivalenceGraph &&) noexcept = default;

std::optional<GraphEClassId>
EquivalenceGraph::addNode(GraphExprNode node) {
  if (!node.operation)
    return std::nullopt;
  impl->canonicalize(node);
  for (GraphEClassId id = 0; id < impl->classes.size(); ++id) {
    if (impl->find(id) != id)
      continue;
    for (const GraphExprNode &existing : impl->classes[id].nodes)
      if (sameNode(existing, node))
        return id;
  }

  if (getENodeCount() >= impl->maxENodes)
    return std::nullopt;
  const GraphEClassId id = impl->classes.size();
  EClass eclass;
  eclass.parent = id;
  eclass.nodes.push_back(std::move(node));
  impl->classes.push_back(std::move(eclass));
  ++impl->generation;
  return id;
}

GraphEClassId EquivalenceGraph::merge(GraphEClassId lhs, GraphEClassId rhs) {
  lhs = impl->find(lhs);
  rhs = impl->find(rhs);
  if (lhs == rhs)
    return lhs;
  if (rhs < lhs)
    std::swap(lhs, rhs);
  impl->classes[rhs].parent = lhs;
  impl->classes[lhs].nodes.append(impl->classes[rhs].nodes.begin(),
                                  impl->classes[rhs].nodes.end());
  impl->classes[rhs].nodes.clear();
  ++impl->generation;
  return lhs;
}

void EquivalenceGraph::rebuild() {
  bool congruenceChanged = false;
  do {
    congruenceChanged = false;
    for (GraphEClassId id = 0; id < impl->classes.size(); ++id) {
      if (impl->find(id) != id)
        continue;
      SmallVector<GraphExprNode, 4> unique;
      for (GraphExprNode node : impl->classes[id].nodes) {
        impl->canonicalize(node);
        if (llvm::none_of(unique, [&](const GraphExprNode &existing) {
              return sameNode(existing, node);
            }))
          unique.push_back(std::move(node));
      }
      impl->classes[id].nodes = std::move(unique);
    }

    for (GraphEClassId lhs = 0;
         lhs < impl->classes.size() && !congruenceChanged; ++lhs) {
      if (impl->find(lhs) != lhs)
        continue;
      for (GraphEClassId rhs = lhs + 1; rhs < impl->classes.size(); ++rhs) {
        if (impl->find(rhs) != rhs)
          continue;
        const bool congruent = llvm::any_of(
            impl->classes[lhs].nodes, [&](const GraphExprNode &lhsNode) {
              return llvm::any_of(
                  impl->classes[rhs].nodes,
                  [&](const GraphExprNode &rhsNode) {
                    return sameNode(lhsNode, rhsNode);
                  });
            });
        if (!congruent)
          continue;
        merge(lhs, rhs);
        congruenceChanged = true;
        break;
      }
    }
  } while (congruenceChanged);
}

GraphEClassId EquivalenceGraph::find(GraphEClassId id) {
  return impl->find(id);
}

GraphEClassId EquivalenceGraph::find(GraphEClassId id) const {
  return impl->find(id);
}

SmallVector<GraphEClassId> EquivalenceGraph::getRootEClassIds() const {
  SmallVector<GraphEClassId> roots;
  for (GraphEClassId id = 0; id < impl->classes.size(); ++id)
    if (impl->find(id) == id)
      roots.push_back(id);
  return roots;
}

SmallVector<GraphExprNode, 4>
EquivalenceGraph::getNodes(GraphEClassId id) const {
  return impl->classes[impl->find(id)].nodes;
}

unsigned EquivalenceGraph::getEClassCount() const {
  unsigned count = 0;
  for (GraphEClassId id = 0; id < impl->classes.size(); ++id)
    if (impl->find(id) == id)
      ++count;
  return count;
}

unsigned EquivalenceGraph::getENodeCount() const {
  unsigned count = 0;
  for (GraphEClassId id = 0; id < impl->classes.size(); ++id)
    if (impl->find(id) == id)
      count += impl->classes[id].nodes.size();
  return count;
}

unsigned EquivalenceGraph::getGeneration() const { return impl->generation; }
unsigned EquivalenceGraph::getMaxENodes() const { return impl->maxENodes; }
bool EquivalenceGraph::reachedNodeLimit() const {
  return getENodeCount() >= getMaxENodes();
}

TensorSemanticExpr
TensorSemanticExpr::symbol(MLIRContext *context, StringRef name, Type type) {
  auto node = std::make_shared<Node>();
  node->kind = TensorSemanticKind::Symbol;
  node->type = type;
  node->payload = StringAttr::get(context, name);
  return TensorSemanticExpr(std::move(node));
}

TensorSemanticExpr
TensorSemanticExpr::boundIndex(MLIRContext *context, unsigned depth,
                               Type type) {
  auto node = std::make_shared<Node>();
  node->kind = TensorSemanticKind::BoundIndex;
  node->type = type;
  node->payload = IntegerAttr::get(IntegerType::get(context, 64), depth);
  return TensorSemanticExpr(std::move(node));
}

TensorSemanticExpr TensorSemanticExpr::integer(MLIRContext *context,
                                               int64_t value, Type type) {
  auto node = std::make_shared<Node>();
  node->kind = TensorSemanticKind::Integer;
  node->type = type;
  node->payload = Builder(context).getIntegerAttr(type, value);
  return TensorSemanticExpr(std::move(node));
}

TensorSemanticExpr TensorSemanticExpr::boolean(MLIRContext *context,
                                               bool value) {
  auto node = std::make_shared<Node>();
  node->kind = TensorSemanticKind::Boolean;
  node->type = IntegerType::get(context, 1);
  node->payload = Builder(context).getBoolAttr(value);
  return TensorSemanticExpr(std::move(node));
}

TensorSemanticExpr
TensorSemanticExpr::operation(TensorSemanticKind kind, Type type,
                              ArrayRef<TensorSemanticExpr> operands,
                              Attribute payload) {
  auto node = std::make_shared<Node>();
  node->kind = kind;
  node->type = type;
  node->payload = payload;
  node->operands.append(operands.begin(), operands.end());
  return TensorSemanticExpr(std::move(node));
}

TensorSemanticKind TensorSemanticExpr::getKind() const {
  assert(node && "invalid tensor semantic expression");
  return node->kind;
}

Type TensorSemanticExpr::getType() const {
  return node ? node->type : Type{};
}

Attribute TensorSemanticExpr::getPayload() const {
  return node ? node->payload : Attribute{};
}

ArrayRef<TensorSemanticExpr> TensorSemanticExpr::getOperands() const {
  assert(node && "invalid tensor semantic expression");
  return node->operands;
}

uint64_t TensorSemanticExpr::getStructuralHash() const {
  if (!node)
    return 0;
  if (node->operands.empty()) {
    llvm::hash_code hash = llvm::hash_combine(
        static_cast<uint8_t>(node->kind), node->type.getAsOpaquePointer(),
        node->payload ? node->payload.getAsOpaquePointer() : nullptr, 0u);
    return static_cast<uint64_t>(static_cast<size_t>(hash));
  }

  llvm::SmallDenseMap<const void *, uint64_t, 16> memo;
  std::function<uint64_t(const TensorSemanticExpr &)> compute =
      [&](const TensorSemanticExpr &expression) -> uint64_t {
    if (!expression)
      return 0;
    const void *identity = expression.getAsOpaquePointer();
    auto known = memo.find(identity);
    if (known != memo.end())
      return known->second;

    llvm::hash_code hash = llvm::hash_combine(
        static_cast<uint8_t>(expression.getKind()),
        expression.getType().getAsOpaquePointer(),
        expression.getPayload()
            ? expression.getPayload().getAsOpaquePointer()
            : nullptr,
        expression.getOperands().size());
    for (const TensorSemanticExpr &operand : expression.getOperands())
      hash = llvm::hash_combine(hash, compute(operand));
    uint64_t result = static_cast<uint64_t>(static_cast<size_t>(hash));
    memo.try_emplace(identity, result);
    return result;
  };
  return compute(*this);
}

bool TensorSemanticExpr::structurallyEquals(
    const TensorSemanticExpr &other) const {
  if (node == other.node)
    return true;
  if (!node || !other.node)
    return false;
  if (node->kind != other.node->kind || node->type != other.node->type ||
      node->payload != other.node->payload ||
      node->operands.size() != other.node->operands.size())
    return false;
  if (node->operands.empty())
    return true;

  using NodePair = std::pair<const void *, const void *>;
  llvm::SmallDenseMap<NodePair, bool, 16> memo;
  std::function<bool(const TensorSemanticExpr &, const TensorSemanticExpr &)>
      compare = [&](const TensorSemanticExpr &lhs,
                    const TensorSemanticExpr &rhs) -> bool {
    if (lhs.getAsOpaquePointer() == rhs.getAsOpaquePointer())
      return true;
    if (!lhs || !rhs)
      return false;

    NodePair key{lhs.getAsOpaquePointer(), rhs.getAsOpaquePointer()};
    auto known = memo.find(key);
    if (known != memo.end())
      return known->second;
    if (lhs.getKind() != rhs.getKind() || lhs.getType() != rhs.getType() ||
        lhs.getPayload() != rhs.getPayload() ||
        lhs.getOperands().size() != rhs.getOperands().size()) {
      memo.try_emplace(key, false);
      return false;
    }
    for (auto [lhsOperand, rhsOperand] :
         llvm::zip_equal(lhs.getOperands(), rhs.getOperands())) {
      if (!compare(lhsOperand, rhsOperand)) {
        memo.try_emplace(key, false);
        return false;
      }
    }
    memo.try_emplace(key, true);
    return true;
  };
  return compare(*this, other);
}

namespace {

static StringRef stringifySemanticKind(TensorSemanticKind kind) {
  switch (kind) {
  case TensorSemanticKind::Symbol:
    return "symbol";
  case TensorSemanticKind::Element:
    return "element";
  case TensorSemanticKind::Defined:
    return "defined";
  case TensorSemanticKind::BoundIndex:
    return "index";
  case TensorSemanticKind::Integer:
    return "integer";
  case TensorSemanticKind::Boolean:
    return "boolean";
  case TensorSemanticKind::Not:
    return "not";
  case TensorSemanticKind::And:
    return "and";
  case TensorSemanticKind::Or:
    return "or";
  case TensorSemanticKind::Xor:
    return "xor";
  case TensorSemanticKind::BitwiseAnd:
    return "bit_and";
  case TensorSemanticKind::BitwiseOr:
    return "bit_or";
  case TensorSemanticKind::BitwiseXor:
    return "bit_xor";
  case TensorSemanticKind::Equal:
    return "equal";
  case TensorSemanticKind::Add:
    return "add";
  case TensorSemanticKind::Subtract:
    return "subtract";
  case TensorSemanticKind::Multiply:
    return "multiply";
  case TensorSemanticKind::SignedMultiplyHigh:
    return "signed_multiply_high";
  case TensorSemanticKind::UnsignedMultiplyHigh:
    return "unsigned_multiply_high";
  case TensorSemanticKind::IntegerDivide:
    return "integer_divide";
  case TensorSemanticKind::Select:
    return "select";
  case TensorSemanticKind::Sum:
    return "sum";
  case TensorSemanticKind::Product:
    return "product";
  case TensorSemanticKind::ForAll:
    return "forall";
  case TensorSemanticKind::ZeroExtend:
    return "zero_extend";
  case TensorSemanticKind::SignExtend:
    return "sign_extend";
  case TensorSemanticKind::Truncate:
    return "truncate";
  case TensorSemanticKind::ShiftLeft:
    return "shift_left";
  case TensorSemanticKind::ShiftRightLogical:
    return "shift_right_logical";
  case TensorSemanticKind::ShiftRightArithmetic:
    return "shift_right_arithmetic";
  case TensorSemanticKind::SignedRemainder:
    return "signed_remainder";
  case TensorSemanticKind::SignedLessThan:
    return "signed_less_than";
  case TensorSemanticKind::UnsignedLessThan:
    return "unsigned_less_than";
  case TensorSemanticKind::PointerAdd:
    return "pointer_add";
  case TensorSemanticKind::MemoryRead:
    return "memory_read";
  case TensorSemanticKind::MemoryValid:
    return "memory_valid";
  case TensorSemanticKind::MemoryWrite:
    return "memory_write";
  case TensorSemanticKind::MemoryAtomic:
    return "memory_atomic";
  case TensorSemanticKind::MemoryObservable:
    return "memory_observable";
  case TensorSemanticKind::PartitionableMemory:
    return "partitionable_memory";
  case TensorSemanticKind::BufferRead:
    return "buffer_read";
  case TensorSemanticKind::BufferValid:
    return "buffer_valid";
  case TensorSemanticKind::BufferWrite:
    return "buffer_write";
  case TensorSemanticKind::BufferFill:
    return "buffer_fill";
  case TensorSemanticKind::BufferView:
    return "buffer_view";
  case TensorSemanticKind::BufferAllocate:
    return "buffer_allocate";
  case TensorSemanticKind::ProgramId:
    return "program_id";
  case TensorSemanticKind::NumPrograms:
    return "num_programs";
  case TensorSemanticKind::LaunchBound:
    return "launch_bound";
  case TensorSemanticKind::LaunchRemap:
    return "launch_remap";
  case TensorSemanticKind::KernelEffect:
    return "kernel_effect";
  case TensorSemanticKind::KernelPartition:
    return "kernel_partition";
  case TensorSemanticKind::Unobserved:
    return "unobserved";
  case TensorSemanticKind::MicroscaleDecode:
    return "microscale_decode";
  case TensorSemanticKind::ApplyUnary:
    return "apply_unary";
  case TensorSemanticKind::ApplyBinary:
    return "apply_binary";
  case TensorSemanticKind::ApplyTernary:
    return "apply_ternary";
  case TensorSemanticKind::ApplyNary:
    return "apply_nary";
  }
  llvm_unreachable("unknown tensor semantic kind");
}

static std::string toHex(ArrayRef<uint8_t> digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

static void updateFramed(llvm::SHA256 &hash, StringRef value) {
  hash.update(llvm::Twine(value.size()).concat(":").str());
  hash.update(value);
}

static std::string semanticKey(
    const TensorSemanticExpr &expression,
    llvm::DenseMap<const void *, std::string> &memo) {
  if (!expression) {
    llvm::SHA256 hash;
    hash.update("ags-semantic-expression-v2:<invalid>");
    return toHex(hash.final());
  }
  auto known = memo.find(expression.getAsOpaquePointer());
  if (known != memo.end())
    return known->second;

  std::string header;
  llvm::raw_string_ostream stream(header);
  stream << stringifySemanticKind(expression.getKind()) << '\n'
         << expression.getType() << '\n';
  if (Attribute payload = expression.getPayload())
    stream << payload;
  llvm::SHA256 hash;
  hash.update("ags-semantic-expression-v2\n");
  updateFramed(hash, header);
  hash.update(llvm::Twine(expression.getOperands().size()).concat("\n").str());
  for (const TensorSemanticExpr &operand : expression.getOperands())
    updateFramed(hash, semanticKey(operand, memo));
  std::string digest = toHex(hash.final());
  memo.try_emplace(expression.getAsOpaquePointer(), digest);
  return digest;
}

static std::optional<APInt>
getIntegerSemanticValue(const TensorSemanticExpr &expression) {
  if (!expression ||
      (expression.getKind() != TensorSemanticKind::Integer &&
       expression.getKind() != TensorSemanticKind::Boolean))
    return std::nullopt;
  auto value = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
  return value ? std::optional<APInt>(value.getValue()) : std::nullopt;
}

static std::optional<unsigned> getSemanticBitWidth(Type type) {
  if (auto integer = dyn_cast_or_null<IntegerType>(type))
    return integer.getWidth();
  if (isa_and_nonnull<IndexType>(type))
    return 64;
  return std::nullopt;
}

static bool isCommutativeSemanticKind(TensorSemanticKind kind) {
  return kind == TensorSemanticKind::And ||
         kind == TensorSemanticKind::Or ||
         kind == TensorSemanticKind::Xor ||
         kind == TensorSemanticKind::BitwiseAnd ||
         kind == TensorSemanticKind::BitwiseOr ||
         kind == TensorSemanticKind::BitwiseXor ||
         kind == TensorSemanticKind::Equal ||
         kind == TensorSemanticKind::Add ||
         kind == TensorSemanticKind::Multiply ||
         kind == TensorSemanticKind::SignedMultiplyHigh ||
         kind == TensorSemanticKind::UnsignedMultiplyHigh;
}

// These nodes denote pure, element-wise binary functions.  For every such
// function F, ordinary ITE congruence gives
//
//   F(select(c, a, b), x) == select(c, F(a, x), F(b, x)).
//
// Keeping this classification in the semantic algebra (rather than in an OP
// matcher) lets every imported/candidate TT operation benefit from the same
// theorem.  State transitions and reductions are deliberately excluded.
static bool isSelectCongruentBinarySemanticKind(TensorSemanticKind kind) {
  switch (kind) {
  case TensorSemanticKind::And:
  case TensorSemanticKind::Or:
  case TensorSemanticKind::Xor:
  case TensorSemanticKind::BitwiseAnd:
  case TensorSemanticKind::BitwiseOr:
  case TensorSemanticKind::BitwiseXor:
  case TensorSemanticKind::Equal:
  case TensorSemanticKind::Add:
  case TensorSemanticKind::Subtract:
  case TensorSemanticKind::Multiply:
  case TensorSemanticKind::SignedMultiplyHigh:
  case TensorSemanticKind::UnsignedMultiplyHigh:
  case TensorSemanticKind::IntegerDivide:
  case TensorSemanticKind::ShiftLeft:
  case TensorSemanticKind::ShiftRightLogical:
  case TensorSemanticKind::ShiftRightArithmetic:
  case TensorSemanticKind::SignedRemainder:
  case TensorSemanticKind::SignedLessThan:
  case TensorSemanticKind::UnsignedLessThan:
  case TensorSemanticKind::PointerAdd:
  case TensorSemanticKind::ApplyBinary:
    return true;
  default:
    return false;
  }
}

static bool isSemanticZero(const TensorSemanticExpr &expression) {
  std::optional<APInt> value = getIntegerSemanticValue(expression);
  return value && value->isZero();
}

static bool isSemanticOne(const TensorSemanticExpr &expression) {
  std::optional<APInt> value = getIntegerSemanticValue(expression);
  return value && value->isOne();
}

static bool isSemanticTwo(const TensorSemanticExpr &expression) {
  std::optional<APInt> value = getIntegerSemanticValue(expression);
  return value && value->getActiveBits() <= 2 && value->getZExtValue() == 2;
}

static std::optional<TensorSemanticExpr>
matchBitwiseNotOperand(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::BitwiseXor ||
      expression.getOperands().size() != 2)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
  std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
  if (lhs && lhs->isAllOnes())
    return operands[1];
  if (rhs && rhs->isAllOnes())
    return operands[0];
  return std::nullopt;
}

static bool areBooleanComplements(const TensorSemanticExpr &lhs,
                                  const TensorSemanticExpr &rhs) {
  auto isNegationOf = [](const TensorSemanticExpr &negated,
                         const TensorSemanticExpr &value) {
    return negated.getKind() == TensorSemanticKind::Not &&
           negated.getOperands().size() == 1 &&
           negated.getOperands().front().structurallyEquals(value);
  };
  return isNegationOf(lhs, rhs) || isNegationOf(rhs, lhs);
}

static std::optional<std::pair<TensorSemanticExpr, APInt>>
splitEqualityConstant(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::Equal ||
      expression.getOperands().size() != 2)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  if (std::optional<APInt> value = getIntegerSemanticValue(operands[0]))
    return std::make_pair(operands[1], *value);
  if (std::optional<APInt> value = getIntegerSemanticValue(operands[1]))
    return std::make_pair(operands[0], *value);
  return std::nullopt;
}

static bool areContradictoryEqualities(const TensorSemanticExpr &lhs,
                                       const TensorSemanticExpr &rhs) {
  auto lhsParts = splitEqualityConstant(lhs);
  auto rhsParts = splitEqualityConstant(rhs);
  return lhsParts && rhsParts &&
         lhsParts->first.structurallyEquals(rhsParts->first) &&
         lhsParts->second.getBitWidth() == rhsParts->second.getBitWidth() &&
         lhsParts->second != rhsParts->second;
}

static bool containsSemanticExpr(const TensorSemanticExpr &expression,
                                 const TensorSemanticExpr &needle) {
  if (expression.structurallyEquals(needle))
    return true;
  return llvm::any_of(expression.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return containsSemanticExpr(operand, needle);
                      });
}

static bool containsSemanticKind(const TensorSemanticExpr &expression,
                                 TensorSemanticKind kind) {
  if (!expression)
    return false;
  if (expression.getKind() == kind)
    return true;
  return llvm::any_of(expression.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return containsSemanticKind(operand, kind);
                      });
}

// Conservative integer-range facts used by guarded memory observations.  The
// facts are derived from the common semantic language, not from a TTIR graph
// pattern.  Do not propagate the fact through integer add/multiply: without
// an explicit no-overflow premise, fixed-width arithmetic may wrap negative.
static bool isKnownNonNegative(const TensorSemanticExpr &expression) {
  if (!expression)
    return false;
  if (std::optional<APInt> value = getIntegerSemanticValue(expression))
    return !value->isNegative();
  if (expression.getKind() == TensorSemanticKind::BoundIndex ||
      expression.getKind() == TensorSemanticKind::ProgramId ||
      expression.getKind() == TensorSemanticKind::NumPrograms)
    return true;
  return false;
}

static bool conditionContainsSignedUpperBound(
    const TensorSemanticExpr &condition, const TensorSemanticExpr &value,
    const TensorSemanticExpr &upperBound) {
  if (condition.getKind() == TensorSemanticKind::SignedLessThan &&
      condition.getOperands().size() == 2 &&
      condition.getOperands()[0].structurallyEquals(value) &&
      condition.getOperands()[1].structurallyEquals(upperBound))
    return true;
  if (condition.getKind() != TensorSemanticKind::And)
    return false;
  return llvm::any_of(condition.getOperands(), [&](const TensorSemanticExpr &op) {
    return conditionContainsSignedUpperBound(op, value, upperBound);
  });
}

// A signed upper-bound guard with a non-negative left-hand side proves that
// the upper bound is strictly positive.  Keep this as a fact over the common
// denotation language: it is useful for guarded arithmetic definedness as
// well as for any particular TT operation that produced the comparison.
static bool conditionProvesPositive(const TensorSemanticExpr &condition,
                                    const TensorSemanticExpr &value) {
  if (condition.getKind() == TensorSemanticKind::SignedLessThan &&
      condition.getOperands().size() == 2 &&
      condition.getOperands()[1].structurallyEquals(value) &&
      isKnownNonNegative(condition.getOperands()[0]))
    return true;
  if (condition.getKind() != TensorSemanticKind::And)
    return false;
  return llvm::any_of(condition.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return conditionProvesPositive(operand, value);
                      });
}

static bool conditionContainsConjunct(
    const TensorSemanticExpr &condition,
    const TensorSemanticExpr &conjunct) {
  if (condition.structurallyEquals(conjunct))
    return true;
  if (condition.getKind() != TensorSemanticKind::And)
    return false;
  return llvm::any_of(condition.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return conditionContainsConjunct(operand, conjunct);
                      });
}

static bool conditionProvesNonZero(const TensorSemanticExpr &condition,
                                   const TensorSemanticExpr &value) {
  if (value.getType().isInteger(1) &&
      conditionContainsConjunct(condition, value))
    return true;
  if (condition.getKind() == TensorSemanticKind::And)
    return llvm::any_of(condition.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return conditionProvesNonZero(operand, value);
                        });
  if (condition.getKind() != TensorSemanticKind::Not ||
      condition.getOperands().size() != 1)
    return false;
  const TensorSemanticExpr &negated = condition.getOperands().front();
  if (negated.getKind() != TensorSemanticKind::Equal ||
      negated.getOperands().size() != 2)
    return false;
  ArrayRef<TensorSemanticExpr> compared = negated.getOperands();
  return (compared[0].structurallyEquals(value) &&
          isSemanticZero(compared[1])) ||
         (compared[1].structurallyEquals(value) &&
          isSemanticZero(compared[0]));
}

static bool conditionProvesValidShiftAmount(
    const TensorSemanticExpr &condition,
    const TensorSemanticExpr &amount) {
  if (condition.getKind() == TensorSemanticKind::And)
    return llvm::any_of(condition.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return conditionProvesValidShiftAmount(operand,
                                                                 amount);
                        });
  if (condition.getKind() != TensorSemanticKind::ApplyUnary ||
      condition.getOperands().size() != 1 ||
      !condition.getOperands().front().structurallyEquals(amount))
    return false;
  auto payload = dyn_cast_or_null<DictionaryAttr>(condition.getPayload());
  auto role = payload ? dyn_cast_or_null<StringAttr>(
                            payload.get("semantic_role"))
                      : StringAttr{};
  return role && role.getValue() == "shift_amount_valid";
}

static bool conditionForcesShiftAmountZero(
    const TensorSemanticExpr &condition,
    const TensorSemanticExpr &amount) {
  if (!conditionProvesValidShiftAmount(condition, amount))
    return false;
  if (isSemanticZero(amount))
    return true;
  if (amount.getKind() != TensorSemanticKind::Multiply ||
      amount.getOperands().size() != 2)
    return false;

  ArrayRef<TensorSemanticExpr> factors = amount.getOperands();
  std::optional<APInt> lhs = getIntegerSemanticValue(factors[0]);
  std::optional<APInt> rhs = getIntegerSemanticValue(factors[1]);
  const APInt *constant = lhs ? &*lhs : (rhs ? &*rhs : nullptr);
  std::optional<unsigned> width = getSemanticBitWidth(amount.getType());
  if (!constant || !width || *width == 0)
    return false;
  if (constant->isZero())
    return true;
  // A product by 2^k has k zero low bits even after modular wraparound.  If
  // 2^k is at least the shift-width bound, the only valid amount is zero.
  return constant->isPowerOf2() &&
         constant->getLimitedValue() >= static_cast<uint64_t>(*width);
}

// Simplifies an expression only on a branch where `condition` is known true.
// This is the generic signed-remainder law 0 <= x < d => x % d == x.  It is
// what permits a masked memory observation to compare a wrapped address with
// a linear address without adding a ConvertModuloToMask graph matcher.
static TensorSemanticExpr simplifyUnderTrueCondition(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &condition) {
  if (!expression)
    return {};
  if (expression.getKind() == TensorSemanticKind::Equal &&
      expression.getOperands().size() == 2) {
    ArrayRef<TensorSemanticExpr> compared = expression.getOperands();
    std::optional<APInt> lhs = getIntegerSemanticValue(compared[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(compared[1]);
    if ((lhs && lhs->isNegative() && isKnownNonNegative(compared[1])) ||
        (rhs && rhs->isNegative() && isKnownNonNegative(compared[0])))
      return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                         false);
  }
  if (expression.getType().isInteger(1) &&
      conditionProvesValidShiftAmount(condition, expression)) {
    // A defined one-bit shift amount is strictly less than one, hence zero.
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       false);
  }
  if (expression.getType().isInteger(1) &&
      conditionContainsConjunct(condition, expression))
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       true);
  if (expression.getType().isInteger(1)) {
    TensorSemanticExpr negated = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, expression.getType(), {expression});
    if (conditionContainsConjunct(condition, negated))
      return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                         false);
  }
  if (expression.getKind() == TensorSemanticKind::Not &&
      expression.getOperands().size() == 1 &&
      conditionContainsConjunct(condition, expression.getOperands()[0]))
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       false);
  if (expression.getKind() == TensorSemanticKind::Not &&
      expression.getOperands().size() == 1) {
    const TensorSemanticExpr &negated = expression.getOperands().front();
    if (negated.getKind() == TensorSemanticKind::Equal &&
        negated.getOperands().size() == 2) {
      ArrayRef<TensorSemanticExpr> compared = negated.getOperands();
      const TensorSemanticExpr *nonZero = nullptr;
      if (isSemanticZero(compared[0]))
        nonZero = &compared[1];
      else if (isSemanticZero(compared[1]))
        nonZero = &compared[0];
      if (nonZero && conditionProvesPositive(condition, *nonZero))
        return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                           true);
    }
  }
  if (expression.getKind() == TensorSemanticKind::SignedRemainder &&
      expression.getOperands().size() == 2 &&
      isKnownNonNegative(expression.getOperands()[0]) &&
      conditionContainsSignedUpperBound(condition,
                                        expression.getOperands()[0],
                                        expression.getOperands()[1]))
    return expression.getOperands()[0];
  if (expression.getKind() == TensorSemanticKind::SignedRemainder &&
      expression.getOperands().size() == 2 &&
      conditionProvesNonZero(condition, expression.getOperands()[1]) &&
      (isSemanticZero(expression.getOperands()[0]) ||
       expression.getOperands()[0].structurallyEquals(
           expression.getOperands()[1])))
    return TensorSemanticExpr::integer(expression.getType().getContext(), 0,
                                       expression.getType());
  if (expression.getKind() == TensorSemanticKind::IntegerDivide &&
      expression.getOperands().size() == 2 &&
      isSemanticZero(expression.getOperands()[0]) &&
      conditionProvesNonZero(condition, expression.getOperands()[1]))
    return TensorSemanticExpr::integer(expression.getType().getContext(), 0,
                                       expression.getType());
  if ((expression.getKind() == TensorSemanticKind::ShiftLeft ||
       expression.getKind() == TensorSemanticKind::ShiftRightLogical ||
       expression.getKind() == TensorSemanticKind::ShiftRightArithmetic) &&
      expression.getOperands().size() == 2 &&
      conditionForcesShiftAmountZero(condition,
                                     expression.getOperands()[1])) {
    return expression.getOperands()[0];
  }
  if ((expression.getKind() == TensorSemanticKind::ShiftLeft ||
       expression.getKind() == TensorSemanticKind::ShiftRightLogical ||
       expression.getKind() == TensorSemanticKind::ShiftRightArithmetic) &&
      expression.getOperands().size() == 2 &&
      expression.getType().isInteger(1) &&
      conditionProvesValidShiftAmount(condition,
                                      expression.getOperands()[1])) {
    // A defined shift of a one-bit value can only have amount zero.
    return expression.getOperands()[0];
  }
  if ((expression.getKind() == TensorSemanticKind::ShiftRightLogical ||
       expression.getKind() == TensorSemanticKind::ShiftRightArithmetic) &&
      expression.getOperands().size() == 2 &&
      expression.getOperands()[0].structurallyEquals(
          expression.getOperands()[1]) &&
      conditionProvesValidShiftAmount(condition,
                                      expression.getOperands()[1])) {
    // For 0 <= x < bitwidth, x has fewer than x+1 significant bits. Thus both
    // logical and arithmetic x >> x are zero; x is non-negative so the two
    // right-shift interpretations coincide on this guarded domain.
    return TensorSemanticExpr::integer(expression.getType().getContext(), 0,
                                       expression.getType());
  }

  SmallVector<TensorSemanticExpr, 4> operands;
  bool changed = false;
  for (const TensorSemanticExpr &operand : expression.getOperands()) {
    TensorSemanticExpr simplified =
        simplifyUnderTrueCondition(operand, condition);
    changed |= !simplified.structurallyEquals(operand);
    operands.push_back(std::move(simplified));
  }
  if (!changed)
    return expression;
  return TensorSemanticExpr::operation(expression.getKind(),
                                       expression.getType(), operands,
                                       expression.getPayload());
}

// Dual branch-local boolean substitution.  This is deliberately expressed in
// the denotation algebra rather than in terms of any TT operation: on the
// false arm of select(c, t, f), every occurrence of c is false and every
// occurrence of not(c) is true.  Besides ordinary boolean simplification this
// closes masked-load identities such as select(m, x, m) == select(m, x, 0).
static TensorSemanticExpr simplifyUnderFalseCondition(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &condition) {
  if (!expression)
    return {};
  if (expression.structurallyEquals(condition) &&
      expression.getType().isInteger(1))
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       false);
  if (expression.getKind() == TensorSemanticKind::Not &&
      expression.getOperands().size() == 1 &&
      expression.getOperands()[0].structurallyEquals(condition))
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       true);

  SmallVector<TensorSemanticExpr, 4> operands;
  bool changed = false;
  for (const TensorSemanticExpr &operand : expression.getOperands()) {
    TensorSemanticExpr simplified =
        simplifyUnderFalseCondition(operand, condition);
    changed |= !simplified.structurallyEquals(operand);
    operands.push_back(std::move(simplified));
  }
  if (!changed)
    return expression;
  return TensorSemanticExpr::operation(expression.getKind(),
                                       expression.getType(), operands,
                                       expression.getPayload());
}

static TensorSemanticExpr
substituteSemanticExpr(const TensorSemanticExpr &expression,
                       const TensorSemanticExpr &from,
                       const TensorSemanticExpr &to) {
  if (expression.structurallyEquals(from))
    return to;
  SmallVector<TensorSemanticExpr, 4> operands;
  for (const TensorSemanticExpr &operand : expression.getOperands())
    operands.push_back(substituteSemanticExpr(operand, from, to));
  return TensorSemanticExpr::operation(expression.getKind(),
                                       expression.getType(), operands,
                                       expression.getPayload());
}

static std::optional<int64_t>
getSignedSemanticInteger(const TensorSemanticExpr &expression) {
  std::optional<APInt> value = getIntegerSemanticValue(expression);
  if (!value || value->getBitWidth() > 64)
    return std::nullopt;
  return value->getSExtValue();
}

static bool isSingletonConstantRange(const TensorSemanticExpr &lower,
                                     const TensorSemanticExpr &upper) {
  std::optional<int64_t> lowerValue = getSignedSemanticInteger(lower);
  std::optional<int64_t> upperValue = getSignedSemanticInteger(upper);
  return lowerValue && upperValue &&
         *lowerValue != std::numeric_limits<int64_t>::max() &&
         *upperValue == *lowerValue + 1;
}

// Recognizes the affine fragment `variable + constant`. It deliberately does
// not solve products or relations between multiple variables; unknown means
// that the generic proof simply declines the candidate.
static std::optional<int64_t>
matchUnitAffineOffset(const TensorSemanticExpr &expression,
                      const TensorSemanticExpr &variable) {
  if (expression.structurallyEquals(variable))
    return 0;
  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  if (expression.getKind() == TensorSemanticKind::Add &&
      operands.size() == 2) {
    if (std::optional<int64_t> constant =
            getSignedSemanticInteger(operands[0])) {
      std::optional<int64_t> nested =
          matchUnitAffineOffset(operands[1], variable);
      if (nested)
        return *nested + *constant;
    }
    if (std::optional<int64_t> constant =
            getSignedSemanticInteger(operands[1])) {
      std::optional<int64_t> nested =
          matchUnitAffineOffset(operands[0], variable);
      if (nested)
        return *nested + *constant;
    }
  }
  if (expression.getKind() == TensorSemanticKind::Subtract &&
      operands.size() == 2) {
    std::optional<int64_t> constant =
        getSignedSemanticInteger(operands[1]);
    std::optional<int64_t> nested =
        matchUnitAffineOffset(operands[0], variable);
    if (constant && nested)
      return *nested - *constant;
  }
  return std::nullopt;
}

static TensorSemanticExpr
addSemanticOffset(const TensorSemanticExpr &base, int64_t offset) {
  if (offset == 0)
    return base;
  Type type = base.getType();
  TensorSemanticExpr magnitude = TensorSemanticExpr::integer(
      type.getContext(), offset < 0 ? -offset : offset, type);
  return TensorSemanticExpr::operation(
      offset < 0 ? TensorSemanticKind::Subtract : TensorSemanticKind::Add,
      type, {base, magnitude});
}

static std::optional<std::pair<TensorSemanticExpr, int64_t>>
splitAddConstant(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::Add ||
      expression.getOperands().size() != 2)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  if (std::optional<int64_t> value =
          getSignedSemanticInteger(operands[0]))
    return std::make_pair(operands[1], *value);
  if (std::optional<int64_t> value =
          getSignedSemanticInteger(operands[1]))
    return std::make_pair(operands[0], *value);
  return std::nullopt;
}

// Extract a common symbolic pointer base plus a constant element offset.  The
// proof is intentionally narrow: pointers with unrelated bases, symbolic
// offsets, or nonlinear arithmetic remain may-alias and are never reordered.
static std::pair<TensorSemanticExpr, int64_t>
splitConstantPointerOffset(const TensorSemanticExpr &pointer) {
  if (pointer.getKind() != TensorSemanticKind::PointerAdd ||
      pointer.getOperands().size() != 2)
    return {pointer, 0};
  std::optional<int64_t> offset =
      getSignedSemanticInteger(pointer.getOperands()[1]);
  if (!offset)
    return {pointer, 0};
  auto [base, baseOffset] =
      splitConstantPointerOffset(pointer.getOperands()[0]);
  return {base, baseOffset + *offset};
}

static bool areProvablyDistinctPointers(const TensorSemanticExpr &lhs,
                                        const TensorSemanticExpr &rhs) {
  auto [lhsBase, lhsOffset] = splitConstantPointerOffset(lhs);
  auto [rhsBase, rhsOffset] = splitConstantPointerOffset(rhs);
  return lhsBase.structurallyEquals(rhsBase) && lhsOffset != rhsOffset;
}

static bool isEmptyConstantRange(const TensorSemanticExpr &lower,
                                 const TensorSemanticExpr &upper) {
  std::optional<int64_t> lowerValue = getSignedSemanticInteger(lower);
  std::optional<int64_t> upperValue = getSignedSemanticInteger(upper);
  return lowerValue && upperValue && *lowerValue >= *upperValue;
}

static std::optional<uint64_t>
getSmallConstantRangeLength(const TensorSemanticExpr &lower,
                            const TensorSemanticExpr &upper) {
  std::optional<int64_t> lowerValue = getSignedSemanticInteger(lower);
  std::optional<int64_t> upperValue = getSignedSemanticInteger(upper);
  if (!lowerValue || !upperValue || *upperValue < *lowerValue)
    return std::nullopt;
  uint64_t length = static_cast<uint64_t>(*upperValue) -
                    static_cast<uint64_t>(*lowerValue);
  if (length > 4096)
    return std::nullopt;
  return length;
}

struct DeltaSelection {
  TensorSemanticExpr solution;
  TensorSemanticExpr resultIndex;
  unsigned resultDimension = 0;
  int64_t resultOffset = 0;
};

struct GuardedSummand {
  TensorSemanticExpr condition;
  TensorSemanticExpr value;
};

// Returns `condition` and `value` from either
//
//   select(condition, value, 0)
//
// or a nested finite sum whose body has that form and whose condition is
// independent of the inner binder. The second form is produced when the
// generic select-over-sum identity pushes an outer guard into a nested sum.
static std::optional<GuardedSummand>
extractGuardedSummand(const TensorSemanticExpr &body) {
  if (body.getKind() == TensorSemanticKind::Select &&
      body.getOperands().size() == 3 &&
      isSemanticZero(body.getOperands()[2]))
    return GuardedSummand{body.getOperands()[0], body.getOperands()[1]};

  if (body.getKind() != TensorSemanticKind::Sum ||
      body.getOperands().size() != 4)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> sum = body.getOperands();
  if (sum[3].getKind() != TensorSemanticKind::Select ||
      sum[3].getOperands().size() != 3 ||
      !isSemanticZero(sum[3].getOperands()[2]))
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> selected = sum[3].getOperands();
  if (containsSemanticExpr(selected[0], sum[0]))
    return std::nullopt;
  TensorSemanticExpr unguarded = TensorSemanticExpr::operation(
      TensorSemanticKind::Sum, body.getType(),
      {sum[0], sum[1], sum[2], selected[1]});
  return GuardedSummand{selected[0], unguarded};
}

static std::optional<DeltaSelection>
solveDeltaCondition(const TensorSemanticExpr &condition,
                    const TensorSemanticExpr &binder,
                    ArrayRef<TensorSemanticExpr> resultIndices) {
  if (condition.getKind() != TensorSemanticKind::Equal ||
      condition.getOperands().size() != 2)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> equality = condition.getOperands();
  std::optional<int64_t> lhsBinder =
      matchUnitAffineOffset(equality[0], binder);
  std::optional<int64_t> rhsBinder =
      matchUnitAffineOffset(equality[1], binder);
  if (lhsBinder.has_value() == rhsBinder.has_value())
    return std::nullopt;
  TensorSemanticExpr other = lhsBinder ? equality[1] : equality[0];
  int64_t binderOffset = lhsBinder ? *lhsBinder : *rhsBinder;
  if (containsSemanticExpr(other, binder))
    return std::nullopt;

  DeltaSelection result;
  result.solution = addSemanticOffset(other, -binderOffset);
  for (auto [dimension, index] : llvm::enumerate(resultIndices)) {
    std::optional<int64_t> offset =
        matchUnitAffineOffset(result.solution, index);
    if (!offset)
      continue;
    result.resultIndex = index;
    result.resultDimension = dimension;
    result.resultOffset = *offset;
    return result;
  }
  return std::nullopt;
}

// A deliberately small, sound implication checker for canonical boolean
// formulas. It is used only for identities such as A && (A || B) == A; a
// failed query means "unknown", never "false".
static bool semanticBooleanImplies(const TensorSemanticExpr &antecedent,
                                   const TensorSemanticExpr &consequent) {
  if (antecedent.structurallyEquals(consequent))
    return true;
  if (std::optional<APInt> value =
          getIntegerSemanticValue(consequent))
    return !value->isZero();
  if (std::optional<APInt> value =
          getIntegerSemanticValue(antecedent))
    return value->isZero();

  if (antecedent.getKind() == TensorSemanticKind::And)
    return llvm::any_of(antecedent.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return semanticBooleanImplies(operand,
                                                        consequent);
                        });
  if (antecedent.getKind() == TensorSemanticKind::Or)
    return llvm::all_of(antecedent.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return semanticBooleanImplies(operand,
                                                        consequent);
                        });
  if (consequent.getKind() == TensorSemanticKind::And)
    return llvm::all_of(consequent.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return semanticBooleanImplies(antecedent,
                                                        operand);
                        });
  if (consequent.getKind() == TensorSemanticKind::Or)
    return llvm::any_of(consequent.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return semanticBooleanImplies(antecedent,
                                                        operand);
                        });
  return false;
}

static std::optional<int64_t>
getPrivateBufferSymbol(const TensorSemanticExpr &handle) {
  auto payload = dyn_cast_or_null<DictionaryAttr>(handle.getPayload());
  auto marker = payload
                    ? dyn_cast_or_null<BoolAttr>(
                          payload.get("private_buffer"))
                    : BoolAttr{};
  auto symbol = payload
                    ? dyn_cast_or_null<IntegerAttr>(payload.get("symbol_id"))
                    : IntegerAttr{};
  if (!marker || !marker.getValue() || !symbol)
    return std::nullopt;
  return symbol.getInt();
}

// Buffer handles are opaque values in the proof language.  We may prove two
// handles distinct when they are coordinates of the same buffer and at least
// one coordinate is a different integer constant, or when they come from two
// different private allocations.  Different arbitrary input symbols are not
// assumed disjoint: that would be an unsound alias-analysis shortcut.
static bool
areProvablyDistinctBufferHandles(const TensorSemanticExpr &lhs,
                                 const TensorSemanticExpr &rhs) {
  if (lhs.getKind() != TensorSemanticKind::Element ||
      rhs.getKind() != TensorSemanticKind::Element)
    return false;
  std::optional<int64_t> lhsPrivate = getPrivateBufferSymbol(lhs);
  std::optional<int64_t> rhsPrivate = getPrivateBufferSymbol(rhs);
  // A local private allocation is fresh: it cannot alias either another
  // private allocation or an arbitrary boundary memref.
  if (lhsPrivate != rhsPrivate && (lhsPrivate || rhsPrivate))
    return true;
  if (lhs.getPayload() != rhs.getPayload() ||
      lhs.getOperands().size() != rhs.getOperands().size())
    return false;
  for (auto [lhsIndex, rhsIndex] :
       llvm::zip_equal(lhs.getOperands(), rhs.getOperands())) {
    std::optional<APInt> lhsValue = getIntegerSemanticValue(lhsIndex);
    std::optional<APInt> rhsValue = getIntegerSemanticValue(rhsIndex);
    if (lhsValue && rhsValue && *lhsValue != *rhsValue)
      return true;
  }
  return false;
}

} // namespace

void TensorSemanticExpr::print(llvm::raw_ostream &stream) const {
  if (!node) {
    stream << "<invalid>";
    return;
  }
  stream << '(' << stringifySemanticKind(node->kind);
  if (node->payload)
    stream << ' ' << node->payload;
  for (const TensorSemanticExpr &operand : node->operands) {
    stream << ' ';
    operand.print(stream);
  }
  stream << " : " << node->type << ')';
}

LogicalResult
TTOpSemanticsRegistry::add(std::unique_ptr<TTOpSemantics> semantics) {
  if (!semantics || semantics->getOperationName().empty() ||
      indices.count(semantics->getOperationName()))
    return failure();
  indices[semantics->getOperationName()] = definitions.size();
  definitions.push_back(std::move(semantics));
  return success();
}

const TTOpSemantics *
TTOpSemanticsRegistry::lookup(StringRef operationName) const {
  auto iterator = indices.find(operationName);
  if (iterator == indices.end())
    return nullptr;
  return definitions[iterator->second].get();
}

bool TTOpSemanticsRegistry::supports(Operation *operation) const {
  if (!operation)
    return false;
  const TTOpSemantics *semantics =
      lookup(operation->getName().getStringRef());
  return semantics && semantics->supports(operation);
}

bool TTOpSemanticsRegistry::allowsSharedImport(Operation *operation) const {
  if (!operation)
    return false;
  const TTOpSemantics *semantics =
      lookup(operation->getName().getStringRef());
  return semantics && semantics->allowsSharedImport(operation);
}

SmallVector<StringRef, 16> TTOpSemanticsRegistry::getOperationNames() const {
  SmallVector<StringRef, 16> names;
  names.reserve(definitions.size());
  for (const std::unique_ptr<TTOpSemantics> &definition : definitions)
    names.push_back(definition->getOperationName());
  llvm::sort(names);
  return names;
}

ArrayRef<GraphSemanticDomain>
TTOpSemanticsRegistry::getImplementedDomains(StringRef operationName) const {
  const TTOpSemantics *semantics = lookup(operationName);
  return semantics ? semantics->getImplementedDomains()
                   : ArrayRef<GraphSemanticDomain>{};
}

TTOpSearchPolicy
TTOpSemanticsRegistry::getSearchPolicy(StringRef operationName) const {
  const TTOpSemantics *semantics = lookup(operationName);
  return semantics ? semantics->getSearchPolicy()
                   : TTOpSearchPolicy::Incomplete;
}

namespace {

struct GroupedProgramRewrite {
  TensorSemanticExpr expression;
  SmallVector<int64_t, 4> lanes;
  bool valid = true;
};

static void addLane(SmallVectorImpl<int64_t> &lanes, int64_t lane) {
  if (!llvm::is_contained(lanes, lane))
    lanes.push_back(lane);
}

static std::optional<int64_t>
getProgramAxis(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::ProgramId)
    return std::nullopt;
  auto axis = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
  return axis ? std::optional<int64_t>(axis.getInt()) : std::nullopt;
}

static std::optional<int64_t>
getNumProgramsAxis(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::NumPrograms)
    return std::nullopt;
  auto axis = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
  return axis ? std::optional<int64_t>(axis.getInt()) : std::nullopt;
}

static bool isProgramIdOf(const TensorSemanticExpr &expression,
                          const TensorSemanticExpr &launch, int64_t axis) {
  std::optional<int64_t> programAxis = getProgramAxis(expression);
  return programAxis && *programAxis == axis &&
         expression.getOperands().size() == 1 &&
         expression.getOperands().front().structurallyEquals(launch);
}

static std::optional<int64_t> matchGroupedProgramId(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &groupedLaunch, int64_t axis, int64_t factor) {
  auto matchScaled = [&](const TensorSemanticExpr &scaled) {
    if (scaled.getKind() != TensorSemanticKind::Multiply ||
        scaled.getOperands().size() != 2)
      return false;
    ArrayRef<TensorSemanticExpr> operands = scaled.getOperands();
    std::optional<int64_t> lhs = getSignedSemanticInteger(operands[0]);
    std::optional<int64_t> rhs = getSignedSemanticInteger(operands[1]);
    return (lhs && *lhs == factor &&
            isProgramIdOf(operands[1], groupedLaunch, axis)) ||
           (rhs && *rhs == factor &&
            isProgramIdOf(operands[0], groupedLaunch, axis));
  };

  if (matchScaled(expression))
    return 0;
  if (expression.getKind() != TensorSemanticKind::Add ||
      expression.getOperands().size() != 2)
    return std::nullopt;
  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  std::optional<int64_t> lhs = getSignedSemanticInteger(operands[0]);
  std::optional<int64_t> rhs = getSignedSemanticInteger(operands[1]);
  if (lhs && *lhs >= 0 && *lhs < factor && matchScaled(operands[1]))
    return *lhs;
  if (rhs && *rhs >= 0 && *rhs < factor && matchScaled(operands[0]))
    return *rhs;
  return std::nullopt;
}

static GroupedProgramRewrite rewriteGroupedProgramIds(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &groupedLaunch,
    const TensorSemanticExpr &baseLaunch, int64_t axis, int64_t factor) {
  if (!expression)
    return {{}, {}, false};
  if (std::optional<int64_t> lane = matchGroupedProgramId(
          expression, groupedLaunch, axis, factor)) {
    auto integerType = dyn_cast<IntegerType>(expression.getType());
    if (!integerType)
      return {{}, {}, false};
    auto payload = IntegerAttr::get(integerType, axis);
    TensorSemanticExpr program = TensorSemanticExpr::operation(
        TensorSemanticKind::ProgramId, expression.getType(), {baseLaunch},
        payload);
    return {program, {*lane}, true};
  }
  if (std::optional<int64_t> programAxis = getProgramAxis(expression)) {
    if (expression.getOperands().size() != 1 ||
        !expression.getOperands().front().structurallyEquals(groupedLaunch))
      return {expression, {}, true};
    if (*programAxis == axis)
      return {{}, {}, false};
    return {TensorSemanticExpr::operation(
                TensorSemanticKind::ProgramId, expression.getType(),
                {baseLaunch}, expression.getPayload()),
            {}, true};
  }
  if (std::optional<int64_t> gridAxis = getNumProgramsAxis(expression)) {
    if (expression.getOperands().size() != 1 ||
        !expression.getOperands().front().structurallyEquals(groupedLaunch))
      return {expression, {}, true};
    // A remapped axis exposes a different num_programs value. Until an
    // explicit extent theorem proves the conversion, observing that value is
    // a hard rejection. Other axes are unchanged by the remap.
    if (*gridAxis == axis)
      return {{}, {}, false};
    return {TensorSemanticExpr::operation(
                TensorSemanticKind::NumPrograms, expression.getType(),
                {baseLaunch}, expression.getPayload()),
            {}, true};
  }

  SmallVector<TensorSemanticExpr, 4> operands;
  SmallVector<int64_t, 4> lanes;
  for (const TensorSemanticExpr &operand : expression.getOperands()) {
    GroupedProgramRewrite rewritten = rewriteGroupedProgramIds(
        operand, groupedLaunch, baseLaunch, axis, factor);
    if (!rewritten.valid)
      return {{}, {}, false};
    operands.push_back(rewritten.expression);
    for (int64_t lane : rewritten.lanes)
      addLane(lanes, lane);
  }
  return {TensorSemanticExpr::operation(
              expression.getKind(), expression.getType(), operands,
              expression.getPayload()),
          std::move(lanes), true};
}

static bool hasCompleteLaneSet(ArrayRef<int64_t> lanes, int64_t factor) {
  if (lanes.size() != static_cast<size_t>(factor))
    return false;
  for (int64_t lane = 0; lane < factor; ++lane)
    if (!llvm::is_contained(lanes, lane))
      return false;
  return true;
}

static std::optional<TensorSemanticExpr>
getLaunchBound(const TensorSemanticExpr &launch, int64_t axis) {
  if (launch.getKind() != TensorSemanticKind::LaunchBound ||
      launch.getOperands().size() != 2)
    return std::nullopt;
  auto boundAxis = dyn_cast_or_null<IntegerAttr>(launch.getPayload());
  if (boundAxis && boundAxis.getInt() == axis)
    return launch.getOperands()[1];
  return getLaunchBound(launch.getOperands()[0], axis);
}

static void collectGroupedLogicalProgramIds(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &groupedLaunch, int64_t axis, int64_t factor,
    SmallVectorImpl<TensorSemanticExpr> &logicalIds) {
  if (matchGroupedProgramId(expression, groupedLaunch, axis, factor)) {
    if (!llvm::any_of(logicalIds, [&](const TensorSemanticExpr &known) {
          return known.structurallyEquals(expression);
        }))
      logicalIds.push_back(expression);
    return;
  }
  for (const TensorSemanticExpr &operand : expression.getOperands())
    collectGroupedLogicalProgramIds(operand, groupedLaunch, axis, factor,
                                    logicalIds);
}

static bool hasTailGuard(const TensorSemanticExpr &write,
                         const TensorSemanticExpr &groupedLaunch,
                         int64_t axis, int64_t factor,
                         const TensorSemanticExpr &bound) {
  if (write.getOperands().size() != 4)
    return false;
  const TensorSemanticExpr &mask = write.getOperands().back();
  SmallVector<TensorSemanticExpr, 2> logicalIds;
  collectGroupedLogicalProgramIds(mask, groupedLaunch, axis, factor,
                                  logicalIds);
  return llvm::any_of(logicalIds, [&](const TensorSemanticExpr &logicalId) {
    return conditionContainsSignedUpperBound(mask, logicalId, bound);
  });
}

static bool isTailNeutralBoolean(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &groupedLaunch, int64_t axis, int64_t factor,
    const TensorSemanticExpr &bound) {
  SmallVector<TensorSemanticExpr, 4> logicalIds;
  collectGroupedLogicalProgramIds(expression, groupedLaunch, axis, factor,
                                  logicalIds);
  if (logicalIds.empty())
    return true;
  if (expression.getKind() == TensorSemanticKind::And)
    return llvm::all_of(expression.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return isTailNeutralBoolean(
                              operand, groupedLaunch, axis, factor, bound);
                        });
  if (expression.getKind() == TensorSemanticKind::Or) {
    // Normalization rewrites select(guard, valid, true) to
    // valid || !guard.  The negated upper-bound guard makes the whole
    // expression true for every padded tail lane.
    return llvm::any_of(expression.getOperands(),
                        [&](const TensorSemanticExpr &operand) {
                          return operand.getKind() == TensorSemanticKind::Not &&
                                 operand.getOperands().size() == 1 &&
                                 llvm::all_of(
                                     logicalIds,
                                     [&](const TensorSemanticExpr &logicalId) {
                                       return conditionContainsSignedUpperBound(
                                           operand.getOperands().front(),
                                           logicalId, bound);
                                     });
                        });
  }
  if (expression.getKind() != TensorSemanticKind::Select ||
      expression.getOperands().size() != 3 ||
      !isSemanticOne(expression.getOperands()[2]))
    return false;
  const TensorSemanticExpr &condition = expression.getOperands()[0];
  return llvm::all_of(logicalIds, [&](const TensorSemanticExpr &logicalId) {
    return conditionContainsSignedUpperBound(condition, logicalId, bound);
  });
}

static bool containsMemoryTransition(const TensorSemanticExpr &expression) {
  if (!expression)
    return false;
  if (expression.getKind() == TensorSemanticKind::MemoryWrite ||
      expression.getKind() == TensorSemanticKind::MemoryAtomic)
    return true;
  return llvm::any_of(expression.getOperands(),
                      [](const TensorSemanticExpr &operand) {
                        return containsMemoryTransition(operand);
                      });
}

// Validates the premises of the quantified exact-partition theorem without
// expanding one semantic graph per logical lane. The theorem applies to an
// arbitrary per-program pure MemoryState transition. Atomics and ordered
// observables are excluded because changing inter-program scheduling may be
// visible; observing num_programs on the remapped axis is excluded because
// the physical grid extent changes. Program IDs must be tied to the same
// LaunchState quantified by the kernel root.
static bool hasPartitionableMemoryPremise(
    const TensorSemanticExpr &expression, int64_t axis, int64_t ceilDiv) {
  if (expression.getKind() != TensorSemanticKind::PartitionableMemory)
    return false;
  auto premise = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
  auto premiseAxis =
      premise ? dyn_cast_or_null<IntegerAttr>(premise.get("axis"))
              : IntegerAttr{};
  auto premiseCeilDiv =
      premise ? dyn_cast_or_null<IntegerAttr>(premise.get("ceil_div"))
              : IntegerAttr{};
  return premiseAxis && premiseCeilDiv && premiseAxis.getInt() == axis &&
         premiseCeilDiv.getInt() == ceilDiv;
}

static bool isInitialMemoryState(const TensorSemanticExpr &expression) {
  if (expression.getKind() != TensorSemanticKind::Symbol)
    return false;
  auto name = dyn_cast_or_null<StringAttr>(expression.getPayload());
  return name && name.getValue() == "memory-state-0";
}

static bool isExactKernelPartitionSafe(const TensorSemanticExpr &expression,
                                       const TensorSemanticExpr &launch,
                                       int64_t axis, bool root = true) {
  if (!expression)
    return false;
  if (hasPartitionableMemoryPremise(expression, axis, /*ceilDiv=*/0))
    return true;
  // A bare initial MemoryState at the theorem root is an unconstrained body
  // variable, not evidence that every possible body is partition-safe.  The
  // same symbol nested under a concrete read/write is merely the pre-state.
  if (root && isInitialMemoryState(expression))
    return false;
  if (expression.getKind() == TensorSemanticKind::MemoryAtomic ||
      expression.getKind() == TensorSemanticKind::MemoryObservable ||
      expression.getKind() == TensorSemanticKind::KernelEffect ||
      expression.getKind() == TensorSemanticKind::KernelPartition ||
      expression.getKind() == TensorSemanticKind::LaunchRemap)
    return false;
  if (std::optional<int64_t> observedAxis = getNumProgramsAxis(expression))
    if (*observedAxis == axis)
      return false;
  if (std::optional<int64_t> programAxis = getProgramAxis(expression))
    if (*programAxis == axis &&
        (expression.getOperands().size() != 1 ||
         !expression.getOperands().front().structurallyEquals(launch)))
      return false;
  return llvm::all_of(expression.getOperands(),
                      [&](const TensorSemanticExpr &operand) {
                        return isExactKernelPartitionSafe(operand, launch,
                                                          axis,
                                                          /*root=*/false);
                      });
}

static void collectMappedProgramIds(
    const TensorSemanticExpr &expression, const TensorSemanticExpr &launch,
    int64_t axis, SmallVectorImpl<TensorSemanticExpr> &programIds) {
  if (!expression)
    return;
  if (std::optional<int64_t> programAxis = getProgramAxis(expression)) {
    if (*programAxis == axis && expression.getOperands().size() == 1 &&
        expression.getOperands().front().structurallyEquals(launch) &&
        !llvm::any_of(programIds, [&](const TensorSemanticExpr &known) {
          return known.structurallyEquals(expression);
        }))
      programIds.push_back(expression);
  }
  for (const TensorSemanticExpr &operand : expression.getOperands())
    collectMappedProgramIds(operand, launch, axis, programIds);
}

static bool containsMappedProgramId(const TensorSemanticExpr &expression,
                                    const TensorSemanticExpr &launch,
                                    int64_t axis) {
  SmallVector<TensorSemanticExpr, 2> programIds;
  collectMappedProgramIds(expression, launch, axis, programIds);
  return !programIds.empty();
}

static bool conditionGuardsMappedProgramIds(
    const TensorSemanticExpr &condition,
    ArrayRef<TensorSemanticExpr> expressions,
    const TensorSemanticExpr &launch, int64_t axis,
    const TensorSemanticExpr &bound) {
  SmallVector<TensorSemanticExpr, 4> programIds;
  for (const TensorSemanticExpr &expression : expressions)
    collectMappedProgramIds(expression, launch, axis, programIds);
  return !programIds.empty() &&
         llvm::all_of(programIds, [&](const TensorSemanticExpr &programId) {
           return conditionContainsSignedUpperBound(condition, programId,
                                                    bound);
         });
}

// Quantified tail theorem. Pure arithmetic may execute for a padded logical
// lane, but every memory observation that depends on that lane must be under
// the launch-bound predicate. The traversal is semantic (Select/MemoryState),
// not tied to any TTIR source topology.
static bool isGuardedKernelPartitionSafe(
    const TensorSemanticExpr &expression, const TensorSemanticExpr &launch,
    int64_t axis, const TensorSemanticExpr &bound, bool guarded = false,
    bool root = true) {
  if (!expression)
    return false;
  if (hasPartitionableMemoryPremise(expression, axis, /*ceilDiv=*/1))
    return true;
  if (root && isInitialMemoryState(expression))
    return false;
  TensorSemanticKind kind = expression.getKind();
  if (kind == TensorSemanticKind::MemoryAtomic ||
      kind == TensorSemanticKind::MemoryObservable ||
      kind == TensorSemanticKind::KernelEffect ||
      kind == TensorSemanticKind::KernelPartition ||
      kind == TensorSemanticKind::LaunchRemap)
    return false;
  if (std::optional<int64_t> observedAxis = getNumProgramsAxis(expression))
    if (*observedAxis == axis)
      return false;
  if (std::optional<int64_t> programAxis = getProgramAxis(expression))
    if (*programAxis == axis &&
        (expression.getOperands().size() != 1 ||
         !expression.getOperands().front().structurallyEquals(launch)))
      return false;

  ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
  if (kind == TensorSemanticKind::Select && operands.size() == 3) {
    const bool guardsTrue = conditionGuardsMappedProgramIds(
        operands[0], operands.drop_front(1).take_front(1), launch, axis,
        bound);
    return isGuardedKernelPartitionSafe(operands[0], launch, axis, bound,
                                        guarded, /*root=*/false) &&
           isGuardedKernelPartitionSafe(operands[1], launch, axis, bound,
                                        guarded || guardsTrue,
                                        /*root=*/false) &&
           isGuardedKernelPartitionSafe(operands[2], launch, axis, bound,
                                        guarded, /*root=*/false);
  }
  if ((kind == TensorSemanticKind::MemoryRead ||
       kind == TensorSemanticKind::MemoryValid) &&
      containsMappedProgramId(expression, launch, axis) && !guarded)
    return false;
  if (kind == TensorSemanticKind::MemoryWrite) {
    if (operands.size() != 3 && operands.size() != 4)
      return false;
    ArrayRef<TensorSemanticExpr> currentWrite = operands.drop_front();
    if (containsMappedProgramId(TensorSemanticExpr::operation(
                                    TensorSemanticKind::ApplyNary,
                                    expression.getType(), currentWrite),
                                launch, axis)) {
      if (operands.size() != 4 ||
          !conditionGuardsMappedProgramIds(
              operands.back(), currentWrite, launch, axis, bound))
        return false;
    }
  }
  return llvm::all_of(operands, [&](const TensorSemanticExpr &operand) {
    return isGuardedKernelPartitionSafe(operand, launch, axis, bound,
                                        guarded, /*root=*/false);
  });
}

struct MemoryWriteTemplate {
  Type type;
  Attribute payload;
  SmallVector<TensorSemanticExpr, 3> operands;
  SmallVector<int64_t, 4> lanes;
};

static bool sameMemoryWriteTemplate(const MemoryWriteTemplate &lhs,
                                    const MemoryWriteTemplate &rhs) {
  if (lhs.type != rhs.type || lhs.payload != rhs.payload ||
      lhs.operands.size() != rhs.operands.size())
    return false;
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip_equal(lhs.operands, rhs.operands))
    if (!lhsOperand.structurallyEquals(rhsOperand))
      return false;
  return true;
}

static bool flattenMemoryWrites(const TensorSemanticExpr &expression,
                                TensorSemanticExpr &base,
                                SmallVectorImpl<TensorSemanticExpr> &writes) {
  if (expression.getKind() == TensorSemanticKind::MemoryAtomic)
    return false;
  if (expression.getKind() != TensorSemanticKind::MemoryWrite) {
    base = expression;
    return true;
  }
  if (expression.getOperands().size() != 3 &&
      expression.getOperands().size() != 4)
    return false;
  if (!flattenMemoryWrites(expression.getOperands().front(), base, writes))
    return false;
  writes.push_back(expression);
  return true;
}

static std::optional<TensorSemanticExpr> collapseGroupedMemoryWrites(
    const TensorSemanticExpr &expression,
    const TensorSemanticExpr &groupedLaunch,
    const TensorSemanticExpr &baseLaunch, int64_t axis, int64_t factor,
    std::optional<TensorSemanticExpr> tailBound = std::nullopt) {
  TensorSemanticExpr base;
  SmallVector<TensorSemanticExpr, 8> writes;
  const size_t groupSize = static_cast<size_t>(factor);
  if (!flattenMemoryWrites(expression, base, writes) || writes.empty() ||
      writes.size() % groupSize != 0)
    return std::nullopt;
  GroupedProgramRewrite rewrittenBase = rewriteGroupedProgramIds(
      base, groupedLaunch, baseLaunch, axis, factor);
  if (!rewrittenBase.valid || !rewrittenBase.lanes.empty() ||
      containsMemoryTransition(rewrittenBase.expression))
    return std::nullopt;

  SmallVector<MemoryWriteTemplate, 8> templates;
  for (const TensorSemanticExpr &write : writes) {
    MemoryWriteTemplate item;
    item.type = write.getType();
    item.payload = write.getPayload();
    for (const TensorSemanticExpr &operand : write.getOperands().drop_front()) {
      GroupedProgramRewrite rewritten = rewriteGroupedProgramIds(
          operand, groupedLaunch, baseLaunch, axis, factor);
      if (!rewritten.valid)
        return std::nullopt;
      item.operands.push_back(rewritten.expression);
      for (int64_t lane : rewritten.lanes)
        addLane(item.lanes, lane);
    }
    if (item.lanes.size() > 1)
      return std::nullopt;
    if (tailBound && !item.lanes.empty() &&
        !hasTailGuard(write, groupedLaunch, axis, factor, *tailBound))
      return std::nullopt;
    templates.push_back(std::move(item));
  }

  TensorSemanticExpr state = rewrittenBase.expression;
  for (size_t begin = 0; begin < templates.size(); begin += groupSize) {
    SmallVector<int64_t, 4> lanes;
    for (size_t lane = 0; lane < groupSize; ++lane) {
      const MemoryWriteTemplate &item = templates[begin + lane];
      if (!sameMemoryWriteTemplate(templates[begin], item))
        return std::nullopt;
      for (int64_t residue : item.lanes)
        addLane(lanes, residue);
    }
    const bool anyLane = llvm::any_of(
        ArrayRef<MemoryWriteTemplate>(templates).slice(begin, groupSize),
        [](const MemoryWriteTemplate &item) { return !item.lanes.empty(); });
    if (anyLane && (!llvm::all_of(
                        ArrayRef<MemoryWriteTemplate>(templates)
                            .slice(begin, groupSize),
                        [](const MemoryWriteTemplate &item) {
                          return item.lanes.size() == 1;
                        }) ||
                    !hasCompleteLaneSet(lanes, factor)))
      return std::nullopt;
    SmallVector<TensorSemanticExpr, 4> operands = {state};
    operands.append(templates[begin].operands.begin(),
                    templates[begin].operands.end());
    state = TensorSemanticExpr::operation(
        TensorSemanticKind::MemoryWrite, templates[begin].type, operands,
        templates[begin].payload);
  }
  return state;
}

} // namespace

TensorSemanticExpr TensorSemanticNormalizer::normalizeExpr(
    const TensorSemanticExpr &expression,
    SmallVectorImpl<SemanticProofStep> &proof,
    ArrayRef<int64_t> resultShape,
    ArrayRef<TensorSemanticExpr> resultIndices,
    llvm::DenseMap<const void *, TensorSemanticExpr> &memo,
    llvm::DenseMap<const void *, std::string> &orderingKeys) const {
  if (shouldInterrupt())
    return expression;
  if (!expression)
    return {};

  auto known = memo.find(expression.getAsOpaquePointer());
  if (known != memo.end())
    return known->second;
  TensorSemanticExpr normalized = normalizeExprUncached(
      expression, proof, resultShape, resultIndices, memo, orderingKeys);
  memo.try_emplace(expression.getAsOpaquePointer(), normalized);
  return normalized;
}

TensorSemanticExpr TensorSemanticNormalizer::normalizeExprUncached(
    const TensorSemanticExpr &expression,
    SmallVectorImpl<SemanticProofStep> &proof,
    ArrayRef<int64_t> resultShape,
    ArrayRef<TensorSemanticExpr> resultIndices,
    llvm::DenseMap<const void *, TensorSemanticExpr> &memo,
    llvm::DenseMap<const void *, std::string> &orderingKeys) const {

  ArrayRef<TensorSemanticExpr> originalOperands = expression.getOperands();
  SmallVector<TensorSemanticExpr, 4> operands;
  for (const TensorSemanticExpr &operand : originalOperands)
    operands.push_back(
        normalizeExpr(operand, proof, resultShape, resultIndices, memo,
                      orderingKeys));

  if (isCommutativeSemanticKind(expression.getKind()))
    llvm::sort(operands, [&](const TensorSemanticExpr &lhs,
                            const TensorSemanticExpr &rhs) {
      return semanticKey(lhs, orderingKeys) <
             semanticKey(rhs, orderingKeys);
    });

  bool operandsChanged = operands.size() != originalOperands.size();
  if (!operandsChanged)
    for (auto [normalizedOperand, originalOperand] :
         llvm::zip_equal(operands, originalOperands))
      if (!normalizedOperand.structurallyEquals(originalOperand)) {
        operandsChanged = true;
        break;
      }
  TensorSemanticExpr normalized =
      operandsChanged
          ? TensorSemanticExpr::operation(
                expression.getKind(), expression.getType(), operands,
                expression.getPayload())
          : expression;
  const TensorSemanticExpr recursivelyNormalizedOperands = normalized;
  bool normalizedToClosedForm = false;

  auto makeBoolean = [&](bool value) {
    return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                       value);
  };
  auto makeInteger = [&](int64_t value, Type type) {
    return TensorSemanticExpr::integer(type.getContext(), value, type);
  };
  auto makeOperation = [&](TensorSemanticKind kind, Type type,
                           ArrayRef<TensorSemanticExpr> values) {
    return TensorSemanticExpr::operation(kind, type, values);
  };
  auto narrowIntegerTo = [&](const TensorSemanticExpr &operand, Type type) {
    auto sourceType = dyn_cast<IntegerType>(operand.getType());
    auto targetType = dyn_cast<IntegerType>(type);
    if (!sourceType || !targetType ||
        sourceType.getWidth() <= targetType.getWidth())
      return operand;
    return makeOperation(TensorSemanticKind::Truncate, type, {operand});
  };

  if (expression.getKind() == TensorSemanticKind::LaunchRemap &&
      operands.size() == 1) {
    auto mapping = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
    auto factor =
        mapping ? dyn_cast_or_null<IntegerAttr>(mapping.get("factor"))
                : IntegerAttr{};
    if (factor && factor.getInt() == 1)
      normalized = operands.front();
  } else if (expression.getKind() == TensorSemanticKind::KernelEffect &&
             operands.size() == 2 &&
             operands[1].getKind() == TensorSemanticKind::LaunchRemap &&
             operands[1].getOperands().size() == 1) {
    auto mapping =
        dyn_cast_or_null<DictionaryAttr>(operands[1].getPayload());
    auto axis = mapping
                    ? dyn_cast_or_null<IntegerAttr>(mapping.get("axis"))
                    : IntegerAttr{};
    auto factor = mapping
                      ? dyn_cast_or_null<IntegerAttr>(mapping.get("factor"))
                      : IntegerAttr{};
    auto ceilDiv = mapping
                       ? dyn_cast_or_null<IntegerAttr>(mapping.get("ceil_div"))
                       : IntegerAttr{};
    if (axis && factor && ceilDiv && axis.getInt() >= 0 &&
        axis.getInt() <= 2 && factor.getInt() > 1 &&
        (ceilDiv.getInt() == 0 || ceilDiv.getInt() == 1)) {
      const TensorSemanticExpr &groupedLaunch = operands[1];
      const TensorSemanticExpr &baseLaunch =
          groupedLaunch.getOperands().front();
      std::optional<TensorSemanticExpr> tailBound;
      if (ceilDiv.getInt() == 1)
        tailBound = getLaunchBound(baseLaunch, axis.getInt());
      if (ceilDiv.getInt() == 1 && !tailBound)
        return normalized;
      std::optional<TensorSemanticExpr> body = collapseGroupedMemoryWrites(
          operands[0], groupedLaunch, baseLaunch, axis.getInt(),
          factor.getInt(), tailBound);
      if (!body && !containsMemoryTransition(operands[0])) {
        GroupedProgramRewrite rewritten = rewriteGroupedProgramIds(
            operands[0], groupedLaunch, baseLaunch, axis.getInt(),
            factor.getInt());
        const bool exactPartition =
            ceilDiv.getInt() == 0 &&
            hasCompleteLaneSet(rewritten.lanes, factor.getInt());
        const bool guardedTail =
            ceilDiv.getInt() == 1 && tailBound &&
            hasCompleteLaneSet(rewritten.lanes, factor.getInt()) &&
            isTailNeutralBoolean(operands[0], groupedLaunch,
                                 axis.getInt(), factor.getInt(), *tailBound);
        if (rewritten.valid &&
            (rewritten.lanes.empty() || exactPartition || guardedTail))
          body = normalizeExpr(rewritten.expression, proof, resultShape,
                               resultIndices, memo, orderingKeys);
      }
      if (body) {
        TensorSemanticExpr partitioned = TensorSemanticExpr::operation(
            TensorSemanticKind::KernelEffect, expression.getType(),
            {*body, baseLaunch});
        proof.push_back(SemanticProofStep{
            ceilDiv.getInt() == 0 ? "exact-launch-partition"
                                  : "guarded-tail-launch-partition",
            normalized, partitioned});
        normalized = std::move(partitioned);
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::KernelPartition &&
             operands.size() == 2) {
    auto partition =
        dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
    auto axis = partition
                    ? dyn_cast_or_null<IntegerAttr>(partition.get("axis"))
                    : IntegerAttr{};
    auto factor = partition
                      ? dyn_cast_or_null<IntegerAttr>(partition.get("factor"))
                      : IntegerAttr{};
    auto ceilDiv = partition
                       ? dyn_cast_or_null<IntegerAttr>(
                             partition.get("ceil_div"))
                       : IntegerAttr{};
    bool proved = false;
    StringRef reason;
    if (axis && factor && ceilDiv && axis.getInt() >= 0 &&
        axis.getInt() <= 2 && factor.getInt() > 1) {
      if (ceilDiv.getInt() == 0) {
        proved = isExactKernelPartitionSafe(operands[0], operands[1],
                                             axis.getInt());
        reason = "quantified-exact-launch-partition";
      } else if (ceilDiv.getInt() == 1) {
        std::optional<TensorSemanticExpr> tailBound =
            getLaunchBound(operands[1], axis.getInt());
        proved = tailBound && isGuardedKernelPartitionSafe(
                                  operands[0], operands[1], axis.getInt(),
                                  *tailBound);
        reason = "quantified-guarded-tail-launch-partition";
      }
    }
    if (proved) {
      TensorSemanticExpr theoremResult = TensorSemanticExpr::operation(
          TensorSemanticKind::KernelEffect, expression.getType(), operands);
      proof.push_back(
          SemanticProofStep{reason.str(), normalized, theoremResult});
      normalized = std::move(theoremResult);
    }
  } else if (expression.getKind() == TensorSemanticKind::Integer &&
      expression.getType().isInteger(1)) {
    if (std::optional<APInt> value = getIntegerSemanticValue(expression))
      normalized = makeBoolean(!value->isZero());
  } else if (expression.getKind() == TensorSemanticKind::Not &&
      operands.size() == 1) {
    if (std::optional<APInt> value = getIntegerSemanticValue(operands[0]))
      normalized = makeBoolean(value->isZero());
    else if (operands[0].getKind() == TensorSemanticKind::Not &&
             operands[0].getOperands().size() == 1)
      normalized = operands[0].getOperands().front();
  } else if ((expression.getKind() == TensorSemanticKind::And ||
              expression.getKind() == TensorSemanticKind::Or) &&
             !operands.empty()) {
    const TensorSemanticKind kind = expression.getKind();
    SmallVector<TensorSemanticExpr, 8> terms;
    std::function<void(const TensorSemanticExpr &)> collect =
        [&](const TensorSemanticExpr &term) {
          if (term.getKind() == kind && !term.getOperands().empty()) {
            for (const TensorSemanticExpr &operand : term.getOperands())
              collect(operand);
          } else {
            terms.push_back(term);
          }
        };
    collect(operands[0]);
    collect(operands[1]);

    SmallVector<TensorSemanticExpr, 8> live;
    bool annihilated = false;
    for (const TensorSemanticExpr &term : terms) {
      std::optional<APInt> value = getIntegerSemanticValue(term);
      if (value && ((kind == TensorSemanticKind::And && value->isZero()) ||
                    (kind == TensorSemanticKind::Or && value->isOne()))) {
        annihilated = true;
        break;
      }
      if (value && ((kind == TensorSemanticKind::And && value->isOne()) ||
                    (kind == TensorSemanticKind::Or && value->isZero())))
        continue;
      for (const TensorSemanticExpr &prior : live) {
        if (areBooleanComplements(prior, term) ||
            (kind == TensorSemanticKind::And &&
             areContradictoryEqualities(prior, term))) {
          annihilated = true;
          break;
        }
      }
      if (annihilated)
        break;
      live.push_back(term);
    }
    if (annihilated) {
      normalized = makeBoolean(kind == TensorSemanticKind::Or);
    } else {
      if (kind == TensorSemanticKind::And) {
        // Boolean congruence: a && !(a xor b) == a && b.  Applying it to the
        // flattened conjunction makes the theorem independent of the source
        // OPs that produced a and b.
        for (TensorSemanticExpr &term : live) {
          if (term.getKind() != TensorSemanticKind::Not ||
              term.getOperands().size() != 1)
            continue;
          const TensorSemanticExpr &negated = term.getOperands().front();
          if (negated.getKind() != TensorSemanticKind::Xor ||
              negated.getOperands().size() != 2)
            continue;
          ArrayRef<TensorSemanticExpr> xorOperands = negated.getOperands();
          for (const TensorSemanticExpr &other : live) {
            if (other.structurallyEquals(xorOperands[0])) {
              term = xorOperands[1];
              break;
            }
            if (other.structurallyEquals(xorOperands[1])) {
              term = xorOperands[0];
              break;
            }
          }
        }
      }
      llvm::sort(live, [&](const TensorSemanticExpr &lhs,
                          const TensorSemanticExpr &rhs) {
        return semanticKey(lhs, orderingKeys) <
               semanticKey(rhs, orderingKeys);
      });
      live.erase(std::unique(live.begin(), live.end(),
                             [](const TensorSemanticExpr &lhs,
                                const TensorSemanticExpr &rhs) {
                               return lhs.structurallyEquals(rhs);
                             }),
                 live.end());
      SmallVector<bool, 8> redundant(live.size(), false);
      for (unsigned lhs = 0; lhs < live.size(); ++lhs) {
        if (redundant[lhs])
          continue;
        for (unsigned rhs = lhs + 1; rhs < live.size(); ++rhs) {
          if (redundant[rhs])
            continue;
          if (kind == TensorSemanticKind::And) {
            if (semanticBooleanImplies(live[lhs], live[rhs]))
              redundant[rhs] = true;
            else if (semanticBooleanImplies(live[rhs], live[lhs])) {
              redundant[lhs] = true;
              break;
            }
          } else {
            if (semanticBooleanImplies(live[lhs], live[rhs])) {
              redundant[lhs] = true;
              break;
            }
            if (semanticBooleanImplies(live[rhs], live[lhs]))
              redundant[rhs] = true;
          }
        }
      }
      SmallVector<TensorSemanticExpr, 8> canonical;
      for (auto [index, term] : llvm::enumerate(live))
        if (!redundant[index])
          canonical.push_back(term);
      if (canonical.empty()) {
        normalized = makeBoolean(kind == TensorSemanticKind::And);
      } else {
        normalized = canonical.front();
        for (const TensorSemanticExpr &term :
             ArrayRef(canonical).drop_front())
          normalized = makeOperation(kind, expression.getType(),
                                     {normalized, term});
      }
    }
    // The branch above performs the complete associative/commutative Boolean
    // normalization in one pass: it flattens nested nodes, removes identities,
    // contradictions, duplicates and implied terms, sorts the survivors, and
    // rebuilds the canonical left-deep form. Re-entering the normalizer on that
    // freshly built chain repeats the same work once per prefix and makes a
    // finite lane conjunction super-linear enough to stall online matching.
    normalizedToClosedForm = true;
  } else if (expression.getKind() == TensorSemanticKind::Xor &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (operands[0].structurallyEquals(operands[1]))
      normalized = makeBoolean(false);
    else if (lhs && rhs)
      normalized = makeBoolean(lhs->isZero() != rhs->isZero());
    else if (lhs && lhs->isZero())
      normalized = operands[1];
    else if (rhs && rhs->isZero())
      normalized = operands[0];
    else if (lhs && lhs->isOne())
      normalized = makeOperation(TensorSemanticKind::Not,
                                 expression.getType(), {operands[1]});
    else if (rhs && rhs->isOne())
      normalized = makeOperation(TensorSemanticKind::Not,
                                 expression.getType(), {operands[0]});
  } else if ((expression.getKind() == TensorSemanticKind::BitwiseAnd ||
              expression.getKind() == TensorSemanticKind::BitwiseOr ||
              expression.getKind() == TensorSemanticKind::BitwiseXor) &&
             operands.size() == 2) {
    const TensorSemanticKind kind = expression.getKind();
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto integerType = dyn_cast<IntegerType>(expression.getType());
    if (integerType && integerType.getWidth() == 1) {
      TensorSemanticKind logicalKind =
          kind == TensorSemanticKind::BitwiseAnd
              ? TensorSemanticKind::And
              : (kind == TensorSemanticKind::BitwiseOr
                     ? TensorSemanticKind::Or
                     : TensorSemanticKind::Xor);
      SmallVector<TensorSemanticExpr, 2> lowBits = {
          narrowIntegerTo(operands[0], expression.getType()),
          narrowIntegerTo(operands[1], expression.getType())};
      normalized = makeOperation(logicalKind, expression.getType(), lowBits);
    } else if (lhs && rhs) {
      APInt value = kind == TensorSemanticKind::BitwiseAnd
                        ? (*lhs & *rhs)
                        : (kind == TensorSemanticKind::BitwiseOr
                               ? (*lhs | *rhs)
                               : (*lhs ^ *rhs));
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (operands[0].structurallyEquals(operands[1]) &&
               kind != TensorSemanticKind::BitwiseXor) {
      normalized = operands[0];
    } else if (operands[0].structurallyEquals(operands[1])) {
      normalized = makeInteger(0, expression.getType());
    } else if (kind == TensorSemanticKind::BitwiseAnd &&
               ((lhs && lhs->isZero()) || (rhs && rhs->isZero()))) {
      normalized = makeInteger(0, expression.getType());
    } else if ((kind == TensorSemanticKind::BitwiseOr ||
                kind == TensorSemanticKind::BitwiseXor) &&
               lhs && lhs->isZero()) {
      normalized = operands[1];
    } else if ((kind == TensorSemanticKind::BitwiseOr ||
                kind == TensorSemanticKind::BitwiseXor) &&
               rhs && rhs->isZero()) {
      normalized = operands[0];
    } else if (kind == TensorSemanticKind::BitwiseAnd && lhs &&
               lhs->isAllOnes()) {
      normalized = operands[1];
    } else if (kind == TensorSemanticKind::BitwiseAnd && rhs &&
               rhs->isAllOnes()) {
      normalized = operands[0];
    } else if (kind == TensorSemanticKind::BitwiseOr &&
               ((lhs && lhs->isAllOnes()) ||
                (rhs && rhs->isAllOnes()))) {
      normalized = makeInteger(-1, expression.getType());
    } else if (kind == TensorSemanticKind::BitwiseOr &&
               ((lhs && lhs->isOne()) || (rhs && rhs->isOne()))) {
      const bool lhsIsOne = lhs && lhs->isOne();
      const TensorSemanticExpr &other =
          lhsIsOne ? operands[1] : operands[0];
      if (std::optional<TensorSemanticExpr> inverted =
              matchBitwiseNotOperand(other)) {
        // For y=x|1, y is odd and (~x)|1 is its two's-complement
        // negation.  Expressing both spellings through the ring operation
        // lets ordinary arithmetic identities finish the proof.
        TensorSemanticExpr odd = makeOperation(
            TensorSemanticKind::BitwiseOr, expression.getType(),
            {*inverted, lhsIsOne ? operands[0] : operands[1]});
        TensorSemanticExpr zero = makeInteger(0, expression.getType());
        normalized = makeOperation(TensorSemanticKind::Subtract,
                                   expression.getType(), {zero, odd});
      }
    } else if (kind == TensorSemanticKind::BitwiseAnd && integerType &&
               integerType.getWidth() > 32 &&
               ((lhs && *lhs == APInt::getLowBitsSet(lhs->getBitWidth(), 32)) ||
                (rhs && *rhs == APInt::getLowBitsSet(rhs->getBitWidth(), 32)))) {
      const TensorSemanticExpr &value = lhs ? operands[1] : operands[0];
      Type narrowType = IntegerType::get(expression.getType().getContext(), 32);
      TensorSemanticExpr truncated = makeOperation(
          TensorSemanticKind::Truncate, narrowType, {value});
      normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                 expression.getType(), {truncated});
    } else {
      const TensorSemanticExpr *extended = nullptr;
      const APInt *constant = nullptr;
      if (operands[0].getKind() == TensorSemanticKind::ZeroExtend && rhs) {
        extended = &operands[0];
        constant = &*rhs;
      } else if (operands[1].getKind() == TensorSemanticKind::ZeroExtend &&
                 lhs) {
        extended = &operands[1];
        constant = &*lhs;
      }
      if (operands[0].getKind() == TensorSemanticKind::ZeroExtend &&
          operands[1].getKind() == TensorSemanticKind::ZeroExtend &&
          operands[0].getOperands().size() == 1 &&
          operands[1].getOperands().size() == 1 &&
          operands[0].getOperands()[0].getType() ==
              operands[1].getOperands()[0].getType()) {
        Type narrowType = operands[0].getOperands()[0].getType();
        TensorSemanticExpr inner = makeOperation(
            kind, narrowType,
            {operands[0].getOperands()[0], operands[1].getOperands()[0]});
        normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                   expression.getType(), {inner});
      } else if (extended && constant && extended->getOperands().size() == 1) {
        Type narrowType = extended->getOperands()[0].getType();
        auto narrowInteger = dyn_cast<IntegerType>(narrowType);
        if (narrowInteger && constant->getActiveBits() <=
                                 narrowInteger.getWidth()) {
          TensorSemanticExpr narrowConstant = makeInteger(
              constant->trunc(narrowInteger.getWidth()).getSExtValue(),
              narrowType);
          TensorSemanticExpr inner = makeOperation(
              kind, narrowType,
              {extended->getOperands()[0], narrowConstant});
          normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                     expression.getType(), {inner});
        }
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::Equal &&
             operands.size() == 2) {
    if (operands[0].structurallyEquals(operands[1])) {
      normalized = makeBoolean(true);
    } else if (auto lhs = getIntegerSemanticValue(operands[0]),
                    rhs = getIntegerSemanticValue(operands[1]);
               lhs && rhs) {
      normalized = makeBoolean(*lhs == *rhs);
    } else if (operands[0].getType().isInteger(1)) {
      std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
      std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
      if (lhs)
        normalized = lhs->isZero()
                         ? makeOperation(TensorSemanticKind::Not,
                                         expression.getType(), {operands[1]})
                         : operands[1];
      else if (rhs)
        normalized = rhs->isZero()
                         ? makeOperation(TensorSemanticKind::Not,
                                         expression.getType(), {operands[0]})
                         : operands[0];
    } else if ((operands[0].getKind() == TensorSemanticKind::ZeroExtend ||
                operands[0].getKind() == TensorSemanticKind::SignExtend) &&
               operands[1].getKind() == operands[0].getKind() &&
               operands[0].getOperands().size() == 1 &&
               operands[1].getOperands().size() == 1 &&
               operands[0].getOperands()[0].getType() ==
                   operands[1].getOperands()[0].getType()) {
      // Integer extension is injective when both sides use the same signedness
      // and source type:
      //   ext(a) == ext(b)  <=>  a == b.
      // This is a generic semantic identity.  It lets the equality engine
      // prove narrow comparisons without knowing which TTIR optimization is
      // consuming the result.
      normalized = makeOperation(
          TensorSemanticKind::Equal, expression.getType(),
          {operands[0].getOperands()[0], operands[1].getOperands()[0]});
    } else if (operands[0].getKind() == TensorSemanticKind::Select &&
               operands[0].getOperands().size() == 3) {
      ArrayRef<TensorSemanticExpr> selected = operands[0].getOperands();
      TensorSemanticExpr whenTrue = makeOperation(
          TensorSemanticKind::Equal, expression.getType(),
          {selected[1], operands[1]});
      TensorSemanticExpr whenFalse = makeOperation(
          TensorSemanticKind::Equal, expression.getType(),
          {selected[2], operands[1]});
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {selected[0], whenTrue, whenFalse});
    } else if (operands[1].getKind() == TensorSemanticKind::Select &&
               operands[1].getOperands().size() == 3) {
      ArrayRef<TensorSemanticExpr> selected = operands[1].getOperands();
      TensorSemanticExpr whenTrue = makeOperation(
          TensorSemanticKind::Equal, expression.getType(),
          {operands[0], selected[1]});
      TensorSemanticExpr whenFalse = makeOperation(
          TensorSemanticKind::Equal, expression.getType(),
          {operands[0], selected[2]});
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {selected[0], whenTrue, whenFalse});
    }
  } else if (expression.getKind() == TensorSemanticKind::ZeroExtend &&
             operands.size() == 1) {
    if (std::optional<APInt> value =
            getIntegerSemanticValue(operands[0]);
        value && value->isZero()) {
      // Zero has the same representation in every integer/index carrier.
      normalized = makeInteger(0, expression.getType());
    } else if (operands[0].getType().isInteger(1)) {
      TensorSemanticExpr one = makeInteger(1, expression.getType());
      TensorSemanticExpr zero = makeInteger(0, expression.getType());
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {operands[0], one, zero});
    } else if (std::optional<APInt> value =
                   getIntegerSemanticValue(operands[0])) {
      std::optional<unsigned> targetWidth =
          getSemanticBitWidth(expression.getType());
      if (targetWidth && value->getBitWidth() <= *targetWidth)
        normalized = makeInteger(
            value->zextOrTrunc(*targetWidth).getSExtValue(),
            expression.getType());
    }
  } else if (expression.getKind() == TensorSemanticKind::SignExtend &&
             operands.size() == 1) {
    if (std::optional<APInt> value =
            getIntegerSemanticValue(operands[0]);
        value && value->isZero()) {
      normalized = makeInteger(0, expression.getType());
    } else if (operands[0].getType().isInteger(1)) {
      TensorSemanticExpr allOnes = makeInteger(-1, expression.getType());
      TensorSemanticExpr zero = makeInteger(0, expression.getType());
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {operands[0], allOnes, zero});
    } else if (std::optional<APInt> value =
                   getIntegerSemanticValue(operands[0])) {
      std::optional<unsigned> targetWidth =
          getSemanticBitWidth(expression.getType());
      if (targetWidth && value->getBitWidth() <= *targetWidth)
        normalized = makeInteger(
            value->sextOrTrunc(*targetWidth).getSExtValue(),
            expression.getType());
    } else if (std::optional<unsigned> sourceWidth =
                   getSemanticBitWidth(operands[0].getType());
               sourceWidth &&
               sourceWidth == getSemanticBitWidth(expression.getType())) {
      // Same-width sign/zero extension is a bit-preserving carrier cast. Use
      // zero extension as the unique normal form while retaining result type.
      normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                 expression.getType(), operands);
    }
  } else if (expression.getKind() == TensorSemanticKind::Truncate &&
             operands.size() == 1) {
    auto target = dyn_cast<IntegerType>(expression.getType());
    if (std::optional<APInt> value = getIntegerSemanticValue(operands[0])) {
      if (target && value->getBitWidth() > target.getWidth())
        normalized = makeInteger(
            value->trunc(target.getWidth()).getSExtValue(),
            expression.getType());
    } else if ((operands[0].getKind() == TensorSemanticKind::ZeroExtend ||
                operands[0].getKind() == TensorSemanticKind::SignExtend) &&
               operands[0].getOperands().size() == 1 &&
               operands[0].getOperands()[0].getType() == expression.getType()) {
      normalized = operands[0].getOperands()[0];
    } else if ((operands[0].getKind() == TensorSemanticKind::BitwiseAnd ||
                operands[0].getKind() == TensorSemanticKind::BitwiseOr ||
                operands[0].getKind() == TensorSemanticKind::BitwiseXor) &&
               operands[0].getOperands().size() == 2) {
      TensorSemanticExpr lhs = makeOperation(
          TensorSemanticKind::Truncate, expression.getType(),
          {operands[0].getOperands()[0]});
      TensorSemanticExpr rhs = makeOperation(
          TensorSemanticKind::Truncate, expression.getType(),
          {operands[0].getOperands()[1]});
      normalized = makeOperation(operands[0].getKind(), expression.getType(),
                                 {lhs, rhs});
    } else if (operands[0].getKind() == TensorSemanticKind::Select &&
               operands[0].getOperands().size() == 3) {
      TensorSemanticExpr whenTrue = makeOperation(
          TensorSemanticKind::Truncate, expression.getType(),
          {operands[0].getOperands()[1]});
      TensorSemanticExpr whenFalse = makeOperation(
          TensorSemanticKind::Truncate, expression.getType(),
          {operands[0].getOperands()[2]});
      normalized = makeOperation(
          TensorSemanticKind::Select, expression.getType(),
          {operands[0].getOperands()[0], whenTrue, whenFalse});
    }
  } else if (expression.getKind() == TensorSemanticKind::BufferValid &&
             operands.size() == 2 &&
             (operands[0].getKind() == TensorSemanticKind::BufferWrite ||
              operands[0].getKind() == TensorSemanticKind::BufferFill) &&
             operands[0].getOperands().size() == 3) {
    // A functional buffer write changes contents, not the validity of a
    // reference.  Allocation/lifetime legality is an independent online
    // obligation checked before a BufferState suffix can be removed.
    normalized = makeOperation(TensorSemanticKind::BufferValid,
                               expression.getType(),
                               {operands[0].getOperands()[0], operands[1]});
  } else if (expression.getKind() == TensorSemanticKind::BufferRead &&
             operands.size() == 2 &&
             operands[0].getKind() == TensorSemanticKind::BufferWrite &&
             operands[0].getOperands().size() == 3) {
    ArrayRef<TensorSemanticExpr> write = operands[0].getOperands();
    if (operands[1].structurallyEquals(write[1])) {
      // read(write(S, address, value), address) = value
      normalized = write[2];
    } else if (areProvablyDistinctBufferHandles(operands[1], write[1])) {
      // read(write(S, p, value), q) = read(S, q), when p != q.
      normalized = makeOperation(TensorSemanticKind::BufferRead,
                                 expression.getType(),
                                 {write[0], operands[1]});
    }
  } else if (expression.getKind() == TensorSemanticKind::BufferRead &&
             operands.size() == 2 &&
             operands[0].getKind() == TensorSemanticKind::BufferFill &&
             operands[0].getOperands().size() == 3) {
    ArrayRef<TensorSemanticExpr> fill = operands[0].getOperands();
    // BufferFill binds the formal coordinates in fill[1].  A read of the
    // exact same region template observes the scalar fill value at every
    // coordinate.  Region descriptors in the payload prevent a smaller
    // subview fill from being mistaken for a fill of a larger view that has
    // the same base/offset expression.
    if (expression.getPayload() == operands[0].getPayload() &&
        operands[1].structurallyEquals(fill[1]))
      normalized = fill[2];
    else if (areProvablyDistinctBufferHandles(operands[1], fill[1]))
      normalized = TensorSemanticExpr::operation(
          TensorSemanticKind::BufferRead, expression.getType(),
          {fill[0], operands[1]}, expression.getPayload());
  } else if (expression.getKind() == TensorSemanticKind::BufferRead &&
             operands.size() == 2 && expression.getPayload()) {
    // A read is pointwise; once no compact BufferFill is being matched, the
    // logical view descriptor is not part of the physical address identity.
    normalized = makeOperation(TensorSemanticKind::BufferRead,
                               expression.getType(), operands);
  } else if (expression.getKind() == TensorSemanticKind::BufferWrite &&
             operands.size() == 3 &&
             operands[0].getKind() == TensorSemanticKind::BufferWrite &&
             operands[0].getOperands().size() == 3 &&
             operands[0].getOperands()[1].structurallyEquals(operands[1])) {
    // Last write wins at an identical buffer coordinate.  Definedness keeps
    // the validity of both transitions, so this does not hide an invalid
    // earlier write.
    normalized = TensorSemanticExpr::operation(
        TensorSemanticKind::BufferWrite, expression.getType(),
        {operands[0].getOperands()[0], operands[1], operands[2]},
        expression.getPayload());
  } else if (expression.getKind() == TensorSemanticKind::MemoryValid &&
             operands.size() == 2 &&
             (operands[0].getKind() == TensorSemanticKind::MemoryWrite ||
              operands[0].getKind() == TensorSemanticKind::MemoryAtomic) &&
             !operands[0].getOperands().empty()) {
    // A TTIR write/atomic is not an allocator: it cannot change whether any
    // address is valid.  Transition definedness is carried independently, so
    // looking through the state here never hides an invalid earlier access.
    normalized = makeOperation(TensorSemanticKind::MemoryValid,
                               expression.getType(),
                               {operands[0].getOperands()[0], operands[1]});
  } else if ((expression.getKind() == TensorSemanticKind::MemoryWrite ||
              expression.getKind() == TensorSemanticKind::MemoryAtomic) &&
             (operands.size() == 3 || operands.size() == 4)) {
    bool masked = expression.getKind() == TensorSemanticKind::MemoryWrite;
    if (expression.getKind() == TensorSemanticKind::MemoryAtomic)
      if (auto payload =
              dyn_cast_or_null<DictionaryAttr>(expression.getPayload()))
        if (auto value = dyn_cast_or_null<BoolAttr>(payload.get("masked")))
          masked = value.getValue();
    if (masked && operands.size() == 4) {
      TensorSemanticExpr guardedPointer =
          simplifyUnderTrueCondition(operands[1], operands[3]);
      TensorSemanticExpr guardedValue =
          simplifyUnderTrueCondition(operands[2], operands[3]);
      if (!guardedPointer.structurallyEquals(operands[1]) ||
          !guardedValue.structurallyEquals(operands[2]))
        normalized = TensorSemanticExpr::operation(
            expression.getKind(), expression.getType(),
            {operands[0], guardedPointer, guardedValue, operands[3]},
            expression.getPayload());
      if (std::optional<APInt> mask =
              getIntegerSemanticValue(operands[3])) {
        if (mask->isZero()) {
          // A masked-off access is the identity both for MemoryState and for
          // the effect trace (whose first operand is boolean true).
          normalized = operands[0];
        } else {
          Attribute payload = expression.getPayload();
          if (expression.getKind() == TensorSemanticKind::MemoryAtomic) {
            auto dictionary = cast<DictionaryAttr>(payload);
            Builder builder(expression.getType().getContext());
            payload = builder.getDictionaryAttr({builder.getNamedAttr(
                "identity", dictionary.get("identity"))});
          }
          normalized = TensorSemanticExpr::operation(
              expression.getKind(), expression.getType(),
              ArrayRef<TensorSemanticExpr>(operands).take_front(3),
              payload);
        }
      }
    }
    if (expression.getKind() == TensorSemanticKind::MemoryWrite &&
        operands.size() == 3 &&
        operands[0].getKind() == TensorSemanticKind::MemoryWrite &&
        operands[0].getOperands().size() == 3 &&
        operands[0].getOperands()[1].structurallyEquals(operands[1])) {
      // Functional-store overwrite: the second unmasked write to the exact
      // same address determines the final byte value.  Definedness is tracked
      // separately and still retains both accesses, so this never erases an
      // invalid/poison first transition from the proof obligation.
      normalized = TensorSemanticExpr::operation(
          TensorSemanticKind::MemoryWrite, expression.getType(),
          {operands[0].getOperands()[0], operands[1], operands[2]},
          expression.getPayload());
    }
    if (expression.getKind() == TensorSemanticKind::MemoryWrite &&
        (operands.size() == 3 || operands.size() == 4) &&
        operands[0].getKind() == TensorSemanticKind::MemoryWrite &&
        (operands[0].getOperands().size() == 3 ||
         operands[0].getOperands().size() == 4)) {
      ArrayRef<TensorSemanticExpr> previous = operands[0].getOperands();
      const TensorSemanticExpr &previousPointer = previous[1];
      const TensorSemanticExpr &currentPointer = operands[1];
      if (areProvablyDistinctPointers(previousPointer, currentPointer) &&
          semanticKey(previousPointer, orderingKeys) >
              semanticKey(currentPointer, orderingKeys)) {
        SmallVector<TensorSemanticExpr, 4> innerOperands = {previous[0]};
        innerOperands.append(operands.begin() + 1, operands.end());
        TensorSemanticExpr inner = TensorSemanticExpr::operation(
            TensorSemanticKind::MemoryWrite, expression.getType(),
            innerOperands, expression.getPayload());
        SmallVector<TensorSemanticExpr, 4> outerOperands = {inner};
        outerOperands.append(previous.begin() + 1, previous.end());
        normalized = TensorSemanticExpr::operation(
            TensorSemanticKind::MemoryWrite, expression.getType(),
            outerOperands, operands[0].getPayload());
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::Select &&
             operands.size() == 3) {
    const bool hasUnobservedArm =
        containsSemanticKind(operands[1], TensorSemanticKind::Unobserved) ||
        containsSemanticKind(operands[2], TensorSemanticKind::Unobserved);
    TensorSemanticExpr guardedTrue =
        simplifyUnderTrueCondition(operands[1], operands[0]);
    TensorSemanticExpr guardedFalse =
        simplifyUnderFalseCondition(operands[2], operands[0]);
    if (!guardedTrue.structurallyEquals(operands[1]) ||
        !guardedFalse.structurallyEquals(operands[2]))
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {operands[0], guardedTrue, guardedFalse});
    else if (std::optional<APInt> condition =
            getIntegerSemanticValue(operands[0]))
      normalized = condition->isZero() ? operands[2] : operands[1];
    else if (operands[1].structurallyEquals(operands[2]))
      normalized = operands[1];
    else if (operands[1].getKind() == TensorSemanticKind::Select &&
             operands[1].getOperands().size() == 3 &&
             operands[1].getOperands()[0].structurallyEquals(operands[0]))
      normalized = makeOperation(
          TensorSemanticKind::Select, expression.getType(),
          {operands[0], operands[1].getOperands()[1], operands[2]});
    else if (operands[1].getKind() == TensorSemanticKind::ZeroExtend &&
             operands[2].getKind() == TensorSemanticKind::ZeroExtend &&
             operands[1].getOperands().size() == 1 &&
             operands[2].getOperands().size() == 1 &&
             operands[1].getOperands()[0].getType() ==
                 operands[2].getOperands()[0].getType()) {
      Type narrowType = operands[1].getOperands()[0].getType();
      TensorSemanticExpr selected = makeOperation(
          TensorSemanticKind::Select, narrowType,
          {operands[0], operands[1].getOperands()[0],
           operands[2].getOperands()[0]});
      normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                 expression.getType(), {selected});
    } else if (!hasUnobservedArm && expression.getType().isInteger(1) &&
             isSemanticOne(operands[2])) {
      TensorSemanticExpr logicalNot = makeOperation(
          TensorSemanticKind::Not, operands[0].getType(), {operands[0]});
      normalized = makeOperation(TensorSemanticKind::Or,
                                 expression.getType(),
                                 {logicalNot, operands[1]});
    } else if (!hasUnobservedArm && expression.getType().isInteger(1) &&
               isSemanticZero(operands[2])) {
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(),
                                 {operands[0], operands[1]});
    } else if (!hasUnobservedArm && expression.getType().isInteger(1) &&
               isSemanticOne(operands[1])) {
      normalized = makeOperation(TensorSemanticKind::Or,
                                 expression.getType(),
                                 {operands[0], operands[2]});
    } else if (!hasUnobservedArm && expression.getType().isInteger(1) &&
               isSemanticZero(operands[1])) {
      TensorSemanticExpr logicalNot = makeOperation(
          TensorSemanticKind::Not, operands[0].getType(), {operands[0]});
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(),
                                 {logicalNot, operands[2]});
    } else if (isSemanticZero(operands[2]) &&
             operands[1].getKind() == TensorSemanticKind::Select &&
             operands[1].getOperands().size() == 3 &&
             isSemanticOne(operands[1].getOperands()[1]) &&
             isSemanticZero(operands[1].getOperands()[2])) {
      TensorSemanticExpr guard = makeOperation(
          TensorSemanticKind::And, operands[0].getType(),
          {operands[0], operands[1].getOperands()[0]});
      normalized = makeOperation(
          TensorSemanticKind::Select, expression.getType(),
          {guard, operands[1].getOperands()[1], operands[2]});
    } else if (isSemanticZero(operands[2]) &&
               operands[1].getKind() == TensorSemanticKind::Sum &&
               operands[1].getOperands().size() == 4 &&
               !containsSemanticExpr(operands[0],
                                     operands[1].getOperands()[0])) {
      ArrayRef<TensorSemanticExpr> sum = operands[1].getOperands();
      TensorSemanticExpr guardedBody = makeOperation(
          TensorSemanticKind::Select, sum[3].getType(),
          {operands[0], sum[3], makeInteger(0, sum[3].getType())});
      normalized = makeOperation(TensorSemanticKind::Sum,
                                 expression.getType(),
                                 {sum[0], sum[1], sum[2], guardedBody});
    } else if (operands[0].getKind() == TensorSemanticKind::Select &&
               operands[0].getOperands().size() == 3) {
      ArrayRef<TensorSemanticExpr> condition = operands[0].getOperands();
      TensorSemanticExpr whenTrue = makeOperation(
          TensorSemanticKind::Select, expression.getType(),
          {condition[1], operands[1], operands[2]});
      TensorSemanticExpr whenFalse = makeOperation(
          TensorSemanticKind::Select, expression.getType(),
          {condition[2], operands[1], operands[2]});
      normalized = makeOperation(TensorSemanticKind::Select,
                                 expression.getType(),
                                 {condition[0], whenTrue, whenFalse});
    }
  } else if (expression.getKind() == TensorSemanticKind::ApplyUnary &&
             operands.size() == 1) {
    auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
    auto role = payload ? dyn_cast_or_null<StringAttr>(
                              payload.get("semantic_role"))
                        : StringAttr{};
    if (role && role.getValue() == "shift_amount_valid") {
      std::optional<APInt> amount = getIntegerSemanticValue(operands.front());
      auto amountType = dyn_cast<IntegerType>(operands.front().getType());
      if (amount && amountType)
        normalized = makeBoolean(amount->getLimitedValue() <
                                 amountType.getWidth());
      else if (operands.front().getType().isInteger(1))
        // A one-bit shift is defined iff its unsigned amount is below one,
        // which is exactly !amount.  Keeping this in the semantic algebra
        // also canonicalizes validity against ordinary Boolean predicates.
        normalized = makeOperation(TensorSemanticKind::Not,
                                   expression.getType(), operands);
    }
  } else if (expression.getKind() == TensorSemanticKind::ShiftLeft &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto integerType = dyn_cast<IntegerType>(expression.getType());
    if (lhs && lhs->isZero()) {
      // Invalid shift amounts remain unobservable through the independent
      // validity denotation, so the value component may use zero here.
      normalized = makeInteger(0, expression.getType());
    } else if (rhs && rhs->isZero()) {
      normalized = narrowIntegerTo(operands[0], expression.getType());
    } else if (lhs && rhs && integerType &&
               rhs->getLimitedValue() < integerType.getWidth()) {
      APInt value = lhs->zextOrTrunc(integerType.getWidth())
                        .shl(rhs->getLimitedValue());
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (rhs && integerType &&
               rhs->getLimitedValue() < integerType.getWidth()) {
      APInt factor(integerType.getWidth(), 1);
      factor <<= rhs->getLimitedValue();
      TensorSemanticExpr multiplier =
          makeInteger(factor.getSExtValue(), expression.getType());
      normalized = makeOperation(TensorSemanticKind::Multiply,
                                 expression.getType(),
                                 {operands[0], multiplier});
    }
  } else if ((expression.getKind() == TensorSemanticKind::ShiftRightLogical ||
              expression.getKind() ==
                  TensorSemanticKind::ShiftRightArithmetic) &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto integerType = dyn_cast<IntegerType>(expression.getType());
    if (lhs && lhs->isZero()) {
      normalized = makeInteger(0, expression.getType());
    } else if (rhs && rhs->isZero()) {
      // Both logical and arithmetic shifts have zero as a right identity.
      // Definedness is handled independently by shift_amount_valid.
      normalized = narrowIntegerTo(operands[0], expression.getType());
    } else if (lhs && rhs && integerType && rhs->getZExtValue() <
                                                integerType.getWidth()) {
      APInt value = expression.getKind() ==
                            TensorSemanticKind::ShiftRightLogical
                        ? lhs->lshr(rhs->getZExtValue())
                        : lhs->ashr(rhs->getZExtValue());
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (expression.getKind() ==
                   TensorSemanticKind::ShiftRightArithmetic &&
               lhs && !lhs->isNegative()) {
      // Arithmetic and logical right shift differ only in the replicated sign
      // bit. A non-negative constant has sign bit zero, so both operators are
      // identical for every shift amount on which the shared validity
      // predicate is true.
      normalized = makeOperation(TensorSemanticKind::ShiftRightLogical,
                                 expression.getType(), operands);
    } else if (operands[0].getKind() == TensorSemanticKind::ZeroExtend &&
               operands[0].getOperands().size() == 1 && rhs) {
      Type narrowType = operands[0].getOperands()[0].getType();
      auto narrowInteger = dyn_cast<IntegerType>(narrowType);
      if (narrowInteger && rhs->getZExtValue() < narrowInteger.getWidth()) {
        TensorSemanticExpr amount = makeInteger(
            rhs->trunc(narrowInteger.getWidth()).getSExtValue(), narrowType);
        TensorSemanticExpr shifted = makeOperation(
            TensorSemanticKind::ShiftRightLogical, narrowType,
            {operands[0].getOperands()[0], amount});
        normalized = makeOperation(TensorSemanticKind::ZeroExtend,
                                   expression.getType(), {shifted});
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::Product &&
             operands.size() == 4 &&
             (isEmptyConstantRange(operands[1], operands[2]) ||
              isSemanticOne(operands[3]))) {
    // One is the multiplicative identity for an empty product and for every
    // finite product whose body is identically one.
    normalized = makeInteger(1, expression.getType());
  } else if ((expression.getKind() == TensorSemanticKind::Sum ||
              expression.getKind() == TensorSemanticKind::Product) &&
             operands.size() == 4 &&
             getSmallConstantRangeLength(operands[1], operands[2]) &&
             getIntegerSemanticValue(operands[3])) {
    std::optional<uint64_t> length =
        getSmallConstantRangeLength(operands[1], operands[2]);
    std::optional<APInt> body = getIntegerSemanticValue(operands[3]);
    if (length && body) {
      APInt value = expression.getKind() == TensorSemanticKind::Sum
                        ? APInt(body->getBitWidth(), 0)
                        : APInt(body->getBitWidth(), 1);
      for (uint64_t index = 0; index < *length; ++index)
        value = expression.getKind() == TensorSemanticKind::Sum
                    ? value + *body
                    : value * *body;
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    }
  } else if ((expression.getKind() == TensorSemanticKind::Sum ||
              expression.getKind() == TensorSemanticKind::Product ||
              expression.getKind() == TensorSemanticKind::ForAll) &&
             operands.size() == 4 &&
             isSingletonConstantRange(operands[1], operands[2])) {
    // A singleton finite binder evaluates its body exactly once. This generic
    // law applies to every supported fold kind and avoids leaving semantically
    // identical scalar expressions behind opaque Sum/Product/ForAll nodes.
    normalized =
        substituteSemanticExpr(operands[3], operands[0], operands[1]);
  } else if (expression.getKind() == TensorSemanticKind::Sum &&
             operands.size() == 4 &&
             (isEmptyConstantRange(operands[1], operands[2]) ||
              isSemanticZero(operands[3]))) {
    // Empty finite ranges and sums of zero are both the additive identity.
    normalized = makeInteger(0, expression.getType());
  } else if (expression.getKind() == TensorSemanticKind::Sum &&
             operands.size() == 4 &&
             extractGuardedSummand(operands[3]).has_value()) {
    // Kronecker-delta elimination:
    //   Sum_j select(k == j + c, f(j), 0) = f(k - c).
    // The equality must have one unit-affine solution. If that solution falls
    // outside the finite reduction range at a boundary, f must independently
    // normalize to zero there. This is a generic guarded-sum identity; it does
    // not know about scan, diagonal masks, or any TT operation.
    GuardedSummand selected = *extractGuardedSummand(operands[3]);
    std::optional<DeltaSelection> delta =
        solveDeltaCondition(selected.condition, operands[0], resultIndices);
    std::optional<int64_t> lower = getSignedSemanticInteger(operands[1]);
    std::optional<int64_t> upper = getSignedSemanticInteger(operands[2]);
    if (delta && delta->resultIndex && lower && upper &&
        delta->resultDimension < resultShape.size() &&
        !ShapedType::isDynamic(resultShape[delta->resultDimension]) &&
        *lower == 0 && *upper == resultShape[delta->resultDimension] &&
        delta->resultOffset >= -8 && delta->resultOffset <= 8) {
      TensorSemanticExpr selectedBody = substituteSemanticExpr(
          selected.value, operands[0], delta->solution);
      const int64_t extent = resultShape[delta->resultDimension];
      SmallVector<int64_t, 8> outsidePoints;
      if (delta->resultOffset < 0) {
        const int64_t count =
            std::min<int64_t>(extent, -delta->resultOffset);
        for (int64_t point = 0; point < count; ++point)
          outsidePoints.push_back(point);
      } else if (delta->resultOffset > 0) {
        const int64_t first =
            std::max<int64_t>(0, extent - delta->resultOffset);
        for (int64_t point = first; point < extent; ++point)
          outsidePoints.push_back(point);
      }
      bool zeroOutside = outsidePoints.size() <= 8;
      for (int64_t point : outsidePoints) {
        TensorSemanticExpr concreteIndex = makeInteger(
            point, delta->resultIndex.getType());
        TensorSemanticExpr concreteBody = substituteSemanticExpr(
            selectedBody, delta->resultIndex, concreteIndex);
        concreteBody = normalizeExpr(concreteBody, proof, resultShape,
                                     resultIndices, memo, orderingKeys);
        zeroOutside &= isSemanticZero(concreteBody);
      }
      if (zeroOutside)
        normalized = selectedBody;
    }
  } else if (expression.getKind() == TensorSemanticKind::Sum &&
             operands.size() == 4 &&
             operands[3].getKind() == TensorSemanticKind::Add &&
             operands[3].getOperands().size() == 2) {
    ArrayRef<TensorSemanticExpr> add = operands[3].getOperands();
    TensorSemanticExpr lhs = makeOperation(
        TensorSemanticKind::Sum, expression.getType(),
        {operands[0], operands[1], operands[2], add[0]});
    TensorSemanticExpr rhs = makeOperation(
        TensorSemanticKind::Sum, expression.getType(),
        {operands[0], operands[1], operands[2], add[1]});
    normalized = makeOperation(TensorSemanticKind::Add,
                               expression.getType(), {lhs, rhs});
  } else if (expression.getKind() == TensorSemanticKind::Sum &&
             operands.size() == 4 &&
             operands[3].getKind() == TensorSemanticKind::Select &&
             operands[3].getOperands().size() == 3 &&
             !isSemanticZero(operands[3].getOperands()[2])) {
    // Linearity over a partitioned domain:
    //   Sum_i select(c_i, x_i, y_i)
    //     = Sum_i select(c_i, x_i, 0)
    //       + Sum_i select(!c_i, y_i, 0).
    // Keeping this identity at the Sum boundary lets Equal first distribute
    // through Select, instead of prematurely encoding a tensor select as an
    // addition and obscuring its value semantics.
    ArrayRef<TensorSemanticExpr> selected = operands[3].getOperands();
    TensorSemanticExpr zero = makeInteger(0, selected[1].getType());
    TensorSemanticExpr logicalNot = makeOperation(
        TensorSemanticKind::Not, selected[0].getType(), {selected[0]});
    TensorSemanticExpr whenTrue = makeOperation(
        TensorSemanticKind::Select, selected[1].getType(),
        {selected[0], selected[1], zero});
    TensorSemanticExpr whenFalse = makeOperation(
        TensorSemanticKind::Select, selected[2].getType(),
        {logicalNot, selected[2], zero});
    TensorSemanticExpr lhs = makeOperation(
        TensorSemanticKind::Sum, expression.getType(),
        {operands[0], operands[1], operands[2], whenTrue});
    TensorSemanticExpr rhs = makeOperation(
        TensorSemanticKind::Sum, expression.getType(),
        {operands[0], operands[1], operands[2], whenFalse});
    normalized = makeOperation(TensorSemanticKind::Add,
                               expression.getType(), {lhs, rhs});
  } else if (expression.getKind() == TensorSemanticKind::ForAll &&
             operands.size() == 4) {
    if (isEmptyConstantRange(operands[1], operands[2])) {
      normalized = makeBoolean(true);
    } else if (std::optional<APInt> body =
            getIntegerSemanticValue(operands[3])) {
      normalized = makeBoolean(!body->isZero());
    } else if (operands[3].getKind() == TensorSemanticKind::And &&
               operands[3].getOperands().size() == 2) {
      ArrayRef<TensorSemanticExpr> conjunction =
          operands[3].getOperands();
      TensorSemanticExpr lhs = makeOperation(
          TensorSemanticKind::ForAll, expression.getType(),
          {operands[0], operands[1], operands[2], conjunction[0]});
      TensorSemanticExpr rhs = makeOperation(
          TensorSemanticKind::ForAll, expression.getType(),
          {operands[0], operands[1], operands[2], conjunction[1]});
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(), {lhs, rhs});
    }
  } else if (expression.getKind() == TensorSemanticKind::PointerAdd &&
             operands.size() == 2) {
    std::optional<APInt> offset = getIntegerSemanticValue(operands[1]);
    if (offset && offset->isZero()) {
      normalized = operands[0];
    } else if (offset) {
      Type offsetType = IntegerType::get(expression.getType().getContext(), 64);
      APInt canonicalValue = offset->zextOrTrunc(64);
      TensorSemanticExpr canonicalOffset = makeInteger(
          canonicalValue.getSExtValue(), offsetType);
      TensorSemanticExpr base = operands[0];
      if (base.getKind() == TensorSemanticKind::PointerAdd &&
          base.getOperands().size() == 2) {
        if (std::optional<APInt> innerOffset =
                getIntegerSemanticValue(base.getOperands()[1])) {
          canonicalValue += innerOffset->zextOrTrunc(64);
          canonicalOffset = makeInteger(canonicalValue.getSExtValue(),
                                        offsetType);
          base = base.getOperands()[0];
        }
      }
      normalized = makeOperation(TensorSemanticKind::PointerAdd,
                                 expression.getType(),
                                 {base, canonicalOffset});
    }
  } else if (expression.getKind() == TensorSemanticKind::Add &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto integerType = dyn_cast<IntegerType>(expression.getType());
    if (integerType && integerType.getWidth() == 1) {
      // Addition modulo 2 is exactly Boolean xor. Reusing the Boolean normal
      // form proves identities such as x+x=0 without an operation-specific
      // graph rewrite.
      SmallVector<TensorSemanticExpr, 2> lowBits = {
          narrowIntegerTo(operands[0], expression.getType()),
          narrowIntegerTo(operands[1], expression.getType())};
      normalized = makeOperation(TensorSemanticKind::Xor,
                                 expression.getType(), lowBits);
    } else if (lhs && rhs) {
      APInt value = *lhs + *rhs;
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (isSemanticZero(operands[0]))
      normalized = operands[1];
    else if (isSemanticZero(operands[1]))
      normalized = operands[0];
    else if (integerType && integerType.getWidth() > 1 &&
             operands[0].structurallyEquals(operands[1])) {
      TensorSemanticExpr two = makeInteger(2, expression.getType());
      normalized = makeOperation(TensorSemanticKind::Multiply,
                                 expression.getType(), {operands[0], two});
    }
    else if (std::optional<std::pair<TensorSemanticExpr, int64_t>> split =
                 splitAddConstant(normalized)) {
      TensorSemanticExpr base = split->first;
      int64_t constant = split->second;
      if (base.getKind() == TensorSemanticKind::Subtract &&
          base.getOperands().size() == 2) {
        std::optional<int64_t> subtrahend =
            getSignedSemanticInteger(base.getOperands()[1]);
        if (subtrahend && *subtrahend == constant)
          normalized = base.getOperands()[0];
      } else if (std::optional<std::pair<TensorSemanticExpr, int64_t>>
                     nested = splitAddConstant(base)) {
        normalized = addSemanticOffset(nested->first,
                                       nested->second + constant);
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::Multiply &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto resultType = dyn_cast<IntegerType>(expression.getType());
    auto lhsInteger = dyn_cast<IntegerType>(operands[0].getType());
    auto rhsInteger = dyn_cast<IntegerType>(operands[1].getType());
    if (resultType && resultType.getWidth() == 1 && lhsInteger && rhsInteger) {
      // A one-bit product depends only on the parity of each operand. Narrow
      // wider semantic operands to their low bit, then reuse the Boolean And
      // normal form. This covers x*x=x and annihilation by every even value
      // without knowing which TT operation produced the expression.
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(),
                                 {narrowIntegerTo(operands[0],
                                                  expression.getType()),
                                  narrowIntegerTo(operands[1],
                                                  expression.getType())});
    } else if (lhs && rhs) {
      APInt value = *lhs * *rhs;
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (isSemanticZero(operands[0]) || isSemanticZero(operands[1])) {
      normalized = makeInteger(0, expression.getType());
    } else if (isSemanticOne(operands[0])) {
      normalized = operands[1];
    } else if (isSemanticOne(operands[1])) {
      normalized = operands[0];
    } else if (lhs && lhs->isAllOnes()) {
      TensorSemanticExpr zero = makeInteger(0, expression.getType());
      normalized = makeOperation(TensorSemanticKind::Subtract,
                                 expression.getType(), {zero, operands[1]});
    } else if (rhs && rhs->isAllOnes()) {
      TensorSemanticExpr zero = makeInteger(0, expression.getType());
      normalized = makeOperation(TensorSemanticKind::Subtract,
                                 expression.getType(), {zero, operands[0]});
    } else if (operands[0].getKind() == TensorSemanticKind::Subtract &&
               operands[1].getKind() == TensorSemanticKind::Subtract &&
               operands[0].getOperands().size() == 2 &&
               operands[1].getOperands().size() == 2 &&
               isSemanticZero(operands[0].getOperands()[0]) &&
               isSemanticZero(operands[1].getOperands()[0])) {
      // (-a) * (-b) == a * b in every fixed-width integer ring.
      normalized = makeOperation(
          TensorSemanticKind::Multiply, expression.getType(),
          {operands[0].getOperands()[1], operands[1].getOperands()[1]});
    }
  } else if ((expression.getKind() ==
                  TensorSemanticKind::SignedMultiplyHigh ||
              expression.getKind() ==
                  TensorSemanticKind::UnsignedMultiplyHigh) &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    auto resultType = dyn_cast<IntegerType>(expression.getType());
    if (lhs && rhs && resultType && resultType.getWidth() <= 64) {
      unsigned width = resultType.getWidth();
      APInt wideLhs = expression.getKind() ==
                              TensorSemanticKind::SignedMultiplyHigh
                          ? lhs->sext(width * 2)
                          : lhs->zext(width * 2);
      APInt wideRhs = expression.getKind() ==
                              TensorSemanticKind::SignedMultiplyHigh
                          ? rhs->sext(width * 2)
                          : rhs->zext(width * 2);
      APInt high = (wideLhs * wideRhs).lshr(width).trunc(width);
      normalized = makeInteger(high.getSExtValue(), expression.getType());
    } else if (isSemanticZero(operands[0]) ||
               isSemanticZero(operands[1])) {
      normalized = makeInteger(0, expression.getType());
    } else if (resultType && resultType.getWidth() == 1) {
      // The product of two one-bit integers fits in the low bit for both
      // signed {-1,0} and unsigned {0,1} interpretations.
      normalized = makeInteger(0, expression.getType());
    } else if (expression.getKind() ==
                   TensorSemanticKind::UnsignedMultiplyHigh &&
               (isSemanticOne(operands[0]) ||
                isSemanticOne(operands[1]))) {
      // For an unsigned w-bit operand x, x * 1 fits in exactly w bits, so the
      // upper half of the 2w-bit product is zero.
      normalized = makeInteger(0, expression.getType());
    } else if (resultType && resultType.getWidth() > 1) {
      const TensorSemanticExpr *value = nullptr;
      bool factorIsTwo = false;
      if (isSemanticOne(operands[0]) || isSemanticTwo(operands[0])) {
        value = &operands[1];
        factorIsTwo = isSemanticTwo(operands[0]);
      } else if (isSemanticOne(operands[1]) ||
                 isSemanticTwo(operands[1])) {
        value = &operands[0];
        factorIsTwo = isSemanticTwo(operands[1]);
      }
      if (value &&
          (expression.getKind() == TensorSemanticKind::SignedMultiplyHigh ||
           factorIsTwo)) {
        Type i1 = IntegerType::get(expression.getType().getContext(), 1);
        TensorSemanticExpr zero = makeInteger(0, value->getType());
        TensorSemanticExpr signBit = makeOperation(
            TensorSemanticKind::SignedLessThan, i1, {*value, zero});
        normalized = makeOperation(
            expression.getKind() == TensorSemanticKind::SignedMultiplyHigh
                ? TensorSemanticKind::SignExtend
                : TensorSemanticKind::ZeroExtend,
            expression.getType(), {signBit});
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::IntegerDivide &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (lhs && rhs && !rhs->isZero())
      normalized = makeInteger(lhs->udiv(*rhs).getSExtValue(),
                               expression.getType());
    else if (isSemanticOne(operands[1]))
      normalized = operands[0];
    else if (rhs && rhs->isPowerOf2()) {
      TensorSemanticExpr amount =
          makeInteger(rhs->exactLogBase2(), operands[1].getType());
      normalized = makeOperation(TensorSemanticKind::ShiftRightLogical,
                                 expression.getType(),
                                 {operands[0], amount});
    }
  } else if (expression.getKind() == TensorSemanticKind::SignedRemainder &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (lhs && rhs && !rhs->isZero()) {
      APInt value = lhs->srem(*rhs);
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (isSemanticOne(operands[1]))
      normalized = makeInteger(0, expression.getType());
  } else if (expression.getKind() == TensorSemanticKind::SignedLessThan &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (operands[0].structurallyEquals(operands[1]))
      normalized = makeBoolean(false);
    else if (lhs && rhs)
      normalized = makeBoolean(lhs->slt(*rhs));
    else if (operands[0].getType().isInteger(1) && lhs)
      // Signed i1 has only {-1, 0}: 0 < x is false, while -1 < x is !x.
      normalized = lhs->isZero()
                       ? makeBoolean(false)
                       : makeOperation(TensorSemanticKind::Not,
                                       expression.getType(), {operands[1]});
    else if (operands[0].getType().isInteger(1) && rhs)
      // x < 0 is x for signed i1; x < -1 is always false.
      normalized = rhs->isZero() ? operands[0] : makeBoolean(false);
    else if (operands[0].getType().isInteger(1)) {
      // Signed i1 contains {0, -1}.  Therefore x < y is true exactly for
      // x=-1 and y=0, i.e. x && !y in the one-bit Boolean representation.
      TensorSemanticExpr notRhs = makeOperation(
          TensorSemanticKind::Not, expression.getType(), {operands[1]});
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(), {operands[0], notRhs});
    }
  } else if (expression.getKind() == TensorSemanticKind::UnsignedLessThan &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (operands[0].structurallyEquals(operands[1]))
      normalized = makeBoolean(false);
    else if (lhs && rhs)
      normalized = makeBoolean(lhs->ult(*rhs));
    else if (rhs && rhs->isZero())
      normalized = makeBoolean(false);
    else if (operands[0].getType().isInteger(1) && lhs)
      normalized = lhs->isZero() ? operands[1] : makeBoolean(false);
    else if (operands[0].getType().isInteger(1) && rhs)
      normalized = rhs->isZero()
                       ? makeBoolean(false)
                       : makeOperation(TensorSemanticKind::Not,
                                       expression.getType(), {operands[0]});
    else if (operands[0].getType().isInteger(1)) {
      TensorSemanticExpr notLhs = makeOperation(
          TensorSemanticKind::Not, expression.getType(), {operands[0]});
      normalized = makeOperation(TensorSemanticKind::And,
                                 expression.getType(), {notLhs, operands[1]});
    } else if (operands[0].getKind() == TensorSemanticKind::Multiply &&
               operands[0].getOperands().size() == 2 &&
               ((operands[1].structurallyEquals(
                      operands[0].getOperands()[0]) &&
                 isSemanticTwo(operands[0].getOperands()[1])) ||
                (operands[1].structurallyEquals(
                      operands[0].getOperands()[1]) &&
                 isSemanticTwo(operands[0].getOperands()[0])))) {
      // (x * 2) <u x is exactly the high bit of x. Add(x,x) and x<<1 both
      // use this canonical multiplication form.
      TensorSemanticExpr zero = makeInteger(0, operands[1].getType());
      normalized = makeOperation(TensorSemanticKind::SignedLessThan,
                                 expression.getType(),
                                 {operands[1], zero});
    } else if (operands[0].getKind() == TensorSemanticKind::Add &&
               operands[0].getOperands().size() == 2 &&
               (operands[1].structurallyEquals(
                    operands[0].getOperands()[0]) ||
                operands[1].structurallyEquals(
                    operands[0].getOperands()[1]))) {
      ArrayRef<TensorSemanticExpr> add = operands[0].getOperands();
      if (add[0].structurallyEquals(add[1])) {
        TensorSemanticExpr zero = makeInteger(0, add[0].getType());
        normalized = makeOperation(TensorSemanticKind::SignedLessThan,
                                   expression.getType(), {add[0], zero});
      } else if (isSemanticOne(add[0]) || isSemanticOne(add[1])) {
        const TensorSemanticExpr &value =
            isSemanticOne(add[0]) ? add[1] : add[0];
        TensorSemanticExpr allOnes = makeInteger(-1, value.getType());
        normalized = makeOperation(TensorSemanticKind::Equal,
                                   expression.getType(),
                                   {value, allOnes});
      } else {
        // In unsigned modular arithmetic, sum < a and sum < b are equivalent
        // carry predicates for sum=a+b. Canonicalize both spellings to the
        // first already-sorted Add operand.
        normalized = makeOperation(
            TensorSemanticKind::UnsignedLessThan, expression.getType(),
            {operands[0], add[0]});
      }
    }
  } else if (expression.getKind() == TensorSemanticKind::Subtract &&
             operands.size() == 2) {
    std::optional<APInt> lhs = getIntegerSemanticValue(operands[0]);
    std::optional<APInt> rhs = getIntegerSemanticValue(operands[1]);
    if (expression.getType().isInteger(1)) {
      // In the one-bit ring, subtraction and addition are both xor because
      // negation is the identity modulo 2.
      SmallVector<TensorSemanticExpr, 2> lowBits = {
          narrowIntegerTo(operands[0], expression.getType()),
          narrowIntegerTo(operands[1], expression.getType())};
      normalized = makeOperation(TensorSemanticKind::Xor,
                                 expression.getType(), lowBits);
    } else if (lhs && rhs) {
      APInt value = *lhs - *rhs;
      normalized = makeInteger(value.getSExtValue(), expression.getType());
    } else if (isSemanticZero(operands[1])) {
      // Bit-vector subtraction has zero as a right identity at every width.
      // This does not require signedness or an overflow promise.
      normalized = operands[0];
    } else if (operands[0].structurallyEquals(operands[1])) {
      normalized = makeInteger(0, expression.getType());
    } else if (lhs && matchBitwiseNotOperand(operands[1])) {
      std::optional<TensorSemanticExpr> inverted =
          matchBitwiseNotOperand(operands[1]);
      // c - (~x) == x + (c + 1) modulo 2^w.
      APInt offset = *lhs + 1;
      TensorSemanticExpr constant =
          makeInteger(offset.getSExtValue(), expression.getType());
      normalized = makeOperation(TensorSemanticKind::Add,
                                 expression.getType(),
                                 {*inverted, constant});
    } else if (lhs && lhs->isAllOnes()) {
      // In a fixed-width bit-vector ring, -1 - x is exactly bitwise-not x,
      // represented canonically as x xor -1.
      normalized = makeOperation(TensorSemanticKind::BitwiseXor,
                                 expression.getType(), operands);
    } else if (rhs) {
      // Convert subtraction by a constant into addition of its modular
      // additive inverse. Commutative Add canonicalization then unifies
      // x - 1 with x + (-1) at every integer width.
      APInt inverse = -(*rhs);
      TensorSemanticExpr constant =
          makeInteger(inverse.getSExtValue(), expression.getType());
      normalized = makeOperation(TensorSemanticKind::Add,
                                 expression.getType(),
                                 {operands[0], constant});
    } else if (operands[0].getKind() == TensorSemanticKind::Sum &&
               operands[0].getOperands().size() == 4) {
      ArrayRef<TensorSemanticExpr> sum = operands[0].getOperands();
      std::optional<std::pair<TensorSemanticExpr, int64_t>> upper =
          splitAddConstant(sum[2]);
      if (upper && upper->second == 1) {
        TensorSemanticExpr endpoint = substituteSemanticExpr(
            sum[3], sum[0], upper->first);
        endpoint = normalizeExpr(endpoint, proof, resultShape,
                                 resultIndices, memo, orderingKeys);
        if (endpoint.structurallyEquals(operands[1]))
          normalized = makeOperation(
              TensorSemanticKind::Sum, expression.getType(),
              {sum[0], sum[1], upper->first, sum[3]});
      }
      if (normalized.structurallyEquals(expression)) {
        TensorSemanticExpr endpoint = substituteSemanticExpr(
            sum[3], sum[0], sum[1]);
        endpoint = normalizeExpr(endpoint, proof, resultShape,
                                 resultIndices, memo, orderingKeys);
        if (endpoint.structurallyEquals(operands[1]))
          normalized = makeOperation(
              TensorSemanticKind::Sum, expression.getType(),
              {sum[0], addSemanticOffset(sum[1], 1), sum[2], sum[3]});
      }
    } else if (operands[0].getKind() == TensorSemanticKind::Add &&
               operands[0].getOperands().size() == 2) {
      ArrayRef<TensorSemanticExpr> add = operands[0].getOperands();
      if (add[0].structurallyEquals(operands[1]))
        normalized = add[1];
      else if (add[1].structurallyEquals(operands[1]))
        normalized = add[0];
    }
  }

  // Canonicalize a pure pointwise binary function over Select by pushing the
  // function into the selected arms.  This is the generic ITE congruence law,
  // not an optimization-specific rewrite.  Two Select operands are combined
  // only when they share the same condition; expanding different conditions
  // here would form a Cartesian product and make normalization exponential.
  // Lazy Unobserved arms remain untouched because evaluating an operation in
  // such an arm would erase the distinction between bottom and a real value.
  if (isSelectCongruentBinarySemanticKind(normalized.getKind()) &&
      normalized.getOperands().size() == 2) {
    ArrayRef<TensorSemanticExpr> binary = normalized.getOperands();
    const bool lhsSelect = binary[0].getKind() == TensorSemanticKind::Select &&
                           binary[0].getOperands().size() == 3;
    const bool rhsSelect = binary[1].getKind() == TensorSemanticKind::Select &&
                           binary[1].getOperands().size() == 3;
    auto hasUnobservedArm = [](const TensorSemanticExpr &selected) {
      ArrayRef<TensorSemanticExpr> arms = selected.getOperands();
      return containsSemanticKind(arms[1], TensorSemanticKind::Unobserved) ||
             containsSemanticKind(arms[2], TensorSemanticKind::Unobserved);
    };
    auto rebuildBinary = [&](const TensorSemanticExpr &lhs,
                             const TensorSemanticExpr &rhs) {
      return TensorSemanticExpr::operation(
          normalized.getKind(), normalized.getType(), {lhs, rhs},
          normalized.getPayload());
    };

    if (lhsSelect && rhsSelect &&
        !hasUnobservedArm(binary[0]) && !hasUnobservedArm(binary[1])) {
      ArrayRef<TensorSemanticExpr> lhs = binary[0].getOperands();
      ArrayRef<TensorSemanticExpr> rhs = binary[1].getOperands();
      if (lhs[0].structurallyEquals(rhs[0]))
        normalized = makeOperation(
            TensorSemanticKind::Select, normalized.getType(),
            {lhs[0], rebuildBinary(lhs[1], rhs[1]),
             rebuildBinary(lhs[2], rhs[2])});
      else if (lhs[1].structurallyEquals(binary[1]) ||
               lhs[2].structurallyEquals(binary[1]))
        normalized = makeOperation(
            TensorSemanticKind::Select, normalized.getType(),
            {lhs[0], rebuildBinary(lhs[1], binary[1]),
             rebuildBinary(lhs[2], binary[1])});
      else if (rhs[1].structurallyEquals(binary[0]) ||
               rhs[2].structurallyEquals(binary[0]))
        normalized = makeOperation(
            TensorSemanticKind::Select, normalized.getType(),
            {rhs[0], rebuildBinary(binary[0], rhs[1]),
             rebuildBinary(binary[0], rhs[2])});
    } else if (lhsSelect && !rhsSelect &&
               !hasUnobservedArm(binary[0])) {
      ArrayRef<TensorSemanticExpr> selected = binary[0].getOperands();
      normalized = makeOperation(
          TensorSemanticKind::Select, normalized.getType(),
          {selected[0], rebuildBinary(selected[1], binary[1]),
           rebuildBinary(selected[2], binary[1])});
    } else if (!lhsSelect && rhsSelect &&
               !hasUnobservedArm(binary[1])) {
      ArrayRef<TensorSemanticExpr> selected = binary[1].getOperands();
      normalized = makeOperation(
          TensorSemanticKind::Select, normalized.getType(),
          {selected[0], rebuildBinary(binary[0], selected[1]),
           rebuildBinary(binary[0], selected[2])});
    }
  }

  if (!normalized.structurallyEquals(expression)) {
    proof.push_back(
        SemanticProofStep{"canonical-normalization", expression, normalized});
    // Rebuilding a parent around operands that have already reached normal
    // form does not expose another rewrite at that parent. Re-enter only when
    // a local algebraic rule changed the rebuilt expression; otherwise a deep
    // semantic DAG is walked once instead of once again for every ancestor.
    if (normalizedToClosedForm ||
        normalized.structurallyEquals(recursivelyNormalizedOperands))
      return normalized;
    return normalizeExpr(normalized, proof, resultShape, resultIndices, memo,
                         orderingKeys);
  }
  // Preserve the original DAG node when normalization made no structural
  // change.  Prefix replay keeps denotations for every operand e-class; deep
  // cloning an already-normal expression here turns shared semantic DAGs into
  // millions of private trees and dominates production memory.
  return expression;
}

SemanticNormalizationResult
TensorSemanticNormalizer::normalize(TensorDenotation input) const {
  workItems = 0;
  interrupted = false;
  SemanticNormalizationResult result;
  llvm::DenseMap<const void *, TensorSemanticExpr> memo;
  llvm::DenseMap<const void *, std::string> orderingKeys;
  result.denotation.shape = input.shape;
  result.denotation.shapeSymbols = input.shapeSymbols;
  result.denotation.layoutIdentity = input.layoutIdentity;
  for (const TensorSemanticExpr &index : input.resultIndices)
    result.denotation.resultIndices.push_back(
        normalizeExpr(index, result.proof, input.shape,
                      input.resultIndices, memo, orderingKeys));
  result.denotation.value = normalizeExpr(input.value, result.proof,
                                          input.shape, input.resultIndices,
                                          memo, orderingKeys);
  result.denotation.defined = normalizeExpr(input.defined, result.proof,
                                            input.shape, input.resultIndices,
                                            memo, orderingKeys);
  if (input.effect)
    result.denotation.effect = normalizeExpr(input.effect, result.proof,
                                             input.shape, input.resultIndices,
                                             memo, orderingKeys);
  if (input.memory)
    result.denotation.memory = normalizeExpr(input.memory, result.proof,
                                             input.shape, input.resultIndices,
                                             memo, orderingKeys);
  if (input.memoryDefined)
    result.denotation.memoryDefined = normalizeExpr(
        input.memoryDefined, result.proof, input.shape, input.resultIndices,
        memo, orderingKeys);
  result.denotation.isMemoryStateToken = input.isMemoryStateToken;
  if (input.launch)
    result.denotation.launch = normalizeExpr(input.launch, result.proof,
                                             input.shape, input.resultIndices,
                                             memo, orderingKeys);
  if (input.launchDefined)
    result.denotation.launchDefined = normalizeExpr(
        input.launchDefined, result.proof, input.shape, input.resultIndices,
        memo, orderingKeys);
  result.denotation.isLaunchStateToken = input.isLaunchStateToken;
  if (input.buffer)
    result.denotation.buffer = normalizeExpr(input.buffer, result.proof,
                                             input.shape, input.resultIndices,
                                             memo, orderingKeys);
  if (input.bufferDefined)
    result.denotation.bufferDefined = normalizeExpr(
        input.bufferDefined, result.proof, input.shape, input.resultIndices,
        memo, orderingKeys);
  result.denotation.isBufferStateToken = input.isBufferStateToken;

  auto simplifyOnDefinedDomain =
      [&](TensorSemanticExpr &value, const TensorSemanticExpr &validity) {
        // Unobserved is the explicit bottom value of a lazy branch.  Retain
        // that branch structure instead of replacing it from an outer
        // validity premise; doing so would make normalization eager.
        if (!value || !validity || isSemanticZero(validity) ||
            isSemanticOne(validity) ||
            containsSemanticKind(value, TensorSemanticKind::Unobserved))
          return;
        TensorSemanticExpr previous = value;
        TensorSemanticExpr simplified =
            simplifyUnderTrueCondition(value, validity);
        if (simplified.structurallyEquals(previous))
          return;
        value = normalizeExpr(simplified, result.proof, input.shape,
                              input.resultIndices, memo, orderingKeys);
        result.proof.push_back(SemanticProofStep{
            "defined-domain-simplification", previous, value});
      };
  // TensorDenotation is a partial function: a value/state is observed only
  // where its corresponding validity predicate is true. Canonicalize under
  // that premise before comparing two denotations, rather than requiring
  // arbitrary values outside the defined domain to match.
  simplifyOnDefinedDomain(result.denotation.value,
                          result.denotation.defined);
  simplifyOnDefinedDomain(result.denotation.memory,
                          result.denotation.memoryDefined);
  simplifyOnDefinedDomain(result.denotation.launch,
                          result.denotation.launchDefined);
  simplifyOnDefinedDomain(result.denotation.buffer,
                          result.denotation.bufferDefined);

  // Effects form a conjunction/sequence whose leaf for the current memory
  // operation is the corresponding memory transition with the incoming state
  // replaced by the effect identity `true`.  If the current operation is
  // statically undefined, that leaf is unobservable, but already-evaluated
  // operand effects must remain.  Remove exactly the root leaf rather than
  // dropping the complete effect tree.
  if (isSemanticZero(result.denotation.defined) &&
      result.denotation.effect && result.denotation.memory &&
      (result.denotation.memory.getKind() ==
           TensorSemanticKind::MemoryWrite ||
       result.denotation.memory.getKind() ==
           TensorSemanticKind::MemoryAtomic) &&
      !result.denotation.memory.getOperands().empty()) {
    SmallVector<TensorSemanticExpr, 4> effectOperands =
        llvm::to_vector(result.denotation.memory.getOperands());
    effectOperands[0] = TensorSemanticExpr::boolean(
        result.denotation.memory.getType().getContext(), true);
    TensorSemanticExpr rootEffect = TensorSemanticExpr::operation(
        result.denotation.memory.getKind(),
        result.denotation.memory.getType(), effectOperands,
        result.denotation.memory.getPayload());
    bool removed = false;
    std::function<TensorSemanticExpr(const TensorSemanticExpr &)> strip =
        [&](const TensorSemanticExpr &expression) -> TensorSemanticExpr {
      if (!removed && expression.structurallyEquals(rootEffect)) {
        removed = true;
        return TensorSemanticExpr::boolean(expression.getType().getContext(),
                                           true);
      }
      if (expression.getKind() != TensorSemanticKind::And ||
          expression.getOperands().size() != 2)
        return expression;
      TensorSemanticExpr lhs = strip(expression.getOperands()[0]);
      TensorSemanticExpr rhs = strip(expression.getOperands()[1]);
      return normalizeExpr(
          TensorSemanticExpr::operation(TensorSemanticKind::And,
                                        expression.getType(), {lhs, rhs}),
          result.proof, input.shape, input.resultIndices, memo,
          orderingKeys);
    };
    TensorSemanticExpr previous = result.denotation.effect;
    TensorSemanticExpr stripped = strip(previous);
    if (removed && !previous.structurallyEquals(stripped)) {
      result.proof.push_back(SemanticProofStep{
          "undefined-root-effect-elision", previous, stripped});
      result.denotation.effect = std::move(stripped);
    }
  }

  auto canonicalizeUnobservable =
      [&](TensorSemanticExpr &value, const TensorSemanticExpr &validity) {
        if (!value || !validity || !isSemanticZero(validity))
          return;
        TensorSemanticExpr unobserved = TensorSemanticExpr::operation(
            TensorSemanticKind::Unobserved, value.getType(), {});
        if (value.structurallyEquals(unobserved))
          return;
        result.proof.push_back(SemanticProofStep{
            "undefined-domain-canonicalization", value, unobserved});
        value = std::move(unobserved);
      };
  // A denotation is a partial function.  Once its validity predicate is
  // statically false, the associated value/state has no observable meaning;
  // canonicalize every such bottom value to the same representation.
  canonicalizeUnobservable(result.denotation.value,
                           result.denotation.defined);
  canonicalizeUnobservable(result.denotation.memory,
                           result.denotation.memoryDefined);
  canonicalizeUnobservable(result.denotation.launch,
                           result.denotation.launchDefined);
  canonicalizeUnobservable(result.denotation.buffer,
                           result.denotation.bufferDefined);
  return result;
}

bool TensorSemanticNormalizer::shouldInterrupt() const {
  if (!shouldStop || (workItems++ & 1023) != 0)
    return interrupted;
  if (!shouldStop())
    return false;
  interrupted = true;
  return true;
}

SemanticProofResult
SemanticEquivalenceOracle::compare(const TensorDenotation &lhs,
                                   const TensorDenotation &rhs) const {
  return compareImpl(lhs, rhs, /*inputsAlreadyNormalized=*/false,
                     /*allowFallback=*/true);
}

SemanticProofResult
SemanticEquivalenceOracle::comparePrimary(const TensorDenotation &lhs,
                                          const TensorDenotation &rhs) const {
  return compareImpl(lhs, rhs, /*inputsAlreadyNormalized=*/false,
                     /*allowFallback=*/false);
}

SemanticProofResult SemanticEquivalenceOracle::compareNormalized(
    const TensorDenotation &lhs, const TensorDenotation &rhs) const {
  return compareImpl(lhs, rhs, /*inputsAlreadyNormalized=*/true,
                     /*allowFallback=*/true);
}

SemanticProofResult SemanticEquivalenceOracle::comparePrimaryNormalized(
    const TensorDenotation &lhs, const TensorDenotation &rhs) const {
  return compareImpl(lhs, rhs, /*inputsAlreadyNormalized=*/true,
                     /*allowFallback=*/false);
}

SemanticProofResult SemanticEquivalenceOracle::compareImpl(
    const TensorDenotation &lhs, const TensorDenotation &rhs,
    bool inputsAlreadyNormalized, bool allowFallback) const {
  SemanticProofResult result;
  if (!lhs.isValid() || !rhs.isValid()) {
    result.reason = "invalid or missing denotation";
    return result;
  }
  if (lhs.shape != rhs.shape ||
      lhs.resultIndices.size() != rhs.resultIndices.size()) {
    result.status = SemanticProofStatus::Refuted;
    result.reason = "shape/rank mismatch";
    return result;
  }
  if (lhs.layoutIdentity != rhs.layoutIdentity) {
    result.status = SemanticProofStatus::Refuted;
    result.reason = "physical layout/encoding mismatch";
    return result;
  }
  for (auto [dimension, extent] : llvm::enumerate(lhs.shape)) {
    if (!ShapedType::isDynamic(extent))
      continue;
    if (lhs.shapeSymbols.size() != lhs.shape.size() ||
        rhs.shapeSymbols.size() != rhs.shape.size() ||
        lhs.shapeSymbols[dimension].empty() ||
        rhs.shapeSymbols[dimension].empty() ||
        lhs.shapeSymbols[dimension] == "?" ||
        rhs.shapeSymbols[dimension] == "?" ||
        lhs.shapeSymbols[dimension] == "!conflict" ||
        rhs.shapeSymbols[dimension] == "!conflict" ||
        lhs.shapeSymbols[dimension] != rhs.shapeSymbols[dimension]) {
      result.reason = "symbolic shape relation not proved";
      return result;
    }
  }

  // RuleDB compilation certifies every imported graph with a reflexive
  // proof. When both references are the same denotation, normalize once and
  // retain its normalization steps in the certificate.
  if (&lhs == &rhs) {
    SemanticNormalizationResult normalized = normalizer.normalize(lhs);
    SemanticEquivalenceProof proof;
    proof.verifiedDomains = {
        GraphSemanticDomain::Shape, GraphSemanticDomain::Index,
        GraphSemanticDomain::Layout, GraphSemanticDomain::Value,
        GraphSemanticDomain::Definedness};
    if (normalized.denotation.effect)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Effect);
    if (normalized.denotation.memory)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Memory);
    if (normalized.denotation.launch)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Launch);
    if (normalized.denotation.isBufferStateToken &&
        normalized.denotation.buffer)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Buffer);
    proof.lhs = normalized.denotation;
    proof.rhs = normalized.denotation;
    proof.steps = std::move(normalized.proof);
    result.status = SemanticProofStatus::Proved;
    result.reason = "reflexive normalized tensor denotation";
    result.proof = std::move(proof);
    return result;
  }

  auto structurallyEqual = [](const TensorDenotation &left,
                              const TensorDenotation &right) {
    if (left.shape != right.shape ||
        left.shapeSymbols != right.shapeSymbols ||
        left.layoutIdentity != right.layoutIdentity ||
        left.resultIndices.size() != right.resultIndices.size() ||
        left.isMemoryStateToken != right.isMemoryStateToken ||
        left.isLaunchStateToken != right.isLaunchStateToken ||
        left.isBufferStateToken != right.isBufferStateToken)
      return false;
    for (auto [leftIndex, rightIndex] :
         llvm::zip_equal(left.resultIndices, right.resultIndices))
      if (!leftIndex.structurallyEquals(rightIndex))
        return false;
    return left.value.structurallyEquals(right.value) &&
           left.defined.structurallyEquals(right.defined) &&
           left.effect.structurallyEquals(right.effect) &&
           left.memory.structurallyEquals(right.memory) &&
           left.memoryDefined.structurallyEquals(right.memoryDefined) &&
           left.launch.structurallyEquals(right.launch) &&
           left.launchDefined.structurallyEquals(right.launchDefined) &&
           left.buffer.structurallyEquals(right.buffer) &&
           left.bufferDefined.structurallyEquals(right.bufferDefined);
  };

  // Exact structural equality between distinct denotation objects is already
  // a complete proof and needs no second normalization pass. Offline
  // fingerprinting stores normalized denotations, so this covers every member
  // edge inside one exact semantic quotient group.
  if (structurallyEqual(lhs, rhs)) {
    SemanticEquivalenceProof proof;
    proof.verifiedDomains = {
        GraphSemanticDomain::Shape, GraphSemanticDomain::Index,
        GraphSemanticDomain::Layout, GraphSemanticDomain::Value,
        GraphSemanticDomain::Definedness};
    if (lhs.effect)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Effect);
    if (lhs.memory)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Memory);
    if (lhs.launch)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Launch);
    if (lhs.isBufferStateToken && lhs.buffer)
      proof.verifiedDomains.push_back(GraphSemanticDomain::Buffer);
    proof.lhs = lhs;
    proof.rhs = rhs;
    result.status = SemanticProofStatus::Proved;
    result.reason = "structurally identical tensor denotation";
    result.proof = std::move(proof);
    return result;
  }

  SemanticNormalizationResult normalizedLhs;
  SemanticNormalizationResult normalizedRhs;
  if (inputsAlreadyNormalized) {
    normalizedLhs.denotation = lhs;
    normalizedRhs.denotation = rhs;
  } else {
    normalizedLhs = normalizer.normalize(lhs);
    normalizedRhs = normalizer.normalize(rhs);
  }
  for (auto [lhsIndex, rhsIndex] :
       llvm::zip_equal(normalizedLhs.denotation.resultIndices,
                       normalizedRhs.denotation.resultIndices))
    if (!lhsIndex.structurallyEquals(rhsIndex)) {
      result.status = SemanticProofStatus::Refuted;
      result.reason = "result index domain mismatch";
      return result;
    }

  bool valueAndDefinedEqual =
      normalizedLhs.denotation.value.structurallyEquals(
          normalizedRhs.denotation.value) &&
      normalizedLhs.denotation.defined.structurallyEquals(
          normalizedRhs.denotation.defined);
  if (!valueAndDefinedEqual && !lhs.shape.empty()) {
    // Static tensors are finite functions.  When their symbolic normal forms
    // differ, compare every coordinate extensionally.  This is a generic
    // proof rule over the tensor domain; it is what lets read-after-write
    // algebra establish a full-buffer copy without knowing memref.copy (or
    // any other producer) by name.
    constexpr int64_t kMaxExtensionalElements = 4096;
    int64_t elementCount = 1;
    bool finite = true;
    for (int64_t extent : lhs.shape) {
      if (ShapedType::isDynamic(extent) || extent <= 0 ||
          elementCount > kMaxExtensionalElements / extent) {
        finite = false;
        break;
      }
      elementCount *= extent;
    }
    if (finite) {
      valueAndDefinedEqual = true;
      for (int64_t linear = 0; linear < elementCount; ++linear) {
        int64_t remainder = linear;
        SmallVector<TensorSemanticExpr, 4> coordinates(lhs.shape.size());
        for (int64_t dimension = static_cast<int64_t>(lhs.shape.size()) - 1;
             dimension >= 0; --dimension) {
          int64_t coordinate = remainder % lhs.shape[dimension];
          remainder /= lhs.shape[dimension];
          Type indexType = normalizedLhs.denotation
                               .resultIndices[dimension]
                               .getType();
          coordinates[dimension] = TensorSemanticExpr::integer(
              indexType.getContext(), coordinate, indexType);
        }
        auto instantiate = [&](TensorSemanticExpr expression) {
          for (auto [index, coordinate] : llvm::zip_equal(
                   normalizedLhs.denotation.resultIndices, coordinates))
            expression = substituteSemanticExpr(expression, index, coordinate);
          return expression;
        };
        TensorDenotation lhsLane;
        lhsLane.value = instantiate(normalizedLhs.denotation.value);
        lhsLane.defined = instantiate(normalizedLhs.denotation.defined);
        TensorDenotation rhsLane;
        rhsLane.value = instantiate(normalizedRhs.denotation.value);
        rhsLane.defined = instantiate(normalizedRhs.denotation.defined);
        SemanticNormalizationResult lhsLaneNormal =
            normalizer.normalize(std::move(lhsLane));
        SemanticNormalizationResult rhsLaneNormal =
            normalizer.normalize(std::move(rhsLane));
        if (!lhsLaneNormal.denotation.value.structurallyEquals(
                rhsLaneNormal.denotation.value) ||
            !lhsLaneNormal.denotation.defined.structurallyEquals(
                rhsLaneNormal.denotation.defined)) {
          valueAndDefinedEqual = false;
          break;
        }
      }
      if (valueAndDefinedEqual)
        normalizedLhs.proof.push_back(SemanticProofStep{
            "finite-tensor-extensionality", normalizedLhs.denotation.value,
            normalizedRhs.denotation.value});
    }
  }
  if (!valueAndDefinedEqual) {
    LLVM_DEBUG({
      llvm::dbgs() << "AGS semantic mismatch\n  lhs value: ";
      normalizedLhs.denotation.value.print(llvm::dbgs());
      llvm::dbgs() << "\n  rhs value: ";
      normalizedRhs.denotation.value.print(llvm::dbgs());
      llvm::dbgs() << "\n  lhs valid: ";
      normalizedLhs.denotation.defined.print(llvm::dbgs());
      llvm::dbgs() << "\n  rhs valid: ";
      normalizedRhs.denotation.defined.print(llvm::dbgs());
      llvm::dbgs() << '\n';
    });
  }
  const bool compareBufferState =
      normalizedLhs.denotation.isBufferStateToken ||
      normalizedRhs.denotation.isBufferStateToken;
  if (static_cast<bool>(normalizedLhs.denotation.effect) !=
          static_cast<bool>(normalizedRhs.denotation.effect) ||
      static_cast<bool>(normalizedLhs.denotation.memory) !=
          static_cast<bool>(normalizedRhs.denotation.memory) ||
      static_cast<bool>(normalizedLhs.denotation.memoryDefined) !=
          static_cast<bool>(normalizedRhs.denotation.memoryDefined) ||
      static_cast<bool>(normalizedLhs.denotation.launch) !=
          static_cast<bool>(normalizedRhs.denotation.launch) ||
      static_cast<bool>(normalizedLhs.denotation.launchDefined) !=
          static_cast<bool>(normalizedRhs.denotation.launchDefined) ||
      (compareBufferState &&
       (static_cast<bool>(normalizedLhs.denotation.buffer) !=
            static_cast<bool>(normalizedRhs.denotation.buffer) ||
        static_cast<bool>(normalizedLhs.denotation.bufferDefined) !=
            static_cast<bool>(normalizedRhs.denotation.bufferDefined)))) {
    result.reason =
        valueAndDefinedEqual
            ? "effect/memory/launch/buffer semantics are incomplete"
            : "normal forms differ; counterexample required";
    return result;
  }
  if ((normalizedLhs.denotation.effect &&
       !normalizedLhs.denotation.effect.structurallyEquals(
           normalizedRhs.denotation.effect)) ||
      (normalizedLhs.denotation.memory &&
       !normalizedLhs.denotation.memory.structurallyEquals(
           normalizedRhs.denotation.memory)) ||
      (normalizedLhs.denotation.memoryDefined &&
       !normalizedLhs.denotation.memoryDefined.structurallyEquals(
           normalizedRhs.denotation.memoryDefined)) ||
      (normalizedLhs.denotation.launch &&
       !normalizedLhs.denotation.launch.structurallyEquals(
           normalizedRhs.denotation.launch)) ||
      (normalizedLhs.denotation.launchDefined &&
       !normalizedLhs.denotation.launchDefined.structurallyEquals(
           normalizedRhs.denotation.launchDefined)) ||
      (compareBufferState && normalizedLhs.denotation.buffer &&
       !normalizedLhs.denotation.buffer.structurallyEquals(
           normalizedRhs.denotation.buffer)) ||
      (compareBufferState && normalizedLhs.denotation.bufferDefined &&
       !normalizedLhs.denotation.bufferDefined.structurallyEquals(
           normalizedRhs.denotation.bufferDefined)) ||
      normalizedLhs.denotation.isMemoryStateToken !=
          normalizedRhs.denotation.isMemoryStateToken ||
      normalizedLhs.denotation.isLaunchStateToken !=
          normalizedRhs.denotation.isLaunchStateToken ||
      normalizedLhs.denotation.isBufferStateToken !=
          normalizedRhs.denotation.isBufferStateToken) {
    result.reason =
        valueAndDefinedEqual
            ? "effect, memory, launch, or buffer normal forms differ"
            : "normal forms differ; counterexample required";
    return result;
  }

  // The algebraic normalizer is deliberately small and predictable.  Exact
  // backends such as the offline Z3 bit-vector prover get a chance only after
  // shape/index/layout and every observable state domain have passed the same
  // checks as a normalizer proof.  This keeps an SMT result from hiding an
  // unsupported side effect or memory-state difference.
  if (!valueAndDefinedEqual) {
    if (allowFallback && fallbackBackend)
      return fallbackBackend->compareNormalized(normalizedLhs.denotation,
                                                normalizedRhs.denotation);
    result.reason = "normal forms differ; counterexample required";
    return result;
  }

  SemanticEquivalenceProof proof;
  proof.verifiedDomains = {
      GraphSemanticDomain::Shape, GraphSemanticDomain::Index,
      GraphSemanticDomain::Layout, GraphSemanticDomain::Value,
      GraphSemanticDomain::Definedness};
  if (normalizedLhs.denotation.effect)
    proof.verifiedDomains.push_back(GraphSemanticDomain::Effect);
  if (normalizedLhs.denotation.memory)
    proof.verifiedDomains.push_back(GraphSemanticDomain::Memory);
  if (normalizedLhs.denotation.launch)
    proof.verifiedDomains.push_back(GraphSemanticDomain::Launch);
  if (compareBufferState && normalizedLhs.denotation.buffer)
    proof.verifiedDomains.push_back(GraphSemanticDomain::Buffer);
  proof.lhs = std::move(normalizedLhs.denotation);
  proof.rhs = std::move(normalizedRhs.denotation);
  proof.steps.append(normalizedLhs.proof.begin(), normalizedLhs.proof.end());
  proof.steps.append(normalizedRhs.proof.begin(), normalizedRhs.proof.end());
  result.status = SemanticProofStatus::Proved;
  result.reason = "all enabled tensor denotation domains are equal";
  result.proof = std::move(proof);
  return result;
}

std::optional<SemanticEquivalenceProof>
SemanticEquivalenceOracle::prove(const TensorDenotation &lhs,
                                 const TensorDenotation &rhs) const {
  SemanticProofResult result = compare(lhs, rhs);
  return result.isProved() ? std::move(result.proof) : std::nullopt;
}

SemanticProofResult SemanticEquivalenceOracle::replay(
    const SemanticEquivalenceProof &proof) const {
  if (proof.proofBackend != "normalizer-v1") {
    if (!fallbackBackend ||
        proof.proofBackend != fallbackBackend->getProofBackendName()) {
      SemanticProofResult result;
      result.reason = "proof backend '" + proof.proofBackend +
                      "' is unavailable for certificate replay";
      return result;
    }
    return fallbackBackend->replay(proof);
  }
  return compare(proof.lhs, proof.rhs);
}

std::string mlir::triton::cfg::getStableTensorDenotationKey(
    const TensorDenotation &denotation) {
  llvm::DenseMap<const void *, std::string> expressionDigests;
  llvm::SHA256 hash;
  hash.update("ags-tensor-denotation-key-v2\n");
  auto append = [&](StringRef name, StringRef value) {
    updateFramed(hash, name);
    updateFramed(hash, value);
  };
  auto appendExpression = [&](StringRef name,
                              const TensorSemanticExpr &expression) {
    append(name, semanticKey(expression, expressionDigests));
  };

  std::string metadata;
  llvm::raw_string_ostream stream(metadata);
  stream << "shape[";
  for (auto [index, dimension] : llvm::enumerate(denotation.shape)) {
    stream << dimension;
    if (ShapedType::isDynamic(dimension)) {
      stream << ':';
      if (index < denotation.shapeSymbols.size())
        stream << denotation.shapeSymbols[index];
      else
        stream << '?';
    }
    stream << ',';
  }
  stream << "]layout[" << denotation.layoutIdentity << ']';
  append("metadata", metadata);
  for (auto [index, expression] :
       llvm::enumerate(denotation.resultIndices))
    appendExpression((llvm::Twine("index[") + llvm::Twine(index) + "]").str(),
                     expression);
  appendExpression("value", denotation.value);
  appendExpression("valid", denotation.defined);
  appendExpression("effect", denotation.effect);
  appendExpression("memory", denotation.memory);
  appendExpression("memory_valid", denotation.memoryDefined);
  append("memory_state_token",
         denotation.isMemoryStateToken ? "true" : "false");
  appendExpression("launch", denotation.launch);
  appendExpression("launch_valid", denotation.launchDefined);
  append("launch_state_token",
         denotation.isLaunchStateToken ? "true" : "false");
  appendExpression("buffer", denotation.buffer);
  appendExpression("buffer_valid", denotation.bufferDefined);
  append("buffer_state_token",
         denotation.isBufferStateToken ? "true" : "false");
  return "sha256:" + toHex(hash.final());
}

static std::optional<TensorSemanticKind>
parseSemanticKind(llvm::StringRef spelling) {
  return llvm::StringSwitch<std::optional<TensorSemanticKind>>(spelling)
      .Case("symbol", TensorSemanticKind::Symbol)
      .Case("element", TensorSemanticKind::Element)
      .Case("defined", TensorSemanticKind::Defined)
      .Case("index", TensorSemanticKind::BoundIndex)
      .Case("integer", TensorSemanticKind::Integer)
      .Case("boolean", TensorSemanticKind::Boolean)
      .Case("not", TensorSemanticKind::Not)
      .Case("and", TensorSemanticKind::And)
      .Case("or", TensorSemanticKind::Or)
      .Case("xor", TensorSemanticKind::Xor)
      .Case("bit_and", TensorSemanticKind::BitwiseAnd)
      .Case("bit_or", TensorSemanticKind::BitwiseOr)
      .Case("bit_xor", TensorSemanticKind::BitwiseXor)
      .Case("equal", TensorSemanticKind::Equal)
      .Case("add", TensorSemanticKind::Add)
      .Case("subtract", TensorSemanticKind::Subtract)
      .Case("multiply", TensorSemanticKind::Multiply)
      .Case("signed_multiply_high", TensorSemanticKind::SignedMultiplyHigh)
      .Case("unsigned_multiply_high",
            TensorSemanticKind::UnsignedMultiplyHigh)
      .Case("integer_divide", TensorSemanticKind::IntegerDivide)
      .Case("select", TensorSemanticKind::Select)
      .Case("sum", TensorSemanticKind::Sum)
      .Case("product", TensorSemanticKind::Product)
      .Case("forall", TensorSemanticKind::ForAll)
      .Case("zero_extend", TensorSemanticKind::ZeroExtend)
      .Case("sign_extend", TensorSemanticKind::SignExtend)
      .Case("truncate", TensorSemanticKind::Truncate)
      .Case("shift_left", TensorSemanticKind::ShiftLeft)
      .Case("shift_right_logical", TensorSemanticKind::ShiftRightLogical)
      .Case("shift_right_arithmetic",
            TensorSemanticKind::ShiftRightArithmetic)
      .Case("signed_remainder", TensorSemanticKind::SignedRemainder)
      .Case("signed_less_than", TensorSemanticKind::SignedLessThan)
      .Case("unsigned_less_than", TensorSemanticKind::UnsignedLessThan)
      .Case("pointer_add", TensorSemanticKind::PointerAdd)
      .Case("memory_read", TensorSemanticKind::MemoryRead)
      .Case("memory_valid", TensorSemanticKind::MemoryValid)
      .Case("memory_write", TensorSemanticKind::MemoryWrite)
      .Case("memory_atomic", TensorSemanticKind::MemoryAtomic)
      .Case("memory_observable", TensorSemanticKind::MemoryObservable)
      .Case("partitionable_memory",
            TensorSemanticKind::PartitionableMemory)
      .Case("buffer_read", TensorSemanticKind::BufferRead)
      .Case("buffer_valid", TensorSemanticKind::BufferValid)
      .Case("buffer_write", TensorSemanticKind::BufferWrite)
      .Case("buffer_fill", TensorSemanticKind::BufferFill)
      .Case("buffer_view", TensorSemanticKind::BufferView)
      .Case("buffer_allocate", TensorSemanticKind::BufferAllocate)
      .Case("program_id", TensorSemanticKind::ProgramId)
      .Case("num_programs", TensorSemanticKind::NumPrograms)
      .Case("launch_bound", TensorSemanticKind::LaunchBound)
      .Case("launch_remap", TensorSemanticKind::LaunchRemap)
      .Case("kernel_effect", TensorSemanticKind::KernelEffect)
      .Case("kernel_partition", TensorSemanticKind::KernelPartition)
      .Case("unobserved", TensorSemanticKind::Unobserved)
      .Case("microscale_decode", TensorSemanticKind::MicroscaleDecode)
      .Case("apply_unary", TensorSemanticKind::ApplyUnary)
      .Case("apply_binary", TensorSemanticKind::ApplyBinary)
      .Case("apply_ternary", TensorSemanticKind::ApplyTernary)
      .Case("apply_nary", TensorSemanticKind::ApplyNary)
      .Default(std::nullopt);
}

static std::string printTypeForCertificate(Type type) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << type;
  return storage;
}

static std::string printAttributeForCertificate(Attribute attribute) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << attribute;
  return storage;
}

static llvm::json::Value expressionToJSON(
    const TensorSemanticExpr &expression) {
  llvm::json::Array operands;
  for (const TensorSemanticExpr &operand : expression.getOperands())
    operands.push_back(expressionToJSON(operand));
  llvm::json::Object object;
  object["kind"] = stringifySemanticKind(expression.getKind());
  object["type"] = printTypeForCertificate(expression.getType());
  if (expression.getPayload())
    object["payload"] =
        printAttributeForCertificate(expression.getPayload());
  object["operands"] = std::move(operands);
  return object;
}

static std::optional<TensorSemanticExpr>
expressionFromJSON(const llvm::json::Value &serialized, MLIRContext &context,
                   std::string &error) {
  const llvm::json::Object *object = serialized.getAsObject();
  if (!object) {
    error = "semantic expression must be an object";
    return std::nullopt;
  }
  std::optional<llvm::StringRef> kindName = object->getString("kind");
  std::optional<llvm::StringRef> typeName = object->getString("type");
  const llvm::json::Array *operandArray = object->getArray("operands");
  if (!kindName || !typeName || !operandArray) {
    error = "semantic expression is missing kind, type, or operands";
    return std::nullopt;
  }
  std::optional<TensorSemanticKind> kind = parseSemanticKind(*kindName);
  if (!kind) {
    error = (llvm::Twine("unknown semantic expression kind: ") + *kindName)
                .str();
    return std::nullopt;
  }
  Type type = mlir::parseType(*typeName, &context);
  if (!type) {
    error = (llvm::Twine("cannot parse semantic expression type: ") +
             *typeName)
                .str();
    return std::nullopt;
  }
  Attribute payload;
  if (std::optional<llvm::StringRef> payloadText =
          object->getString("payload")) {
    payload = mlir::parseAttribute(*payloadText, &context);
    if (!payload) {
      error = (llvm::Twine("cannot parse semantic expression payload: ") +
               *payloadText)
                  .str();
      return std::nullopt;
    }
  }
  SmallVector<TensorSemanticExpr, 4> operands;
  for (const llvm::json::Value &operandValue : *operandArray) {
    std::optional<TensorSemanticExpr> operand =
        expressionFromJSON(operandValue, context, error);
    if (!operand)
      return std::nullopt;
    operands.push_back(std::move(*operand));
  }
  return TensorSemanticExpr::operation(*kind, type, operands, payload);
}

static llvm::json::Value denotationToJSON(const TensorDenotation &denotation) {
  llvm::json::Array shape;
  for (int64_t dimension : denotation.shape)
    shape.push_back(dimension);
  llvm::json::Array shapeSymbols;
  for (auto [index, dimension] : llvm::enumerate(denotation.shape)) {
    if (index < denotation.shapeSymbols.size())
      shapeSymbols.push_back(denotation.shapeSymbols[index]);
    else
      shapeSymbols.push_back(ShapedType::isDynamic(dimension) ? "?" : "");
  }
  llvm::json::Array indices;
  for (const TensorSemanticExpr &index : denotation.resultIndices)
    indices.push_back(expressionToJSON(index));
  llvm::json::Object object;
  object["shape"] = std::move(shape);
  object["shape_symbols"] = std::move(shapeSymbols);
  object["layout_identity"] = denotation.layoutIdentity;
  object["result_indices"] = std::move(indices);
  object["value"] = expressionToJSON(denotation.value);
  object["defined"] = expressionToJSON(denotation.defined);
  if (denotation.effect)
    object["effect"] = expressionToJSON(denotation.effect);
  if (denotation.memory)
    object["memory"] = expressionToJSON(denotation.memory);
  if (denotation.memoryDefined)
    object["memory_valid"] = expressionToJSON(denotation.memoryDefined);
  object["memory_state_token"] = denotation.isMemoryStateToken;
  if (denotation.launch)
    object["launch"] = expressionToJSON(denotation.launch);
  if (denotation.launchDefined)
    object["launch_valid"] = expressionToJSON(denotation.launchDefined);
  object["launch_state_token"] = denotation.isLaunchStateToken;
  if (denotation.buffer)
    object["buffer"] = expressionToJSON(denotation.buffer);
  if (denotation.bufferDefined)
    object["buffer_valid"] = expressionToJSON(denotation.bufferDefined);
  object["buffer_state_token"] = denotation.isBufferStateToken;
  return object;
}

static std::optional<TensorDenotation>
denotationFromJSON(const llvm::json::Value &serialized, MLIRContext &context,
                   std::string &error) {
  const llvm::json::Object *object = serialized.getAsObject();
  if (!object) {
    error = "tensor denotation must be an object";
    return std::nullopt;
  }
  const llvm::json::Array *shape = object->getArray("shape");
  const llvm::json::Array *shapeSymbols = object->getArray("shape_symbols");
  const llvm::json::Array *indices = object->getArray("result_indices");
  const llvm::json::Value *value = object->get("value");
  const llvm::json::Value *defined = object->get("defined");
  if (!shape || !indices || !value || !defined) {
    error = "tensor denotation is missing a required field";
    return std::nullopt;
  }
  TensorDenotation denotation;
  if (std::optional<llvm::StringRef> layout =
          object->getString("layout_identity"))
    denotation.layoutIdentity = layout->str();
  for (const llvm::json::Value &dimensionValue : *shape) {
    std::optional<int64_t> dimension = dimensionValue.getAsInteger();
    if (!dimension) {
      error = "tensor denotation shape must contain integers";
      return std::nullopt;
    }
    denotation.shape.push_back(*dimension);
  }
  if (shapeSymbols) {
    if (shapeSymbols->size() != denotation.shape.size()) {
      error = "tensor denotation shape_symbols size mismatch";
      return std::nullopt;
    }
    for (const llvm::json::Value &symbolValue : *shapeSymbols) {
      std::optional<llvm::StringRef> symbol = symbolValue.getAsString();
      if (!symbol) {
        error = "tensor denotation shape_symbols must contain strings";
        return std::nullopt;
      }
      denotation.shapeSymbols.push_back(symbol->str());
    }
  } else {
    for (int64_t dimension : denotation.shape)
      denotation.shapeSymbols.push_back(
          ShapedType::isDynamic(dimension) ? "?" : "");
  }
  for (const llvm::json::Value &indexValue : *indices) {
    std::optional<TensorSemanticExpr> index =
        expressionFromJSON(indexValue, context, error);
    if (!index)
      return std::nullopt;
    denotation.resultIndices.push_back(std::move(*index));
  }
  std::optional<TensorSemanticExpr> parsedValue =
      expressionFromJSON(*value, context, error);
  if (!parsedValue)
    return std::nullopt;
  std::optional<TensorSemanticExpr> parsedDefined =
      expressionFromJSON(*defined, context, error);
  if (!parsedDefined)
    return std::nullopt;
  denotation.value = std::move(*parsedValue);
  denotation.defined = std::move(*parsedDefined);
  if (const llvm::json::Value *effect = object->get("effect")) {
    std::optional<TensorSemanticExpr> parsedEffect =
        expressionFromJSON(*effect, context, error);
    if (!parsedEffect)
      return std::nullopt;
    denotation.effect = std::move(*parsedEffect);
  }
  if (const llvm::json::Value *memory = object->get("memory")) {
    std::optional<TensorSemanticExpr> parsedMemory =
        expressionFromJSON(*memory, context, error);
    if (!parsedMemory)
      return std::nullopt;
    denotation.memory = std::move(*parsedMemory);
  }
  if (const llvm::json::Value *memoryDefined = object->get("memory_valid")) {
    std::optional<TensorSemanticExpr> parsedMemoryDefined =
        expressionFromJSON(*memoryDefined, context, error);
    if (!parsedMemoryDefined)
      return std::nullopt;
    denotation.memoryDefined = std::move(*parsedMemoryDefined);
  }
  denotation.isMemoryStateToken =
      object->getBoolean("memory_state_token").value_or(false);
  if (const llvm::json::Value *launch = object->get("launch")) {
    std::optional<TensorSemanticExpr> parsedLaunch =
        expressionFromJSON(*launch, context, error);
    if (!parsedLaunch)
      return std::nullopt;
    denotation.launch = std::move(*parsedLaunch);
  }
  if (const llvm::json::Value *launchDefined =
          object->get("launch_valid")) {
    std::optional<TensorSemanticExpr> parsedLaunchDefined =
        expressionFromJSON(*launchDefined, context, error);
    if (!parsedLaunchDefined)
      return std::nullopt;
    denotation.launchDefined = std::move(*parsedLaunchDefined);
  }
  denotation.isLaunchStateToken =
      object->getBoolean("launch_state_token").value_or(false);
  if (const llvm::json::Value *buffer = object->get("buffer")) {
    std::optional<TensorSemanticExpr> parsedBuffer =
        expressionFromJSON(*buffer, context, error);
    if (!parsedBuffer)
      return std::nullopt;
    denotation.buffer = std::move(*parsedBuffer);
  }
  if (const llvm::json::Value *bufferDefined =
          object->get("buffer_valid")) {
    std::optional<TensorSemanticExpr> parsedBufferDefined =
        expressionFromJSON(*bufferDefined, context, error);
    if (!parsedBufferDefined)
      return std::nullopt;
    denotation.bufferDefined = std::move(*parsedBufferDefined);
  }
  denotation.isBufferStateToken =
      object->getBoolean("buffer_state_token").value_or(false);
  return denotation;
}

std::string SemanticEquivalenceProof::getStableHash() const {
  std::string canonical = getStableTensorDenotationKey(lhs) + "==" +
                          getStableTensorDenotationKey(rhs);
  llvm::raw_string_ostream stream(canonical);
  // Preserve hashes of legacy normalizer-v1 certificates while making proof
  // procedures part of every new backend certificate's identity.
  if (proofBackend != "normalizer-v1")
    stream << "|proof_backend=" << proofBackend;
  stream << "|domains=";
  for (GraphSemanticDomain domain : verifiedDomains)
    stream << stringifyGraphSemanticDomain(domain) << ',';
  for (const SemanticProofStep &step : steps) {
    stream << "|step=" << step.reason << ':';
    step.before.print(stream);
    stream << "->";
    step.after.print(stream);
  }
  stream.flush();

  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char byte : canonical) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

std::string SemanticEquivalenceProof::toJSON() const {
  llvm::json::Array domains;
  for (GraphSemanticDomain domain : verifiedDomains)
    domains.push_back(stringifyGraphSemanticDomain(domain));
  llvm::json::Array proofSteps;
  for (const SemanticProofStep &step : steps) {
    llvm::json::Object entry;
    entry["reason"] = step.reason;
    entry["before"] = expressionToJSON(step.before);
    entry["after"] = expressionToJSON(step.after);
    proofSteps.push_back(std::move(entry));
  }
  llvm::json::Object certificate;
  certificate["version"] = 2;
  certificate["stable_hash"] = getStableHash();
  certificate["proof_backend"] = proofBackend;
  certificate["lhs_semantic_key"] = getStableTensorDenotationKey(lhs);
  certificate["rhs_semantic_key"] = getStableTensorDenotationKey(rhs);
  certificate["lhs"] = denotationToJSON(lhs);
  certificate["rhs"] = denotationToJSON(rhs);
  certificate["verified_domains"] = std::move(domains);
  certificate["steps"] = std::move(proofSteps);
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << llvm::formatv("{0}", llvm::json::Value(std::move(certificate)));
  return storage;
}

std::optional<SemanticEquivalenceProof> SemanticEquivalenceProof::fromJSON(
    StringRef serialized, MLIRContext &context, std::string &error) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(serialized);
  if (!parsed) {
    error = "invalid proof certificate JSON: " +
            llvm::toString(parsed.takeError());
    return std::nullopt;
  }
  llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->getInteger("version") != 2) {
    error = "unsupported or missing proof certificate version";
    return std::nullopt;
  }
  llvm::json::Value *lhsValue = object->get("lhs");
  llvm::json::Value *rhsValue = object->get("rhs");
  const llvm::json::Array *domains = object->getArray("verified_domains");
  const llvm::json::Array *proofSteps = object->getArray("steps");
  std::optional<llvm::StringRef> expectedHash =
      object->getString("stable_hash");
  std::optional<llvm::StringRef> expectedLhsKey =
      object->getString("lhs_semantic_key");
  std::optional<llvm::StringRef> expectedRhsKey =
      object->getString("rhs_semantic_key");
  if (!lhsValue || !rhsValue || !domains || !proofSteps || !expectedHash ||
      !expectedLhsKey || !expectedRhsKey) {
    error = "proof certificate is missing a required field";
    return std::nullopt;
  }

  std::optional<TensorDenotation> lhs =
      denotationFromJSON(*lhsValue, context, error);
  if (!lhs)
    return std::nullopt;
  std::optional<TensorDenotation> rhs =
      denotationFromJSON(*rhsValue, context, error);
  if (!rhs)
    return std::nullopt;

  SemanticEquivalenceProof proof;
  proof.proofBackend =
      object->getString("proof_backend").value_or("normalizer-v1").str();
  proof.lhs = std::move(*lhs);
  proof.rhs = std::move(*rhs);
  if (getStableTensorDenotationKey(proof.lhs) != *expectedLhsKey ||
      getStableTensorDenotationKey(proof.rhs) != *expectedRhsKey) {
    error = "proof certificate semantic key mismatch";
    return std::nullopt;
  }
  for (const llvm::json::Value &domainValue : *domains) {
    std::optional<llvm::StringRef> spelling = domainValue.getAsString();
    std::optional<GraphSemanticDomain> domain =
        spelling ? parseGraphSemanticDomain(*spelling) : std::nullopt;
    if (!domain) {
      error = "proof certificate contains an unknown semantic domain";
      return std::nullopt;
    }
    proof.verifiedDomains.push_back(*domain);
  }
  for (const llvm::json::Value &stepValue : *proofSteps) {
    const llvm::json::Object *stepObject = stepValue.getAsObject();
    if (!stepObject) {
      error = "proof step must be an object";
      return std::nullopt;
    }
    std::optional<llvm::StringRef> reason = stepObject->getString("reason");
    const llvm::json::Value *beforeValue = stepObject->get("before");
    const llvm::json::Value *afterValue = stepObject->get("after");
    if (!reason || !beforeValue || !afterValue) {
      error = "proof step is missing reason, before, or after";
      return std::nullopt;
    }
    std::optional<TensorSemanticExpr> before =
        expressionFromJSON(*beforeValue, context, error);
    if (!before)
      return std::nullopt;
    std::optional<TensorSemanticExpr> after =
        expressionFromJSON(*afterValue, context, error);
    if (!after)
      return std::nullopt;
    proof.steps.push_back(
        SemanticProofStep{reason->str(), std::move(*before),
                          std::move(*after)});
  }
  if (proof.getStableHash() != *expectedHash) {
    error = "proof certificate stable hash mismatch";
    return std::nullopt;
  }
  return proof;
}

CandidateEnumerationResult TypedCandidateEnumerator::enumerate(
    EquivalenceGraph &graph, ArrayRef<Type> resultTypes,
    ArrayRef<GraphEClassId> availableOperands,
    DenseMap<GraphEClassId, TensorDenotation> &denotations) const {
  CandidateEnumerationResult result;
  SmallVector<GraphEClassId, 32> available(availableOperands);
  llvm::StringMap<SmallVector<GraphEClassId, 2>> classesByDenotation;
  for (const auto &entry : denotations) {
    TensorDenotation canonical = normalizer.normalize(entry.second).denotation;
    classesByDenotation[getStableTensorDenotationKey(canonical)].push_back(
        graph.find(entry.first));
  }
  for (unsigned round = 0; round < options.maxRounds; ++round) {
    SmallVector<GraphExprNode, 32> candidates;
    const unsigned perTypeLimit = std::max(
        1u, options.maxCandidates /
                std::max(1u, static_cast<unsigned>(resultTypes.size())));
    for (Type resultType : resultTypes) {
      language.enumerate(resultType, available, graph, candidates,
                         perTypeLimit);
    }
    if (candidates.empty())
      break;

    const unsigned generationBefore = graph.getGeneration();
    for (GraphExprNode &candidate : candidates) {
      ++result.generated;

      const std::string candidateOperation =
          candidate.getOperationName().str();

      const TTOpSemantics *definition =
          semantics.lookup(candidate.getOperationName());
      if (!definition) {
        ++result.rejected;
        continue;
      }

      SmallVector<TensorDenotation, 4> operandDenotations;
      bool operandsKnown = true;
      for (GraphEClassId operand : candidate.operands) {
        operand = graph.find(operand);
        auto known = denotations.find(operand);
        if (known == denotations.end()) {
          operandsKnown = false;
          break;
        }
        operandDenotations.push_back(known->second);
      }
      if (!operandsKnown) {
        ++result.rejected;
        continue;
      }

      FailureOr<TensorDenotation> candidateDenotation =
          definition->denote(candidate, operandDenotations);
      if (failed(candidateDenotation)) {
        ++result.rejected;
        continue;
      }
      TensorDenotation canonical =
          normalizer.normalize(*candidateDenotation).denotation;

      SmallVector<std::pair<GraphEClassId, SemanticEquivalenceProof>, 2>
          matches;
      llvm::DenseSet<GraphEClassId> comparedClasses;
      auto sameDenotation = classesByDenotation.find(
          getStableTensorDenotationKey(canonical));
      if (sameDenotation != classesByDenotation.end())
        for (GraphEClassId indexedClass : sameDenotation->second) {
        GraphEClassId existingClass = graph.find(indexedClass);
        if (!comparedClasses.insert(existingClass).second)
          continue;
        auto entry = denotations.find(existingClass);
        if (entry == denotations.end()) {
          for (auto candidateEntry = denotations.begin();
               candidateEntry != denotations.end(); ++candidateEntry)
            if (graph.find(candidateEntry->first) == existingClass) {
              entry = candidateEntry;
              break;
            }
        }
        if (entry == denotations.end())
          continue;
        std::optional<SemanticEquivalenceProof> proof =
            oracle.prove(canonical, entry->second);
        if (proof)
          matches.emplace_back(existingClass, std::move(*proof));
      }

      // A store-mask observation is an effect boundary: the candidate only
      // needs to agree with the selected root on lanes that the store can
      // observe.  Intermediate nodes are still compared globally.  This
      // keeps contextual equality local to a single-use root and prevents an
      // under-mask proof from leaking into another consumer.
      if (contextualRoot && observation) {
        GraphEClassId root = graph.find(*contextualRoot);
        bool alreadyMatched = llvm::any_of(
            matches, [root](const auto &match) { return match.first == root; });
        auto rootDenotation = denotations.find(root);
        // Contextual equality may hide lanes that the enclosing store cannot
        // observe, but it must not turn an eagerly evaluated candidate into a
        // predicated one.  Require the candidate to close the observation
        // boundary by itself before comparing it with the root.  A masked
        // load satisfies this because its false lanes are already unobserved;
        // an unmasked load does not.  The check is entirely denotational and
        // applies to every candidate operation uniformly.
        TensorDenotation observedCandidate = observation(canonical);
        bool closesObservation =
            static_cast<bool>(oracle.prove(canonical, observedCandidate));
        if (!alreadyMatched && closesObservation &&
            rootDenotation != denotations.end() &&
            canonical.shape == rootDenotation->second.shape) {
          TensorDenotation observedRoot = observation(rootDenotation->second);
          std::optional<SemanticEquivalenceProof> proof =
              oracle.prove(observedCandidate, observedRoot);
          if (proof)
            matches.emplace_back(root, std::move(*proof));
        }
      }

      // Unary projections are useful generic bridges. Declaratively marked
      // exploratory constructors extend that beam to multi-step candidates
      // such as pointer-add -> masked-load, while the shared budgets still
      // bound K^N growth.
      bool exploratory = false;
      if (candidate.attributes)
        if (auto flag = dyn_cast_or_null<BoolAttr>(
                candidate.attributes.get("ags.exploratory")))
          exploratory = flag.getValue();
      if (matches.empty() && candidate.operands.size() != 1 &&
          !exploratory) {
        ++result.rejected;
        continue;
      }
      std::optional<GraphEClassId> candidateClass =
          graph.addNode(std::move(candidate));
      if (!candidateClass) {
        result.hitBudget = true;
        break;
      }

      bool merged = !matches.empty();
      for (auto &[existingClass, proof] : matches) {
        *candidateClass = graph.merge(*candidateClass, existingClass);
        result.merges.push_back(SemanticMergeRecord{
            candidateOperation, graph.find(existingClass),
            std::move(proof)});
        ++result.provedEquivalent;
      }
      denotations[graph.find(*candidateClass)] = std::move(canonical);
      classesByDenotation[getStableTensorDenotationKey(
          denotations[graph.find(*candidateClass)])]
          .push_back(graph.find(*candidateClass));
      if (!llvm::is_contained(available, graph.find(*candidateClass)))
        available.push_back(graph.find(*candidateClass));
      if (!merged)
        ++result.rejected;
    }

    graph.rebuild();
    result.hitBudget |= candidates.size() >= options.maxCandidates;
    if (graph.getGeneration() == generationBefore)
      break;
  }
  return result;
}

const GraphExprNode *GraphExtraction::lookup(GraphEClassId eclass) const {
  auto iterator = choices.find(eclass);
  return iterator == choices.end() ? nullptr : &iterator->second;
}

std::optional<unsigned>
StructuralGraphCostModel::getNodeCost(const GraphExprNode &node) const {
  // Only explicit island inputs are free. Imported executable operations also
  // carry their source Value so a selected graph can reuse them, but they
  // still contribute cost. Wide integer operations receive a small structural
  // penalty; this makes the POC prefer an equally proven i32 computation over
  // its i64 form without embedding an operation-specific rule.
  if (node.getOperationName() == "ags.input" ||
      node.getOperationName() == "ags.private_buffer")
    return 0u;
  if (node.attributes)
    if (auto reusable = dyn_cast_or_null<BoolAttr>(
            node.attributes.get("ags.reusable_source"));
        reusable && reusable.getValue())
      return 0u;
  if (node.getOperationName() == "arith.constant")
    return 1u;
  // Integer width casts are data-movement edges rather than wide arithmetic.
  // Charging them by destination bit width makes a proven low-bit graph look
  // more expensive than the wide bitwise/shift graph it replaces. Keep the
  // POC model structural and operation-class based; StageCostModel will later
  // supply calibrated target costs.
  if (node.getOperationName() == "arith.extui" ||
      node.getOperationName() == "arith.extsi" ||
      node.getOperationName() == "arith.trunci")
    return 1u;
  Type type;
  if (node.attributes)
    if (auto resultType = dyn_cast_or_null<TypeAttr>(
            node.attributes.get("ags.result_type")))
      type = resultType.getValue();
  Type elementType = type;
  if (auto shaped = dyn_cast_or_null<ShapedType>(type))
    elementType = shaped.getElementType();
  unsigned bitWidth = 8;
  if (auto integer = dyn_cast_or_null<IntegerType>(elementType))
    bitWidth = integer.getWidth();
  else if (auto floating = dyn_cast_or_null<FloatType>(elementType))
    bitWidth = floating.getWidth();
  // This remains a deliberately small structural model, but charges wider
  // tensors uniformly.  It therefore prefers moving a narrowing pointwise op
  // before a transpose without encoding that graph pattern in the model.
  unsigned widthCost = std::max(1u, (bitWidth + 15u) / 16u);
  // Division and remainder are multi-cycle arithmetic rather than one cheap
  // pointwise instruction. This is a generic structural weight (independent
  // of any source/replacement graph) until StageCostModel supplies calibrated
  // target costs.
  if (node.getOperationName() == "arith.divsi" ||
      node.getOperationName() == "arith.divui" ||
      node.getOperationName() == "arith.remsi" ||
      node.getOperationName() == "arith.remui")
    return 5u * widthCost;
  return widthCost;
}

namespace {

struct ExtractionState {
  DenseMap<GraphEClassId, unsigned> costs;
  DenseMap<GraphEClassId, GraphExprNode> choices;
  llvm::DenseSet<GraphEClassId> active;
};

static std::optional<unsigned>
extractEClass(const EquivalenceGraph &graph, GraphEClassId eclass,
              const GraphCostModel &costModel, ExtractionState &state) {
  eclass = graph.find(eclass);
  auto known = state.costs.find(eclass);
  if (known != state.costs.end())
    return known->second;
  if (!state.active.insert(eclass).second)
    return std::nullopt;

  std::optional<unsigned> bestCost;
  std::optional<GraphExprNode> bestNode;
  for (const GraphExprNode &node : graph.getNodes(eclass)) {
    std::optional<unsigned> ownCost = costModel.getNodeCost(node);
    if (!ownCost)
      continue;
    unsigned cost = *ownCost;
    bool valid = true;
    bool reusableSource = false;
    if (node.attributes)
      if (auto reusable = dyn_cast_or_null<BoolAttr>(
              node.attributes.get("ags.reusable_source")))
        reusableSource = reusable.getValue();
    ArrayRef<GraphEClassId> childClasses = node.operands;
    if (reusableSource)
      childClasses = {};
    for (GraphEClassId operand : childClasses) {
      std::optional<unsigned> childCost =
          extractEClass(graph, operand, costModel, state);
      if (!childCost ||
          *childCost > std::numeric_limits<unsigned>::max() - cost) {
        valid = false;
        break;
      }
      cost += *childCost;
    }
    const bool preferGeneratedTie =
        valid && bestCost && cost == *bestCost && bestNode &&
        bestNode->input && !node.input;
    if (valid && (!bestCost || cost < *bestCost || preferGeneratedTie)) {
      bestCost = cost;
      bestNode = node;
    }
  }

  state.active.erase(eclass);
  if (!bestCost)
    return std::nullopt;
  state.costs[eclass] = *bestCost;
  state.choices[eclass] = *bestNode;
  return bestCost;
}

} // namespace

std::optional<GraphExtraction>
GraphExtractor::extract(const EquivalenceGraph &graph,
                        GraphEClassId root) const {
  root = graph.find(root);
  ExtractionState state;
  std::optional<unsigned> cost =
      extractEClass(graph, root, costModel, state);
  if (!cost)
    return std::nullopt;
  GraphExtraction extraction;
  extraction.cost = *cost;
  extraction.root = root;
  extraction.choices = std::move(state.choices);
  return extraction;
}

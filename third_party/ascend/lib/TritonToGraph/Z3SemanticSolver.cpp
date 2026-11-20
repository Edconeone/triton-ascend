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

#include "TritonToGraph/Z3SemanticSolver.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include <z3++.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::triton::cfg {
namespace {

constexpr llvm::StringLiteral kProofBackendName = "z3-bv-v1";

static std::optional<unsigned> getBitWidth(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (isa<IndexType>(type))
    return 64;
  // TensorIndexSemantics models a pointer as its fixed-width address value.
  // Keep this identical to the concrete fingerprint evaluator: pointer
  // values are 64-bit bit-vectors and pointer arithmetic wraps modulo 2^64.
  if (isa<triton::PointerType>(type))
    return 64;
  return std::nullopt;
}

static std::optional<bool>
getBooleanLiteral(const TensorSemanticExpr &expression) {
  if (!expression || expression.getKind() != TensorSemanticKind::Boolean)
    return std::nullopt;
  auto attribute = dyn_cast_or_null<BoolAttr>(expression.getPayload());
  if (!attribute)
    return std::nullopt;
  return attribute.getValue();
}

static bool sameExpression(const TensorSemanticExpr &lhs,
                           const TensorSemanticExpr &rhs) {
  if (static_cast<bool>(lhs) != static_cast<bool>(rhs))
    return false;
  return !lhs || lhs.structurallyEquals(rhs);
}

static bool sameNonValueDomains(const TensorDenotation &lhs,
                                const TensorDenotation &rhs) {
  if (lhs.shape != rhs.shape || lhs.layoutIdentity != rhs.layoutIdentity ||
      lhs.resultIndices.size() != rhs.resultIndices.size() ||
      lhs.isMemoryStateToken != rhs.isMemoryStateToken ||
      lhs.isLaunchStateToken != rhs.isLaunchStateToken ||
      lhs.isBufferStateToken != rhs.isBufferStateToken)
    return false;
  // Keep this gate identical to SemanticEquivalenceOracle::compareImpl.
  // Static dimensions do not observe shape-symbol spellings. Dynamic
  // dimensions do, and the primary oracle has already rejected an absent,
  // unknown, conflicting, or unequal symbol before invoking this fallback.
  for (auto [dimension, extent] : llvm::enumerate(lhs.shape)) {
    if (!ShapedType::isDynamic(extent))
      continue;
    if (lhs.shapeSymbols.size() != lhs.shape.size() ||
        rhs.shapeSymbols.size() != rhs.shape.size() ||
        lhs.shapeSymbols[dimension] != rhs.shapeSymbols[dimension])
      return false;
  }
  for (auto [lhsIndex, rhsIndex] :
       llvm::zip_equal(lhs.resultIndices, rhs.resultIndices))
    if (!lhsIndex.structurallyEquals(rhsIndex))
      return false;
  if (!sameExpression(lhs.effect, rhs.effect) ||
      !sameExpression(lhs.memory, rhs.memory) ||
      !sameExpression(lhs.memoryDefined, rhs.memoryDefined) ||
      !sameExpression(lhs.launch, rhs.launch) ||
      !sameExpression(lhs.launchDefined, rhs.launchDefined))
    return false;
  // Buffer is observable only for buffer-state tokens. Ordinary value
  // denotations may retain bookkeeping expressions that are intentionally
  // ignored by the primary oracle.
  return !lhs.isBufferStateToken ||
         (sameExpression(lhs.buffer, rhs.buffer) &&
          sameExpression(lhs.bufferDefined, rhs.bufferDefined));
}

static bool isShiftAmountValidity(
    const TensorSemanticExpr &expression,
    TensorSemanticExpr &amount) {
  if (expression.getKind() != TensorSemanticKind::ApplyUnary ||
      expression.getOperands().size() != 1)
    return false;
  auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
  auto role = payload ? dyn_cast_or_null<StringAttr>(
                            payload.get("semantic_role"))
                      : StringAttr{};
  if (!role || role.getValue() != "shift_amount_valid")
    return false;
  amount = expression.getOperands().front();
  return true;
}

static void collectConjunctiveShiftAmounts(
    const TensorSemanticExpr &condition,
    SmallVectorImpl<TensorSemanticExpr> &amounts) {
  if (condition.getKind() == TensorSemanticKind::And) {
    for (const TensorSemanticExpr &operand : condition.getOperands())
      collectConjunctiveShiftAmounts(operand, amounts);
    return;
  }
  TensorSemanticExpr amount;
  if (!isShiftAmountValidity(condition, amount))
    return;
  if (!llvm::any_of(amounts, [&](const TensorSemanticExpr &existing) {
        return existing.structurallyEquals(amount);
      }))
    amounts.push_back(std::move(amount));
}

static SmallVector<GraphSemanticDomain, 10>
getVerifiedDomains(const TensorDenotation &denotation) {
  SmallVector<GraphSemanticDomain, 10> domains = {
      GraphSemanticDomain::Shape, GraphSemanticDomain::Index,
      GraphSemanticDomain::Layout, GraphSemanticDomain::Value,
      GraphSemanticDomain::Definedness};
  if (denotation.effect)
    domains.push_back(GraphSemanticDomain::Effect);
  if (denotation.memory)
    domains.push_back(GraphSemanticDomain::Memory);
  if (denotation.launch)
    domains.push_back(GraphSemanticDomain::Launch);
  if (denotation.isBufferStateToken && denotation.buffer)
    domains.push_back(GraphSemanticDomain::Buffer);
  return domains;
}

struct CachedTranslation {
  TensorSemanticExpr source;
  z3::expr translated;
};

using TranslationCache = std::unordered_map<const void *, CachedTranslation>;
using AtomCache = std::map<std::string, z3::expr>;
using UninterpretedFunctionCache = std::map<std::string, z3::func_decl>;

class ExpressionTranslator {
public:
  ExpressionTranslator(z3::context &context, TranslationCache &translated,
                       AtomCache &atoms,
                       UninterpretedFunctionCache &uninterpretedFunctions)
      : context(context), translated(translated), atoms(atoms),
        uninterpretedFunctions(uninterpretedFunctions) {}

  std::optional<z3::expr> translate(const TensorSemanticExpr &expression) {
    if (!expression) {
      error = "missing semantic expression";
      return std::nullopt;
    }
    auto cached = translated.find(expression.getAsOpaquePointer());
    if (cached != translated.end() &&
        cached->second.source.structurallyEquals(expression))
      return cached->second.translated;

    std::optional<z3::expr> result = translateImpl(expression);
    if (result)
      translated.insert_or_assign(expression.getAsOpaquePointer(),
                                  CachedTranslation{expression, *result});
    return result;
  }

  StringRef getError() const { return error; }

private:
  std::optional<z3::expr> fail(const TensorSemanticExpr &expression,
                               StringRef reason) {
    std::string rendered;
    llvm::raw_string_ostream stream(rendered);
    expression.print(stream);
    stream.flush();
    error = (llvm::Twine(reason) + ": " + rendered).str();
    return std::nullopt;
  }

  std::optional<z3::expr> atom(const TensorSemanticExpr &expression,
                               bool booleanSort) {
    std::string key;
    llvm::raw_string_ostream stream(key);
    stream << (booleanSort ? "bool:" : "bv:");
    expression.print(stream);
    stream.flush();
    auto found = atoms.find(key);
    if (found != atoms.end())
      return found->second;

    std::optional<unsigned> width = getBitWidth(expression.getType());
    if (!booleanSort && (!width || *width == 0))
      return fail(expression, "Z3 backend does not support this atom type");
    z3::expr variable = booleanSort ? context.bool_const(key.c_str())
                                    : context.bv_const(key.c_str(), *width);
    atoms.emplace(key, variable);
    return variable;
  }

  std::optional<z3::expr> operand(const TensorSemanticExpr &expression,
                                  unsigned index) {
    if (index >= expression.getOperands().size())
      return fail(expression, "semantic node has too few operands");
    return translate(expression.getOperands()[index]);
  }

  std::optional<z3::expr> requireBool(const TensorSemanticExpr &expression,
                                      unsigned index) {
    std::optional<z3::expr> value = operand(expression, index);
    if (!value || !value->is_bool()) {
      if (value)
        fail(expression, "expected a Boolean operand");
      return std::nullopt;
    }
    return value;
  }

  std::optional<z3::expr> requireBitVector(const TensorSemanticExpr &expression,
                                           unsigned index) {
    std::optional<z3::expr> value = operand(expression, index);
    if (!value)
      return std::nullopt;
    // TensorIndexSemantics uses MLIR i1 both as a predicate and as a one-bit
    // integer.  Keep predicates as native Z3 Bool values, but convert them to
    // BV1 exactly when an integer operation consumes the value.
    if (value->is_bool())
      return z3::ite(*value, context.bv_val(1, 1), context.bv_val(0, 1));
    if (!value->is_bv())
      return fail(expression, "expected a bit-vector operand");
    return value;
  }

  std::optional<z3::expr>
  requireUnsignedBitVector(const TensorSemanticExpr &expression,
                           unsigned index, unsigned width) {
    std::optional<z3::expr> value = requireBitVector(expression, index);
    if (!value || width == 0)
      return std::nullopt;
    unsigned inputWidth = value->get_sort().bv_size();
    if (inputWidth == width)
      return value;
    if (inputWidth < width)
      return z3::zext(*value, width - inputWidth);
    return value->extract(width - 1, 0);
  }

  // Memory contents and address validity are arbitrary functions of the
  // incoming MemoryState and the byte address.  Encoding them as shared
  // uninterpreted functions gives Z3 exactly the congruence property required
  // by masked-address proofs: equal states and equal addresses imply equal
  // reads/validity, while no value relationship is invented for distinct
  // addresses.  A per-expression atom would lose that implication and leave
  // otherwise elementary load/store equivalences Unknown.
  std::optional<z3::expr>
  uninterpretedApplication(const TensorSemanticExpr &expression,
                           bool booleanResult) {
    const ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
    if (operands.empty())
      return fail(expression,
                  "uninterpreted semantic primitive has no operands");

    std::vector<z3::expr> arguments;
    std::vector<z3::sort> domain;
    arguments.reserve(operands.size());
    domain.reserve(operands.size());
    for (const TensorSemanticExpr &operandExpression : operands) {
      std::optional<z3::expr> argument = translate(operandExpression);
      if (!argument)
        return std::nullopt;
      domain.push_back(argument->get_sort());
      arguments.push_back(*argument);
    }

    z3::sort range = context.bool_sort();
    if (!booleanResult) {
      std::optional<unsigned> width = getBitWidth(expression.getType());
      if (!width || *width == 0)
        return fail(expression,
                    "uninterpreted semantic result has no bit width");
      range = context.bv_sort(*width);
    }

    std::string key =
        (llvm::Twine(static_cast<unsigned>(expression.getKind())) + ":")
            .str();
    for (const z3::sort &sort : domain)
      key += sort.to_string() + ":";
    key += "->" + range.to_string();

    auto found = uninterpretedFunctions.find(key);
    if (found == uninterpretedFunctions.end()) {
      std::string name =
          (llvm::Twine("ags_uf_") +
           llvm::Twine(static_cast<unsigned>(expression.getKind())) + "_" +
           llvm::Twine(uninterpretedFunctions.size()))
              .str();
      z3::func_decl declaration = context.function(
          name.c_str(), domain.size(), domain.data(), range);
      found = uninterpretedFunctions.emplace(key, declaration).first;
    }
    return found->second(arguments.size(), arguments.data());
  }

  std::optional<z3::expr> translateImpl(const TensorSemanticExpr &expression) {
    const TensorSemanticKind kind = expression.getKind();
    const ArrayRef<TensorSemanticExpr> operands = expression.getOperands();
    switch (kind) {
    case TensorSemanticKind::Symbol:
    case TensorSemanticKind::Element:
    case TensorSemanticKind::BoundIndex:
      return atom(expression, getBitWidth(expression.getType()) == 1);
    case TensorSemanticKind::Defined:
    case TensorSemanticKind::BufferValid:
      // Definedness is an independent predicate unless its construction was
      // reduced by the semantic normalizer.  Modeling it as a shared Boolean
      // atom is conservative: it may miss a proof but cannot invent one.
      return atom(expression, true);
    case TensorSemanticKind::MemoryValid:
      return uninterpretedApplication(expression, /*booleanResult=*/true);
    case TensorSemanticKind::MemoryRead:
      return uninterpretedApplication(expression, /*booleanResult=*/false);
    case TensorSemanticKind::Integer: {
      auto attribute = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
      std::optional<unsigned> width = getBitWidth(expression.getType());
      if (!attribute || !width || *width == 0 || *width > 64)
        return fail(expression, "unsupported integer literal");
      if (*width == 1)
        return context.bool_val(attribute.getValue().isOne());
      return context.bv_val(attribute.getValue().getZExtValue(), *width);
    }
    case TensorSemanticKind::Boolean: {
      auto attribute = dyn_cast_or_null<BoolAttr>(expression.getPayload());
      if (!attribute)
        return fail(expression, "Boolean literal has no BoolAttr payload");
      return context.bool_val(attribute.getValue());
    }
    case TensorSemanticKind::Unobserved: {
      // TensorDenotation is a partial function. The final inequivalence query
      // compares Value only under lhsValid && rhsValid, so a bottom value in
      // an inactive lazy-select arm may be totalized to one canonical zero.
      // This matches concrete fingerprint evaluation and does not assign an
      // observable value outside the denotation's Defined domain.
      std::optional<unsigned> width = getBitWidth(expression.getType());
      if (!width || *width == 0)
        return fail(expression, "unobserved value has no integer bit width");
      if (*width == 1)
        return context.bool_val(false);
      return context.bv_val(0, *width);
    }
    case TensorSemanticKind::Not: {
      std::optional<z3::expr> value = operand(expression, 0);
      if (!value)
        return std::nullopt;
      if (value->is_bool())
        return !*value;
      if (value->is_bv())
        return ~*value;
      return fail(expression, "not expects Boolean or bit-vector input");
    }
    case TensorSemanticKind::And:
    case TensorSemanticKind::Or:
    case TensorSemanticKind::Xor: {
      if (operands.empty())
        return fail(expression, "Boolean connective has no operands");
      std::optional<z3::expr> result = requireBool(expression, 0);
      if (!result)
        return std::nullopt;
      for (unsigned index = 1; index < operands.size(); ++index) {
        std::optional<z3::expr> rhs = requireBool(expression, index);
        if (!rhs)
          return std::nullopt;
        if (kind == TensorSemanticKind::And)
          result = *result && *rhs;
        else if (kind == TensorSemanticKind::Or)
          result = *result || *rhs;
        else
          result = *result != *rhs;
      }
      return result;
    }
    case TensorSemanticKind::BitwiseAnd:
    case TensorSemanticKind::BitwiseOr:
    case TensorSemanticKind::BitwiseXor:
    case TensorSemanticKind::Add:
    case TensorSemanticKind::Subtract:
    case TensorSemanticKind::Multiply:
    case TensorSemanticKind::IntegerDivide:
    case TensorSemanticKind::SignedRemainder:
    case TensorSemanticKind::ShiftLeft:
    case TensorSemanticKind::ShiftRightLogical:
    case TensorSemanticKind::ShiftRightArithmetic: {
      std::optional<z3::expr> lhs = requireBitVector(expression, 0);
      std::optional<z3::expr> rhs = requireBitVector(expression, 1);
      if (!lhs || !rhs)
        return std::nullopt;
      if ((kind == TensorSemanticKind::ShiftLeft ||
           kind == TensorSemanticKind::ShiftRightLogical ||
           kind == TensorSemanticKind::ShiftRightArithmetic) &&
          lhs->get_sort().bv_size() != rhs->get_sort().bv_size()) {
        unsigned lhsWidth = lhs->get_sort().bv_size();
        unsigned rhsWidth = rhs->get_sort().bv_size();
        if (rhsWidth < lhsWidth)
          rhs = z3::zext(*rhs, lhsWidth - rhsWidth);
        else
          rhs = rhs->extract(lhsWidth - 1, 0);
      }
      if (lhs->get_sort().bv_size() != rhs->get_sort().bv_size())
        return fail(expression, "bit-vector widths differ");
      std::optional<z3::expr> computed;
      switch (kind) {
      case TensorSemanticKind::BitwiseAnd:
        computed = *lhs & *rhs;
        break;
      case TensorSemanticKind::BitwiseOr:
        computed = *lhs | *rhs;
        break;
      case TensorSemanticKind::BitwiseXor:
        computed = *lhs ^ *rhs;
        break;
      case TensorSemanticKind::Add:
        computed = *lhs + *rhs;
        break;
      case TensorSemanticKind::Subtract:
        computed = *lhs - *rhs;
        break;
      case TensorSemanticKind::Multiply:
        computed = *lhs * *rhs;
        break;
      case TensorSemanticKind::IntegerDivide:
        computed = z3::udiv(*lhs, *rhs);
        break;
      case TensorSemanticKind::SignedRemainder:
        computed = z3::srem(*lhs, *rhs);
        break;
      case TensorSemanticKind::ShiftLeft:
        computed = z3::shl(*lhs, *rhs);
        break;
      case TensorSemanticKind::ShiftRightLogical:
        computed = z3::lshr(*lhs, *rhs);
        break;
      case TensorSemanticKind::ShiftRightArithmetic:
        computed = z3::ashr(*lhs, *rhs);
        break;
      default:
        llvm_unreachable("handled bit-vector operation");
      }
      if (getBitWidth(expression.getType()) == 1)
        return *computed == context.bv_val(1, 1);
      return computed;
    }
    case TensorSemanticKind::PointerAdd: {
      std::optional<unsigned> width = getBitWidth(expression.getType());
      if (!width || *width == 0)
        return fail(expression, "pointer_add has an unsupported result type");
      std::optional<z3::expr> pointer =
          requireUnsignedBitVector(expression, 0, *width);
      std::optional<z3::expr> offset =
          requireUnsignedBitVector(expression, 1, *width);
      if (!pointer || !offset)
        return std::nullopt;
      return *pointer + *offset;
    }
    case TensorSemanticKind::SignedMultiplyHigh:
    case TensorSemanticKind::UnsignedMultiplyHigh: {
      std::optional<z3::expr> lhs = requireBitVector(expression, 0);
      std::optional<z3::expr> rhs = requireBitVector(expression, 1);
      if (!lhs || !rhs ||
          lhs->get_sort().bv_size() != rhs->get_sort().bv_size())
        return fail(expression, "multiply-high operand widths differ");
      unsigned width = lhs->get_sort().bv_size();
      z3::expr wideLhs = kind == TensorSemanticKind::SignedMultiplyHigh
                             ? z3::sext(*lhs, width)
                             : z3::zext(*lhs, width);
      z3::expr wideRhs = kind == TensorSemanticKind::SignedMultiplyHigh
                             ? z3::sext(*rhs, width)
                             : z3::zext(*rhs, width);
      z3::expr high = (wideLhs * wideRhs).extract(2 * width - 1, width);
      if (getBitWidth(expression.getType()) == 1)
        return high == context.bv_val(1, 1);
      return high;
    }
    case TensorSemanticKind::Equal: {
      std::optional<z3::expr> lhs = operand(expression, 0);
      std::optional<z3::expr> rhs = operand(expression, 1);
      if (!lhs || !rhs || !z3::eq(lhs->get_sort(), rhs->get_sort()))
        return fail(expression, "comparison operand sorts differ");
      return *lhs == *rhs;
    }
    case TensorSemanticKind::SignedLessThan:
    case TensorSemanticKind::UnsignedLessThan: {
      std::optional<z3::expr> lhs = requireBitVector(expression, 0);
      std::optional<z3::expr> rhs = requireBitVector(expression, 1);
      if (!lhs || !rhs || !z3::eq(lhs->get_sort(), rhs->get_sort()))
        return fail(expression, "ordered comparison operand sorts differ");
      if (kind == TensorSemanticKind::SignedLessThan)
        return *lhs < *rhs;
      return z3::ult(*lhs, *rhs);
    }
    case TensorSemanticKind::Select: {
      std::optional<z3::expr> condition = requireBool(expression, 0);
      std::optional<z3::expr> trueValue = operand(expression, 1);
      std::optional<z3::expr> falseValue = operand(expression, 2);
      if (!condition || !trueValue || !falseValue ||
          !z3::eq(trueValue->get_sort(), falseValue->get_sort()))
        return fail(expression, "select operands are not type-compatible");
      return z3::ite(*condition, *trueValue, *falseValue);
    }
    case TensorSemanticKind::ZeroExtend:
    case TensorSemanticKind::SignExtend:
    case TensorSemanticKind::Truncate: {
      std::optional<z3::expr> input = operand(expression, 0);
      std::optional<unsigned> width = getBitWidth(expression.getType());
      if (!input || !width || *width == 0)
        return fail(expression, "integer cast has an unsupported result type");
      if (input->is_bool()) {
        if (*width == 1)
          return input;
        if (kind == TensorSemanticKind::Truncate)
          return fail(expression, "truncate widens its Boolean operand");
        z3::expr zero = context.bv_val(0, *width);
        z3::expr one = kind == TensorSemanticKind::ZeroExtend
                           ? context.bv_val(1, *width)
                           : ~context.bv_val(0, *width);
        return z3::ite(*input, one, zero);
      }
      if (!input->is_bv())
        return fail(expression, "integer cast expects Bool or bit-vector");
      unsigned inputWidth = input->get_sort().bv_size();
      if (kind == TensorSemanticKind::Truncate) {
        if (*width > inputWidth)
          return fail(expression, "truncate widens its operand");
        if (*width == 1)
          return input->extract(0, 0) == context.bv_val(1, 1);
        return input->extract(*width - 1, 0);
      }
      if (*width < inputWidth)
        return fail(expression, "extension narrows its operand");
      if (*width == inputWidth)
        return input;
      return kind == TensorSemanticKind::ZeroExtend
                 ? std::optional<z3::expr>(
                       z3::zext(*input, *width - inputWidth))
                 : std::optional<z3::expr>(
                       z3::sext(*input, *width - inputWidth));
    }
    case TensorSemanticKind::ApplyUnary: {
      auto payload = dyn_cast_or_null<DictionaryAttr>(expression.getPayload());
      auto role =
          payload ? dyn_cast_or_null<StringAttr>(payload.get("semantic_role"))
                  : StringAttr{};
      if (!role || role.getValue() != "shift_amount_valid")
        return fail(expression, "unsupported unary semantic primitive");
      std::optional<z3::expr> amount = requireBitVector(expression, 0);
      std::optional<unsigned> width =
          getBitWidth(expression.getOperands().front().getType());
      if (!amount || !width || *width == 0)
        return fail(expression, "invalid shift-amount predicate");
      return z3::ult(*amount, context.bv_val(static_cast<uint64_t>(*width),
                                             amount->get_sort().bv_size()));
    }
    default:
      return fail(expression, (llvm::Twine("unsupported semantic kind #") +
                               llvm::Twine(static_cast<unsigned>(kind)))
                                  .str());
    }
  }

  z3::context &context;
  TranslationCache &translated;
  AtomCache &atoms;
  UninterpretedFunctionCache &uninterpretedFunctions;
  std::string error;
};

} // namespace

class Z3SemanticSolver::Impl {
public:
  explicit Impl(unsigned timeoutMilliseconds,
                unsigned maxQueriesPerWorkerContext,
                unsigned retryTimeoutMilliseconds,
                unsigned bitBlastTimeoutMilliseconds,
                unsigned maxBitBlastWorkers)
      : timeoutMilliseconds(timeoutMilliseconds),
        maxQueriesPerWorkerContext(
            std::max(1u, maxQueriesPerWorkerContext)),
        retryTimeoutMilliseconds(retryTimeoutMilliseconds),
        bitBlastTimeoutMilliseconds(bitBlastTimeoutMilliseconds),
        maxBitBlastWorkers(std::max(1u, maxBitBlastWorkers)) {}

  SemanticProofResult compare(const TensorDenotation &lhs,
                              const TensorDenotation &rhs) const {
    ++queries;
    SemanticProofResult result;
    if (!lhs.isValid() || !rhs.isValid()) {
      ++unknown;
      ++unsupported;
      result.reason = "Z3 proof requires complete Value and Valid denotations";
      return result;
    }
    if (!sameNonValueDomains(lhs, rhs)) {
      ++unknown;
      ++unsupported;
      result.reason =
          "Z3 fallback does not replace shape/index/layout/state proofs";
      return result;
    }

    // Offline proof classifies independent pairs in parallel. Z3 contexts are
    // not shared across threads, so retain one incremental context/solver/cache
    // per worker instead of serializing the whole stage behind a global mutex.
    std::shared_ptr<WorkerState> worker = acquireWorker();
    struct QueryCompletion {
      WorkerState &worker;
      ~QueryCompletion() { ++worker.completedQueries; }
    } queryCompletion{*worker};
    ExpressionTranslator translator(worker->context, worker->translated,
                                    worker->atoms,
                                    worker->uninterpretedFunctions);
    std::optional<z3::expr> lhsValid = translator.translate(lhs.defined);
    std::optional<z3::expr> rhsValid = translator.translate(rhs.defined);
    if (!lhsValid || !rhsValid || !lhsValid->is_bool() ||
        !rhsValid->is_bool()) {
      ++unknown;
      ++unsupported;
      result.reason =
          "Z3 translation unsupported: " + translator.getError().str();
      trimTranslationCache(*worker);
      return result;
    }

    std::string lastCounterexampleModel;
    std::string lastUnknownReason;
    auto check = [&](const z3::expr &condition) {
      z3::expr simplified = condition.simplify();
      z3::expr assumptions[] = {simplified};
      z3::check_result status = worker->solver.check(1, assumptions);
      if (status == z3::sat)
        lastCounterexampleModel = worker->solver.get_model().to_string();
      if (status == z3::unknown)
        lastUnknownReason = worker->solver.reason_unknown();
      if (status != z3::unknown)
        return status;
      bool retryableUnknown =
          lastUnknownReason.find("timeout") != std::string::npos ||
          lastUnknownReason == "canceled" || lastUnknownReason == "incomplete";
      if (!retryableUnknown)
        return status;

      if (retryTimeoutMilliseconds > timeoutMilliseconds) {
        ++retryChecks;
        z3::params retryParameters(worker->context);
        retryParameters.set("timeout", retryTimeoutMilliseconds);
        worker->solver.set(retryParameters);
        status = worker->solver.check(1, assumptions);
        if (status == z3::unsat)
          ++retryUnsatChecks;
        else if (status == z3::sat) {
          ++retrySatChecks;
          lastCounterexampleModel = worker->solver.get_model().to_string();
        } else {
          ++retryUnknown;
          lastUnknownReason = worker->solver.reason_unknown();
        }
        z3::params primaryParameters(worker->context);
        primaryParameters.set("timeout", timeoutMilliseconds);
        worker->solver.set(primaryParameters);
      }

      if (status == z3::unknown && bitBlastTimeoutMilliseconds != 0 &&
          (lastUnknownReason.find("timeout") != std::string::npos ||
           lastUnknownReason == "canceled")) {
        acquireBitBlastWorker();
        auto releaseBitBlastWorkerOnExit =
            llvm::make_scope_exit([&] { releaseBitBlastWorker(); });
        ++bitBlastChecks;
        z3::tactic pipeline =
            z3::tactic(worker->context, "simplify") &
            z3::tactic(worker->context, "bit-blast") &
            z3::tactic(worker->context, "sat");
        z3::solver bitBlastSolver = pipeline.mk_solver();
        z3::params parameters(worker->context);
        parameters.set("timeout", bitBlastTimeoutMilliseconds);
        bitBlastSolver.set(parameters);
        bitBlastSolver.add(simplified);
        status = bitBlastSolver.check();
        if (status == z3::unsat)
          ++bitBlastUnsatChecks;
        else if (status == z3::sat) {
          ++bitBlastSatChecks;
          lastCounterexampleModel = bitBlastSolver.get_model().to_string();
        } else {
          ++bitBlastUnknown;
          lastUnknownReason = bitBlastSolver.reason_unknown();
        }
      }

      // The regular solver can report "incomplete" for formulas in the
      // quantifier-free AUF+BV fragment.  This is common for semantic
      // denotations containing MemoryRead/MemoryValid uninterpreted
      // functions: plain bit-blasting cannot eliminate those functions, but
      // Z3's dedicated qfaufbv tactic can decide the same formula.  Run it
      // for every remaining unknown result, including incomplete results, so
      // the offline proof stage does not leave a resolvable pair pending.
      if (status == z3::unknown) {
        acquireBitBlastWorker();
        auto releaseQfaufbvWorkerOnExit =
            llvm::make_scope_exit([&] { releaseBitBlastWorker(); });
        z3::tactic qfaufbv(worker->context, "qfaufbv");
        z3::solver qfaufbvSolver = qfaufbv.mk_solver();
        z3::params parameters(worker->context);
        unsigned qfaufbvTimeoutMilliseconds = std::max(
            timeoutMilliseconds,
            std::max(retryTimeoutMilliseconds, bitBlastTimeoutMilliseconds));
        parameters.set("timeout", qfaufbvTimeoutMilliseconds);
        qfaufbvSolver.set(parameters);
        qfaufbvSolver.add(simplified);
        status = qfaufbvSolver.check();
        if (status == z3::sat) {
          lastCounterexampleModel = qfaufbvSolver.get_model().to_string();
        } else if (status == z3::unknown) {
          lastUnknownReason = qfaufbvSolver.reason_unknown();
        }
      }
      return status;
    };
    auto makeProved = [&](StringRef step, StringRef reason) {
      ++proved;
      SemanticEquivalenceProof proof;
      proof.proofBackend = kProofBackendName.str();
      proof.lhs = lhs;
      proof.rhs = rhs;
      proof.verifiedDomains = getVerifiedDomains(lhs);
      proof.steps.push_back(
          SemanticProofStep{step.str(), lhs.value, rhs.value});
      SemanticProofResult provedResult;
      provedResult.status = SemanticProofStatus::Proved;
      provedResult.proof = std::move(proof);
      provedResult.reason = reason.str();
      trimTranslationCache(*worker);
      return provedResult;
    };
    auto makeRefuted = [&](StringRef reason) {
      ++refuted;
      SemanticProofResult refutedResult;
      refutedResult.status = SemanticProofStatus::Refuted;
      std::string model = lastCounterexampleModel;
      constexpr size_t kMaximumModelCharacters = 2048;
      if (model.size() > kMaximumModelCharacters)
        model.resize(kMaximumModelCharacters);
      refutedResult.reason = reason.str() + ": " + model;
      trimTranslationCache(*worker);
      return refutedResult;
    };
    auto makeUnknown = [&]() {
      ++unknown;
      std::string reason = lastUnknownReason;
      // Z3 4.8 reports a per-check timeout as "canceled" while newer builds
      // may report "timeout". This backend never calls Z3_interrupt, so both
      // spellings describe the configured solver timeout here.
      if (reason.find("timeout") != std::string::npos || reason == "canceled")
        ++timedOut;
      else
        ++otherUnknown;
      SemanticProofResult unknownResult;
      unknownResult.reason = "Z3 returned unknown: " + reason;
      trimTranslationCache(*worker);
      return unknownResult;
    };

    // Definedness is observable even when the value expression itself uses a
    // semantic kind outside the current BV translator.  Prove/refute this
    // domain first, and translate Value only when a jointly defined model
    // exists.
    if (!lhs.defined.structurallyEquals(rhs.defined)) {
      z3::expr validityDifference = (*lhsValid != *rhsValid).simplify();
      if (!validityDifference.is_false()) {
        z3::check_result validityStatus = check(validityDifference);
        if (validityStatus == z3::sat) {
          if (validityDifference.is_true())
            ++simplifierRefuted;
          return makeRefuted("Z3 found a definedness counterexample");
        }
        if (validityStatus == z3::unknown)
          return makeUnknown();
      }
    }
    std::optional<bool> literalValidity = getBooleanLiteral(lhs.defined);
    if (literalValidity && !*literalValidity)
      return makeProved("z3-unsat-defined-domain",
                        "Z3 proved both denotations are never defined");
    if (!literalValidity || !*literalValidity) {
      z3::expr jointlyDefined = (*lhsValid && *rhsValid).simplify();
      if (jointlyDefined.is_false()) {
        ++simplifierProved;
        return makeProved("z3-simplified-unobservable-domain",
                          "Z3 simplifier proved both denotations are never "
                          "jointly defined");
      }
      if (!jointlyDefined.is_true()) {
        z3::check_result observableStatus = check(jointlyDefined);
        if (observableStatus == z3::unsat)
          return makeProved("z3-unsat-defined-domain",
                            "Z3 proved both denotations are never defined");
        if (observableStatus == z3::unknown)
          return makeUnknown();
      }
    }

    std::optional<z3::expr> lhsValue = translator.translate(lhs.value);
    std::optional<z3::expr> rhsValue = translator.translate(rhs.value);
    if (!lhsValue || !rhsValue ||
        !z3::eq(lhsValue->get_sort(), rhsValue->get_sort())) {
      ++unknown;
      ++unsupported;
      result.reason =
          "Z3 translation unsupported: " + translator.getError().str();
      trimTranslationCache(*worker);
      return result;
    }

    z3::expr inequivalent =
        (*lhsValid != *rhsValid) ||
        (*lhsValid && *rhsValid && (*lhsValue != *rhsValue));
    z3::expr simplifiedInequality = inequivalent.simplify();
    if (simplifiedInequality.is_false()) {
      ++simplifierProved;
      return makeProved("z3-simplified-semantic-inequality",
                        "Z3 simplifier reduced semantic inequality to false");
    }
    z3::check_result status = check(simplifiedInequality);
    if (status == z3::unsat)
      return makeProved("z3-unsat-semantic-inequality",
                        "Z3 proved semantic inequality unsatisfiable");
    if (status == z3::sat) {
      if (simplifiedInequality.is_true())
        ++simplifierRefuted;
      return makeRefuted("Z3 found a value counterexample");
    }

    // Variable shifts combined with nonlinear arithmetic are difficult for a
    // monolithic QF_BV query.  When both denotations share a conjunctive
    // shift-validity predicate, its amount has the exact finite domain
    // [0, bitwidth).  Splitting that domain is complete and remains generic:
    // SAT in any branch is a counterexample, while UNSAT in every branch is a
    // proof.  This fallback runs only after the configured SMT/tactic retries
    // have timed out.
    if (lhs.defined.structurallyEquals(rhs.defined) &&
        (lastUnknownReason.find("timeout") != std::string::npos ||
         lastUnknownReason == "canceled")) {
      SmallVector<TensorSemanticExpr, 2> shiftAmounts;
      collectConjunctiveShiftAmounts(lhs.defined, shiftAmounts);
      const unsigned branchTimeoutMilliseconds = std::max(
          1000u, std::max(timeoutMilliseconds, retryTimeoutMilliseconds));
      for (const TensorSemanticExpr &amount : shiftAmounts) {
        std::optional<unsigned> width = getBitWidth(amount.getType());
        std::optional<z3::expr> translatedAmount = translator.translate(amount);
        if (!width || *width == 0 || *width > 128 || !translatedAmount ||
            !translatedAmount->is_bv())
          continue;

        z3::params branchParameters(worker->context);
        branchParameters.set("timeout", branchTimeoutMilliseconds);
        worker->solver.set(branchParameters);
        bool allBranchesUnsat = true;
        for (uint64_t value = 0; value < *width; ++value) {
          ++finiteShiftDomainChecks;
          z3::expr branch =
              (*translatedAmount ==
               worker->context.bv_val(value,
                                      translatedAmount->get_sort().bv_size()))
                  .simplify();
          z3::expr assumptions[] = {simplifiedInequality, branch};
          z3::check_result branchStatus =
              worker->solver.check(2, assumptions);
          if (branchStatus == z3::sat) {
            ++finiteShiftDomainSatChecks;
            lastCounterexampleModel = worker->solver.get_model().to_string();
            z3::params primaryParameters(worker->context);
            primaryParameters.set("timeout", timeoutMilliseconds);
            worker->solver.set(primaryParameters);
            return makeRefuted(
                "Z3 found a counterexample after finite shift-domain split");
          }
          if (branchStatus == z3::unsat) {
            ++finiteShiftDomainUnsatChecks;
            continue;
          }
          ++finiteShiftDomainUnknownChecks;
          lastUnknownReason = worker->solver.reason_unknown();
          allBranchesUnsat = false;
        }
        z3::params primaryParameters(worker->context);
        primaryParameters.set("timeout", timeoutMilliseconds);
        worker->solver.set(primaryParameters);
        if (allBranchesUnsat)
          return makeProved(
              "z3-finite-shift-domain-unsat",
              "Z3 proved every valid finite shift-amount branch unsatisfiable");
      }
    }
    return makeUnknown();
  }

  struct WorkerState {
    explicit WorkerState(unsigned timeoutMilliseconds)
        : solver(context) {
      z3::params parameters(context);
      parameters.set("timeout", timeoutMilliseconds);
      solver.set(parameters);
    }

    z3::context context;
    z3::solver solver;
    TranslationCache translated;
    AtomCache atoms;
    UninterpretedFunctionCache uninterpretedFunctions;
    unsigned completedQueries = 0;
  };

  void acquireBitBlastWorker() const {
    std::unique_lock<std::mutex> lock(bitBlastPoolMutex);
    bitBlastPoolCondition.wait(
        lock, [&] { return activeBitBlastWorkers < maxBitBlastWorkers; });
    ++activeBitBlastWorkers;
  }

  void releaseBitBlastWorker() const {
    {
      std::lock_guard<std::mutex> lock(bitBlastPoolMutex);
      --activeBitBlastWorkers;
    }
    bitBlastPoolCondition.notify_one();
  }

  std::shared_ptr<WorkerState> acquireWorker() const {
    std::lock_guard<std::mutex> lock(workerPoolMutex);
    // Z3 contexts retain internal AST and solver state even after the C++
    // expression cache is cleared.  A context that serves an unbounded stream
    // of unrelated offline candidates therefore grows in memory and makes
    // later, otherwise cheap queries hit their timeout.  Retire idle workers
    // after a bounded number of queries; the pool still contains at most the
    // proof-stage concurrency, while new workers start from a clean context.
    for (auto worker = workers.begin(); worker != workers.end();) {
      if (worker->use_count() == 1 &&
          (*worker)->completedQueries >= maxQueriesPerWorkerContext) {
        worker = workers.erase(worker);
        ++workerContextRecycles;
      } else {
        ++worker;
      }
    }
    for (const std::shared_ptr<WorkerState> &worker : workers)
      if (worker.use_count() == 1)
        return worker;
    std::shared_ptr<WorkerState> worker =
        std::make_shared<WorkerState>(timeoutMilliseconds);
    workers.push_back(worker);
    ++workerContextCreations;
    return worker;
  }

  static void trimTranslationCache(WorkerState &worker) {
    // Retaining translations makes repeated denotations cheap while bounding
    // the extra proof memory independently of universe size.  Entries own the
    // semantic nodes they key, so allocator address reuse cannot alias a later
    // expression to an obsolete Z3 AST.
    constexpr size_t kMaximumCachedExpressions = 65536;
    if (worker.translated.size() <= kMaximumCachedExpressions)
      return;
    worker.translated.clear();
    worker.atoms.clear();
  }

  Z3SemanticSolverStatistics getStatistics() const {
    std::lock_guard<std::mutex> lock(workerPoolMutex);
    return Z3SemanticSolverStatistics{
        queries.load(),          proved.load(),      refuted.load(),
        unknown.load(),          unsupported.load(), timedOut.load(),
        otherUnknown.load(),     simplifierProved.load(),
        simplifierRefuted.load(), retryChecks.load(),
        retryUnsatChecks.load(),  retrySatChecks.load(),
        retryUnknown.load(),      bitBlastChecks.load(),
        bitBlastUnsatChecks.load(), bitBlastSatChecks.load(),
        bitBlastUnknown.load(),
        finiteShiftDomainChecks.load(),
        finiteShiftDomainUnsatChecks.load(),
        finiteShiftDomainSatChecks.load(),
        finiteShiftDomainUnknownChecks.load(), workers.size(),
        workerContextCreations.load(),
        workerContextRecycles.load()};
  }

  const unsigned timeoutMilliseconds;
  const unsigned maxQueriesPerWorkerContext;
  const unsigned retryTimeoutMilliseconds;
  const unsigned bitBlastTimeoutMilliseconds;
  const unsigned maxBitBlastWorkers;
  mutable std::mutex bitBlastPoolMutex;
  mutable std::condition_variable bitBlastPoolCondition;
  mutable unsigned activeBitBlastWorkers = 0;
  mutable std::mutex workerPoolMutex;
  mutable std::vector<std::shared_ptr<WorkerState>> workers;
  mutable std::atomic<uint64_t> queries{0};
  mutable std::atomic<uint64_t> proved{0};
  mutable std::atomic<uint64_t> refuted{0};
  mutable std::atomic<uint64_t> unknown{0};
  mutable std::atomic<uint64_t> unsupported{0};
  mutable std::atomic<uint64_t> timedOut{0};
  mutable std::atomic<uint64_t> otherUnknown{0};
  mutable std::atomic<uint64_t> simplifierProved{0};
  mutable std::atomic<uint64_t> simplifierRefuted{0};
  mutable std::atomic<uint64_t> retryChecks{0};
  mutable std::atomic<uint64_t> retryUnsatChecks{0};
  mutable std::atomic<uint64_t> retrySatChecks{0};
  mutable std::atomic<uint64_t> retryUnknown{0};
  mutable std::atomic<uint64_t> bitBlastChecks{0};
  mutable std::atomic<uint64_t> bitBlastUnsatChecks{0};
  mutable std::atomic<uint64_t> bitBlastSatChecks{0};
  mutable std::atomic<uint64_t> bitBlastUnknown{0};
  mutable std::atomic<uint64_t> finiteShiftDomainChecks{0};
  mutable std::atomic<uint64_t> finiteShiftDomainUnsatChecks{0};
  mutable std::atomic<uint64_t> finiteShiftDomainSatChecks{0};
  mutable std::atomic<uint64_t> finiteShiftDomainUnknownChecks{0};
  mutable std::atomic<uint64_t> workerContextCreations{0};
  mutable std::atomic<uint64_t> workerContextRecycles{0};
};

Z3SemanticSolver::Z3SemanticSolver(unsigned timeoutMilliseconds,
                                   unsigned maxQueriesPerWorkerContext,
                                   unsigned retryTimeoutMilliseconds,
                                   unsigned bitBlastTimeoutMilliseconds,
                                   unsigned maxBitBlastWorkers)
    : impl(std::make_unique<Impl>(timeoutMilliseconds,
                                  maxQueriesPerWorkerContext,
                                  retryTimeoutMilliseconds,
                                  bitBlastTimeoutMilliseconds,
                                  maxBitBlastWorkers)) {}

Z3SemanticSolver::~Z3SemanticSolver() = default;

StringRef Z3SemanticSolver::getProofBackendName() const {
  return kProofBackendName;
}

SemanticProofResult
Z3SemanticSolver::compareNormalized(const TensorDenotation &lhs,
                                    const TensorDenotation &rhs) const {
  return impl->compare(lhs, rhs);
}

SemanticProofResult
Z3SemanticSolver::replay(const SemanticEquivalenceProof &proof) const {
  SemanticProofResult result;
  if (proof.proofBackend != kProofBackendName) {
    result.reason = "Z3 certificate names a different proof backend";
    return result;
  }
  result = impl->compare(proof.lhs, proof.rhs);
  if (!result.isProved() || !result.proof ||
      result.proof->getStableHash() != proof.getStableHash()) {
    SemanticProofResult failure;
    failure.reason = "Z3 certificate replay did not reproduce its proof hash";
    return failure;
  }
  return result;
}

Z3SemanticSolverStatistics Z3SemanticSolver::getStatistics() const {
  return impl->getStatistics();
}

} // namespace mlir::triton::cfg

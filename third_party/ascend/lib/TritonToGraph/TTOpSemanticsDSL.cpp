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
#include "TritonToGraph/CanonicalRegion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

namespace {

static constexpr llvm::StringLiteral kResultTypeAttr = "ags.result_type";
static constexpr llvm::StringLiteral kSymbolIdAttr = "ags.symbol_id";
static constexpr llvm::StringLiteral kResultIndexAttr = "ags.result_index";
static constexpr llvm::StringLiteral kOperationResultTypesAttr =
    "ags.operation_result_types";
static constexpr llvm::StringLiteral kCanonicalRegionsAttr =
    "ags.canonical_regions";
static constexpr llvm::StringLiteral kCallSummaryRegionsAttr =
    "ags.call_summary_regions";

static constexpr char kBuiltinTTOpSemanticsSource[] =
#include "BuiltinTTOpSemantics.inc"
    ;

enum class TokenKind {
  Identifier,
  Integer,
  LParen,
  RParen,
  LBrace,
  RBrace,
  Comma,
  Equal,
  Semicolon,
  End,
};

struct Token {
  TokenKind kind = TokenKind::End;
  StringRef spelling;
  unsigned line = 1;
};

class SemanticsLexer {
public:
  explicit SemanticsLexer(StringRef source) : remaining(source) {}

  Token next() {
    skipTrivia();
    if (remaining.empty())
      return Token{TokenKind::End, {}, line};

    const char first = remaining.front();
    switch (first) {
    case '(':
      return punctuation(TokenKind::LParen);
    case ')':
      return punctuation(TokenKind::RParen);
    case '{':
      return punctuation(TokenKind::LBrace);
    case '}':
      return punctuation(TokenKind::RBrace);
    case ',':
      return punctuation(TokenKind::Comma);
    case '=':
      return punctuation(TokenKind::Equal);
    case ';':
      return punctuation(TokenKind::Semicolon);
    default:
      break;
    }

    if (std::isdigit(static_cast<unsigned char>(first)) ||
        (first == '-' && remaining.size() > 1 &&
         std::isdigit(static_cast<unsigned char>(remaining[1])))) {
      size_t length = first == '-' ? 1 : 0;
      while (length < remaining.size() &&
             std::isdigit(static_cast<unsigned char>(remaining[length])))
        ++length;
      return take(TokenKind::Integer, length);
    }

    if (std::isalpha(static_cast<unsigned char>(first)) || first == '_') {
      size_t length = 1;
      while (length < remaining.size()) {
        const char current = remaining[length];
        if (!std::isalnum(static_cast<unsigned char>(current)) &&
            current != '_' && current != '.')
          break;
        ++length;
      }
      return take(TokenKind::Identifier, length);
    }

    fail("unexpected character");
  }

private:
  void skipTrivia() {
    while (!remaining.empty()) {
      if (remaining.front() == '#') {
        size_t newline = remaining.find('\n');
        if (newline == StringRef::npos) {
          remaining = {};
          return;
        }
        remaining = remaining.drop_front(newline + 1);
        ++line;
        continue;
      }
      if (!std::isspace(static_cast<unsigned char>(remaining.front())))
        return;
      if (remaining.front() == '\n')
        ++line;
      remaining = remaining.drop_front();
    }
  }

  Token punctuation(TokenKind kind) { return take(kind, 1); }

  Token take(TokenKind kind, size_t length) {
    Token token{kind, remaining.take_front(length), line};
    remaining = remaining.drop_front(length);
    return token;
  }

  [[noreturn]] void fail(StringRef message) const {
    llvm::report_fatal_error("invalid TT semantics at line " +
                             Twine(line) + ": " + message);
  }

  StringRef remaining;
  unsigned line = 1;
};

struct Formula;
using FormulaPtr = std::shared_ptr<const Formula>;

struct Formula {
  enum class Kind { Identifier, Integer, Boolean, Call };

  Kind kind = Kind::Identifier;
  std::string spelling;
  int64_t integer = 0;
  bool boolean = false;
  SmallVector<FormulaPtr, 4> arguments;
};

struct SemanticClause {
  enum class EffectKind {
    Pure,
    Read,
    Write,
    Atomic,
    Observable,
    BufferRead,
    BufferWrite,
  };

  std::string operationName;
  SmallVector<std::string, 4> operands;
  EffectKind effectKind = EffectKind::Pure;
  SmallVector<std::string, 2> effectOperands;
  FormulaPtr signature;
  FormulaPtr shareable;
  FormulaPtr candidate;
  FormulaPtr explore;
  TTOpSearchPolicy searchPolicy = TTOpSearchPolicy::Searchable;
  bool hasExplicitSearchPolicy = false;
  // Multi-result operations are described as one semantic projection per
  // result. `operation_results` is the complete result signature and
  // `result_index` selects the projection described by value/valid.
  FormulaPtr resultIndex;
  FormulaPtr operationResults;
  // Structural region meaning is data, not a matcher. The first supported
  // constructor is binary(combiner_op, terminator_op); more region forms can
  // be added without changing the graph enumerator/materializer.
  FormulaPtr region;
  // When present, the final named operand is a one-or-more variadic tail.
  // The generic schema builder expands it only up to the finite universe
  // operand bound; importer/denoter matching remains arity-generic.
  std::optional<std::string> variadicOperand;
  // Optional arithmetic progression for legal concrete arities of a
  // variadic operation: sequence(first, step). This represents several ODS
  // variadic groups without operation-specific candidate code.
  FormulaPtr operandArity;
  // Optional operand whose logical tensor coordinates define lane-wise state
  // expansion when other shaped operands are index vectors or broadcasts.
  FormulaPtr laneDomain;
  FormulaPtr value;
  FormulaPtr definedness;
  FormulaPtr effect;
  FormulaPtr memory;
  FormulaPtr memoryDefinedness;
  FormulaPtr launch;
  FormulaPtr launchDefinedness;
  FormulaPtr buffer;
  FormulaPtr bufferDefinedness;
  FormulaPtr action;
  FormulaPtr materialization;
};

class SemanticsParser {
public:
  explicit SemanticsParser(StringRef source) : lexer(source) { advance(); }

  SmallVector<SemanticClause, 16> parse() {
    SmallVector<SemanticClause, 16> clauses;
    while (current.kind != TokenKind::End)
      clauses.push_back(parseClause());
    return clauses;
  }

private:
  SemanticClause parseClause() {
    expectIdentifier("semantics");
    SemanticClause clause;
    Token effect = expect(TokenKind::Identifier);
    if (effect.spelling == "pure") {
      clause.effectKind = SemanticClause::EffectKind::Pure;
    } else if (effect.spelling == "observable") {
      clause.effectKind = SemanticClause::EffectKind::Observable;
    } else if (effect.spelling == "read" || effect.spelling == "write" ||
               effect.spelling == "atomic" ||
               effect.spelling == "buffer_read" ||
               effect.spelling == "buffer_write") {
      clause.effectKind =
          effect.spelling == "read"
              ? SemanticClause::EffectKind::Read
          : effect.spelling == "write"
              ? SemanticClause::EffectKind::Write
          : effect.spelling == "atomic"
              ? SemanticClause::EffectKind::Atomic
          : effect.spelling == "buffer_read"
              ? SemanticClause::EffectKind::BufferRead
              : SemanticClause::EffectKind::BufferWrite;
      expect(TokenKind::LParen);
      if (current.kind != TokenKind::RParen) {
        while (true) {
          clause.effectOperands.push_back(
              expect(TokenKind::Identifier).spelling.str());
          if (!consume(TokenKind::Comma))
            break;
        }
      }
      expect(TokenKind::RParen);
      if (clause.effectOperands.empty())
        fail("state effect requires at least one operand");
    } else {
      fail("semantics requires an explicit effect kind");
    }
    clause.operationName = expect(TokenKind::Identifier).spelling.str();
    expect(TokenKind::LParen);
    if (current.kind != TokenKind::RParen) {
      while (true) {
        std::string operand = expect(TokenKind::Identifier).spelling.str();
        if (llvm::is_contained(clause.operands, operand))
          fail("duplicate operand name");
        clause.operands.push_back(std::move(operand));
        if (!consume(TokenKind::Comma))
          break;
      }
    }
    expect(TokenKind::RParen);
    for (StringRef effectOperand : clause.effectOperands)
      if (!llvm::is_contained(clause.operands, effectOperand))
        fail(Twine("unknown effect operand '") + effectOperand + "'");
    expect(TokenKind::LBrace);
    while (current.kind != TokenKind::RBrace) {
      std::string field = expect(TokenKind::Identifier).spelling.str();
      expect(TokenKind::Equal);
      FormulaPtr formula = parseFormula();
      expect(TokenKind::Semicolon);
      FormulaPtr *destination = nullptr;
      if (field == "import")
        destination = &clause.signature;
      else if (field == "shareable")
        destination = &clause.shareable;
      else if (field == "candidate")
        destination = &clause.candidate;
      else if (field == "explore")
        destination = &clause.explore;
      else if (field == "result_index")
        destination = &clause.resultIndex;
      else if (field == "operation_results")
        destination = &clause.operationResults;
      else if (field == "region")
        destination = &clause.region;
      else if (field == "variadic") {
        if (formula->kind != Formula::Kind::Identifier)
          fail("variadic requires the final operand name");
        if (clause.variadicOperand)
          fail("duplicate semantics field 'variadic'");
        clause.variadicOperand = formula->spelling;
        continue;
      }
      else if (field == "operand_arity")
        destination = &clause.operandArity;
      else if (field == "lane_domain")
        destination = &clause.laneDomain;
      else if (field == "search") {
        if (formula->kind != Formula::Kind::Identifier ||
            (formula->spelling != "searchable" &&
             formula->spelling != "import_only" &&
             formula->spelling != "region_only"))
          fail("search requires searchable, import_only or region_only");
        if (clause.hasExplicitSearchPolicy)
          fail("duplicate semantics field 'search'");
        clause.hasExplicitSearchPolicy = true;
        clause.searchPolicy =
            formula->spelling == "searchable"
                ? TTOpSearchPolicy::Searchable
            : formula->spelling == "import_only"
                ? TTOpSearchPolicy::ImportOnly
                : TTOpSearchPolicy::RegionOnly;
        continue;
      }
      else if (field == "value")
        destination = &clause.value;
      else if (field == "valid")
        destination = &clause.definedness;
      else if (field == "effect")
        destination = &clause.effect;
      else if (field == "memory")
        destination = &clause.memory;
      else if (field == "memory_valid")
        destination = &clause.memoryDefinedness;
      else if (field == "launch")
        destination = &clause.launch;
      else if (field == "launch_valid")
        destination = &clause.launchDefinedness;
      else if (field == "buffer")
        destination = &clause.buffer;
      else if (field == "buffer_valid")
        destination = &clause.bufferDefinedness;
      else if (field == "action")
        destination = &clause.action;
      else if (field == "materialize")
        destination = &clause.materialization;
      else
        fail(Twine("unknown semantics field '") + field + "'");
      if (*destination)
        fail(Twine("duplicate semantics field '") + field + "'");
      *destination = std::move(formula);
    }
    expect(TokenKind::RBrace);
    if (clause.variadicOperand &&
        (clause.operands.empty() ||
         clause.operands.back() != *clause.variadicOperand))
      fail("variadic must name the final operation operand");
    if (clause.operandArity &&
        (!clause.variadicOperand ||
         clause.operandArity->kind != Formula::Kind::Call ||
         clause.operandArity->spelling != "sequence" ||
         clause.operandArity->arguments.size() != 2 ||
         llvm::any_of(clause.operandArity->arguments,
                      [](const FormulaPtr &argument) {
                        return argument->kind != Formula::Kind::Integer;
                      }) ||
         clause.operandArity->arguments[0]->integer <
             static_cast<int64_t>(clause.operands.size() - 1) ||
         clause.operandArity->arguments[1]->integer <= 0))
      fail("operand_arity requires variadic plus "
           "sequence(first>=fixed_operand_count, step>0)");
    if (clause.laneDomain &&
        (clause.laneDomain->kind != Formula::Kind::Identifier ||
         !llvm::is_contained(clause.operands,
                             clause.laneDomain->spelling)))
      fail("lane_domain requires one named operation operand");
    if (!clause.signature || !clause.value || !clause.definedness)
      fail("semantics requires import, value and valid fields");
    if ((clause.searchPolicy == TTOpSearchPolicy::ImportOnly ||
         clause.searchPolicy == TTOpSearchPolicy::RegionOnly) &&
        clause.candidate)
      fail("non-searchable semantics must not declare candidate");
    if (static_cast<bool>(clause.resultIndex) !=
        static_cast<bool>(clause.operationResults))
      fail("result_index and operation_results must be specified together");
    const bool importedResults =
        clause.resultIndex && clause.operationResults &&
        clause.resultIndex->kind == Formula::Kind::Identifier &&
        clause.resultIndex->spelling == "any" &&
        clause.operationResults->kind == Formula::Kind::Identifier &&
        clause.operationResults->spelling == "imported";
    if (clause.resultIndex && !importedResults &&
        clause.resultIndex->kind != Formula::Kind::Integer)
      fail("result_index requires a non-negative integer or any");
    if (clause.resultIndex && !importedResults &&
        clause.resultIndex->integer < 0)
      fail("result_index requires a non-negative integer");
    if (clause.operationResults && !importedResults &&
        (clause.operationResults->kind != Formula::Kind::Call ||
         clause.operationResults->spelling != "tuple" ||
         clause.operationResults->arguments.empty()))
      fail("operation_results requires tuple(type constructors...) or "
           "imported");
    if (importedResults && clause.candidate)
      fail("imported result signatures cannot declare candidates");
    if ((clause.effectKind == SemanticClause::EffectKind::Write ||
         clause.effectKind == SemanticClause::EffectKind::Atomic) &&
        (!clause.memory || !clause.memoryDefinedness))
      fail("write/atomic semantics requires memory and memory_valid fields");
    if (clause.effectKind == SemanticClause::EffectKind::BufferWrite &&
        (!clause.buffer || !clause.bufferDefinedness))
      fail("buffer_write semantics requires buffer and buffer_valid fields");
    if (static_cast<bool>(clause.launch) !=
        static_cast<bool>(clause.launchDefinedness))
      fail("launch and launch_valid must be specified together");
    if (clause.materialization &&
        (clause.materialization->kind != Formula::Kind::Identifier ||
         clause.materialization->spelling != "proof_only"))
      fail("materialize currently requires proof_only");
    return clause;
  }

  FormulaPtr parseFormula() {
    if (current.kind == TokenKind::Integer) {
      auto formula = std::make_shared<Formula>();
      formula->kind = Formula::Kind::Integer;
      if (current.spelling.getAsInteger(10, formula->integer))
        fail("invalid integer");
      advance();
      return formula;
    }

    Token identifier = expect(TokenKind::Identifier);
    if (identifier.spelling == "true" || identifier.spelling == "false") {
      auto formula = std::make_shared<Formula>();
      formula->kind = Formula::Kind::Boolean;
      formula->boolean = identifier.spelling == "true";
      return formula;
    }

    auto formula = std::make_shared<Formula>();
    formula->spelling = identifier.spelling.str();
    if (!consume(TokenKind::LParen)) {
      formula->kind = Formula::Kind::Identifier;
      return formula;
    }

    formula->kind = Formula::Kind::Call;
    if (current.kind != TokenKind::RParen) {
      while (true) {
        formula->arguments.push_back(parseFormula());
        if (!consume(TokenKind::Comma))
          break;
      }
    }
    expect(TokenKind::RParen);
    return formula;
  }

  bool consume(TokenKind kind) {
    if (current.kind != kind)
      return false;
    advance();
    return true;
  }

  Token expect(TokenKind kind) {
    if (current.kind != kind)
      fail("unexpected token");
    Token token = current;
    advance();
    return token;
  }

  void expectIdentifier(StringRef spelling) {
    Token token = expect(TokenKind::Identifier);
    if (token.spelling != spelling)
      fail("expected '" + spelling + "'");
  }

  void advance() { current = lexer.next(); }

  [[noreturn]] void fail(const Twine &message) const {
    llvm::report_fatal_error("invalid TT semantics at line " +
                             Twine(current.line) + ": " + message);
  }

  SemanticsLexer lexer;
  Token current;
};

static Type getResultType(const GraphExprNode &node) {
  if (!node.attributes)
    return {};
  auto type = dyn_cast_or_null<TypeAttr>(node.attributes.get(kResultTypeAttr));
  return type ? type.getValue() : Type{};
}

static Type getElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

enum class ConstraintTruth : uint8_t { False, True, Unknown };

struct ConstraintContext {
  Operation *operation = nullptr;
  Type resultType;
  ArrayRef<Type> operandTypes;
  ArrayRef<std::string> operandNames;
  DictionaryAttr attributes;
  bool compileOn91095 = false;
  bool allowUnknownOperands = false;
  bool allowUnknownAttributes = false;
  unsigned resultIndex = 0;
  // Concrete arity of the schema being solved.  During prefix search,
  // operandTypes contains only the assigned prefix; predicates use this field
  // to reject an impossible variadic arity without enumerating Type^N.
  unsigned expectedOperandCount = 0;
};

struct ConstraintType {
  Type type;
  bool unknown = false;
};

static ConstraintTruth negate(ConstraintTruth value) {
  if (value == ConstraintTruth::True)
    return ConstraintTruth::False;
  if (value == ConstraintTruth::False)
    return ConstraintTruth::True;
  return ConstraintTruth::Unknown;
}

static ConstraintType lookupConstraintType(const Formula &formula,
                                           const ConstraintContext &context) {
  if (formula.kind != Formula::Kind::Identifier)
    return {};
  if (formula.spelling == "result")
    return ConstraintType{context.resultType, false};
  for (auto [index, name] : llvm::enumerate(context.operandNames)) {
    if (name != formula.spelling)
      continue;
    if (index < context.operandTypes.size() && context.operandTypes[index])
      return ConstraintType{context.operandTypes[index], false};
    return ConstraintType{{}, context.allowUnknownOperands};
  }
  return {};
}

static Type makeBooleanLike(Type source) {
  if (!source)
    return {};
  MLIRContext *context = source.getContext();
  Type i1 = IntegerType::get(context, 1);
  if (auto tensor = dyn_cast<RankedTensorType>(source))
    return RankedTensorType::get(tensor.getShape(), i1,
                                 tensor.getEncoding());
  return isa<IntegerType>(source) ? i1 : Type{};
}

static Type makeSplitLike(Type source) {
  auto tensor = dyn_cast_or_null<RankedTensorType>(source);
  if (!tensor || tensor.getRank() == 0 || tensor.getShape().back() != 2 ||
      tensor.getEncoding())
    return {};
  ArrayRef<int64_t> shape = tensor.getShape().drop_back();
  return RankedTensorType::get(shape, tensor.getElementType());
}

static Type resolveResultTypeConstructor(const Formula &formula,
                                         const ConstraintContext &context) {
  if (formula.kind == Formula::Kind::Identifier)
    return lookupConstraintType(formula, context).type;
  if (formula.kind != Formula::Kind::Call ||
      formula.arguments.size() != 1)
    return {};
  ConstraintType source =
      lookupConstraintType(*formula.arguments.front(), context);
  if (source.unknown || !source.type)
    return {};
  if (formula.spelling == "same_as")
    return source.type;
  if (formula.spelling == "bool_like")
    return makeBooleanLike(source.type);
  if (formula.spelling == "split_like")
    return makeSplitLike(source.type);
  return {};
}

static SmallVector<Type, 2>
resolveOperationResultTypes(const Formula *formula,
                            const ConstraintContext &context) {
  if (!formula) {
    SmallVector<Type, 2> result;
    if (context.resultType)
      result.push_back(context.resultType);
    return result;
  }
  if (formula->kind != Formula::Kind::Call ||
      formula->spelling != "tuple")
    return {};
  SmallVector<Type, 2> result;
  for (const FormulaPtr &constructor : formula->arguments) {
    Type type = resolveResultTypeConstructor(*constructor, context);
    if (!type)
      return {};
    result.push_back(type);
  }
  return result;
}

static std::optional<int64_t> getConstraintInteger(const Formula &formula) {
  if (formula.kind != Formula::Kind::Integer)
    return std::nullopt;
  return formula.integer;
}

static bool containsCall(const Formula &formula, StringRef name) {
  if (formula.kind == Formula::Kind::Call && formula.spelling == name)
    return true;
  return llvm::any_of(formula.arguments, [&](const FormulaPtr &argument) {
    return containsCall(*argument, name);
  });
}

static void collectAttributeBindings(
    const Formula &formula,
    SmallVectorImpl<TTIRCandidateAttributeBinding> &bindings) {
  if (formula.kind == Formula::Kind::Call &&
      (formula.spelling == "attribute" ||
       formula.spelling == "attribute_i32" ||
       formula.spelling == "attribute_i64_array") &&
      formula.arguments.size() == 2 &&
      formula.arguments[0]->kind == Formula::Kind::Identifier &&
      formula.arguments[1]->kind == Formula::Kind::Identifier) {
    TTIRCandidateAttributeBinding binding;
    binding.attributeName = formula.arguments[0]->spelling;
    binding.domainName = formula.arguments[1]->spelling;
    binding.storageKind =
        formula.spelling == "attribute_i32"
            ? TTIRCandidateAttributeBinding::StorageKind::I32
        : formula.spelling == "attribute_i64_array"
            ? TTIRCandidateAttributeBinding::StorageKind::I64Array
            : TTIRCandidateAttributeBinding::StorageKind::Inferred;
    auto duplicate = llvm::find_if(bindings, [&](const auto &existing) {
      return existing.attributeName == binding.attributeName;
    });
    if (duplicate == bindings.end()) {
      bindings.push_back(std::move(binding));
    } else if (duplicate->domainName != binding.domainName ||
               duplicate->storageKind != binding.storageKind) {
      llvm::report_fatal_error("TT semantics attribute '" +
                               Twine(binding.attributeName) +
                               "' is bound to multiple domains");
    }
  }
  for (const FormulaPtr &argument : formula.arguments)
    collectAttributeBindings(*argument, bindings);
}

static void collectSemanticAttributeNames(
    const Formula &formula, SmallVectorImpl<std::string> &names) {
  if (formula.kind == Formula::Kind::Call && !formula.arguments.empty() &&
      formula.arguments.front()->kind == Formula::Kind::Identifier &&
      (formula.spelling == "attribute" ||
       formula.spelling == "attribute_i32" ||
       formula.spelling == "attribute_i64_array" ||
       formula.spelling == "attribute_optional" ||
       formula.spelling == "attribute_integer_equals" ||
       formula.spelling == "attribute_integer_not_equals" ||
       formula.spelling == "attribute_boolean_equals" ||
       formula.spelling == "attribute_absent" ||
       formula.spelling == "attribute_absent_or_empty" ||
       formula.spelling == "attribute_absent_or_false" ||
       formula.spelling == "attribute_integer_greater" ||
       formula.spelling == "segments_equal")) {
    StringRef name = formula.arguments.front()->spelling;
    if (!llvm::is_contained(names, name))
      names.push_back(name.str());
  }
  for (const FormulaPtr &argument : formula.arguments)
    collectSemanticAttributeNames(*argument, names);
}

static std::optional<TTIRCandidateEffectKind>
parseCandidateEffectKind(StringRef spelling) {
  return llvm::StringSwitch<std::optional<TTIRCandidateEffectKind>>(spelling)
      .Case("pure", TTIRCandidateEffectKind::Pure)
      .Case("read", TTIRCandidateEffectKind::Read)
      .Case("write", TTIRCandidateEffectKind::Write)
      .Case("atomic", TTIRCandidateEffectKind::Atomic)
      .Case("observable", TTIRCandidateEffectKind::Observable)
      .Case("buffer_read", TTIRCandidateEffectKind::BufferRead)
      .Case("buffer_write", TTIRCandidateEffectKind::BufferWrite)
      .Default(std::nullopt);
}

static void collectOperandEffectRequirements(
    const Formula &formula, ArrayRef<std::string> operandNames,
    SmallVectorImpl<TTIRCandidateOperandEffectRequirement> &requirements) {
  if (formula.kind == Formula::Kind::Call &&
      formula.spelling == "producer_effect" &&
      formula.arguments.size() == 2 &&
      formula.arguments[0]->kind == Formula::Kind::Identifier &&
      formula.arguments[1]->kind == Formula::Kind::Identifier) {
    auto operand = llvm::find(operandNames, formula.arguments[0]->spelling);
    std::optional<TTIRCandidateEffectKind> effect =
        parseCandidateEffectKind(formula.arguments[1]->spelling);
    if (operand == operandNames.end() || !effect)
      llvm::report_fatal_error(
          "invalid producer_effect operand or effect kind in TT semantics");
    TTIRCandidateOperandEffectRequirement requirement;
    requirement.operandIndex =
        static_cast<unsigned>(std::distance(operandNames.begin(), operand));
    requirement.effectKind = *effect;
    auto duplicate = llvm::find_if(requirements, [&](const auto &known) {
      return known.operandIndex == requirement.operandIndex;
    });
    if (duplicate == requirements.end()) {
      requirements.push_back(requirement);
    } else if (duplicate->effectKind != requirement.effectKind) {
      llvm::report_fatal_error(
          "one TT semantics operand requires multiple producer effects");
    }
  }
  for (const FormulaPtr &argument : formula.arguments)
    collectOperandEffectRequirements(*argument, operandNames, requirements);
}

static std::optional<TTIRCandidateStateDomain>
parseCandidateStateDomain(StringRef spelling) {
  return llvm::StringSwitch<std::optional<TTIRCandidateStateDomain>>(spelling)
      .Case("memory", TTIRCandidateStateDomain::Memory)
      .Case("launch", TTIRCandidateStateDomain::Launch)
      .Case("buffer", TTIRCandidateStateDomain::Buffer)
      .Default(std::nullopt);
}

static bool isDirectPureCallable(Operation *operation) {
  auto call = dyn_cast_or_null<CallOpInterface>(operation);
  if (!call)
    return false;
  Operation *callable = call.resolveCallable();
  if (!callable || callable->getNumRegions() == 0 ||
      llvm::any_of(callable->getRegions(),
                   [](Region &region) { return region.empty(); }))
    return false;
  WalkResult result = callable->walk([&](Operation *nested) {
    if (nested == callable)
      return WalkResult::advance();
    // Nested calls need their own summary and recursion guard.  Keep this
    // first summary language deliberately acyclic and fail closed.
    if (isa<CallOpInterface>(nested) || !isMemoryEffectFree(nested))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

static void collectImplicitStateDomains(
    const Formula &formula,
    SmallVectorImpl<TTIRCandidateStateDomain> &domains) {
  if (formula.kind == Formula::Kind::Call &&
      formula.spelling == "state_operand" && formula.arguments.size() == 1 &&
      formula.arguments[0]->kind == Formula::Kind::Identifier) {
    std::optional<TTIRCandidateStateDomain> domain =
        parseCandidateStateDomain(formula.arguments[0]->spelling);
    if (!domain)
      llvm::report_fatal_error("unknown state_operand domain in TT semantics");
    if (!llvm::is_contained(domains, *domain))
      domains.push_back(*domain);
  }
  for (const FormulaPtr &argument : formula.arguments)
    collectImplicitStateDomains(*argument, domains);
}

static void collectProducedStateDomain(
    const Formula &formula,
    std::optional<TTIRCandidateStateDomain> &produced) {
  if (formula.kind == Formula::Kind::Call && formula.arguments.size() == 1 &&
      formula.arguments[0]->kind == Formula::Kind::Identifier &&
      formula.arguments[0]->spelling == "result") {
    std::optional<TTIRCandidateStateDomain> domain;
    if (formula.spelling == "memory_state")
      domain = TTIRCandidateStateDomain::Memory;
    else if (formula.spelling == "launch_state")
      domain = TTIRCandidateStateDomain::Launch;
    else if (formula.spelling == "buffer_state")
      domain = TTIRCandidateStateDomain::Buffer;
    if (domain) {
      if (produced && *produced != *domain)
        llvm::report_fatal_error(
            "one TT semantics result produces multiple state domains");
      produced = domain;
    }
  }
  for (const FormulaPtr &argument : formula.arguments)
    collectProducedStateDomain(*argument, produced);
}

static DictionaryAttr buildModuleAttributePatch(const Formula &action,
                                                DictionaryAttr attributes) {
  if (action.kind != Formula::Kind::Call || !attributes)
    return {};
  Builder builder(attributes.getContext());
  if (action.spelling == "launch_partition" &&
      action.arguments.size() == 3 &&
      llvm::all_of(action.arguments, [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      })) {
    auto axis = dyn_cast_or_null<IntegerAttr>(
        attributes.get(action.arguments[0]->spelling));
    auto factor = dyn_cast_or_null<IntegerAttr>(
        attributes.get(action.arguments[1]->spelling));
    auto ceilDiv = dyn_cast_or_null<IntegerAttr>(
        attributes.get(action.arguments[2]->spelling));
    if (!axis || !factor || !ceilDiv)
      return {};
    return builder.getDictionaryAttr(
        {builder.getNamedAttr("hacc.coalesce_axis",
                              builder.getI32IntegerAttr(axis.getInt())),
         builder.getNamedAttr("hacc.coalesce_factor",
                              builder.getI32IntegerAttr(factor.getInt())),
         builder.getNamedAttr(
             "hacc.coalesce_grid_ceil_div",
             builder.getI32IntegerAttr(ceilDiv.getInt()))});
  }
  if (action.spelling != "module_attrs")
    return {};
  SmallVector<NamedAttribute, 4> patch;
  for (const FormulaPtr &binding : action.arguments) {
    if (binding->kind != Formula::Kind::Call ||
        binding->spelling != "set_attr_i32" ||
        binding->arguments.size() != 2 ||
        llvm::any_of(binding->arguments, [](const FormulaPtr &argument) {
          return argument->kind != Formula::Kind::Identifier;
        }))
      return {};
    StringRef destination = binding->arguments[0]->spelling;
    StringRef source = binding->arguments[1]->spelling;
    auto value = dyn_cast_or_null<IntegerAttr>(attributes.get(source));
    if (!value)
      return {};
    patch.push_back(builder.getNamedAttr(
        destination, builder.getI32IntegerAttr(value.getInt())));
  }
  return patch.empty() ? DictionaryAttr{}
                       : builder.getDictionaryAttr(patch);
}

static DictionaryAttr buildCompilerAction(const Formula &action,
                                          DictionaryAttr attributes) {
  if (action.kind != Formula::Kind::Call ||
      action.spelling != "launch_partition" ||
      action.arguments.size() != 3 || !attributes ||
      llvm::any_of(action.arguments, [](const FormulaPtr &argument) {
        return argument->kind != Formula::Kind::Identifier;
      }))
    return {};
  auto axis = dyn_cast_or_null<IntegerAttr>(
      attributes.get(action.arguments[0]->spelling));
  auto factor = dyn_cast_or_null<IntegerAttr>(
      attributes.get(action.arguments[1]->spelling));
  auto ceilDiv = dyn_cast_or_null<IntegerAttr>(
      attributes.get(action.arguments[2]->spelling));
  if (!axis || !factor || !ceilDiv || axis.getInt() < 0 ||
      axis.getInt() > 2 || factor.getInt() <= 1 ||
      (ceilDiv.getInt() != 0 && ceilDiv.getInt() != 1))
    return {};
  Builder builder(attributes.getContext());
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("kind", builder.getStringAttr("launch_partition")),
       builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis.getInt())),
       builder.getNamedAttr("factor",
                            builder.getI32IntegerAttr(factor.getInt())),
       builder.getNamedAttr("ceil_div",
                            builder.getI32IntegerAttr(ceilDiv.getInt()))});
}

static Attribute getConstraintAttribute(const ConstraintContext &context,
                                        StringRef name) {
  if (context.operation)
    return context.operation->getAttr(name);
  return context.attributes ? context.attributes.get(name) : Attribute{};
}

static std::optional<SmallVector<int64_t, 4>>
getIntegerArrayAttribute(Attribute attribute) {
  SmallVector<int64_t, 4> values;
  if (auto array = dyn_cast_or_null<DenseI64ArrayAttr>(attribute)) {
    values.append(array.asArrayRef().begin(), array.asArrayRef().end());
    return values;
  }
  if (auto array = dyn_cast_or_null<DenseI32ArrayAttr>(attribute)) {
    for (int32_t value : array.asArrayRef())
      values.push_back(value);
    return values;
  }
  return std::nullopt;
}

static bool isIntegerConstant(Attribute value) {
  if (isa_and_nonnull<IntegerAttr>(value))
    return true;
  auto dense = dyn_cast_or_null<DenseElementsAttr>(value);
  return dense && dense.isSplat() &&
         isa<IntegerType>(dense.getElementType());
}

static bool hasBinaryCombiner(Operation *operation,
                              StringRef combinerOperation) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1 || operation->getNumRegions() != 1)
    return false;
  if (operation->getRegion(0).empty())
    return false;
  Block &body = operation->getRegion(0).front();
  if (body.getNumArguments() != 2 ||
      !llvm::hasSingleElement(body.without_terminator()))
    return false;
  Operation &combiner = body.front();
  if (combiner.getName().getStringRef() != combinerOperation ||
      combiner.getNumOperands() != 2 || combiner.getNumResults() != 1 ||
      body.getTerminator()->getNumOperands() != 1 ||
      body.getTerminator()->getOperand(0) != combiner.getResult(0))
    return false;
  bool consumesArguments =
      (combiner.getOperand(0) == body.getArgument(0) &&
       combiner.getOperand(1) == body.getArgument(1)) ||
      (combiner.getOperand(0) == body.getArgument(1) &&
       combiner.getOperand(1) == body.getArgument(0));
  if (!consumesArguments)
    return false;
  if (auto add = dyn_cast<arith::AddIOp>(combiner))
    return add.getOverflowFlags() == arith::IntegerOverflowFlags::none;
  if (auto subtract = dyn_cast<arith::SubIOp>(combiner))
    return subtract.getOverflowFlags() ==
           arith::IntegerOverflowFlags::none;
  return true;
}

static bool isBinaryReduction(Operation *operation, StringRef axisAttribute,
                              StringRef combinerOperation) {
  if (!hasBinaryCombiner(operation, combinerOperation))
    return false;
  auto axis = dyn_cast_or_null<IntegerAttr>(operation->getAttr(axisAttribute));
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  Type resultType = operation->getResult(0).getType();
  if (!axis || !input || axis.getInt() < 0 ||
      axis.getInt() >= input.getRank())
    return false;
  SmallVector<int64_t, 4> expectedShape(input.getShape().begin(),
                                        input.getShape().end());
  expectedShape.erase(expectedShape.begin() + axis.getInt());
  if (expectedShape.empty())
    return resultType == input.getElementType();
  auto result = dyn_cast<RankedTensorType>(resultType);
  return result && result.getShape() == ArrayRef<int64_t>(expectedShape) &&
         result.getElementType() == input.getElementType();
}

static bool isBinaryScan(Operation *operation, StringRef axisAttribute,
                         StringRef reverseAttribute,
                         StringRef combinerOperation) {
  if (!hasBinaryCombiner(operation, combinerOperation))
    return false;
  auto axis = dyn_cast_or_null<IntegerAttr>(operation->getAttr(axisAttribute));
  auto reverse = dyn_cast_or_null<BoolAttr>(
      operation->getAttr(reverseAttribute));
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  return axis && reverse && input && result &&
         axis.getInt() >= 0 && axis.getInt() < input.getRank() &&
         input == result;
}

static bool hasExpandDimsShape(Operation *operation, StringRef axisAttribute) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1)
    return false;
  auto axis = dyn_cast_or_null<IntegerAttr>(operation->getAttr(axisAttribute));
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!axis || !input || !result ||
      result.getRank() != input.getRank() + 1 ||
      axis.getInt() < 0 || axis.getInt() >= result.getRank() ||
      result.getDimSize(axis.getInt()) != 1 ||
      input.getElementType() != result.getElementType())
    return false;
  SmallVector<int64_t, 4> expected(result.getShape().begin(),
                                   result.getShape().end());
  expected.erase(expected.begin() + axis.getInt());
  return input.getShape() == ArrayRef<int64_t>(expected);
}

static bool hasBroadcastShape(Operation *operation) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!input || !result || input.getRank() != result.getRank() ||
      input.getElementType() != result.getElementType())
    return false;
  for (int64_t dimension = 0; dimension < input.getRank(); ++dimension)
    if (input.getDimSize(dimension) != result.getDimSize(dimension) &&
        input.getDimSize(dimension) != 1)
      return false;
  return true;
}

static bool hasTransposeShape(Operation *operation, StringRef orderAttribute) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto result = dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  auto order = dyn_cast_or_null<DenseI32ArrayAttr>(
      operation->getAttr(orderAttribute));
  if (!input || !result || !order || input.getRank() != result.getRank() ||
      input.getElementType() != result.getElementType() ||
      input.getEncoding() != result.getEncoding() ||
      static_cast<int64_t>(order.size()) != input.getRank())
    return false;
  SmallVector<bool, 4> seen(order.size(), false);
  for (auto [resultDimension, sourceDimension] :
       llvm::enumerate(order.asArrayRef())) {
    if (sourceDimension < 0 ||
        static_cast<size_t>(sourceDimension) >= order.size() ||
        seen[sourceDimension] ||
        result.getDimSize(resultDimension) !=
            input.getDimSize(sourceDimension))
      return false;
    seen[sourceDimension] = true;
  }
  return true;
}

static bool hasExpandDimsShape(Type inputType, Type resultType,
                               Attribute axisAttribute) {
  auto axis = dyn_cast_or_null<IntegerAttr>(axisAttribute);
  auto input = dyn_cast<RankedTensorType>(inputType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  if (!axis || !input || !result ||
      result.getRank() != input.getRank() + 1 ||
      axis.getInt() < 0 || axis.getInt() >= result.getRank() ||
      result.getDimSize(axis.getInt()) != 1 ||
      input.getElementType() != result.getElementType())
    return false;
  SmallVector<int64_t, 4> expected(result.getShape().begin(),
                                   result.getShape().end());
  expected.erase(expected.begin() + axis.getInt());
  return input.getShape() == ArrayRef<int64_t>(expected);
}

static bool hasBroadcastShape(Type inputType, Type resultType) {
  auto input = dyn_cast<RankedTensorType>(inputType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  if (!input || !result || input.getRank() != result.getRank() ||
      input.getElementType() != result.getElementType())
    return false;
  for (int64_t dimension = 0; dimension < input.getRank(); ++dimension)
    if (input.getDimSize(dimension) != result.getDimSize(dimension) &&
        input.getDimSize(dimension) != 1)
      return false;
  return true;
}

static bool hasTransposeShape(Type inputType, Type resultType,
                              Attribute orderAttribute) {
  auto input = dyn_cast<RankedTensorType>(inputType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  auto order = dyn_cast_or_null<DenseI32ArrayAttr>(orderAttribute);
  if (!input || !result || !order || input.getRank() != result.getRank() ||
      input.getElementType() != result.getElementType() ||
      input.getEncoding() != result.getEncoding() ||
      static_cast<int64_t>(order.size()) != input.getRank())
    return false;
  SmallVector<bool, 4> seen(order.size(), false);
  for (auto [resultDimension, sourceDimension] :
       llvm::enumerate(order.asArrayRef())) {
    if (sourceDimension < 0 ||
        static_cast<size_t>(sourceDimension) >= order.size() ||
        seen[sourceDimension] ||
        result.getDimSize(resultDimension) !=
            input.getDimSize(sourceDimension))
      return false;
    seen[sourceDimension] = true;
  }
  return true;
}

static unsigned countDynamicEntries(ArrayRef<int64_t> values) {
  return llvm::count_if(values,
                        [](int64_t value) {
                          // UniverseSpec serializes dynamic entries as -1,
                          // while current MLIR uses kDynamic internally.
                          return value == -1 || ShapedType::isDynamic(value);
                        });
}

static bool isDynamicSliceEntry(int64_t value) {
  return value == -1 || ShapedType::isDynamic(value);
}

// Validates the generic mixed static/dynamic slice encoding used by tensor
// and memref slice-like operations. The segment vector, rather than operand
// position guesses, determines which variadic operands are offsets, sizes and
// strides. This is an ODS contract, not an operation-specific graph matcher.
static ConstraintTruth hasMixedSliceShape(
    const ConstraintContext &context, bool insertion,
    StringRef segmentsName, StringRef offsetsName, StringRef sizesName,
    StringRef stridesName) {
  const unsigned fixedOperands = insertion ? 2 : 1;
  if (context.operandTypes.size() < fixedOperands)
    return context.allowUnknownOperands ? ConstraintTruth::Unknown
                                        : ConstraintTruth::False;
  Attribute segmentsAttribute =
      getConstraintAttribute(context, segmentsName);
  Attribute offsetsAttribute = getConstraintAttribute(context, offsetsName);
  Attribute sizesAttribute = getConstraintAttribute(context, sizesName);
  Attribute stridesAttribute = getConstraintAttribute(context, stridesName);
  if ((!segmentsAttribute || !offsetsAttribute || !sizesAttribute ||
       !stridesAttribute) &&
      context.allowUnknownAttributes)
    return ConstraintTruth::Unknown;
  std::optional<SmallVector<int64_t, 4>> segments =
      getIntegerArrayAttribute(segmentsAttribute);
  std::optional<SmallVector<int64_t, 4>> offsets =
      getIntegerArrayAttribute(offsetsAttribute);
  std::optional<SmallVector<int64_t, 4>> sizes =
      getIntegerArrayAttribute(sizesAttribute);
  std::optional<SmallVector<int64_t, 4>> strides =
      getIntegerArrayAttribute(stridesAttribute);
  const unsigned expectedSegments = insertion ? 5 : 4;
  if (!segments || !offsets || !sizes || !strides ||
      segments->size() != expectedSegments || (*segments)[0] != 1 ||
      (insertion && (*segments)[1] != 1) ||
      llvm::any_of(*segments, [](int64_t value) { return value < 0; }))
    return ConstraintTruth::False;
  const unsigned segmentBase = insertion ? 2 : 1;
  if ((*segments)[segmentBase] !=
          static_cast<int64_t>(countDynamicEntries(*offsets)) ||
      (*segments)[segmentBase + 1] !=
          static_cast<int64_t>(countDynamicEntries(*sizes)) ||
      (*segments)[segmentBase + 2] !=
          static_cast<int64_t>(countDynamicEntries(*strides)))
    return ConstraintTruth::False;
  uint64_t requiredOperands = 0;
  for (int64_t segment : *segments)
    requiredOperands += static_cast<uint64_t>(segment);
  if (context.expectedOperandCount != 0 &&
      context.expectedOperandCount != requiredOperands)
    return ConstraintTruth::False;
  if (context.operandTypes.size() < requiredOperands)
    return context.allowUnknownOperands ? ConstraintTruth::Unknown
                                        : ConstraintTruth::False;
  if (context.operandTypes.size() != requiredOperands)
    return ConstraintTruth::False;
  bool hasUnknownOperand = false;
  for (Type dynamicType : context.operandTypes.drop_front(fixedOperands)) {
    if (!dynamicType) {
      hasUnknownOperand = true;
      continue;
    }
    if (!isa<IndexType>(dynamicType))
      return ConstraintTruth::False;
  }

  Type sourceType = context.operandTypes[0];
  if (!sourceType)
    return context.allowUnknownOperands ? ConstraintTruth::Unknown
                                        : ConstraintTruth::False;
  auto source = dyn_cast<ShapedType>(sourceType);
  auto result = dyn_cast_or_null<ShapedType>(context.resultType);
  if (!source || !result || !source.hasRank() || !result.hasRank() ||
      source.getRank() != result.getRank() ||
      source.getElementType() != result.getElementType() ||
      offsets->size() != static_cast<size_t>(source.getRank()) ||
      sizes->size() != offsets->size() ||
      strides->size() != offsets->size())
    return ConstraintTruth::False;
  const bool tensorSlice = isa<RankedTensorType>(source) &&
                           isa<RankedTensorType>(result);
  const bool memrefSlice = isa<BaseMemRefType>(source) &&
                           isa<BaseMemRefType>(result);
  if (!tensorSlice && !memrefSlice)
    return ConstraintTruth::False;
  if (tensorSlice &&
      (cast<RankedTensorType>(source).getEncoding() ||
       cast<RankedTensorType>(result).getEncoding()))
    return ConstraintTruth::False;
  if (memrefSlice) {
    auto sourceMemref = dyn_cast<MemRefType>(source);
    auto resultMemref = dyn_cast<MemRefType>(result);
    if (!sourceMemref || !resultMemref ||
        sourceMemref.getMemorySpace() != resultMemref.getMemorySpace())
      return ConstraintTruth::False;
  }

  if (insertion) {
    if (!context.operandTypes[1])
      return context.allowUnknownOperands ? ConstraintTruth::Unknown
                                          : ConstraintTruth::False;
    auto destination =
        dyn_cast<RankedTensorType>(context.operandTypes[1]);
    auto sourceTensor = dyn_cast<RankedTensorType>(source);
    auto resultTensor = dyn_cast<RankedTensorType>(result);
    if (!sourceTensor || !destination || !resultTensor ||
        destination != resultTensor || destination.getEncoding() ||
        sourceTensor.getRank() != destination.getRank() ||
        sourceTensor.getElementType() != destination.getElementType() ||
        (*segments)[4] != 0)
      return ConstraintTruth::False;
  }

  for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
    const int64_t offset = (*offsets)[dimension];
    const int64_t size = (*sizes)[dimension];
    const int64_t stride = (*strides)[dimension];
    if (!isDynamicSliceEntry(offset) && offset < 0)
      return ConstraintTruth::False;
    if (!isDynamicSliceEntry(size) && size <= 0)
      return ConstraintTruth::False;
    if (!isDynamicSliceEntry(stride) && stride <= 0)
      return ConstraintTruth::False;
    if (insertion && stride != 1)
      return ConstraintTruth::False;
    if (isDynamicSliceEntry(size)) {
      if (!ShapedType::isDynamic(result.getDimSize(dimension)))
        return ConstraintTruth::False;
    } else if (result.getDimSize(dimension) != size) {
      return ConstraintTruth::False;
    }
    if (insertion) {
      auto sourceTensor = cast<RankedTensorType>(source);
      if (sourceTensor.getDimSize(dimension) != size)
        return ConstraintTruth::False;
    }
    int64_t containerExtent =
        insertion
            ? cast<RankedTensorType>(context.operandTypes[1])
                  .getDimSize(dimension)
            : source.getDimSize(dimension);
    if (!isDynamicSliceEntry(offset) && !isDynamicSliceEntry(size) &&
        !isDynamicSliceEntry(stride) &&
        !ShapedType::isDynamic(containerExtent) &&
        offset + (size - 1) * stride >= containerExtent)
      return ConstraintTruth::False;
  }
  return hasUnknownOperand ? ConstraintTruth::Unknown : ConstraintTruth::True;
}

static bool isBinaryReduction(Type inputType, Type resultType,
                              Attribute axisAttribute) {
  auto axis = dyn_cast_or_null<IntegerAttr>(axisAttribute);
  auto input = dyn_cast<RankedTensorType>(inputType);
  if (!axis || !input || axis.getInt() < 0 ||
      axis.getInt() >= input.getRank())
    return false;
  SmallVector<int64_t, 4> expectedShape(input.getShape().begin(),
                                        input.getShape().end());
  expectedShape.erase(expectedShape.begin() + axis.getInt());
  if (expectedShape.empty())
    return resultType == input.getElementType();
  auto result = dyn_cast<RankedTensorType>(resultType);
  return result && result.getShape() == ArrayRef<int64_t>(expectedShape) &&
         result.getElementType() == input.getElementType();
}

static bool isBinaryScan(Type inputType, Type resultType,
                         Attribute axisAttribute, Attribute reverseAttribute) {
  auto axis = dyn_cast_or_null<IntegerAttr>(axisAttribute);
  auto reverse = dyn_cast_or_null<BoolAttr>(reverseAttribute);
  auto input = dyn_cast<RankedTensorType>(inputType);
  auto result = dyn_cast<RankedTensorType>(resultType);
  return axis && reverse && input && result &&
         axis.getInt() >= 0 && axis.getInt() < input.getRank() &&
         input == result;
}

static ConstraintTruth evaluateConstraint(const Formula &formula,
                                          const ConstraintContext &context) {
  if (formula.kind == Formula::Kind::Boolean)
    return formula.boolean ? ConstraintTruth::True : ConstraintTruth::False;
  if (formula.kind != Formula::Kind::Call)
    return ConstraintTruth::False;

  StringRef call = formula.spelling;
  ArrayRef<FormulaPtr> arguments = formula.arguments;
  if (call == "and") {
    bool hasUnknown = false;
    for (const FormulaPtr &argument : arguments) {
      ConstraintTruth value = evaluateConstraint(*argument, context);
      if (value == ConstraintTruth::False)
        return ConstraintTruth::False;
      hasUnknown |= value == ConstraintTruth::Unknown;
    }
    return hasUnknown ? ConstraintTruth::Unknown : ConstraintTruth::True;
  }
  if (call == "or") {
    bool hasUnknown = false;
    for (const FormulaPtr &argument : arguments) {
      ConstraintTruth value = evaluateConstraint(*argument, context);
      if (value == ConstraintTruth::True)
        return ConstraintTruth::True;
      hasUnknown |= value == ConstraintTruth::Unknown;
    }
    return hasUnknown ? ConstraintTruth::Unknown : ConstraintTruth::False;
  }
  if (call == "not" && arguments.size() == 1)
    return negate(evaluateConstraint(*arguments.front(), context));
  if (call == "target_compile_on_91095" && arguments.empty())
    return context.compileOn91095 ? ConstraintTruth::True
                                  : ConstraintTruth::False;
  // Candidate discovery consumes this declarative marker. It is not a type
  // predicate, so it is true when the remaining candidate contract is
  // interpreted here.
  if ((call == "witnessed_operation" || call == "offline_only") &&
      arguments.empty())
    return ConstraintTruth::True;
  if (call == "semantic_result" && arguments.size() == 1) {
    std::optional<int64_t> expected = getConstraintInteger(*arguments[0]);
    return expected && *expected >= 0 &&
                   static_cast<uint64_t>(*expected) == context.resultIndex
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "state_operand" && arguments.size() == 1 &&
      arguments[0]->kind == Formula::Kind::Identifier &&
      parseCandidateStateDomain(arguments[0]->spelling))
    return ConstraintTruth::True;
  // The offline enumerator checks this marker against producer metadata. It
  // is intentionally orthogonal to MLIR type legality interpreted here.
  if (call == "producer_effect" && arguments.size() == 2 &&
      arguments[0]->kind == Formula::Kind::Identifier &&
      arguments[1]->kind == Formula::Kind::Identifier)
    return ConstraintTruth::True;
  if ((call == "attribute" || call == "attribute_i32" ||
       call == "attribute_i64_array") &&
      arguments.size() == 2 &&
      arguments[0]->kind == Formula::Kind::Identifier &&
      arguments[1]->kind == Formula::Kind::Identifier) {
    if (getConstraintAttribute(context, arguments[0]->spelling))
      return ConstraintTruth::True;
    return context.allowUnknownAttributes ? ConstraintTruth::Unknown
                                          : ConstraintTruth::False;
  }
  if (call == "attribute_integer_greater" && arguments.size() == 2 &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute attribute =
        getConstraintAttribute(context, arguments[0]->spelling);
    if (!attribute)
      return context.allowUnknownAttributes ? ConstraintTruth::Unknown
                                            : ConstraintTruth::False;
    auto integer = dyn_cast<IntegerAttr>(attribute);
    std::optional<int64_t> bound = getConstraintInteger(*arguments[1]);
    return integer && bound && integer.getInt() > *bound
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "attribute_optional" && arguments.size() == 1 &&
      arguments.front()->kind == Formula::Kind::Identifier)
    return ConstraintTruth::True;
  if (call == "segments_equal" && arguments.size() >= 2 &&
      arguments.front()->kind == Formula::Kind::Identifier) {
    Attribute attribute = getConstraintAttribute(
        context, arguments.front()->spelling);
    if (!attribute)
      return context.allowUnknownAttributes ? ConstraintTruth::Unknown
                                            : ConstraintTruth::False;
    std::optional<SmallVector<int64_t, 4>> actual =
        getIntegerArrayAttribute(attribute);
    if (!actual || actual->size() != arguments.size() - 1)
      return ConstraintTruth::False;
    for (auto [value, expectedFormula] :
         llvm::zip_equal(*actual, arguments.drop_front())) {
      std::optional<int64_t> expected =
          getConstraintInteger(*expectedFormula);
      if (!expected || value != *expected)
        return ConstraintTruth::False;
    }
    return ConstraintTruth::True;
  }
  if ((call == "attribute_absent" ||
       call == "attribute_absent_or_empty" ||
       call == "attribute_absent_or_false") &&
      arguments.size() == 1 &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute attribute =
        getConstraintAttribute(context, arguments[0]->spelling);
    if (!attribute)
      return ConstraintTruth::True;
    if (call == "attribute_absent")
      return ConstraintTruth::False;
    if (call == "attribute_absent_or_false") {
      auto boolean = dyn_cast<BoolAttr>(attribute);
      return boolean && !boolean.getValue() ? ConstraintTruth::True
                                            : ConstraintTruth::False;
    }
    if (auto array = dyn_cast<DenseI32ArrayAttr>(attribute))
      return array.empty() ? ConstraintTruth::True : ConstraintTruth::False;
    if (auto array = dyn_cast<ArrayAttr>(attribute))
      return array.empty() ? ConstraintTruth::True : ConstraintTruth::False;
    return ConstraintTruth::False;
  }

  auto unaryType = [&]() -> ConstraintType {
    if (arguments.empty())
      return {};
    return lookupConstraintType(*arguments.front(), context);
  };
  if (call == "rank" && arguments.size() == 2) {
    ConstraintType value = unaryType();
    std::optional<int64_t> rank = getConstraintInteger(*arguments[1]);
    if (value.unknown)
      return ConstraintTruth::Unknown;
    auto shaped = dyn_cast_or_null<ShapedType>(value.type);
    return shaped && shaped.hasRank() && rank && shaped.getRank() == *rank
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "static_shape" && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    auto shaped = dyn_cast_or_null<ShapedType>(value.type);
    return shaped && shaped.hasStaticShape() ? ConstraintTruth::True
                                             : ConstraintTruth::False;
  }
  if ((call == "scalar" || call == "ranked_tensor" ||
       call == "ranked_memref") &&
      arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    bool isTensor = isa_and_nonnull<RankedTensorType>(value.type);
    bool isMemRef = isa_and_nonnull<MemRefType>(value.type);
    bool accepted = call == "ranked_tensor" ? isTensor
                    : call == "ranked_memref" ? isMemRef
                                               : !isa_and_nonnull<ShapedType>(
                                                     value.type);
    return accepted ? ConstraintTruth::True : ConstraintTruth::False;
  }
  if (call == "rectangular_index_pack" && arguments.size() >= 4) {
    ConstraintType shaped = lookupConstraintType(*arguments[0], context);
    ConstraintType offset = lookupConstraintType(*arguments[1], context);
    if (shaped.unknown || offset.unknown)
      return ConstraintTruth::Unknown;
    auto tensor = dyn_cast_or_null<RankedTensorType>(shaped.type);
    auto indexType = dyn_cast_or_null<IntegerType>(offset.type);
    if (!tensor || tensor.getRank() <= 0 ||
        arguments.size() != 2 + 2 * static_cast<size_t>(tensor.getRank()) ||
        !indexType ||
        (indexType.getWidth() != 32 && indexType.getWidth() != 64))
      return ConstraintTruth::False;
    for (const FormulaPtr &argument : arguments.drop_front(2)) {
      ConstraintType metadata = lookupConstraintType(*argument, context);
      if (metadata.unknown)
        return ConstraintTruth::Unknown;
      if (!metadata.type || metadata.type != offset.type ||
          isa<ShapedType>(metadata.type))
        return ConstraintTruth::False;
    }
    return ConstraintTruth::True;
  }
  if (call == "positive_dim" && arguments.size() == 2) {
    ConstraintType value = unaryType();
    std::optional<int64_t> dimension =
        getConstraintInteger(*arguments[1]);
    if (value.unknown)
      return ConstraintTruth::Unknown;
    auto shaped = dyn_cast_or_null<ShapedType>(value.type);
    if (!shaped || !shaped.hasRank() || !dimension || *dimension < 0 ||
        *dimension >= shaped.getRank())
      return ConstraintTruth::False;
    return shaped.getDimSize(*dimension) > 0 ? ConstraintTruth::True
                                             : ConstraintTruth::False;
  }
  if (call == "integer_element" && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    return value.type && isa<IntegerType>(getElementType(value.type))
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "integer_or_index_element" && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    Type element = getElementType(value.type);
    return element && isa<IntegerType, IndexType>(element)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "float_element" && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    return value.type && isa<FloatType>(getElementType(value.type))
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "integer_or_float_element" && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    Type element = getElementType(value.type);
    return element && isa<IntegerType, FloatType>(element)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if ((call == "pointer_element" ||
       call == "integer_or_pointer_element") &&
      arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    Type element = getElementType(value.type);
    bool accepted = isa_and_nonnull<triton::PointerType>(element);
    if (call == "integer_or_pointer_element")
      accepted |= isa_and_nonnull<IntegerType>(element);
    return accepted ? ConstraintTruth::True : ConstraintTruth::False;
  }
  if (call == "load_result" && arguments.size() == 2) {
    ConstraintType result = lookupConstraintType(*arguments[0], context);
    ConstraintType pointer = lookupConstraintType(*arguments[1], context);
    if (result.unknown || pointer.unknown)
      return ConstraintTruth::Unknown;
    if (auto scalarPointer =
            dyn_cast_or_null<triton::PointerType>(pointer.type))
      return result.type == scalarPointer.getPointeeType()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    auto pointerContainer = dyn_cast_or_null<RankedTensorType>(pointer.type);
    auto resultContainer = dyn_cast_or_null<RankedTensorType>(result.type);
    auto pointerElement = pointerContainer
                              ? dyn_cast<triton::PointerType>(
                                    pointerContainer.getElementType())
                              : triton::PointerType{};
    return pointerContainer && resultContainer && pointerElement &&
                   pointerContainer.getShape() == resultContainer.getShape() &&
                   pointerContainer.getEncoding() ==
                       resultContainer.getEncoding() &&
                   pointerElement.getPointeeType() ==
                       resultContainer.getElementType()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if ((call == "memory_state" || call == "buffer_state" ||
       call == "launch_state") && arguments.size() == 1) {
    ConstraintType value = unaryType();
    if (value.unknown)
      return ConstraintTruth::Unknown;
    return isa_and_nonnull<NoneType>(value.type) ? ConstraintTruth::True
                                                 : ConstraintTruth::False;
  }
  if (call == "store_value" && arguments.size() == 2) {
    ConstraintType pointer = lookupConstraintType(*arguments[0], context);
    ConstraintType value = lookupConstraintType(*arguments[1], context);
    if (pointer.unknown || value.unknown)
      return ConstraintTruth::Unknown;
    if (auto scalarPointer =
            dyn_cast_or_null<triton::PointerType>(pointer.type))
      return value.type == scalarPointer.getPointeeType()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    auto pointerContainer = dyn_cast_or_null<RankedTensorType>(pointer.type);
    auto valueContainer = dyn_cast_or_null<RankedTensorType>(value.type);
    auto pointerElement = pointerContainer
                              ? dyn_cast<triton::PointerType>(
                                    pointerContainer.getElementType())
                              : triton::PointerType{};
    return pointerContainer && valueContainer && pointerElement &&
                   pointerContainer.getShape() == valueContainer.getShape() &&
                   pointerContainer.getEncoding() ==
                       valueContainer.getEncoding() &&
                   pointerElement.getPointeeType() ==
                       valueContainer.getElementType()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "integer_bits" && arguments.size() == 2) {
    ConstraintType value = unaryType();
    std::optional<int64_t> width = getConstraintInteger(*arguments[1]);
    if (value.unknown)
      return ConstraintTruth::Unknown;
    auto integer = dyn_cast_or_null<IntegerType>(getElementType(value.type));
    return integer && width && integer.getWidth() == *width
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if ((call == "same_type" || call == "same_shape" ||
       call == "same_shape_or_scalar" || call == "same_encoding" ||
       call == "same_layout" || call == "same_memory_space" ||
       call == "same_container" || call == "same_element_type" ||
       call == "pointer_pointee_element_type" ||
       call == "same_element_bits" || call == "element_bits_less" ||
       call == "element_bits_greater") &&
      arguments.size() == 2) {
    ConstraintType lhs = lookupConstraintType(*arguments[0], context);
    ConstraintType rhs = lookupConstraintType(*arguments[1], context);
    if (lhs.unknown || rhs.unknown)
      return ConstraintTruth::Unknown;
    if (!lhs.type || !rhs.type)
      return ConstraintTruth::False;
    if (call == "same_type")
      return lhs.type == rhs.type ? ConstraintTruth::True
                                  : ConstraintTruth::False;
    if (call == "same_element_type")
      return getElementType(lhs.type) == getElementType(rhs.type)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    if (call == "pointer_pointee_element_type") {
      Type pointerElement = getElementType(rhs.type);
      auto pointer = dyn_cast_or_null<triton::PointerType>(pointerElement);
      return pointer && pointer.getPointeeType() == getElementType(lhs.type)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    }
    if (call == "same_element_bits" || call == "element_bits_less" ||
        call == "element_bits_greater") {
      Type lhsElement = getElementType(lhs.type);
      Type rhsElement = getElementType(rhs.type);
      auto getWidth = [](Type type) -> std::optional<unsigned> {
        if (auto integer = dyn_cast<IntegerType>(type))
          return integer.getWidth();
        if (auto floating = dyn_cast<FloatType>(type))
          return floating.getWidth();
        return std::nullopt;
      };
      std::optional<unsigned> lhsWidth = getWidth(lhsElement);
      std::optional<unsigned> rhsWidth = getWidth(rhsElement);
      if (!lhsWidth || !rhsWidth)
        return ConstraintTruth::False;
      if (call == "same_element_bits")
        return *lhsWidth == *rhsWidth ? ConstraintTruth::True
                                      : ConstraintTruth::False;
      if (call == "element_bits_less")
        return *lhsWidth < *rhsWidth ? ConstraintTruth::True
                                     : ConstraintTruth::False;
      return *lhsWidth > *rhsWidth ? ConstraintTruth::True
                                   : ConstraintTruth::False;
    }
    if (call == "same_shape_or_scalar") {
      if (!isa<ShapedType>(lhs.type) && !isa<ShapedType>(rhs.type))
        return ConstraintTruth::True;
      auto lhsShaped = dyn_cast<ShapedType>(lhs.type);
      auto rhsShaped = dyn_cast<ShapedType>(rhs.type);
      return lhsShaped && rhsShaped && lhsShaped.hasRank() &&
                     rhsShaped.hasRank() &&
                     lhsShaped.getShape() == rhsShaped.getShape()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    }
    if (call == "same_container") {
      if (!isa<ShapedType>(lhs.type) && !isa<ShapedType>(rhs.type))
        return ConstraintTruth::True;
      auto lhsTensor = dyn_cast<RankedTensorType>(lhs.type);
      auto rhsTensor = dyn_cast<RankedTensorType>(rhs.type);
      return lhsTensor && rhsTensor && lhsTensor.hasRank() &&
                     rhsTensor.hasRank() &&
                     lhsTensor.getShape() == rhsTensor.getShape() &&
                     lhsTensor.getEncoding() == rhsTensor.getEncoding()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    }
    auto lhsShaped = dyn_cast<ShapedType>(lhs.type);
    auto rhsShaped = dyn_cast<ShapedType>(rhs.type);
    if (!lhsShaped || !rhsShaped || !lhsShaped.hasRank() ||
        !rhsShaped.hasRank())
      return ConstraintTruth::False;
    if (call == "same_shape")
      return lhsShaped.getShape() == rhsShaped.getShape()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    if (call == "same_encoding") {
      auto lhsTensor = dyn_cast<RankedTensorType>(lhs.type);
      auto rhsTensor = dyn_cast<RankedTensorType>(rhs.type);
      return lhsTensor && rhsTensor &&
                     lhsTensor.getEncoding() == rhsTensor.getEncoding()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    }
    auto lhsMemRef = dyn_cast<MemRefType>(lhs.type);
    auto rhsMemRef = dyn_cast<MemRefType>(rhs.type);
    if (!lhsMemRef || !rhsMemRef)
      return ConstraintTruth::False;
    if (call == "same_memory_space")
      return lhsMemRef.getMemorySpace() == rhsMemRef.getMemorySpace()
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    return lhsMemRef.getLayout() == rhsMemRef.getLayout() &&
                   lhsMemRef.getMemorySpace() == rhsMemRef.getMemorySpace()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }

  if (call == "integer_constant" && arguments.size() == 1 &&
      context.operation &&
      arguments.front()->kind == Formula::Kind::Identifier)
    return isIntegerConstant(
               context.operation->getAttr(arguments.front()->spelling))
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  if (call == "no_integer_overflow" && arguments.empty()) {
    // Synthesized arith.addi/subi operations carry no overflow promise.
    if (!context.operation)
      return ConstraintTruth::True;
    if (auto add = dyn_cast<arith::AddIOp>(context.operation))
      return add.getOverflowFlags() == arith::IntegerOverflowFlags::none
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    if (auto subtract = dyn_cast<arith::SubIOp>(context.operation))
      return subtract.getOverflowFlags() ==
                     arith::IntegerOverflowFlags::none
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    if (auto multiply = dyn_cast<arith::MulIOp>(context.operation))
      return multiply.getOverflowFlags() ==
                     arith::IntegerOverflowFlags::none
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    return ConstraintTruth::False;
  }
  if (call == "callee_is_pure" && arguments.empty()) {
    if (context.operation)
      return isDirectPureCallable(context.operation)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    if (context.attributes &&
        context.attributes.get(kCallSummaryRegionsAttr))
      return ConstraintTruth::True;
    return context.allowUnknownAttributes ? ConstraintTruth::Unknown
                                          : ConstraintTruth::False;
  }
  if (call == "attribute_integer_equals" && arguments.size() == 2 &&
      arguments.front()->kind == Formula::Kind::Identifier) {
    std::optional<int64_t> expected =
        getConstraintInteger(*arguments[1]);
    auto actual = dyn_cast_or_null<IntegerAttr>(
        getConstraintAttribute(context, arguments.front()->spelling));
    if (!actual && context.allowUnknownAttributes)
      return ConstraintTruth::Unknown;
    return expected && actual && actual.getInt() == *expected
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "attribute_integer_not_equals" && arguments.size() == 2 &&
      arguments.front()->kind == Formula::Kind::Identifier) {
    std::optional<int64_t> expected =
        getConstraintInteger(*arguments[1]);
    Attribute attribute =
        getConstraintAttribute(context, arguments.front()->spelling);
    if (!attribute)
      return context.allowUnknownAttributes ? ConstraintTruth::Unknown
                                            : ConstraintTruth::False;
    auto integer = dyn_cast<IntegerAttr>(attribute);
    return integer && expected && integer.getInt() != *expected
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "attribute_boolean_equals" && arguments.size() == 2 &&
      arguments.front()->kind == Formula::Kind::Identifier &&
      arguments[1]->kind == Formula::Kind::Boolean) {
    auto actual = dyn_cast_or_null<BoolAttr>(
        getConstraintAttribute(context, arguments.front()->spelling));
    if (!actual && context.allowUnknownAttributes)
      return ConstraintTruth::Unknown;
    return actual && actual.getValue() == arguments[1]->boolean
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "supported_integer_compare" && arguments.size() == 1 &&
      arguments.front()->kind == Formula::Kind::Identifier) {
    auto predicate = dyn_cast_or_null<IntegerAttr>(
        getConstraintAttribute(context, arguments.front()->spelling));
    if (!predicate && context.allowUnknownAttributes)
      return ConstraintTruth::Unknown;
    if (!predicate)
      return ConstraintTruth::False;
    int64_t value = predicate.getInt();
    return (value == static_cast<int64_t>(arith::CmpIPredicate::eq) ||
            value == static_cast<int64_t>(arith::CmpIPredicate::slt))
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "binary_reduction" && arguments.size() == 2 &&
      arguments[0]->kind == Formula::Kind::Identifier &&
      arguments[1]->kind == Formula::Kind::Identifier) {
    if (context.operation)
      return isBinaryReduction(context.operation, arguments[0]->spelling,
                               arguments[1]->spelling)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    Attribute axis =
        getConstraintAttribute(context, arguments[0]->spelling);
    if (((context.operandTypes.empty() || !context.operandTypes[0]) &&
         context.allowUnknownOperands) ||
        (!axis && context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    return context.operandTypes.size() == 1 &&
                   isBinaryReduction(context.operandTypes[0],
                                     context.resultType, axis)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "binary_scan" && arguments.size() == 3 &&
      arguments[0]->kind == Formula::Kind::Identifier &&
      arguments[1]->kind == Formula::Kind::Identifier &&
      arguments[2]->kind == Formula::Kind::Identifier) {
    if (context.operation)
      return isBinaryScan(context.operation, arguments[0]->spelling,
                          arguments[1]->spelling, arguments[2]->spelling)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    Attribute axis =
        getConstraintAttribute(context, arguments[0]->spelling);
    Attribute reverse =
        getConstraintAttribute(context, arguments[1]->spelling);
    if (((context.operandTypes.empty() || !context.operandTypes[0]) &&
         context.allowUnknownOperands) ||
        ((!axis || !reverse) && context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    return context.operandTypes.size() == 1 &&
                   isBinaryScan(context.operandTypes[0], context.resultType,
                                axis, reverse)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "expand_dims_shape" && arguments.size() == 3 &&
      arguments[2]->kind == Formula::Kind::Identifier) {
    if (context.operation)
      return hasExpandDimsShape(context.operation, arguments[2]->spelling)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    Attribute axis =
        getConstraintAttribute(context, arguments[2]->spelling);
    if (((context.operandTypes.empty() || !context.operandTypes[0]) &&
         context.allowUnknownOperands) ||
        (!axis && context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    return context.operandTypes.size() == 1 &&
                   hasExpandDimsShape(context.operandTypes[0],
                                      context.resultType, axis)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "broadcast_shape" && arguments.size() == 2) {
    if (context.operation)
      return hasBroadcastShape(context.operation) ? ConstraintTruth::True
                                                  : ConstraintTruth::False;
    if ((context.operandTypes.empty() || !context.operandTypes[0]) &&
        context.allowUnknownOperands)
      return ConstraintTruth::Unknown;
    return context.operandTypes.size() == 1 &&
                   hasBroadcastShape(context.operandTypes[0],
                                     context.resultType)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "transpose_shape" && arguments.size() == 3 &&
      arguments[2]->kind == Formula::Kind::Identifier) {
    if (context.operation)
      return hasTransposeShape(context.operation, arguments[2]->spelling)
                 ? ConstraintTruth::True
                 : ConstraintTruth::False;
    Attribute order =
        getConstraintAttribute(context, arguments[2]->spelling);
    if (((context.operandTypes.empty() || !context.operandTypes[0]) &&
         context.allowUnknownOperands) ||
        (!order && context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    return context.operandTypes.size() == 1 &&
                   hasTransposeShape(context.operandTypes[0],
                                     context.resultType, order)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "reshape_shape" && arguments.size() == 2) {
    ConstraintType input = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    if (input.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    auto inputTensor = dyn_cast_or_null<RankedTensorType>(input.type);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!inputTensor || !resultTensor || !inputTensor.hasStaticShape() ||
        !resultTensor.hasStaticShape() ||
        inputTensor.getElementType() != resultTensor.getElementType() ||
        inputTensor.getNumElements() != resultTensor.getNumElements())
      return ConstraintTruth::False;
    // The initial reshape denotation is logical row-major. Encoded tensors
    // need an explicit layout semantic before they can enter this schema.
    return !inputTensor.getEncoding() && !resultTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "empty_shape" && arguments.size() == 1) {
    ConstraintType result = lookupConstraintType(*arguments[0], context);
    if (result.unknown)
      return ConstraintTruth::Unknown;
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    return resultTensor && resultTensor.hasStaticShape() &&
                   !resultTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "singleton_tensor_to_scalar" && arguments.size() == 2) {
    ConstraintType source = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    if (source.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    auto sourceTensor = dyn_cast_or_null<RankedTensorType>(source.type);
    if (!sourceTensor || isa_and_nonnull<ShapedType>(result.type) ||
        sourceTensor.getElementType() != result.type ||
        !sourceTensor.hasStaticShape())
      return ConstraintTruth::False;
    return sourceTensor.getNumElements() == 1 ? ConstraintTruth::True
                                               : ConstraintTruth::False;
  }
  if (call == "join_shape" && arguments.size() == 3) {
    ConstraintType lhs = lookupConstraintType(*arguments[0], context);
    ConstraintType rhs = lookupConstraintType(*arguments[1], context);
    ConstraintType result = lookupConstraintType(*arguments[2], context);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!result.unknown &&
        (!resultTensor || resultTensor.getRank() == 0 ||
         resultTensor.getShape().back() != 2 || resultTensor.getEncoding()))
      return ConstraintTruth::False;
    auto operandCompatible = [&](ConstraintType operand) {
      if (operand.unknown)
        return true;
      auto tensor = dyn_cast_or_null<RankedTensorType>(operand.type);
      return tensor && resultTensor &&
             resultTensor.getRank() == tensor.getRank() + 1 &&
             resultTensor.getElementType() == tensor.getElementType() &&
             resultTensor.getShape().drop_back() == tensor.getShape() &&
             !tensor.getEncoding();
    };
    if (!operandCompatible(lhs) || !operandCompatible(rhs) ||
        (!lhs.unknown && !rhs.unknown && lhs.type != rhs.type))
      return ConstraintTruth::False;
    if (lhs.unknown || rhs.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    auto lhsTensor = dyn_cast_or_null<RankedTensorType>(lhs.type);
    auto rhsTensor = dyn_cast_or_null<RankedTensorType>(rhs.type);
    if (!lhsTensor || !rhsTensor || !resultTensor || lhsTensor != rhsTensor ||
        lhsTensor.getRank() == 0 ||
        resultTensor.getRank() != lhsTensor.getRank() + 1 ||
        resultTensor.getElementType() != lhsTensor.getElementType() ||
        resultTensor.getShape().drop_back() != lhsTensor.getShape() ||
        resultTensor.getShape().back() != 2)
      return ConstraintTruth::False;
    // Encoded join/split need the dialect layout-inference interface.  The
    // first executable semantics deliberately covers logical unencoded TTIR;
    // physical layout coverage remains visible in the catalog gap.
    return !lhsTensor.getEncoding() && !resultTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "split_shape" && arguments.size() == 2) {
    ConstraintType source = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    if (source.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    auto sourceTensor = dyn_cast_or_null<RankedTensorType>(source.type);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!sourceTensor || !resultTensor || sourceTensor.getRank() == 0 ||
        sourceTensor.getShape().back() != 2 ||
        sourceTensor.getShape().drop_back() != resultTensor.getShape() ||
        sourceTensor.getElementType() != resultTensor.getElementType())
      return ConstraintTruth::False;
    return !sourceTensor.getEncoding() && !resultTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "cat_shape" && arguments.size() == 3) {
    ConstraintType lhs = lookupConstraintType(*arguments[0], context);
    ConstraintType rhs = lookupConstraintType(*arguments[1], context);
    ConstraintType result = lookupConstraintType(*arguments[2], context);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!result.unknown &&
        (!resultTensor || resultTensor.getRank() != 1 ||
         !resultTensor.hasStaticShape() || resultTensor.getEncoding()))
      return ConstraintTruth::False;
    auto operandCompatible = [&](ConstraintType operand) {
      if (operand.unknown)
        return true;
      auto tensor = dyn_cast_or_null<RankedTensorType>(operand.type);
      return tensor && resultTensor && tensor.getRank() == 1 &&
             tensor.hasStaticShape() && !tensor.getEncoding() &&
             tensor.getElementType() == resultTensor.getElementType() &&
             tensor.getDimSize(0) <= resultTensor.getDimSize(0);
    };
    if (!operandCompatible(lhs) || !operandCompatible(rhs))
      return ConstraintTruth::False;
    if (lhs.unknown || rhs.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    auto lhsTensor = dyn_cast_or_null<RankedTensorType>(lhs.type);
    auto rhsTensor = dyn_cast_or_null<RankedTensorType>(rhs.type);
    if (!lhsTensor || !rhsTensor || !resultTensor ||
        lhsTensor.getRank() != 1 || rhsTensor.getRank() != 1 ||
        resultTensor.getRank() != 1 || !lhsTensor.hasStaticShape() ||
        !rhsTensor.hasStaticShape() || !resultTensor.hasStaticShape() ||
        lhsTensor.getElementType() != rhsTensor.getElementType() ||
        lhsTensor.getElementType() != resultTensor.getElementType() ||
        resultTensor.getDimSize(0) !=
            lhsTensor.getDimSize(0) + rhsTensor.getDimSize(0))
      return ConstraintTruth::False;
    return !lhsTensor.getEncoding() && !rhsTensor.getEncoding() &&
                   !resultTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "gather_shape" && arguments.size() == 4 &&
      arguments[3]->kind == Formula::Kind::Identifier) {
    ConstraintType source = lookupConstraintType(*arguments[0], context);
    ConstraintType indices = lookupConstraintType(*arguments[1], context);
    ConstraintType result = lookupConstraintType(*arguments[2], context);
    Attribute axisAttribute =
        getConstraintAttribute(context, arguments[3]->spelling);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    auto axis = dyn_cast_or_null<IntegerAttr>(axisAttribute);
    if (!result.unknown &&
        (!resultTensor || !resultTensor.hasStaticShape()))
      return ConstraintTruth::False;
    if (axisAttribute && !axis)
      return ConstraintTruth::False;
    auto sourceTensor = dyn_cast_or_null<RankedTensorType>(source.type);
    if (!source.unknown &&
        (!sourceTensor || !sourceTensor.hasStaticShape() || !resultTensor ||
         sourceTensor.getRank() != resultTensor.getRank() || !axis ||
         axis.getInt() < 0 || axis.getInt() >= sourceTensor.getRank() ||
         sourceTensor.getElementType() != resultTensor.getElementType()))
      return ConstraintTruth::False;
    auto indicesTensor = dyn_cast_or_null<RankedTensorType>(indices.type);
    if (!indices.unknown &&
        (!indicesTensor || !indicesTensor.hasStaticShape() || !resultTensor ||
         indicesTensor.getShape() != resultTensor.getShape() ||
         indicesTensor.getEncoding() != resultTensor.getEncoding() ||
         !indicesTensor.getElementType().isInteger(32)))
      return ConstraintTruth::False;
    if (source.unknown || indices.unknown || result.unknown ||
        (!axisAttribute && context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    if (!sourceTensor || !indicesTensor || !resultTensor || !axis ||
        !sourceTensor.hasStaticShape() || !indicesTensor.hasStaticShape() ||
        sourceTensor.getRank() != indicesTensor.getRank() ||
        indicesTensor.getShape() != resultTensor.getShape() ||
        indicesTensor.getEncoding() != resultTensor.getEncoding() ||
        sourceTensor.getElementType() != resultTensor.getElementType() ||
        !indicesTensor.getElementType().isInteger(32) || axis.getInt() < 0 ||
        axis.getInt() >= sourceTensor.getRank())
      return ConstraintTruth::False;
    for (int64_t dimension = 0; dimension < sourceTensor.getRank();
         ++dimension)
      if (dimension != axis.getInt() &&
          sourceTensor.getDimSize(dimension) !=
              indicesTensor.getDimSize(dimension))
        return ConstraintTruth::False;
    return ConstraintTruth::True;
  }
  if (call == "dot_shape" && arguments.size() == 4) {
    ConstraintType lhs = lookupConstraintType(*arguments[0], context);
    ConstraintType rhs = lookupConstraintType(*arguments[1], context);
    ConstraintType acc = lookupConstraintType(*arguments[2], context);
    ConstraintType result = lookupConstraintType(*arguments[3], context);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!result.unknown &&
        (!resultTensor || resultTensor.getRank() != 2 ||
         !resultTensor.hasStaticShape() || resultTensor.getEncoding()))
      return ConstraintTruth::False;
    auto lhsTensor = dyn_cast_or_null<RankedTensorType>(lhs.type);
    if (!lhs.unknown &&
        (!lhsTensor || lhsTensor.getRank() != 2 ||
         !lhsTensor.hasStaticShape() || lhsTensor.getEncoding() ||
         !isa<IntegerType, FloatType>(lhsTensor.getElementType()) ||
         !resultTensor ||
         lhsTensor.getDimSize(0) != resultTensor.getDimSize(0)))
      return ConstraintTruth::False;
    auto rhsTensor = dyn_cast_or_null<RankedTensorType>(rhs.type);
    if (!rhs.unknown &&
        (!rhsTensor || rhsTensor.getRank() != 2 ||
         !rhsTensor.hasStaticShape() || rhsTensor.getEncoding() ||
         !isa<IntegerType, FloatType>(rhsTensor.getElementType()) ||
         !resultTensor ||
         rhsTensor.getDimSize(1) != resultTensor.getDimSize(1)))
      return ConstraintTruth::False;
    auto accTensor = dyn_cast_or_null<RankedTensorType>(acc.type);
    if (!acc.unknown && acc.type != result.type)
      return ConstraintTruth::False;
    if (!lhs.unknown && !rhs.unknown &&
        (lhsTensor.getDimSize(1) != rhsTensor.getDimSize(0) ||
         lhsTensor.getElementType().getIntOrFloatBitWidth() !=
             rhsTensor.getElementType().getIntOrFloatBitWidth()))
      return ConstraintTruth::False;
    if (lhs.unknown || rhs.unknown || acc.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    Type lhsElement = lhsTensor ? lhsTensor.getElementType() : Type{};
    Type rhsElement = rhsTensor ? rhsTensor.getElementType() : Type{};
    if (!lhsTensor || !rhsTensor || !accTensor || !resultTensor ||
        !isa_and_nonnull<IntegerType, FloatType>(lhsElement) ||
        !isa_and_nonnull<IntegerType, FloatType>(rhsElement) ||
        lhsTensor.getRank() != 2 || rhsTensor.getRank() != 2 ||
        accTensor.getRank() != 2 || !lhsTensor.hasStaticShape() ||
        !rhsTensor.hasStaticShape() || !accTensor.hasStaticShape() ||
        lhsTensor.getDimSize(1) != rhsTensor.getDimSize(0) ||
        lhsTensor.getDimSize(0) != accTensor.getDimSize(0) ||
        rhsTensor.getDimSize(1) != accTensor.getDimSize(1) ||
        accTensor != resultTensor ||
        lhsElement.getIntOrFloatBitWidth() !=
            rhsElement.getIntOrFloatBitWidth())
      return ConstraintTruth::False;
    return !lhsTensor.getEncoding() && !rhsTensor.getEncoding() &&
                   !accTensor.getEncoding()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "tensor_pointer_constructor" && arguments.size() == 3 &&
      arguments[2]->kind == Formula::Kind::Identifier) {
    ConstraintType base = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    Attribute orderAttribute =
        getConstraintAttribute(context, arguments[2]->spelling);
    auto resultPointer = dyn_cast_or_null<triton::PointerType>(result.type);
    auto pointee = resultPointer
                       ? dyn_cast<RankedTensorType>(
                             resultPointer.getPointeeType())
                       : RankedTensorType{};
    auto order = dyn_cast_or_null<DenseI32ArrayAttr>(orderAttribute);
    if (!result.unknown &&
        (!resultPointer || !pointee || pointee.getRank() == 0))
      return ConstraintTruth::False;
    if (!orderAttribute && context.allowUnknownAttributes)
      return ConstraintTruth::Unknown;
    if (!order || !pointee ||
        order.size() != static_cast<size_t>(pointee.getRank()))
      return ConstraintTruth::False;
    const size_t expected = 1 + 3 * pointee.getRank();
    if (context.expectedOperandCount != 0 &&
        context.expectedOperandCount != expected)
      return ConstraintTruth::False;
    SmallVector<bool, 4> seen(pointee.getRank(), false);
    for (int32_t dimension : order.asArrayRef()) {
      if (dimension < 0 || dimension >= pointee.getRank() || seen[dimension])
        return ConstraintTruth::False;
      seen[dimension] = true;
    }
    if (base.unknown)
      return ConstraintTruth::Unknown;
    auto basePointer = dyn_cast_or_null<triton::PointerType>(base.type);
    if (!basePointer || basePointer.getPointeeType() != pointee.getElementType() ||
        basePointer.getAddressSpace() != resultPointer.getAddressSpace())
      return ConstraintTruth::False;
    bool hasUnknownOperand = false;
    for (size_t operand = 1; operand < context.operandTypes.size(); ++operand) {
      if (!context.operandTypes[operand]) {
        hasUnknownOperand = true;
        continue;
      }
      size_t field = (operand - 1) / pointee.getRank();
      if ((field < 2 && !context.operandTypes[operand].isInteger(64)) ||
          (field == 2 && !context.operandTypes[operand].isInteger(32)))
        return ConstraintTruth::False;
    }
    if (context.operandTypes.size() < expected && context.allowUnknownOperands)
      return ConstraintTruth::Unknown;
    if (context.operandTypes.size() != expected)
      return ConstraintTruth::False;
    return hasUnknownOperand ? ConstraintTruth::Unknown : ConstraintTruth::True;
  }
  if (call == "tensor_pointer_advance" && arguments.size() == 2) {
    ConstraintType pointer = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    auto resultPointer = dyn_cast_or_null<triton::PointerType>(result.type);
    auto pointee = resultPointer
                       ? dyn_cast<RankedTensorType>(
                             resultPointer.getPointeeType())
                       : RankedTensorType{};
    if (!result.unknown && (!resultPointer || !pointee))
      return ConstraintTruth::False;
    const size_t expected = pointee ? 1 + pointee.getRank() : 0;
    if (context.expectedOperandCount != 0 && expected != 0 &&
        context.expectedOperandCount != expected)
      return ConstraintTruth::False;
    if (pointer.unknown)
      return ConstraintTruth::Unknown;
    auto pointerType = dyn_cast_or_null<triton::PointerType>(pointer.type);
    if (!pointerType || pointer.type != result.type)
      return ConstraintTruth::False;
    bool hasUnknownOperand = false;
    for (Type offset : ArrayRef<Type>(context.operandTypes).drop_front()) {
      if (!offset) {
        hasUnknownOperand = true;
        continue;
      }
      if (!offset.isInteger(32))
        return ConstraintTruth::False;
    }
    if (context.operandTypes.size() < expected && context.allowUnknownOperands)
      return ConstraintTruth::Unknown;
    if (context.operandTypes.size() != expected)
      return ConstraintTruth::False;
    return hasUnknownOperand ? ConstraintTruth::Unknown : ConstraintTruth::True;
  }
  if (call == "tensor_descriptor" && arguments.size() == 1) {
    ConstraintType descriptor = lookupConstraintType(*arguments[0], context);
    if (descriptor.unknown)
      return ConstraintTruth::Unknown;
    return isa_and_nonnull<triton::TensorDescType>(descriptor.type)
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "descriptor_base" && arguments.size() == 2) {
    ConstraintType base = lookupConstraintType(*arguments[0], context);
    ConstraintType descriptor = lookupConstraintType(*arguments[1], context);
    if (base.unknown || descriptor.unknown)
      return ConstraintTruth::Unknown;
    auto pointer = dyn_cast_or_null<triton::PointerType>(base.type);
    auto descriptorType =
        dyn_cast_or_null<triton::TensorDescType>(descriptor.type);
    return pointer && descriptorType &&
                   pointer.getPointeeType() ==
                       descriptorType.getBlockType().getElementType()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "descriptor_block_same" && arguments.size() == 2) {
    ConstraintType descriptor = lookupConstraintType(*arguments[0], context);
    ConstraintType block = lookupConstraintType(*arguments[1], context);
    if (descriptor.unknown || block.unknown)
      return ConstraintTruth::Unknown;
    auto descriptorType =
        dyn_cast_or_null<triton::TensorDescType>(descriptor.type);
    return descriptorType && descriptorType.getSignlessBlockType() == block.type
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  if (call == "descriptor_metadata" && arguments.size() == 2) {
    ConstraintType descriptor = lookupConstraintType(*arguments[0], context);
    std::optional<int64_t> start = getConstraintInteger(*arguments[1]);
    if (descriptor.unknown || !start || *start < 0)
      return ConstraintTruth::Unknown;
    auto descriptorType =
        dyn_cast_or_null<triton::TensorDescType>(descriptor.type);
    if (!descriptorType)
      return ConstraintTruth::False;
    int64_t rank = descriptorType.getBlockType().getRank();
    const size_t expected = *start + 2 * rank;
    if (context.expectedOperandCount != 0 &&
        context.expectedOperandCount != expected)
      return ConstraintTruth::False;
    bool hasUnknownOperand = false;
    for (size_t operand = *start; operand < context.operandTypes.size();
         ++operand) {
      if (!context.operandTypes[operand]) {
        hasUnknownOperand = true;
        continue;
      }
      bool shape = operand < static_cast<size_t>(*start + rank);
      if ((shape && !context.operandTypes[operand].isInteger(32)) ||
          (!shape && !context.operandTypes[operand].isInteger(64)))
        return ConstraintTruth::False;
    }
    if (context.operandTypes.size() < expected && context.allowUnknownOperands)
      return ConstraintTruth::Unknown;
    if (context.operandTypes.size() != expected)
      return ConstraintTruth::False;
    return hasUnknownOperand ? ConstraintTruth::Unknown : ConstraintTruth::True;
  }
  if (call == "descriptor_indices" && arguments.size() == 2) {
    ConstraintType descriptor = lookupConstraintType(*arguments[0], context);
    std::optional<int64_t> start = getConstraintInteger(*arguments[1]);
    if (descriptor.unknown || !start || *start < 0)
      return ConstraintTruth::Unknown;
    auto descriptorType =
        dyn_cast_or_null<triton::TensorDescType>(descriptor.type);
    if (!descriptorType)
      return ConstraintTruth::False;
    int64_t rank = descriptorType.getBlockType().getRank();
    const size_t expected = *start + rank;
    if (context.expectedOperandCount != 0 &&
        context.expectedOperandCount != expected)
      return ConstraintTruth::False;
    bool hasUnknownOperand = false;
    if (context.operandTypes.size() > static_cast<size_t>(*start))
      for (Type type :
           ArrayRef<Type>(context.operandTypes).drop_front(*start)) {
        if (!type) {
          hasUnknownOperand = true;
          continue;
        }
        if (!type.isInteger(32))
          return ConstraintTruth::False;
      }
    if (context.operandTypes.size() < expected && context.allowUnknownOperands)
      return ConstraintTruth::Unknown;
    if (context.operandTypes.size() != expected)
      return ConstraintTruth::False;
    return hasUnknownOperand ? ConstraintTruth::Unknown : ConstraintTruth::True;
  }
  if (call == "descriptor_gather_shape" && arguments.size() == 3) {
    ConstraintType descriptor = lookupConstraintType(*arguments[0], context);
    ConstraintType offsets = lookupConstraintType(*arguments[1], context);
    ConstraintType result = lookupConstraintType(*arguments[2], context);
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    if (!result.unknown &&
        (!resultTensor || !resultTensor.hasStaticShape() ||
         resultTensor.getRank() != 2))
      return ConstraintTruth::False;
    auto descriptorType =
        dyn_cast_or_null<triton::TensorDescType>(descriptor.type);
    if (!descriptor.unknown) {
      if (!descriptorType || !resultTensor)
        return ConstraintTruth::False;
      RankedTensorType block = descriptorType.getSignlessBlockType();
      if (!block.hasStaticShape() || block.getRank() != 2 ||
          resultTensor.getElementType() != block.getElementType() ||
          resultTensor.getDimSize(1) != block.getDimSize(1))
        return ConstraintTruth::False;
    }
    auto offsetTensor = dyn_cast_or_null<RankedTensorType>(offsets.type);
    if (!offsets.unknown &&
        (!offsetTensor || !offsetTensor.hasStaticShape() ||
         offsetTensor.getRank() != 1 ||
         !offsetTensor.getElementType().isInteger(32) || !resultTensor ||
         resultTensor.getDimSize(0) != offsetTensor.getDimSize(0)))
      return ConstraintTruth::False;
    if (descriptor.unknown || offsets.unknown || result.unknown)
      return ConstraintTruth::Unknown;
    if (!descriptorType || !offsetTensor || !resultTensor)
      return ConstraintTruth::False;
    RankedTensorType block = descriptorType.getSignlessBlockType();
    if (!block.hasStaticShape() || !offsetTensor.hasStaticShape() ||
        !resultTensor.hasStaticShape() || block.getRank() != 2 ||
        offsetTensor.getRank() != 1 || resultTensor.getRank() != 2 ||
        !offsetTensor.getElementType().isInteger(32) ||
        resultTensor.getElementType() != block.getElementType() ||
        resultTensor.getDimSize(0) != offsetTensor.getDimSize(0) ||
        resultTensor.getDimSize(1) != block.getDimSize(1))
      return ConstraintTruth::False;
    return ConstraintTruth::True;
  }
  if (call == "mixed_slice_shape" && arguments.size() == 6 &&
      llvm::all_of(arguments.drop_front(2), [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      }))
    return hasMixedSliceShape(
        context, /*insertion=*/false, arguments[2]->spelling,
        arguments[3]->spelling, arguments[4]->spelling,
        arguments[5]->spelling);
  if (call == "mixed_insert_slice_shape" && arguments.size() == 7 &&
      llvm::all_of(arguments.drop_front(3), [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      }))
    return hasMixedSliceShape(
        context, /*insertion=*/true, arguments[3]->spelling,
        arguments[4]->spelling, arguments[5]->spelling,
        arguments[6]->spelling);
  if (call == "slice_shape" && arguments.size() == 5 &&
      arguments[2]->kind == Formula::Kind::Identifier &&
      arguments[3]->kind == Formula::Kind::Identifier &&
      arguments[4]->kind == Formula::Kind::Identifier) {
    ConstraintType source = lookupConstraintType(*arguments[0], context);
    ConstraintType result = lookupConstraintType(*arguments[1], context);
    Attribute offsetsAttribute =
        getConstraintAttribute(context, arguments[2]->spelling);
    Attribute sizesAttribute =
        getConstraintAttribute(context, arguments[3]->spelling);
    Attribute stridesAttribute =
        getConstraintAttribute(context, arguments[4]->spelling);
    if (source.unknown || result.unknown ||
        ((!offsetsAttribute || !sizesAttribute || !stridesAttribute) &&
         context.allowUnknownAttributes))
      return ConstraintTruth::Unknown;
    auto sourceShape = dyn_cast_or_null<ShapedType>(source.type);
    auto resultShape = dyn_cast_or_null<ShapedType>(result.type);
    std::optional<SmallVector<int64_t, 4>> offsets =
        getIntegerArrayAttribute(offsetsAttribute);
    std::optional<SmallVector<int64_t, 4>> sizes =
        getIntegerArrayAttribute(sizesAttribute);
    std::optional<SmallVector<int64_t, 4>> strides =
        getIntegerArrayAttribute(stridesAttribute);
    const bool bothTensors = isa<RankedTensorType>(source.type) &&
                             isa<RankedTensorType>(result.type);
    const bool bothMemrefs = isa<BaseMemRefType>(source.type) &&
                             isa<BaseMemRefType>(result.type);
    if (!sourceShape || !resultShape || (!bothTensors && !bothMemrefs) ||
        !offsets || !sizes || !strides ||
        !sourceShape.hasStaticShape() || !resultShape.hasStaticShape() ||
        sourceShape.getRank() != resultShape.getRank() ||
        sourceShape.getElementType() != resultShape.getElementType() ||
        offsets->size() != static_cast<size_t>(sourceShape.getRank()) ||
        sizes->size() != offsets->size() ||
        strides->size() != offsets->size())
      return ConstraintTruth::False;
    if (auto sourceTensor = dyn_cast<RankedTensorType>(source.type))
      if (sourceTensor.getEncoding() ||
          cast<RankedTensorType>(result.type).getEncoding())
        return ConstraintTruth::False;
    if (bothMemrefs) {
      auto sourceMemref = dyn_cast<MemRefType>(source.type);
      auto resultMemref = dyn_cast<MemRefType>(result.type);
      if (!sourceMemref || !resultMemref ||
          sourceMemref.getMemorySpace() != resultMemref.getMemorySpace())
        return ConstraintTruth::False;
      auto [sourceStrides, sourceOffset] =
          sourceMemref.getStridesAndOffset();
      auto [resultStrides, resultOffset] =
          resultMemref.getStridesAndOffset();
      if (ShapedType::isDynamic(sourceOffset))
        return ConstraintTruth::False;
      int64_t expectedOffset = sourceOffset;
      for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
        if (ShapedType::isDynamic(sourceStrides[dimension]) ||
            ShapedType::isDynamic(resultStrides[dimension]))
          return ConstraintTruth::False;
        expectedOffset += (*offsets)[dimension] * sourceStrides[dimension];
        if (resultStrides[dimension] !=
            sourceStrides[dimension] * (*strides)[dimension])
          return ConstraintTruth::False;
      }
      if (resultOffset != expectedOffset)
        return ConstraintTruth::False;
    }
    for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
      if ((*offsets)[dimension] < 0 || (*sizes)[dimension] <= 0 ||
          (*strides)[dimension] <= 0 ||
          (*sizes)[dimension] != resultShape.getDimSize(dimension) ||
          (*offsets)[dimension] +
                  ((*sizes)[dimension] - 1) * (*strides)[dimension] >=
              sourceShape.getDimSize(dimension))
        return ConstraintTruth::False;
    }
    return ConstraintTruth::True;
  }
  if (call == "insert_slice_shape" && arguments.size() == 6 &&
      arguments[3]->kind == Formula::Kind::Identifier &&
      arguments[4]->kind == Formula::Kind::Identifier &&
      arguments[5]->kind == Formula::Kind::Identifier) {
    ConstraintType source = lookupConstraintType(*arguments[0], context);
    ConstraintType destination = lookupConstraintType(*arguments[1], context);
    ConstraintType result = lookupConstraintType(*arguments[2], context);
    Attribute offsetsAttribute =
        getConstraintAttribute(context, arguments[3]->spelling);
    Attribute sizesAttribute =
        getConstraintAttribute(context, arguments[4]->spelling);
    Attribute stridesAttribute =
        getConstraintAttribute(context, arguments[5]->spelling);
    if (((!offsetsAttribute || !sizesAttribute || !stridesAttribute) &&
         context.allowUnknownAttributes) ||
        result.unknown)
      return ConstraintTruth::Unknown;
    auto resultTensor = dyn_cast_or_null<RankedTensorType>(result.type);
    std::optional<SmallVector<int64_t, 4>> offsets =
        getIntegerArrayAttribute(offsetsAttribute);
    std::optional<SmallVector<int64_t, 4>> sizes =
        getIntegerArrayAttribute(sizesAttribute);
    std::optional<SmallVector<int64_t, 4>> strides =
        getIntegerArrayAttribute(stridesAttribute);
    if (!resultTensor || !offsets || !sizes || !strides ||
        !resultTensor.hasStaticShape() || resultTensor.getEncoding() ||
        offsets->size() != static_cast<size_t>(resultTensor.getRank()) ||
        sizes->size() != offsets->size() ||
        strides->size() != offsets->size())
      return ConstraintTruth::False;
    auto sourceTensor = dyn_cast_or_null<RankedTensorType>(source.type);
    if (!source.unknown &&
        (!sourceTensor || !sourceTensor.hasStaticShape() ||
         sourceTensor.getEncoding() ||
         sourceTensor.getRank() != resultTensor.getRank() ||
         sourceTensor.getElementType() != resultTensor.getElementType()))
      return ConstraintTruth::False;
    auto destinationTensor =
        dyn_cast_or_null<RankedTensorType>(destination.type);
    if (!destination.unknown &&
        (!destinationTensor || !destinationTensor.hasStaticShape() ||
         destinationTensor.getEncoding() || destinationTensor != resultTensor))
      return ConstraintTruth::False;
    for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
      if ((*offsets)[dimension] < 0 || (*sizes)[dimension] <= 0 ||
          (*strides)[dimension] != 1 ||
          (*offsets)[dimension] + (*sizes)[dimension] >
              resultTensor.getDimSize(dimension) ||
          (!source.unknown &&
           (*sizes)[dimension] != sourceTensor.getDimSize(dimension)))
        return ConstraintTruth::False;
    }
    return source.unknown || destination.unknown ? ConstraintTruth::Unknown
                                                 : ConstraintTruth::True;
  }
  if (call == "make_range_shape" && arguments.size() == 3 &&
      arguments[1]->kind == Formula::Kind::Identifier &&
      arguments[2]->kind == Formula::Kind::Identifier) {
    auto result = dyn_cast_or_null<RankedTensorType>(context.resultType);
    auto start = dyn_cast_or_null<IntegerAttr>(
        getConstraintAttribute(context, arguments[1]->spelling));
    auto end = dyn_cast_or_null<IntegerAttr>(
        getConstraintAttribute(context, arguments[2]->spelling));
    if ((!start || !end) && context.allowUnknownAttributes)
      return ConstraintTruth::Unknown;
    return result && result.hasStaticShape() && result.getRank() == 1 &&
                   start && end && end.getInt() > start.getInt() &&
                   result.getDimSize(0) == end.getInt() - start.getInt()
               ? ConstraintTruth::True
               : ConstraintTruth::False;
  }
  return ConstraintTruth::False;
}

static TensorSemanticExpr canonicalIndex(MLIRContext *context, unsigned id) {
  return TensorSemanticExpr::boundIndex(
      context, id, IntegerType::get(context, 32));
}

static TensorDenotation makeDenotationSkeleton(Type type,
                                                const GraphExprNode &node) {
  TensorDenotation result;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    if (Attribute encoding = tensor.getEncoding())
      stream << encoding;
    else
      stream << "<identity>";
    result.layoutIdentity = stream.str();
  } else if (auto memref = dyn_cast<MemRefType>(type)) {
    std::string storage;
    llvm::raw_string_ostream stream(storage);
    stream << "layout=" << memref.getLayout()
           << ";space=" << memref.getMemorySpace();
    result.layoutIdentity = stream.str();
  }
  if (auto shaped = dyn_cast<ShapedType>(type); shaped && shaped.hasRank()) {
    result.shape.assign(shaped.getShape().begin(), shaped.getShape().end());
    std::optional<int64_t> symbolId =
        (node.getOperationName() == "ags.input" ||
         node.getOperationName() == "ags.private_buffer")
            ? node.getIntegerAttribute(kSymbolIdAttr)
            : std::nullopt;
    for (unsigned dimension = 0; dimension < result.shape.size(); ++dimension) {
      if (!ShapedType::isDynamic(result.shape[dimension])) {
        result.shapeSymbols.push_back("");
      } else if (symbolId) {
        StringRef prefix = node.getOperationName() == "ags.private_buffer"
                               ? "private-buffer-"
                               : "input-";
        result.shapeSymbols.push_back(
            (llvm::Twine(prefix) + llvm::Twine(*symbolId) + "-dim-" +
             llvm::Twine(dimension))
                .str());
      } else {
        result.shapeSymbols.push_back("?");
      }
      result.resultIndices.push_back(canonicalIndex(type.getContext(),
                                                    dimension));
    }
  }
  return result;
}

static TensorSemanticExpr
substituteMany(const TensorSemanticExpr &expression,
               ArrayRef<TensorSemanticExpr> from,
               ArrayRef<TensorSemanticExpr> to) {
  assert(from.size() == to.size());
  for (auto [source, replacement] : llvm::zip_equal(from, to))
    if (expression.structurallyEquals(source))
      return replacement;
  SmallVector<TensorSemanticExpr, 4> operands;
  for (const TensorSemanticExpr &operand : expression.getOperands())
    operands.push_back(substituteMany(operand, from, to));
  return TensorSemanticExpr::operation(expression.getKind(),
                                       expression.getType(), operands,
                                       expression.getPayload());
}

static void findLargestIndex(const TensorSemanticExpr &expression,
                             std::optional<uint64_t> &largest) {
  if (expression.getKind() == TensorSemanticKind::BoundIndex) {
    auto value = dyn_cast_or_null<IntegerAttr>(expression.getPayload());
    if (value) {
      uint64_t id = value.getValue().getZExtValue();
      largest = largest ? std::max(*largest, id) : id;
    }
  }
  for (const TensorSemanticExpr &operand : expression.getOperands())
    findLargestIndex(operand, largest);
}

struct EvaluationContext {
  const GraphExprNode &node;
  ArrayRef<TensorDenotation> operands;
  const SemanticClause &clause;
  TensorDenotation &result;
  Type resultElementType;
  llvm::StringMap<TensorSemanticExpr> binders;
  TensorSemanticExpr inputEffect;
  TensorSemanticExpr inputMemory;
  TensorSemanticExpr inputMemoryDefined;
  TensorSemanticExpr inputLaunch;
  TensorSemanticExpr inputBuffer;
};

// A memory-effect operation over tensor operands denotes one transition per
// logical lane.  Keeping this expansion in the generic interpreter is
// essential: operation semantics describe only a scalar memory write, while
// the common engine supplies the tensor iteration space.  Consequently a
// wider store and several adjacent narrow stores can normalize to the same
// functional MemoryState without a StoreCoalescing-specific matcher.
struct StaticLaneDomain {
  SmallVector<int64_t, 4> shape;
  SmallVector<TensorSemanticExpr, 4> indices;
  int64_t elementCount = 1;
};

static std::optional<unsigned>
findOperand(const EvaluationContext &context, const Formula &formula);

static FailureOr<std::optional<StaticLaneDomain>>
getStaticLaneDomain(EvaluationContext &context) {
  std::optional<StaticLaneDomain> domain;
  constexpr int64_t kMaxExpandedMemoryLanes = 4096;
  for (auto [operandIndex, operand] : llvm::enumerate(context.operands)) {
    if (context.clause.laneDomain) {
      std::optional<unsigned> selected =
          findOperand(context, *context.clause.laneDomain);
      if (!selected || operandIndex != *selected)
        continue;
    }
    if (operand.shape.empty())
      continue;
    if (operand.shape.size() != operand.resultIndices.size())
      return failure();
    if (!domain) {
      StaticLaneDomain candidate;
      candidate.shape = operand.shape;
      candidate.indices = operand.resultIndices;
      candidate.elementCount = 1;
      for (int64_t extent : candidate.shape) {
        if (ShapedType::isDynamic(extent) || extent <= 0 ||
            candidate.elementCount > kMaxExpandedMemoryLanes / extent)
          return failure();
        candidate.elementCount *= extent;
      }
      domain = std::move(candidate);
      continue;
    }
    if (operand.shape != domain->shape ||
        operand.resultIndices.size() != domain->indices.size())
      return failure();
    for (auto [lhs, rhs] :
         llvm::zip_equal(operand.resultIndices, domain->indices))
      if (!lhs.structurallyEquals(rhs))
        return failure();
  }
  return domain;
}

static SmallVector<TensorSemanticExpr, 4>
getLaneCoordinates(const StaticLaneDomain &domain, int64_t linearIndex) {
  SmallVector<TensorSemanticExpr, 4> coordinates(domain.shape.size());
  for (int64_t dimension = static_cast<int64_t>(domain.shape.size()) - 1;
       dimension >= 0; --dimension) {
    int64_t coordinate = linearIndex % domain.shape[dimension];
    linearIndex /= domain.shape[dimension];
    Type type = domain.indices[dimension].getType();
    coordinates[dimension] = TensorSemanticExpr::integer(
        type.getContext(), coordinate, type);
  }
  return coordinates;
}

static FailureOr<TensorSemanticExpr>
foldAllLanes(const TensorSemanticExpr &expression,
             EvaluationContext &context) {
  FailureOr<std::optional<StaticLaneDomain>> maybeDomain =
      getStaticLaneDomain(context);
  constexpr int64_t kMaxExpandedValidityLanes = 64;
  if (succeeded(maybeDomain) && *maybeDomain &&
      (**maybeDomain).elementCount <= kMaxExpandedValidityLanes) {
    const StaticLaneDomain &domain = **maybeDomain;
    TensorSemanticExpr result = TensorSemanticExpr::boolean(
        expression.getType().getContext(), true);
    for (int64_t lane = 0; lane < domain.elementCount; ++lane) {
      SmallVector<TensorSemanticExpr, 4> coordinates =
          getLaneCoordinates(domain, lane);
      TensorSemanticExpr laneExpression = substituteMany(
          expression, domain.indices, coordinates);
      result = TensorSemanticExpr::operation(
          TensorSemanticKind::And, expression.getType(),
          {result, laneExpression});
    }
    return result;
  }
  if (succeeded(maybeDomain) && *maybeDomain) {
    const StaticLaneDomain &domain = **maybeDomain;
    TensorSemanticExpr result = expression;
    for (int64_t dimension = static_cast<int64_t>(domain.shape.size()) - 1;
         dimension >= 0; --dimension) {
      TensorSemanticExpr binder = domain.indices[dimension];
      TensorSemanticExpr lower = TensorSemanticExpr::integer(
          expression.getType().getContext(), 0, binder.getType());
      TensorSemanticExpr upper = TensorSemanticExpr::integer(
          expression.getType().getContext(), domain.shape[dimension],
          binder.getType());
      result = TensorSemanticExpr::operation(
          TensorSemanticKind::ForAll, expression.getType(),
          {binder, lower, upper, result});
    }
    return result;
  }

  // A dynamic or deliberately non-expanded domain is represented with the
  // proof language's finite ForAll binder.  This is the quantified analogue
  // of static lane expansion and prevents result-index binders from escaping
  // into a scalar BufferState token.
  const TensorDenotation *domain = nullptr;
  for (auto [operandIndex, operand] : llvm::enumerate(context.operands)) {
    if (context.clause.laneDomain) {
      std::optional<unsigned> selected =
          findOperand(context, *context.clause.laneDomain);
      if (!selected || operandIndex != *selected)
        continue;
    }
    if (operand.shape.empty())
      continue;
    if (operand.shape.size() != operand.resultIndices.size() ||
        operand.shapeSymbols.size() != operand.shape.size())
      return failure();
    if (!domain) {
      domain = &operand;
      continue;
    }
    if (operand.shape != domain->shape ||
        operand.shapeSymbols != domain->shapeSymbols ||
        operand.resultIndices.size() != domain->resultIndices.size())
      return failure();
    for (auto [lhs, rhs] :
         llvm::zip_equal(operand.resultIndices, domain->resultIndices))
      if (!lhs.structurallyEquals(rhs))
        return failure();
  }
  if (!domain)
    return expression;

  TensorSemanticExpr result = expression;
  for (int64_t dimension = static_cast<int64_t>(domain->shape.size()) - 1;
       dimension >= 0; --dimension) {
    TensorSemanticExpr binder = domain->resultIndices[dimension];
    TensorSemanticExpr lower = TensorSemanticExpr::integer(
        expression.getType().getContext(), 0, binder.getType());
    TensorSemanticExpr upper;
    if (!ShapedType::isDynamic(domain->shape[dimension])) {
      upper = TensorSemanticExpr::integer(expression.getType().getContext(),
                                          domain->shape[dimension],
                                          binder.getType());
    } else {
      StringRef symbol = domain->shapeSymbols[dimension];
      if (symbol.empty() || symbol == "?" || symbol == "!conflict")
        return failure();
      upper = TensorSemanticExpr::symbol(
          expression.getType().getContext(),
          (llvm::Twine("extent-") + symbol).str(), binder.getType());
    }
    result = TensorSemanticExpr::operation(
        TensorSemanticKind::ForAll, expression.getType(),
        {binder, lower, upper, result});
  }
  return result;
}

static void appendShapeRelationAtoms(StringRef relation,
                                     SmallVectorImpl<std::string> &atoms) {
  if (relation.consume_front("eq{") && relation.consume_back("}")) {
    SmallVector<StringRef, 4> members;
    relation.split(members, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef member : members)
      atoms.push_back(member.str());
    return;
  }
  atoms.push_back(relation.str());
}

static std::string mergeShapeRelations(StringRef lhs, StringRef rhs) {
  SmallVector<std::string, 4> atoms;
  appendShapeRelationAtoms(lhs, atoms);
  appendShapeRelationAtoms(rhs, atoms);
  llvm::sort(atoms);
  atoms.erase(std::unique(atoms.begin(), atoms.end()), atoms.end());
  if (atoms.size() == 1)
    return atoms.front();

  std::string relation = "eq{";
  for (auto [index, atom] : llvm::enumerate(atoms)) {
    if (index != 0)
      relation += ',';
    relation += atom;
  }
  relation += '}';
  return relation;
}

static void mergeSameShapeSymbols(TensorDenotation &result,
                                  const TensorDenotation &source) {
  if (result.shape.size() != source.shape.size() ||
      result.shapeSymbols.size() != result.shape.size() ||
      source.shapeSymbols.size() != source.shape.size())
    return;
  for (unsigned dimension = 0; dimension < result.shape.size(); ++dimension) {
    if (!ShapedType::isDynamic(result.shape[dimension]) ||
        !ShapedType::isDynamic(source.shape[dimension]))
      continue;
    StringRef incoming = source.shapeSymbols[dimension];
    if (incoming.empty() || incoming == "?" || incoming == "!conflict")
      continue;
    std::string &current = result.shapeSymbols[dimension];
    if (current == "?")
      current = incoming.str();
    else if (!current.empty() && current != "!conflict")
      current = mergeShapeRelations(current, incoming);
  }
}

static std::optional<unsigned> findOperand(const EvaluationContext &context,
                                           const Formula &formula) {
  if (formula.kind != Formula::Kind::Identifier)
    return std::nullopt;
  for (auto [index, name] : llvm::enumerate(context.clause.operands))
    if (name == formula.spelling)
      return index;
  return std::nullopt;
}

static std::optional<int64_t> getIntegerLiteral(const Formula &formula) {
  if (formula.kind != Formula::Kind::Integer)
    return std::nullopt;
  return formula.integer;
}

static FailureOr<TensorSemanticExpr>
evaluate(const Formula &formula, Type expectedType,
         EvaluationContext &context);

static FailureOr<TensorSemanticExpr>
evaluateOperand(const Formula &formula, bool validity,
                EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, formula);
  if (!operand || *operand >= context.operands.size())
    return failure();
  mergeSameShapeSymbols(context.result, context.operands[*operand]);
  return validity ? context.operands[*operand].defined
                  : context.operands[*operand].value;
}

// Executes a captured, single-block scalar region by recursively applying the
// same declarative semantics registry used by the outer graph.  The region is
// data (`ags.canonical_regions`); this interpreter contains no operation-name
// cases.  It deliberately accepts only side-effect-free scalar bodies.  A
// region with memory/control effects fails closed and therefore cannot be used
// as an equality proof.
static FailureOr<TensorSemanticExpr>
evaluateImportedPureRegion(EvaluationContext &context,
                           StringRef regionAttribute, bool validity) {
  if (!context.node.attributes)
    return failure();
  auto regions = dyn_cast_or_null<ArrayAttr>(
      context.node.attributes.get(regionAttribute));
  std::string validationError;
  if (!regions || regions.size() != 1 ||
      failed(validateCanonicalRegions(regions, validationError)))
    return failure();
  auto region = dyn_cast<DictionaryAttr>(regions[0]);
  auto blocks = region ? dyn_cast_or_null<ArrayAttr>(region.get("blocks"))
                       : ArrayAttr{};
  if (!blocks || blocks.size() != 1)
    return failure();
  auto block = dyn_cast<DictionaryAttr>(blocks[0]);
  auto argumentTypes =
      block ? dyn_cast_or_null<ArrayAttr>(block.get("argument_types"))
            : ArrayAttr{};
  auto operations =
      block ? dyn_cast_or_null<ArrayAttr>(block.get("operations"))
            : ArrayAttr{};
  auto terminator =
      block ? dyn_cast_or_null<DictionaryAttr>(block.get("terminator"))
            : DictionaryAttr{};
  if (!argumentTypes || !operations || !terminator ||
      argumentTypes.size() != context.operands.size())
    return failure();

  MLIRContext *mlirContext = context.resultElementType.getContext();
  Type boolType = IntegerType::get(mlirContext, 1);
  TensorSemanticExpr semanticTrue =
      TensorSemanticExpr::boolean(mlirContext, true);
  TensorSemanticExpr initialMemory =
      TensorSemanticExpr::symbol(mlirContext, "memory-state-0", boolType);
  TensorSemanticExpr initialLaunch =
      TensorSemanticExpr::symbol(mlirContext, "launch-state-0", boolType);
  TensorSemanticExpr initialBuffer =
      TensorSemanticExpr::symbol(mlirContext, "buffer-state-0", boolType);

  SmallVector<TensorDenotation, 8> arguments;
  for (auto [typeAttribute, source] :
       llvm::zip_equal(argumentTypes, context.operands)) {
    auto type = dyn_cast<TypeAttr>(typeAttribute);
    if (!type || !source.isValid())
      return failure();
    TensorDenotation argument = source;
    if (auto shaped = dyn_cast<ShapedType>(type.getValue())) {
      if (!shaped.hasRank() ||
          !llvm::equal(source.shape, shaped.getShape()) ||
          source.value.getType() != shaped.getElementType())
        return failure();
    } else {
      if (source.value.getType() != type.getValue())
        return failure();
      argument.shape.clear();
      argument.shapeSymbols.clear();
      argument.layoutIdentity.clear();
      argument.resultIndices.clear();
    }
    argument.effect = semanticTrue;
    argument.memory = initialMemory;
    argument.memoryDefined = semanticTrue;
    argument.launch = initialLaunch;
    argument.launchDefined = semanticTrue;
    argument.buffer = initialBuffer;
    argument.bufferDefined = semanticTrue;
    arguments.push_back(std::move(argument));
  }

  auto parseReferences = [](Attribute attribute)
      -> std::optional<SmallVector<int32_t, 4>> {
    auto references = dyn_cast_or_null<DenseI32ArrayAttr>(attribute);
    if (!references)
      return std::nullopt;
    return SmallVector<int32_t, 4>(references.asArrayRef().begin(),
                                   references.asArrayRef().end());
  };
  auto parseTypes = [](Attribute attribute)
      -> std::optional<SmallVector<Type, 4>> {
    auto attributes = dyn_cast_or_null<ArrayAttr>(attribute);
    if (!attributes)
      return std::nullopt;
    SmallVector<Type, 4> types;
    for (Attribute element : attributes) {
      auto type = dyn_cast<TypeAttr>(element);
      if (!type)
        return std::nullopt;
      types.push_back(type.getValue());
    }
    return types;
  };

  SmallVector<TensorDenotation, 16> results;
  auto resolve = [&](int32_t reference)
      -> std::optional<TensorDenotation> {
    if (reference < 0) {
      int64_t ordinal = -static_cast<int64_t>(reference) - 1;
      if (ordinal < 0 || static_cast<uint64_t>(ordinal) >= arguments.size())
        return std::nullopt;
      return arguments[static_cast<unsigned>(ordinal)];
    }
    if (static_cast<uint32_t>(reference) >= results.size())
      return std::nullopt;
    return results[static_cast<unsigned>(reference)];
  };

  TTOpSemanticsRegistry nestedRegistry = createBuiltinTTOpSemanticsRegistry();
  TensorSemanticNormalizer normalizer;
  Builder builder(mlirContext);
  for (Attribute operationAttribute : operations) {
    auto operation = dyn_cast<DictionaryAttr>(operationAttribute);
    auto name = operation
                    ? dyn_cast_or_null<StringAttr>(operation.get("name"))
                    : StringAttr{};
    auto attributes =
        operation ? dyn_cast_or_null<DictionaryAttr>(
                        operation.get("attributes"))
                  : DictionaryAttr{};
    std::optional<SmallVector<int32_t, 4>> operandReferences =
        operation ? parseReferences(operation.get("operands"))
                  : std::nullopt;
    std::optional<SmallVector<Type, 4>> resultTypes =
        operation ? parseTypes(operation.get("result_types"))
                  : std::nullopt;
    auto successors =
        operation ? dyn_cast_or_null<DenseI32ArrayAttr>(
                        operation.get("successors"))
                  : DenseI32ArrayAttr{};
    auto nestedRegions =
        operation ? dyn_cast_or_null<ArrayAttr>(operation.get("regions"))
                  : ArrayAttr{};
    if (!name || !attributes || !operandReferences || !resultTypes ||
        !successors || !successors.empty() || !nestedRegions ||
        !nestedRegions.empty() || resultTypes->empty())
      return failure();
    const TTOpSemantics *definition =
        nestedRegistry.lookup(name.getValue());
    if (!definition)
      return failure();
    SmallVector<TensorDenotation, 4> operands;
    for (int32_t reference : *operandReferences) {
      std::optional<TensorDenotation> operand = resolve(reference);
      if (!operand)
        return failure();
      operands.push_back(std::move(*operand));
    }
    SmallVector<Attribute, 4> encodedResultTypes;
    for (Type type : *resultTypes)
      encodedResultTypes.push_back(TypeAttr::get(type));
    for (auto [resultIndex, resultType] : llvm::enumerate(*resultTypes)) {
      SmallVector<NamedAttribute, 8> semanticAttributes(attributes.begin(),
                                                         attributes.end());
      semanticAttributes.push_back(builder.getNamedAttr(
          kResultTypeAttr, TypeAttr::get(resultType)));
      if (resultTypes->size() > 1) {
        semanticAttributes.push_back(builder.getNamedAttr(
            kResultIndexAttr,
            builder.getI64IntegerAttr(resultIndex)));
        semanticAttributes.push_back(builder.getNamedAttr(
            kOperationResultTypesAttr,
            builder.getArrayAttr(encodedResultTypes)));
      }
      GraphExprNode node = GraphExprNode::create(
          mlirContext, name.getValue(), /*operands=*/{}, semanticAttributes);
      FailureOr<TensorDenotation> denotation =
          definition->denote(node, operands);
      if (failed(denotation))
        return failure();
      TensorDenotation normalized =
          normalizer.normalize(std::move(*denotation)).denotation;
      if (!normalized.effect ||
          !normalized.effect.structurallyEquals(semanticTrue) ||
          !normalized.memory ||
          !normalized.memory.structurallyEquals(initialMemory) ||
          !normalized.memoryDefined ||
          !normalized.memoryDefined.structurallyEquals(semanticTrue) ||
          !normalized.launch ||
          !normalized.launch.structurallyEquals(initialLaunch) ||
          !normalized.launchDefined ||
          !normalized.launchDefined.structurallyEquals(semanticTrue) ||
          !normalized.buffer ||
          !normalized.buffer.structurallyEquals(initialBuffer) ||
          !normalized.bufferDefined ||
          !normalized.bufferDefined.structurallyEquals(semanticTrue))
        return failure();
      results.push_back(std::move(normalized));
    }
  }

  std::optional<SmallVector<int32_t, 4>> returned =
      parseReferences(terminator.get("operands"));
  unsigned resultIndex = 0;
  if (auto index = dyn_cast_or_null<IntegerAttr>(
          context.node.attributes.get(kResultIndexAttr))) {
    if (index.getInt() < 0)
      return failure();
    resultIndex = static_cast<unsigned>(index.getInt());
  }
  if (!returned || resultIndex >= returned->size())
    return failure();
  std::optional<TensorDenotation> selected = resolve((*returned)[resultIndex]);
  if (!selected || selected->value.getType() != context.resultElementType)
    return failure();
  mergeSameShapeSymbols(context.result, *selected);
  return validity ? selected->defined : selected->value;
}

static FailureOr<TensorSemanticExpr>
evaluateAt(const Formula &operandFormula, ArrayRef<FormulaPtr> indexFormulas,
           bool validity, EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, operandFormula);
  if (!operand || *operand >= context.operands.size())
    return failure();
  const TensorDenotation &denotation = context.operands[*operand];
  if (denotation.resultIndices.size() != indexFormulas.size())
    return failure();
  SmallVector<TensorSemanticExpr, 4> indices;
  for (auto [formula, formalIndex] :
       llvm::zip_equal(indexFormulas, denotation.resultIndices)) {
    FailureOr<TensorSemanticExpr> index =
        evaluate(*formula, formalIndex.getType(), context);
    if (failed(index))
      return failure();
    indices.push_back(*index);
  }
  return substituteMany(validity ? denotation.defined : denotation.value,
                        denotation.resultIndices, indices);
}

static FailureOr<int64_t> getNodeIntegerAttribute(
    const Formula &formula, const EvaluationContext &context) {
  if (formula.kind != Formula::Kind::Identifier)
    return failure();
  std::optional<int64_t> value =
      context.node.getIntegerAttribute(formula.spelling);
  if (!value)
    return failure();
  return *value;
}

static FailureOr<bool> getNodeBooleanAttribute(
    const Formula &formula, const EvaluationContext &context) {
  if (formula.kind != Formula::Kind::Identifier || !context.node.attributes)
    return failure();
  auto value = dyn_cast_or_null<BoolAttr>(
      context.node.attributes.get(formula.spelling));
  if (!value)
    return failure();
  return value.getValue();
}

static TensorSemanticExpr makeFreshBinder(EvaluationContext &context) {
  static constexpr uint64_t kFirstBinderId = 1ULL << 20;
  std::optional<uint64_t> largest;
  for (const TensorSemanticExpr &index : context.result.resultIndices)
    findLargestIndex(index, largest);
  for (const TensorDenotation &operand : context.operands) {
    findLargestIndex(operand.value, largest);
    findLargestIndex(operand.defined, largest);
  }
  for (const auto &entry : context.binders)
    findLargestIndex(entry.getValue(), largest);
  uint64_t next = largest ? *largest + 1 : kFirstBinderId;
  next = std::max(next, kFirstBinderId);
  return canonicalIndex(context.resultElementType.getContext(),
                        static_cast<unsigned>(next));
}

static FailureOr<TensorSemanticExpr>
evaluateReshape(const Formula &operandFormula,
                const Formula *axisFormula, bool broadcast, bool validity,
                EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, operandFormula);
  if (!operand || *operand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*operand];
  SmallVector<TensorSemanticExpr, 4> indices;
  if (broadcast) {
    if (source.shape.size() != context.result.shape.size())
      return failure();
    for (auto [dimension, extent] : llvm::enumerate(source.shape)) {
      if (extent == 1)
        indices.push_back(TensorSemanticExpr::integer(
            context.resultElementType.getContext(), 0,
            source.resultIndices[dimension].getType()));
      else if (extent == context.result.shape[dimension])
        indices.push_back(context.result.resultIndices[dimension]);
      else
        return failure();
      if (ShapedType::isDynamic(extent) &&
          ShapedType::isDynamic(context.result.shape[dimension]) &&
          source.shapeSymbols.size() == source.shape.size() &&
          context.result.shapeSymbols.size() == context.result.shape.size())
        context.result.shapeSymbols[dimension] =
            source.shapeSymbols[dimension];
    }
  } else {
    if (!axisFormula ||
        source.shape.size() + 1 != context.result.shape.size())
      return failure();
    FailureOr<int64_t> axis = getNodeIntegerAttribute(*axisFormula, context);
    if (failed(axis) || *axis < 0 ||
        static_cast<size_t>(*axis) >= context.result.shape.size())
      return failure();
    unsigned sourceDimension = 0;
    for (unsigned dimension = 0; dimension < context.result.shape.size();
         ++dimension) {
      if (dimension != static_cast<unsigned>(*axis)) {
        indices.push_back(context.result.resultIndices[dimension]);
        if (ShapedType::isDynamic(context.result.shape[dimension]) &&
            source.shapeSymbols.size() == source.shape.size() &&
            context.result.shapeSymbols.size() == context.result.shape.size())
          context.result.shapeSymbols[dimension] =
              source.shapeSymbols[sourceDimension];
        ++sourceDimension;
      }
    }
  }
  if (indices.size() != source.resultIndices.size())
    return failure();
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, indices);
}

static FailureOr<TensorSemanticExpr>
evaluateTranspose(const Formula &operandFormula,
                  const Formula &orderFormula, bool validity,
                  EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, operandFormula);
  if (!operand || *operand >= context.operands.size() ||
      orderFormula.kind != Formula::Kind::Identifier)
    return failure();
  const TensorDenotation &source = context.operands[*operand];
  auto order = dyn_cast_or_null<DenseI32ArrayAttr>(
      context.node.attributes.get(orderFormula.spelling));
  if (!order || order.size() != source.shape.size() ||
      order.size() != context.result.shape.size())
    return failure();

  SmallVector<TensorSemanticExpr, 4> sourceIndices(order.size());
  SmallVector<bool, 4> seen(order.size(), false);
  for (auto [resultDimension, sourceDimension] :
       llvm::enumerate(order.asArrayRef())) {
    if (sourceDimension < 0 ||
        static_cast<size_t>(sourceDimension) >= order.size() ||
        seen[sourceDimension] ||
        source.shape[sourceDimension] != context.result.shape[resultDimension])
      return failure();
    seen[sourceDimension] = true;
    sourceIndices[sourceDimension] =
        context.result.resultIndices[resultDimension];
    if (ShapedType::isDynamic(context.result.shape[resultDimension]) &&
        source.shapeSymbols.size() == source.shape.size() &&
        context.result.shapeSymbols.size() == context.result.shape.size())
      context.result.shapeSymbols[resultDimension] =
          source.shapeSymbols[sourceDimension];
  }
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, sourceIndices);
}

static FailureOr<TensorSemanticExpr>
evaluateSingletonTensor(const Formula &operandFormula, bool validity,
                        EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, operandFormula);
  if (!operand || *operand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*operand];
  if (source.shape.empty() ||
      source.shape.size() != source.resultIndices.size() ||
      llvm::any_of(source.shape, [](int64_t extent) { return extent != 1; }))
    return failure();
  SmallVector<TensorSemanticExpr, 4> zeros;
  zeros.reserve(source.resultIndices.size());
  for (const TensorSemanticExpr &index : source.resultIndices)
    zeros.push_back(TensorSemanticExpr::integer(
        context.resultElementType.getContext(), 0, index.getType()));
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, zeros);
}

static FailureOr<TensorSemanticExpr>
evaluateJoin(const Formula &lhsFormula, const Formula &rhsFormula,
             bool validity, EvaluationContext &context) {
  std::optional<unsigned> lhsIndex = findOperand(context, lhsFormula);
  std::optional<unsigned> rhsIndex = findOperand(context, rhsFormula);
  if (!lhsIndex || !rhsIndex || *lhsIndex >= context.operands.size() ||
      *rhsIndex >= context.operands.size())
    return failure();
  const TensorDenotation &lhs = context.operands[*lhsIndex];
  const TensorDenotation &rhs = context.operands[*rhsIndex];
  if (lhs.shape != rhs.shape ||
      context.result.resultIndices.size() != lhs.resultIndices.size() + 1)
    return failure();
  ArrayRef<TensorSemanticExpr> coordinates =
      ArrayRef<TensorSemanticExpr>(context.result.resultIndices).drop_back();
  TensorSemanticExpr lhsValue = substituteMany(
      validity ? lhs.defined : lhs.value, lhs.resultIndices, coordinates);
  TensorSemanticExpr rhsValue = substituteMany(
      validity ? rhs.defined : rhs.value, rhs.resultIndices, coordinates);
  TensorSemanticExpr lane = context.result.resultIndices.back();
  TensorSemanticExpr zero = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), 0, lane.getType());
  TensorSemanticExpr selectLhs = TensorSemanticExpr::operation(
      TensorSemanticKind::Equal,
      IntegerType::get(context.resultElementType.getContext(), 1),
      {lane, zero});
  return TensorSemanticExpr::operation(
      TensorSemanticKind::Select,
      validity ? IntegerType::get(context.resultElementType.getContext(), 1)
               : context.resultElementType,
      {selectLhs, lhsValue, rhsValue});
}

static FailureOr<TensorSemanticExpr>
evaluateSplit(const Formula &sourceFormula, const Formula &laneFormula,
              bool validity, EvaluationContext &context) {
  std::optional<unsigned> sourceIndex = findOperand(context, sourceFormula);
  std::optional<int64_t> lane = getIntegerLiteral(laneFormula);
  if (!sourceIndex || *sourceIndex >= context.operands.size() || !lane ||
      (*lane != 0 && *lane != 1))
    return failure();
  const TensorDenotation &source = context.operands[*sourceIndex];
  if (source.resultIndices.size() != context.result.resultIndices.size() + 1)
    return failure();
  SmallVector<TensorSemanticExpr, 4> coordinates(
      context.result.resultIndices.begin(),
      context.result.resultIndices.end());
  coordinates.push_back(TensorSemanticExpr::integer(
      context.resultElementType.getContext(), *lane,
      source.resultIndices.back().getType()));
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, coordinates);
}

static FailureOr<TensorSemanticExpr>
evaluateCat(const Formula &lhsFormula, const Formula &rhsFormula,
            bool validity, EvaluationContext &context) {
  std::optional<unsigned> lhsIndex = findOperand(context, lhsFormula);
  std::optional<unsigned> rhsIndex = findOperand(context, rhsFormula);
  if (!lhsIndex || !rhsIndex || *lhsIndex >= context.operands.size() ||
      *rhsIndex >= context.operands.size())
    return failure();
  const TensorDenotation &lhs = context.operands[*lhsIndex];
  const TensorDenotation &rhs = context.operands[*rhsIndex];
  if (lhs.shape.size() != 1 || rhs.shape.size() != 1 ||
      context.result.resultIndices.size() != 1 ||
      ShapedType::isDynamic(lhs.shape.front()) || lhs.shape.front() <= 0)
    return failure();
  TensorSemanticExpr index = context.result.resultIndices.front();
  TensorSemanticExpr lhsExtent = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), lhs.shape.front(),
      index.getType());
  TensorSemanticExpr inLhs = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan,
      IntegerType::get(context.resultElementType.getContext(), 1),
      {index, lhsExtent});
  TensorSemanticExpr rhsIndexValue = TensorSemanticExpr::operation(
      TensorSemanticKind::Subtract, index.getType(), {index, lhsExtent});
  TensorSemanticExpr lhsValue = substituteMany(
      validity ? lhs.defined : lhs.value, lhs.resultIndices, {index});
  TensorSemanticExpr rhsValue = substituteMany(
      validity ? rhs.defined : rhs.value, rhs.resultIndices,
      {rhsIndexValue});
  return TensorSemanticExpr::operation(
      TensorSemanticKind::Select,
      validity ? IntegerType::get(context.resultElementType.getContext(), 1)
               : context.resultElementType,
      {inLhs, lhsValue, rhsValue});
}

static FailureOr<TensorSemanticExpr>
evaluateGather(const Formula &sourceFormula, const Formula &indicesFormula,
               const Formula &axisFormula, bool validity,
               EvaluationContext &context) {
  std::optional<unsigned> sourceIndex = findOperand(context, sourceFormula);
  std::optional<unsigned> indicesIndex = findOperand(context, indicesFormula);
  if (!sourceIndex || !indicesIndex ||
      *sourceIndex >= context.operands.size() ||
      *indicesIndex >= context.operands.size())
    return failure();
  FailureOr<int64_t> axis = getNodeIntegerAttribute(axisFormula, context);
  const TensorDenotation &source = context.operands[*sourceIndex];
  const TensorDenotation &indices = context.operands[*indicesIndex];
  if (failed(axis) || *axis < 0 ||
      static_cast<size_t>(*axis) >= source.shape.size() ||
      source.resultIndices.size() != context.result.resultIndices.size() ||
      indices.resultIndices.size() != context.result.resultIndices.size() ||
      ShapedType::isDynamic(source.shape[*axis]) || source.shape[*axis] <= 0)
    return failure();
  TensorSemanticExpr gatheredIndex = substituteMany(
      indices.value, indices.resultIndices, context.result.resultIndices);
  SmallVector<TensorSemanticExpr, 4> sourceCoordinates(
      context.result.resultIndices.begin(),
      context.result.resultIndices.end());
  sourceCoordinates[*axis] = gatheredIndex;
  TensorSemanticExpr sourceValue = substituteMany(
      validity ? source.defined : source.value, source.resultIndices,
      sourceCoordinates);
  if (!validity)
    return sourceValue;
  TensorSemanticExpr indicesValid = substituteMany(
      indices.defined, indices.resultIndices, context.result.resultIndices);
  Type boolType = IntegerType::get(context.resultElementType.getContext(), 1);
  TensorSemanticExpr zero = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), 0, gatheredIndex.getType());
  TensorSemanticExpr upper = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), source.shape[*axis],
      gatheredIndex.getType());
  TensorSemanticExpr belowZero = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, boolType, {gatheredIndex, zero});
  TensorSemanticExpr nonnegative = TensorSemanticExpr::operation(
      TensorSemanticKind::Not, boolType, {belowZero});
  TensorSemanticExpr belowUpper = TensorSemanticExpr::operation(
      TensorSemanticKind::SignedLessThan, boolType, {gatheredIndex, upper});
  return TensorSemanticExpr::operation(
      TensorSemanticKind::And, boolType,
      {indicesValid, nonnegative, belowUpper, sourceValue});
}

static FailureOr<TensorSemanticExpr>
evaluateDot(const Formula &lhsFormula, const Formula &rhsFormula,
            const Formula &accFormula, bool validity,
            EvaluationContext &context) {
  std::optional<unsigned> lhsIndex = findOperand(context, lhsFormula);
  std::optional<unsigned> rhsIndex = findOperand(context, rhsFormula);
  std::optional<unsigned> accIndex = findOperand(context, accFormula);
  if (!lhsIndex || !rhsIndex || !accIndex ||
      *lhsIndex >= context.operands.size() ||
      *rhsIndex >= context.operands.size() ||
      *accIndex >= context.operands.size())
    return failure();
  const TensorDenotation &lhs = context.operands[*lhsIndex];
  const TensorDenotation &rhs = context.operands[*rhsIndex];
  const TensorDenotation &acc = context.operands[*accIndex];
  if (lhs.shape.size() != 2 || rhs.shape.size() != 2 ||
      acc.shape.size() != 2 || context.result.resultIndices.size() != 2 ||
      lhs.resultIndices.size() != 2 || rhs.resultIndices.size() != 2 ||
      acc.resultIndices.size() != 2 || ShapedType::isDynamic(lhs.shape[1]) ||
      lhs.shape[1] <= 0 || lhs.shape[1] != rhs.shape[0])
    return failure();

  TensorSemanticExpr reductionIndex = makeFreshBinder(context);
  TensorSemanticExpr lhsValue = substituteMany(
      validity ? lhs.defined : lhs.value, lhs.resultIndices,
      {context.result.resultIndices[0], reductionIndex});
  TensorSemanticExpr rhsValue = substituteMany(
      validity ? rhs.defined : rhs.value, rhs.resultIndices,
      {reductionIndex, context.result.resultIndices[1]});
  TensorSemanticExpr accValue = substituteMany(
      validity ? acc.defined : acc.value, acc.resultIndices,
      context.result.resultIndices);
  Type indexType = reductionIndex.getType();
  TensorSemanticExpr lower = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), 0, indexType);
  TensorSemanticExpr upper = TensorSemanticExpr::integer(
      context.resultElementType.getContext(), lhs.shape[1], indexType);
  if (validity) {
    Type boolType = IntegerType::get(context.resultElementType.getContext(), 1);
    TensorSemanticExpr both = TensorSemanticExpr::operation(
        TensorSemanticKind::And, boolType, {lhsValue, rhsValue});
    TensorSemanticExpr all = TensorSemanticExpr::operation(
        TensorSemanticKind::ForAll, boolType,
        {reductionIndex, lower, upper, both});
    return TensorSemanticExpr::operation(TensorSemanticKind::And, boolType,
                                         {accValue, all});
  }
  TensorSemanticExpr product = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, context.resultElementType,
      {lhsValue, rhsValue});
  TensorSemanticExpr sum = TensorSemanticExpr::operation(
      TensorSemanticKind::Sum, context.resultElementType,
      {reductionIndex, lower, upper, product});
  return TensorSemanticExpr::operation(TensorSemanticKind::Add,
                                       context.resultElementType,
                                       {sum, accValue});
}

static FailureOr<TensorSemanticExpr>
evaluateDotScaled(const Formula &formula, bool validity,
                  EvaluationContext &context) {
  if (formula.arguments.size() < 3 || formula.arguments.size() > 5 ||
      context.operands.size() < 3 || context.operands.size() > 5)
    return failure();
  for (unsigned index = 0; index < 3; ++index) {
    std::optional<unsigned> operand =
        findOperand(context, *formula.arguments[index]);
    if (!operand || *operand != index)
      return failure();
  }
  const TensorDenotation &lhs = context.operands[0];
  const TensorDenotation &rhs = context.operands[1];
  const TensorDenotation &acc = context.operands[2];
  const unsigned rank = lhs.shape.size();
  if ((rank != 2 && rank != 3) || rhs.shape.size() != rank ||
      acc.shape.size() != rank || context.result.shape.size() != rank ||
      lhs.resultIndices.size() != rank || rhs.resultIndices.size() != rank ||
      acc.resultIndices.size() != rank ||
      context.result.resultIndices.size() != rank)
    return failure();

  std::optional<SmallVector<int64_t, 4>> segments =
      getIntegerArrayAttribute(
          context.node.attributes.get("operandSegmentSizes"));
  if (!segments || segments->size() != 5 || (*segments)[0] != 1 ||
      (*segments)[1] != 1 || (*segments)[2] != 1 ||
      ((*segments)[3] != 0 && (*segments)[3] != 1) ||
      ((*segments)[4] != 0 && (*segments)[4] != 1) ||
      3 + (*segments)[3] + (*segments)[4] != context.operands.size())
    return failure();
  unsigned nextOptional = 3;
  const TensorDenotation *lhsScale =
      (*segments)[3] ? &context.operands[nextOptional++] : nullptr;
  const TensorDenotation *rhsScale =
      (*segments)[4] ? &context.operands[nextOptional++] : nullptr;

  std::optional<int64_t> lhsFormat =
      context.node.getIntegerAttribute("a_elem_type");
  std::optional<int64_t> rhsFormat =
      context.node.getIntegerAttribute("b_elem_type");
  auto fastMath = dyn_cast_or_null<BoolAttr>(
      context.node.attributes.get("fastMath"));
  auto readPack = [&](StringRef name) -> std::optional<bool> {
    Attribute attribute = context.node.attributes.get(name);
    if (!attribute)
      return true;
    auto boolean = dyn_cast<BoolAttr>(attribute);
    return boolean ? std::optional<bool>(boolean.getValue()) : std::nullopt;
  };
  std::optional<bool> lhsKPack = readPack("lhs_k_pack");
  std::optional<bool> rhsKPack = readPack("rhs_k_pack");
  if (!lhsFormat || !rhsFormat || !fastMath || !lhsKPack || !rhsKPack)
    return failure();

  constexpr int64_t kE2M1 = 4;
  auto logicalK = [&](const TensorDenotation &operand, int64_t format,
                      bool kPack, bool lhsSide) -> std::optional<int64_t> {
    int64_t physical = operand.shape[lhsSide ? rank - 1 : rank - 2];
    if (ShapedType::isDynamic(physical) || physical <= 0)
      return std::nullopt;
    return format == kE2M1 && kPack
               ? std::optional<int64_t>(physical * 2)
               : std::optional<int64_t>(physical);
  };
  std::optional<int64_t> lhsK =
      logicalK(lhs, *lhsFormat, *lhsKPack, /*lhsSide=*/true);
  std::optional<int64_t> rhsK =
      logicalK(rhs, *rhsFormat, *rhsKPack, /*lhsSide=*/false);
  if (!lhsK || !rhsK || *lhsK != *rhsK)
    return failure();
  int64_t logicalM = lhs.shape[rank - 2];
  int64_t logicalN = rhs.shape[rank - 1];
  if (*lhsFormat == kE2M1 && !*lhsKPack)
    logicalM *= 2;
  if (*rhsFormat == kE2M1 && !*rhsKPack)
    logicalN *= 2;
  if (logicalM <= 0 || logicalN <= 0 ||
      acc.shape[rank - 2] != logicalM ||
      acc.shape[rank - 1] != logicalN ||
      !llvm::equal(context.result.shape, acc.shape) ||
      !llvm::equal(ArrayRef<int64_t>(lhs.shape).take_front(rank - 2),
                   ArrayRef<int64_t>(context.result.shape)
                       .take_front(rank - 2)) ||
      !llvm::equal(ArrayRef<int64_t>(rhs.shape).take_front(rank - 2),
                   ArrayRef<int64_t>(context.result.shape)
                       .take_front(rank - 2)))
    return failure();

  TensorSemanticExpr reductionIndex = makeFreshBinder(context);
  Type indexType = reductionIndex.getType();
  auto integer = [&](int64_t value) {
    return TensorSemanticExpr::integer(indexType.getContext(), value,
                                       indexType);
  };
  auto divide = [&](TensorSemanticExpr value, int64_t divisor) {
    return divisor == 1
               ? value
               : TensorSemanticExpr::operation(
                     TensorSemanticKind::IntegerDivide, indexType,
                     {value, integer(divisor)});
  };
  auto remainder = [&](TensorSemanticExpr value, int64_t divisor) {
    return TensorSemanticExpr::operation(TensorSemanticKind::SignedRemainder,
                                         indexType,
                                         {value, integer(divisor)});
  };
  SmallVector<TensorSemanticExpr, 4> prefix(
      context.result.resultIndices.begin(),
      context.result.resultIndices.begin() + rank - 2);
  TensorSemanticExpr row = context.result.resultIndices[rank - 2];
  TensorSemanticExpr column = context.result.resultIndices[rank - 1];

  struct DecodedOperand {
    TensorSemanticExpr value;
    TensorSemanticExpr valid;
  };
  auto decode = [&](const TensorDenotation &operand,
                    const TensorDenotation *scale, int64_t format,
                    bool kPack, bool lhsSide) -> FailureOr<DecodedOperand> {
    SmallVector<TensorSemanticExpr, 4> coordinates(prefix);
    TensorSemanticExpr outer = lhsSide ? row : column;
    TensorSemanticExpr physicalOuter = outer;
    TensorSemanticExpr physicalK = reductionIndex;
    std::optional<TensorSemanticExpr> nibble;
    if (format == kE2M1) {
      if (kPack) {
        physicalK = divide(reductionIndex, 2);
        nibble = remainder(reductionIndex, 2);
      } else {
        physicalOuter = divide(outer, 2);
        nibble = remainder(outer, 2);
      }
    }
    if (lhsSide) {
      coordinates.push_back(physicalOuter);
      coordinates.push_back(physicalK);
    } else {
      coordinates.push_back(physicalK);
      coordinates.push_back(physicalOuter);
    }
    TensorSemanticExpr raw = substituteMany(
        operand.value, operand.resultIndices, coordinates);
    TensorSemanticExpr rawValid = substituteMany(
        operand.defined, operand.resultIndices, coordinates);
    SmallVector<TensorSemanticExpr, 3> decodedOperands{raw};
    TensorSemanticExpr combinedValid = rawValid;
    int64_t scaleFactor = 0;
    Type scaleType;
    if (scale) {
      if (scale->shape.size() != rank ||
          scale->resultIndices.size() != rank)
        return failure();
      scaleType = scale->value.getType();
      scaleFactor = isa<Float8E4M3FNType>(scaleType) ? 16 : 32;
      SmallVector<TensorSemanticExpr, 4> scaleCoordinates(prefix);
      scaleCoordinates.push_back(outer);
      scaleCoordinates.push_back(divide(reductionIndex, scaleFactor));
      TensorSemanticExpr scaleValue = substituteMany(
          scale->value, scale->resultIndices, scaleCoordinates);
      TensorSemanticExpr scaleValid = substituteMany(
          scale->defined, scale->resultIndices, scaleCoordinates);
      decodedOperands.push_back(scaleValue);
      combinedValid = TensorSemanticExpr::operation(
          TensorSemanticKind::And,
          IntegerType::get(context.resultElementType.getContext(), 1),
          {combinedValid, scaleValid});
    }
    if (nibble)
      decodedOperands.push_back(*nibble);
    Builder builder(context.resultElementType.getContext());
    SmallVector<NamedAttribute, 8> payload{
        builder.getNamedAttr("format", builder.getI32IntegerAttr(format)),
        builder.getNamedAttr("side",
                             builder.getStringAttr(lhsSide ? "lhs" : "rhs")),
        builder.getNamedAttr("k_pack", builder.getBoolAttr(kPack)),
        builder.getNamedAttr("fast_math", fastMath),
        builder.getNamedAttr("scale_factor",
                             builder.getI64IntegerAttr(scaleFactor))};
    if (scaleType)
      payload.push_back(builder.getNamedAttr("scale_type",
                                             TypeAttr::get(scaleType)));
    return DecodedOperand{TensorSemanticExpr::operation(
                              TensorSemanticKind::MicroscaleDecode,
                              context.resultElementType, decodedOperands,
                              builder.getDictionaryAttr(payload)),
                          combinedValid};
  };

  FailureOr<DecodedOperand> decodedLhs =
      decode(lhs, lhsScale, *lhsFormat, *lhsKPack, /*lhsSide=*/true);
  FailureOr<DecodedOperand> decodedRhs =
      decode(rhs, rhsScale, *rhsFormat, *rhsKPack, /*lhsSide=*/false);
  if (failed(decodedLhs) || failed(decodedRhs))
    return failure();
  TensorSemanticExpr accExpr = substituteMany(
      validity ? acc.defined : acc.value, acc.resultIndices,
      context.result.resultIndices);
  TensorSemanticExpr lower = integer(0);
  TensorSemanticExpr upper = integer(*lhsK);
  if (validity) {
    Type boolType = IntegerType::get(context.resultElementType.getContext(), 1);
    TensorSemanticExpr both = TensorSemanticExpr::operation(
        TensorSemanticKind::And, boolType,
        {decodedLhs->valid, decodedRhs->valid});
    TensorSemanticExpr all = TensorSemanticExpr::operation(
        TensorSemanticKind::ForAll, boolType,
        {reductionIndex, lower, upper, both});
    return TensorSemanticExpr::operation(TensorSemanticKind::And, boolType,
                                         {accExpr, all});
  }
  TensorSemanticExpr product = TensorSemanticExpr::operation(
      TensorSemanticKind::Multiply, context.resultElementType,
      {decodedLhs->value, decodedRhs->value});
  TensorSemanticExpr sum = TensorSemanticExpr::operation(
      TensorSemanticKind::Sum, context.resultElementType,
      {reductionIndex, lower, upper, product});
  return TensorSemanticExpr::operation(TensorSemanticKind::Add,
                                       context.resultElementType,
                                       {sum, accExpr});
}

static FailureOr<TensorSemanticExpr>
evaluateStaticReshape(const Formula &operandFormula, bool validity,
                      EvaluationContext &context) {
  std::optional<unsigned> operand = findOperand(context, operandFormula);
  if (!operand || *operand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*operand];
  if (source.shape.empty() || context.result.shape.empty() ||
      source.shape.size() != source.resultIndices.size() ||
      context.result.shape.size() != context.result.resultIndices.size() ||
      llvm::any_of(source.shape, [](int64_t extent) {
        return ShapedType::isDynamic(extent);
      }) ||
      llvm::any_of(context.result.shape, [](int64_t extent) {
        return ShapedType::isDynamic(extent);
      }))
    return failure();

  Type indexType = context.result.resultIndices.front().getType();
  auto integer = [&](int64_t value) {
    return TensorSemanticExpr::integer(indexType.getContext(), value,
                                       indexType);
  };
  TensorSemanticExpr linear = integer(0);
  for (auto [dimension, index] :
       llvm::enumerate(context.result.resultIndices)) {
    linear = TensorSemanticExpr::operation(
        TensorSemanticKind::Multiply, indexType,
        {linear, integer(context.result.shape[dimension])});
    linear = TensorSemanticExpr::operation(TensorSemanticKind::Add,
                                           indexType, {linear, index});
  }

  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  int64_t trailing = 1;
  SmallVector<int64_t, 4> divisors(source.shape.size(), 1);
  for (int64_t dimension = static_cast<int64_t>(source.shape.size()) - 1;
       dimension >= 0; --dimension) {
    divisors[dimension] = trailing;
    trailing *= source.shape[dimension];
  }
  for (unsigned dimension = 0; dimension < source.shape.size(); ++dimension) {
    TensorSemanticExpr coordinate = linear;
    if (divisors[dimension] != 1)
      coordinate = TensorSemanticExpr::operation(
          TensorSemanticKind::IntegerDivide, indexType,
          {coordinate, integer(divisors[dimension])});
    if (dimension + 1 != source.shape.size())
      coordinate = TensorSemanticExpr::operation(
          TensorSemanticKind::SignedRemainder, indexType,
          {coordinate, integer(source.shape[dimension])});
    sourceIndices.push_back(std::move(coordinate));
  }
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, sourceIndices);
}

static FailureOr<TensorSemanticExpr>
evaluateStaticSlice(const Formula &formula, bool validity,
                    EvaluationContext &context) {
  if (formula.arguments.size() != 4 ||
      formula.arguments[1]->kind != Formula::Kind::Identifier ||
      formula.arguments[2]->kind != Formula::Kind::Identifier ||
      formula.arguments[3]->kind != Formula::Kind::Identifier)
    return failure();
  std::optional<unsigned> sourceOperand =
      findOperand(context, *formula.arguments[0]);
  if (!sourceOperand || *sourceOperand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*sourceOperand];
  std::optional<SmallVector<int64_t, 4>> offsets =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[1]->spelling));
  std::optional<SmallVector<int64_t, 4>> sizes =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[2]->spelling));
  std::optional<SmallVector<int64_t, 4>> strides =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[3]->spelling));
  if (!offsets || !sizes || !strides ||
      offsets->size() != context.result.resultIndices.size() ||
      sizes->size() != offsets->size() ||
      strides->size() != offsets->size() ||
      source.resultIndices.size() != offsets->size())
    return failure();

  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
    Type indexType = source.resultIndices[dimension].getType();
    TensorSemanticExpr coordinate = context.result.resultIndices[dimension];
    if ((*strides)[dimension] != 1) {
      TensorSemanticExpr stride = TensorSemanticExpr::integer(
          indexType.getContext(), (*strides)[dimension], indexType);
      coordinate = TensorSemanticExpr::operation(
          TensorSemanticKind::Multiply, indexType, {coordinate, stride});
    }
    if ((*offsets)[dimension] != 0) {
      TensorSemanticExpr offset = TensorSemanticExpr::integer(
          indexType.getContext(), (*offsets)[dimension], indexType);
      coordinate = TensorSemanticExpr::operation(
          TensorSemanticKind::Add, indexType, {coordinate, offset});
    }
    sourceIndices.push_back(std::move(coordinate));
  }
  return substituteMany(validity ? source.defined : source.value,
                        source.resultIndices, sourceIndices);
}

struct MixedSliceCoordinates {
  SmallVector<TensorSemanticExpr, 4> offsets;
  SmallVector<TensorSemanticExpr, 4> sizes;
  SmallVector<TensorSemanticExpr, 4> strides;
};

static std::string printSemanticExpr(const TensorSemanticExpr &expression) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  expression.print(stream);
  return storage;
}

static FailureOr<MixedSliceCoordinates> getMixedSliceCoordinates(
    ArrayRef<FormulaPtr> attributeFormulas, unsigned fixedOperands,
    EvaluationContext &context) {
  if (attributeFormulas.size() != 4 ||
      llvm::any_of(attributeFormulas, [](const FormulaPtr &argument) {
        return argument->kind != Formula::Kind::Identifier;
      }) ||
      context.operands.size() < fixedOperands ||
      context.result.resultIndices.empty())
    return failure();
  std::optional<SmallVector<int64_t, 4>> segments =
      getIntegerArrayAttribute(
          context.node.attributes.get(attributeFormulas[0]->spelling));
  std::optional<SmallVector<int64_t, 4>> offsets =
      getIntegerArrayAttribute(
          context.node.attributes.get(attributeFormulas[1]->spelling));
  std::optional<SmallVector<int64_t, 4>> sizes =
      getIntegerArrayAttribute(
          context.node.attributes.get(attributeFormulas[2]->spelling));
  std::optional<SmallVector<int64_t, 4>> strides =
      getIntegerArrayAttribute(
          context.node.attributes.get(attributeFormulas[3]->spelling));
  const unsigned expectedSegments = fixedOperands == 2 ? 5 : 4;
  if (!segments || !offsets || !sizes || !strides ||
      segments->size() != expectedSegments ||
      offsets->size() != context.result.resultIndices.size() ||
      sizes->size() != offsets->size() ||
      strides->size() != offsets->size() || (*segments)[0] != 1 ||
      (fixedOperands == 2 && (*segments)[1] != 1) ||
      llvm::any_of(*segments, [](int64_t value) { return value < 0; }))
    return failure();
  const unsigned segmentBase = fixedOperands;
  if ((*segments)[segmentBase] !=
          static_cast<int64_t>(countDynamicEntries(*offsets)) ||
      (*segments)[segmentBase + 1] !=
          static_cast<int64_t>(countDynamicEntries(*sizes)) ||
      (*segments)[segmentBase + 2] !=
          static_cast<int64_t>(countDynamicEntries(*strides)))
    return failure();
  uint64_t requiredOperands = 0;
  for (int64_t segment : *segments)
    requiredOperands += static_cast<uint64_t>(segment);
  if (requiredOperands != context.operands.size())
    return failure();

  unsigned dynamicOperand = fixedOperands;
  Type indexType = context.result.resultIndices.front().getType();
  auto materialize = [&](ArrayRef<int64_t> values,
                         SmallVectorImpl<TensorSemanticExpr> &result) {
    for (int64_t value : values) {
      if (!isDynamicSliceEntry(value)) {
        result.push_back(TensorSemanticExpr::integer(
            indexType.getContext(), value, indexType));
        continue;
      }
      if (dynamicOperand >= context.operands.size() ||
          !context.operands[dynamicOperand].value)
        return false;
      TensorSemanticExpr dynamicValue =
          context.operands[dynamicOperand++].value;
      if (dynamicValue.getType() != indexType) {
        if (!isa<IndexType>(dynamicValue.getType()))
          return false;
        Builder builder(indexType.getContext());
        DictionaryAttr identity = builder.getDictionaryAttr(
            {builder.getNamedAttr("semantic_role",
                                  builder.getStringAttr("index_coordinate"))});
        dynamicValue = TensorSemanticExpr::operation(
            TensorSemanticKind::ApplyUnary, indexType, {dynamicValue},
            identity);
      }
      result.push_back(dynamicValue);
    }
    return true;
  };
  MixedSliceCoordinates coordinates;
  if (!materialize(*offsets, coordinates.offsets) ||
      !materialize(*sizes, coordinates.sizes) ||
      !materialize(*strides, coordinates.strides) ||
      dynamicOperand != context.operands.size())
    return failure();
  return coordinates;
}

static FailureOr<TensorSemanticExpr>
getShapedExtent(const TensorDenotation &denotation, unsigned dimension,
                Type indexType) {
  if (dimension >= denotation.shape.size())
    return failure();
  if (!ShapedType::isDynamic(denotation.shape[dimension]))
    return TensorSemanticExpr::integer(indexType.getContext(),
                                       denotation.shape[dimension],
                                       indexType);
  if (denotation.shapeSymbols.size() != denotation.shape.size() ||
      denotation.shapeSymbols[dimension].empty() ||
      denotation.shapeSymbols[dimension] == "?" ||
      denotation.shapeSymbols[dimension] == "!conflict")
    return failure();
  return TensorSemanticExpr::symbol(
      indexType.getContext(),
      (llvm::Twine("extent-") + denotation.shapeSymbols[dimension]).str(),
      indexType);
}

static TensorSemanticExpr conjunction(TensorSemanticExpr lhs,
                                      TensorSemanticExpr rhs) {
  return TensorSemanticExpr::operation(
      TensorSemanticKind::And, lhs.getType(), {lhs, rhs});
}

static FailureOr<TensorSemanticExpr>
evaluateMixedSlice(const Formula &formula, bool validity,
                   EvaluationContext &context) {
  if (formula.arguments.size() != 5)
    return failure();
  std::optional<unsigned> sourceOperand =
      findOperand(context, *formula.arguments[0]);
  if (!sourceOperand || *sourceOperand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*sourceOperand];
  FailureOr<MixedSliceCoordinates> coordinates = getMixedSliceCoordinates(
      ArrayRef<FormulaPtr>(formula.arguments).drop_front(),
      /*fixedOperands=*/1, context);
  if (failed(coordinates) ||
      source.resultIndices.size() != coordinates->offsets.size())
    return failure();

  Type indexType = context.result.resultIndices.front().getType();
  Type booleanType = IntegerType::get(indexType.getContext(), 1);
  TensorSemanticExpr zero =
      TensorSemanticExpr::integer(indexType.getContext(), 0, indexType);
  TensorSemanticExpr parameterValid =
      TensorSemanticExpr::boolean(indexType.getContext(), true);
  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  for (unsigned dimension = 0; dimension < coordinates->offsets.size();
       ++dimension) {
    TensorSemanticExpr coordinate = TensorSemanticExpr::operation(
        TensorSemanticKind::Multiply, indexType,
        {context.result.resultIndices[dimension],
         coordinates->strides[dimension]});
    coordinate = TensorSemanticExpr::operation(
        TensorSemanticKind::Add, indexType,
        {coordinate, coordinates->offsets[dimension]});
    sourceIndices.push_back(coordinate);

    TensorSemanticExpr offsetNonNegative = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType,
        {TensorSemanticExpr::operation(
            TensorSemanticKind::SignedLessThan, booleanType,
            {coordinates->offsets[dimension], zero})});
    TensorSemanticExpr sizePositive = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType,
        {zero, coordinates->sizes[dimension]});
    TensorSemanticExpr stridePositive = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType,
        {zero, coordinates->strides[dimension]});
    FailureOr<TensorSemanticExpr> sourceExtent =
        getShapedExtent(source, dimension, indexType);
    if (failed(sourceExtent))
      return failure();
    TensorSemanticExpr coordinateInBounds = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType,
        {coordinate, *sourceExtent});
    parameterValid = conjunction(
        parameterValid,
        conjunction(offsetNonNegative,
                    conjunction(sizePositive,
                                conjunction(stridePositive,
                                            coordinateInBounds))));
    if (ShapedType::isDynamic(context.result.shape[dimension]) &&
        context.result.shapeSymbols.size() == context.result.shape.size())
      context.result.shapeSymbols[dimension] =
          "slice-size(" + printSemanticExpr(coordinates->sizes[dimension]) +
          ")";
  }
  TensorSemanticExpr selected = substituteMany(
      validity ? source.defined : source.value, source.resultIndices,
      sourceIndices);
  return validity ? conjunction(parameterValid, selected) : selected;
}

static FailureOr<TensorSemanticExpr>
evaluateMixedInsertSlice(const Formula &formula, bool validity,
                         EvaluationContext &context) {
  if (formula.arguments.size() != 6)
    return failure();
  std::optional<unsigned> sourceOperand =
      findOperand(context, *formula.arguments[0]);
  std::optional<unsigned> destinationOperand =
      findOperand(context, *formula.arguments[1]);
  if (!sourceOperand || !destinationOperand ||
      *sourceOperand >= context.operands.size() ||
      *destinationOperand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*sourceOperand];
  const TensorDenotation &destination = context.operands[*destinationOperand];
  FailureOr<MixedSliceCoordinates> coordinates = getMixedSliceCoordinates(
      ArrayRef<FormulaPtr>(formula.arguments).drop_front(2),
      /*fixedOperands=*/2, context);
  if (failed(coordinates) ||
      source.resultIndices.size() != coordinates->offsets.size() ||
      destination.resultIndices.size() != coordinates->offsets.size())
    return failure();

  Type indexType = context.result.resultIndices.front().getType();
  Type booleanType = IntegerType::get(indexType.getContext(), 1);
  TensorSemanticExpr zero =
      TensorSemanticExpr::integer(indexType.getContext(), 0, indexType);
  TensorSemanticExpr inSlice =
      TensorSemanticExpr::boolean(indexType.getContext(), true);
  TensorSemanticExpr parameterValid = inSlice;
  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  for (unsigned dimension = 0; dimension < coordinates->offsets.size();
       ++dimension) {
    TensorSemanticExpr index = context.result.resultIndices[dimension];
    TensorSemanticExpr lower = coordinates->offsets[dimension];
    TensorSemanticExpr upper = TensorSemanticExpr::operation(
        TensorSemanticKind::Add, indexType,
        {lower, coordinates->sizes[dimension]});
    TensorSemanticExpr belowLower = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType, {index, lower});
    TensorSemanticExpr atOrAboveLower = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType, {belowLower});
    TensorSemanticExpr belowUpper = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType, {index, upper});
    inSlice = conjunction(
        inSlice, conjunction(atOrAboveLower, belowUpper));
    sourceIndices.push_back(TensorSemanticExpr::operation(
        TensorSemanticKind::Subtract, indexType, {index, lower}));

    FailureOr<TensorSemanticExpr> destinationExtent =
        getShapedExtent(destination, dimension, indexType);
    FailureOr<TensorSemanticExpr> sourceExtent =
        getShapedExtent(source, dimension, indexType);
    if (failed(destinationExtent) || failed(sourceExtent))
      return failure();
    TensorSemanticExpr offsetNonNegative = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType,
        {TensorSemanticExpr::operation(
            TensorSemanticKind::SignedLessThan, booleanType, {lower, zero})});
    TensorSemanticExpr sizePositive = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType,
        {zero, coordinates->sizes[dimension]});
    TensorSemanticExpr upperInBounds = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType,
        {TensorSemanticExpr::operation(
            TensorSemanticKind::SignedLessThan, booleanType,
            {*destinationExtent, upper})});
    TensorSemanticExpr sourceSizeMatches = TensorSemanticExpr::operation(
        TensorSemanticKind::Equal, booleanType,
        {*sourceExtent, coordinates->sizes[dimension]});
    parameterValid = conjunction(
        parameterValid,
        conjunction(offsetNonNegative,
                    conjunction(sizePositive,
                                conjunction(upperInBounds,
                                            sourceSizeMatches))));
  }
  TensorSemanticExpr sourceValue = substituteMany(
      validity ? source.defined : source.value, source.resultIndices,
      sourceIndices);
  TensorSemanticExpr destinationValue = substituteMany(
      validity ? destination.defined : destination.value,
      destination.resultIndices, context.result.resultIndices);
  TensorSemanticExpr selected = TensorSemanticExpr::operation(
      TensorSemanticKind::Select,
      validity ? booleanType : context.resultElementType,
      {inSlice, sourceValue, destinationValue});
  return validity ? conjunction(parameterValid, selected) : selected;
}

static FailureOr<TensorSemanticExpr>
evaluateStaticInsertSlice(const Formula &formula, bool validity,
                          EvaluationContext &context) {
  if (formula.arguments.size() != 5 ||
      formula.arguments[2]->kind != Formula::Kind::Identifier ||
      formula.arguments[3]->kind != Formula::Kind::Identifier ||
      formula.arguments[4]->kind != Formula::Kind::Identifier)
    return failure();
  std::optional<unsigned> sourceOperand =
      findOperand(context, *formula.arguments[0]);
  std::optional<unsigned> destinationOperand =
      findOperand(context, *formula.arguments[1]);
  if (!sourceOperand || !destinationOperand ||
      *sourceOperand >= context.operands.size() ||
      *destinationOperand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*sourceOperand];
  const TensorDenotation &destination = context.operands[*destinationOperand];
  std::optional<SmallVector<int64_t, 4>> offsets =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[2]->spelling));
  std::optional<SmallVector<int64_t, 4>> sizes =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[3]->spelling));
  std::optional<SmallVector<int64_t, 4>> strides =
      getIntegerArrayAttribute(
          context.node.attributes.get(formula.arguments[4]->spelling));
  if (!offsets || !sizes || !strides ||
      offsets->size() != context.result.resultIndices.size() ||
      source.resultIndices.size() != offsets->size() ||
      destination.resultIndices.size() != offsets->size())
    return failure();

  Type indexType = context.result.resultIndices.front().getType();
  Type booleanType = IntegerType::get(indexType.getContext(), 1);
  TensorSemanticExpr inSlice =
      TensorSemanticExpr::boolean(indexType.getContext(), true);
  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  for (unsigned dimension = 0; dimension < offsets->size(); ++dimension) {
    if ((*strides)[dimension] != 1)
      return failure();
    TensorSemanticExpr index = context.result.resultIndices[dimension];
    TensorSemanticExpr lower = TensorSemanticExpr::integer(
        indexType.getContext(), (*offsets)[dimension], indexType);
    TensorSemanticExpr upper = TensorSemanticExpr::integer(
        indexType.getContext(),
        (*offsets)[dimension] + (*sizes)[dimension], indexType);
    TensorSemanticExpr belowLower = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType, {index, lower});
    TensorSemanticExpr atOrAboveLower = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType, {belowLower});
    TensorSemanticExpr belowUpper = TensorSemanticExpr::operation(
        TensorSemanticKind::SignedLessThan, booleanType, {index, upper});
    TensorSemanticExpr insideDimension = TensorSemanticExpr::operation(
        TensorSemanticKind::And, booleanType,
        {atOrAboveLower, belowUpper});
    inSlice = TensorSemanticExpr::operation(TensorSemanticKind::And,
                                            booleanType,
                                            {inSlice, insideDimension});
    sourceIndices.push_back(TensorSemanticExpr::operation(
        TensorSemanticKind::Subtract, indexType, {index, lower}));
  }
  TensorSemanticExpr sourceValue = substituteMany(
      validity ? source.defined : source.value, source.resultIndices,
      sourceIndices);
  TensorSemanticExpr destinationValue = substituteMany(
      validity ? destination.defined : destination.value,
      destination.resultIndices, context.result.resultIndices);
  return TensorSemanticExpr::operation(
      TensorSemanticKind::Select,
      validity ? booleanType : context.resultElementType,
      {inSlice, sourceValue, destinationValue});
}

static DictionaryAttr getSemanticOperationIdentity(
    const GraphExprNode &node, StringRef semanticRole = {}) {
  Builder builder(node.operation.getContext());
  SmallVector<NamedAttribute, 8> identity;
  identity.push_back(builder.getNamedAttr("operation", node.operation));
  if (node.attributes)
    for (NamedAttribute attribute : node.attributes)
      if (!attribute.getName().getValue().starts_with("ags.") ||
          attribute.getName().getValue() == kResultIndexAttr ||
          attribute.getName().getValue() == kCanonicalRegionsAttr)
        identity.push_back(attribute);
  if (!semanticRole.empty())
    identity.push_back(builder.getNamedAttr(
        "semantic_role", builder.getStringAttr(semanticRole)));
  return builder.getDictionaryAttr(identity);
}

// A region descriptor is part of the proof meaning of a quantified buffer
// update/read.  The coordinate expression alone is insufficient: two
// subviews can have the same base and offset but different extents.  Keeping
// this metadata in a generic descriptor lets BufferFill remain independent of
// the operation that requested it.
static DictionaryAttr
getBufferRegionDescriptor(const TensorDenotation &denotation) {
  MLIRContext *context = denotation.value.getType().getContext();
  Builder builder(context);
  SmallVector<Attribute, 4> shapeSymbols;
  shapeSymbols.reserve(denotation.shapeSymbols.size());
  for (const std::string &symbol : denotation.shapeSymbols)
    shapeSymbols.push_back(builder.getStringAttr(symbol));
  return builder.getDictionaryAttr(
      {builder.getNamedAttr("shape",
                            builder.getDenseI64ArrayAttr(denotation.shape)),
       builder.getNamedAttr("shape_symbols",
                            builder.getArrayAttr(shapeSymbols)),
       builder.getNamedAttr("layout",
                            builder.getStringAttr(denotation.layoutIdentity))});
}

static FailureOr<TensorSemanticExpr>
evaluateAxisFold(const Formula &formula, TensorSemanticKind foldKind, bool scan,
                 EvaluationContext &context) {
  const size_t expectedArguments = scan ? 4 : 3;
  if (formula.arguments.size() != expectedArguments ||
      formula.arguments[0]->kind != Formula::Kind::Identifier ||
      formula.arguments[2]->kind != Formula::Kind::Identifier)
    return failure();
  std::optional<unsigned> operand =
      findOperand(context, *formula.arguments[1]);
  if (!operand || *operand >= context.operands.size())
    return failure();
  const TensorDenotation &source = context.operands[*operand];
  FailureOr<int64_t> axis =
      getNodeIntegerAttribute(*formula.arguments[2], context);
  if (failed(axis) || *axis < 0 ||
      static_cast<size_t>(*axis) >= source.shape.size() ||
      (!ShapedType::isDynamic(source.shape[*axis]) &&
       source.shape[*axis] <= 0))
    return failure();

  StringRef binderName = formula.arguments[0]->spelling;
  if (context.binders.contains(binderName))
    return failure();
  TensorSemanticExpr binder = makeFreshBinder(context);
  context.binders.insert({binderName, binder});

  SmallVector<TensorSemanticExpr, 4> sourceIndices;
  unsigned resultDimension = 0;
  for (unsigned dimension = 0; dimension < source.shape.size(); ++dimension) {
    if (dimension == static_cast<unsigned>(*axis)) {
      sourceIndices.push_back(binder);
      continue;
    }
    if (resultDimension >= context.result.resultIndices.size()) {
      context.binders.erase(binderName);
      return failure();
    }
    sourceIndices.push_back(context.result.resultIndices[resultDimension++]);
  }
  if (scan) {
    // A scan preserves rank, so the scan-axis free index was skipped above.
    sourceIndices[*axis] = context.result.resultIndices[*axis];
    for (unsigned dimension = 0; dimension < source.shape.size(); ++dimension)
      if (dimension != static_cast<unsigned>(*axis))
        sourceIndices[dimension] = context.result.resultIndices[dimension];
    sourceIndices[*axis] = binder;
    mergeSameShapeSymbols(context.result, source);
  } else if (resultDimension != context.result.resultIndices.size()) {
    context.binders.erase(binderName);
    return failure();
  } else if (source.shapeSymbols.size() == source.shape.size() &&
             context.result.shapeSymbols.size() ==
                 context.result.shape.size()) {
    unsigned targetDimension = 0;
    for (unsigned sourceDimension = 0; sourceDimension < source.shape.size();
         ++sourceDimension) {
      if (sourceDimension == static_cast<unsigned>(*axis))
        continue;
      if (ShapedType::isDynamic(context.result.shape[targetDimension]))
        context.result.shapeSymbols[targetDimension] =
            source.shapeSymbols[sourceDimension];
      ++targetDimension;
    }
  }

  const bool forall = foldKind == TensorSemanticKind::ForAll;
  TensorSemanticExpr body = substituteMany(
      forall ? source.defined : source.value, source.resultIndices,
      sourceIndices);
  context.binders.erase(binderName);

  Type indexType = binder.getType();
  TensorSemanticExpr zero = TensorSemanticExpr::integer(
      indexType.getContext(), 0, indexType);
  TensorSemanticExpr extent;
  if (ShapedType::isDynamic(source.shape[*axis])) {
    if (source.shapeSymbols.size() != source.shape.size() ||
        source.shapeSymbols[*axis].empty() ||
        source.shapeSymbols[*axis] == "?" ||
        source.shapeSymbols[*axis] == "!conflict")
      return failure();
    extent = TensorSemanticExpr::symbol(
        indexType.getContext(),
        (llvm::Twine("extent-") + source.shapeSymbols[*axis]).str(),
        indexType);
  } else {
    extent = TensorSemanticExpr::integer(
        indexType.getContext(), source.shape[*axis], indexType);
  }
  TensorSemanticExpr lower = zero;
  TensorSemanticExpr upper = extent;
  if (scan) {
    if (formula.arguments[3]->kind != Formula::Kind::Identifier)
      return failure();
    FailureOr<bool> reverse =
        getNodeBooleanAttribute(*formula.arguments[3], context);
    if (failed(reverse))
      return failure();
    TensorSemanticExpr position = context.result.resultIndices[*axis];
    if (*reverse) {
      lower = position;
    } else {
      TensorSemanticExpr one = TensorSemanticExpr::integer(
          indexType.getContext(), 1, indexType);
      upper = TensorSemanticExpr::operation(TensorSemanticKind::Add,
                                            indexType, {position, one});
    }
  }
  return TensorSemanticExpr::operation(
      foldKind,
      forall ? IntegerType::get(indexType.getContext(), 1)
             : context.resultElementType,
      {binder, lower, upper, body});
}

static FailureOr<TensorSemanticExpr>
evaluateFold(const Formula &formula, TensorSemanticKind kind, Type resultType,
             Type operandType, EvaluationContext &context) {
  if (formula.arguments.empty())
    return failure();
  FailureOr<TensorSemanticExpr> result =
      evaluate(*formula.arguments.front(), operandType, context);
  if (failed(result))
    return failure();
  for (const FormulaPtr &argument : ArrayRef(formula.arguments).drop_front()) {
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*argument, operandType, context);
    if (failed(rhs))
      return failure();
    *result = TensorSemanticExpr::operation(kind, resultType, {*result, *rhs});
  }
  return result;
}

static FailureOr<TensorSemanticExpr>
evaluateBinder(const Formula &formula, bool forall,
               EvaluationContext &context) {
  if (formula.arguments.size() != 3 ||
      formula.arguments[0]->kind != Formula::Kind::Identifier)
    return failure();
  std::optional<unsigned> rangeOperand =
      findOperand(context, *formula.arguments[1]);
  if (!rangeOperand || *rangeOperand >= context.operands.size())
    return failure();
  const TensorDenotation &range = context.operands[*rangeOperand];
  if (range.shape.size() != 1 || ShapedType::isDynamic(range.shape.front()) ||
      range.shape.front() <= 0)
    return failure();

  StringRef binderName = formula.arguments[0]->spelling;
  if (context.binders.contains(binderName))
    return failure();
  // Pick an ID outside every free and nested bound index already present in
  // the operand denotations. This keeps nested reductions alpha-disjoint.
  TensorSemanticExpr index = makeFreshBinder(context);
  context.binders.insert({binderName, index});
  Type bodyType = forall ? IntegerType::get(index.getType().getContext(), 1)
                         : context.resultElementType;
  FailureOr<TensorSemanticExpr> body =
      evaluate(*formula.arguments[2], bodyType, context);
  context.binders.erase(binderName);
  if (failed(body))
    return failure();

  TensorSemanticKind kind = forall ? TensorSemanticKind::ForAll
                                   : TensorSemanticKind::Sum;
  Type resultType = forall ? IntegerType::get(index.getType().getContext(), 1)
                           : context.resultElementType;
  TensorSemanticExpr lower = TensorSemanticExpr::integer(
      index.getType().getContext(), 0, index.getType());
  TensorSemanticExpr upper = TensorSemanticExpr::integer(
      index.getType().getContext(), range.shape.front(), index.getType());
  return TensorSemanticExpr::operation(kind, resultType,
                                       {index, lower, upper, *body});
}

static FailureOr<TensorSemanticExpr>
evaluate(const Formula &formula, Type expectedType,
         EvaluationContext &context) {
  MLIRContext *mlirContext = context.resultElementType.getContext();
  Type booleanType = IntegerType::get(mlirContext, 1);

  switch (formula.kind) {
  case Formula::Kind::Integer:
    if (!expectedType)
      return failure();
    return TensorSemanticExpr::integer(mlirContext, formula.integer,
                                       expectedType);
  case Formula::Kind::Boolean:
    return TensorSemanticExpr::boolean(mlirContext, formula.boolean);
  case Formula::Kind::Identifier: {
    auto binder = context.binders.find(formula.spelling);
    if (binder == context.binders.end())
      return failure();
    return binder->second;
  }
  case Formula::Kind::Call:
    break;
  }

  StringRef call = formula.spelling;
  ArrayRef<FormulaPtr> arguments = formula.arguments;
  if (call == "value" && arguments.size() == 1)
    return evaluateOperand(*arguments[0], /*validity=*/false, context);
  if (call == "valid" && arguments.size() == 1)
    return evaluateOperand(*arguments[0], /*validity=*/true, context);
  if (call == "region_value" && arguments.empty())
    return evaluateImportedPureRegion(context, kCanonicalRegionsAttr,
                                      /*validity=*/false);
  if (call == "region_valid" && arguments.empty())
    return evaluateImportedPureRegion(context, kCanonicalRegionsAttr,
                                      /*validity=*/true);
  if (call == "call_summary_value" && arguments.empty())
    return evaluateImportedPureRegion(context, kCallSummaryRegionsAttr,
                                      /*validity=*/false);
  if (call == "call_summary_valid" && arguments.empty())
    return evaluateImportedPureRegion(context, kCallSummaryRegionsAttr,
                                      /*validity=*/true);
  if (call == "at" && arguments.size() >= 2)
    return evaluateAt(*arguments[0], arguments.drop_front(),
                      /*validity=*/false, context);
  if (call == "valid_at" && arguments.size() >= 2)
    return evaluateAt(*arguments[0], arguments.drop_front(),
                      /*validity=*/true, context);
  if ((call == "expand_dims_value" || call == "expand_dims_valid") &&
      arguments.size() == 2)
    return evaluateReshape(*arguments[0], arguments[1].get(),
                           /*broadcast=*/false,
                           /*validity=*/call == "expand_dims_valid", context);
  if ((call == "broadcast_value" || call == "broadcast_valid") &&
      arguments.size() == 1)
    return evaluateReshape(*arguments[0], /*axisFormula=*/nullptr,
                           /*broadcast=*/true,
                           /*validity=*/call == "broadcast_valid", context);
  if ((call == "transpose_value" || call == "transpose_valid") &&
      arguments.size() == 2)
    return evaluateTranspose(*arguments[0], *arguments[1],
                             /*validity=*/call == "transpose_valid", context);
  if ((call == "reshape_value" || call == "reshape_valid") &&
      arguments.size() == 1)
    return evaluateStaticReshape(*arguments[0],
                                 /*validity=*/call == "reshape_valid",
                                 context);
  if ((call == "singleton_value" || call == "singleton_valid") &&
      arguments.size() == 1)
    return evaluateSingletonTensor(
        *arguments[0], /*validity=*/call == "singleton_valid", context);
  if ((call == "join_value" || call == "join_valid") &&
      arguments.size() == 2)
    return evaluateJoin(*arguments[0], *arguments[1],
                        /*validity=*/call == "join_valid", context);
  if ((call == "split_value" || call == "split_valid") &&
      arguments.size() == 2)
    return evaluateSplit(*arguments[0], *arguments[1],
                         /*validity=*/call == "split_valid", context);
  if ((call == "cat_value" || call == "cat_valid") &&
      arguments.size() == 2)
    return evaluateCat(*arguments[0], *arguments[1],
                       /*validity=*/call == "cat_valid", context);
  if ((call == "gather_value" || call == "gather_valid") &&
      arguments.size() == 3)
    return evaluateGather(*arguments[0], *arguments[1], *arguments[2],
                          /*validity=*/call == "gather_valid", context);
  if ((call == "dot_value" || call == "dot_valid") &&
      arguments.size() == 3)
    return evaluateDot(*arguments[0], *arguments[1], *arguments[2],
                       /*validity=*/call == "dot_valid", context);
  if (call == "dot_scaled_value")
    return evaluateDotScaled(formula, /*validity=*/false, context);
  if (call == "dot_scaled_valid")
    return evaluateDotScaled(formula, /*validity=*/true, context);
  if (call == "slice_value")
    return evaluateStaticSlice(formula, /*validity=*/false, context);
  if (call == "slice_valid")
    return evaluateStaticSlice(formula, /*validity=*/true, context);
  if (call == "mixed_slice_value")
    return evaluateMixedSlice(formula, /*validity=*/false, context);
  if (call == "mixed_slice_valid")
    return evaluateMixedSlice(formula, /*validity=*/true, context);
  if (call == "insert_slice_value")
    return evaluateStaticInsertSlice(formula, /*validity=*/false, context);
  if (call == "insert_slice_valid")
    return evaluateStaticInsertSlice(formula, /*validity=*/true, context);
  if (call == "mixed_insert_slice_value")
    return evaluateMixedInsertSlice(formula, /*validity=*/false, context);
  if (call == "mixed_insert_slice_valid")
    return evaluateMixedInsertSlice(formula, /*validity=*/true, context);
  if (call == "reduce_sum")
    return evaluateAxisFold(formula, TensorSemanticKind::Sum,
                            /*scan=*/false, context);
  if (call == "reduce_product")
    return evaluateAxisFold(formula, TensorSemanticKind::Product,
                            /*scan=*/false, context);
  if (call == "reduce_all")
    return evaluateAxisFold(formula, TensorSemanticKind::ForAll,
                            /*scan=*/false, context);
  if (call == "scan_sum")
    return evaluateAxisFold(formula, TensorSemanticKind::Sum,
                            /*scan=*/true, context);
  if (call == "scan_product")
    return evaluateAxisFold(formula, TensorSemanticKind::Product,
                            /*scan=*/true, context);
  if (call == "scan_all")
    return evaluateAxisFold(formula, TensorSemanticKind::ForAll,
                            /*scan=*/true, context);
  auto getIndexVector = [&](const Formula &descriptor)
      -> FailureOr<SmallVector<TensorSemanticExpr, 4>> {
    if (descriptor.kind == Formula::Kind::Call &&
        descriptor.spelling == "result_indices" &&
        descriptor.arguments.empty())
      return SmallVector<TensorSemanticExpr, 4>(
          context.result.resultIndices.begin(),
          context.result.resultIndices.end());
    if (descriptor.kind == Formula::Kind::Call &&
        descriptor.spelling == "operand_indices" &&
        descriptor.arguments.size() == 1) {
      std::optional<unsigned> operand =
          findOperand(context, *descriptor.arguments.front());
      if (!operand || *operand >= context.operands.size())
        return failure();
      return SmallVector<TensorSemanticExpr, 4>(
          context.operands[*operand].resultIndices.begin(),
          context.operands[*operand].resultIndices.end());
    }
    return failure();
  };
  auto signedCast = [&](TensorSemanticExpr source,
                        Type targetType) -> FailureOr<TensorSemanticExpr> {
    auto sourceType = dyn_cast<IntegerType>(source.getType());
    auto targetInteger = dyn_cast<IntegerType>(targetType);
    if (!sourceType || !targetInteger)
      return failure();
    if (sourceType == targetInteger)
      return source;
    TensorSemanticKind kind =
        sourceType.getWidth() < targetInteger.getWidth()
            ? TensorSemanticKind::SignExtend
            : TensorSemanticKind::Truncate;
    return TensorSemanticExpr::operation(kind, targetInteger, {source});
  };
  if (call == "linear_address" && arguments.size() >= 4) {
    FailureOr<TensorSemanticExpr> pointer =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> offset =
        evaluate(*arguments[1], {}, context);
    FailureOr<SmallVector<TensorSemanticExpr, 4>> indices =
        getIndexVector(*arguments[2]);
    if (failed(pointer) || failed(offset) || failed(indices) ||
        !isa<IntegerType>(offset->getType()) ||
        indices->size() != arguments.size() - 3)
      return failure();
    TensorSemanticExpr linear = *offset;
    for (auto [index, strideFormula] :
         llvm::zip_equal(*indices, arguments.drop_front(3))) {
      FailureOr<TensorSemanticExpr> coordinate =
          signedCast(index, offset->getType());
      FailureOr<TensorSemanticExpr> stride =
          evaluate(*strideFormula, offset->getType(), context);
      if (failed(coordinate) || failed(stride) ||
          stride->getType() != offset->getType())
        return failure();
      TensorSemanticExpr scaled = TensorSemanticExpr::operation(
          TensorSemanticKind::Multiply, offset->getType(),
          {*coordinate, *stride});
      linear = TensorSemanticExpr::operation(
          TensorSemanticKind::Add, offset->getType(), {linear, scaled});
    }
    return TensorSemanticExpr::operation(TensorSemanticKind::PointerAdd,
                                         pointer->getType(),
                                         {*pointer, linear});
  }
  if (call == "rectangular_in_bounds" && arguments.size() >= 2) {
    FailureOr<SmallVector<TensorSemanticExpr, 4>> indices =
        getIndexVector(*arguments[0]);
    if (failed(indices) || indices->size() != arguments.size() - 1)
      return failure();
    TensorSemanticExpr inBounds =
        TensorSemanticExpr::boolean(mlirContext, true);
    for (auto [index, numelFormula] :
         llvm::zip_equal(*indices, arguments.drop_front())) {
      FailureOr<TensorSemanticExpr> numel =
          evaluate(*numelFormula, {}, context);
      if (failed(numel) || !isa<IntegerType>(numel->getType()))
        return failure();
      FailureOr<TensorSemanticExpr> coordinate =
          signedCast(index, numel->getType());
      if (failed(coordinate))
        return failure();
      TensorSemanticExpr upper = TensorSemanticExpr::operation(
          TensorSemanticKind::SignedLessThan, booleanType,
          {*coordinate, *numel});
      inBounds = TensorSemanticExpr::operation(
          TensorSemanticKind::And, booleanType, {inBounds, upper});
    }
    return inBounds;
  }
  if (call == "result_index" && arguments.size() == 1) {
    std::optional<int64_t> dimension = getIntegerLiteral(*arguments[0]);
    if (!dimension || *dimension < 0 ||
        static_cast<size_t>(*dimension) >= context.result.resultIndices.size())
      return failure();
    return context.result.resultIndices[*dimension];
  }
  if (call == "operand_index" && arguments.size() == 2) {
    std::optional<unsigned> operand = findOperand(context, *arguments[0]);
    std::optional<int64_t> dimension = getIntegerLiteral(*arguments[1]);
    if (!operand || *operand >= context.operands.size() || !dimension ||
        *dimension < 0 ||
        static_cast<size_t>(*dimension) >=
            context.operands[*operand].resultIndices.size())
      return failure();
    return context.operands[*operand].resultIndices[*dimension];
  }
  if (call == "signed_cast_like" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> source =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> exemplar =
        evaluate(*arguments[1], {}, context);
    if (failed(source) || failed(exemplar))
      return failure();
    auto sourceType = dyn_cast<IntegerType>(source->getType());
    auto targetType = dyn_cast<IntegerType>(exemplar->getType());
    if (!sourceType || !targetType)
      return failure();
    if (sourceType == targetType)
      return *source;
    return signedCast(*source, targetType);
  }
  if ((call == "signed_integer_cast" || call == "unsigned_integer_cast") &&
      arguments.size() == 1 && expectedType) {
    FailureOr<TensorSemanticExpr> source =
        evaluate(*arguments[0], {}, context);
    if (failed(source))
      return failure();
    auto bitWidth = [](Type type) -> std::optional<unsigned> {
      if (auto integer = dyn_cast<IntegerType>(type))
        return integer.getWidth();
      if (isa<IndexType>(type))
        return 64;
      return std::nullopt;
    };
    std::optional<unsigned> sourceWidth = bitWidth(source->getType());
    std::optional<unsigned> targetWidth = bitWidth(expectedType);
    if (!sourceWidth || !targetWidth)
      return failure();
    if (source->getType() == expectedType)
      return *source;
    if (*sourceWidth > *targetWidth)
      return TensorSemanticExpr::operation(TensorSemanticKind::Truncate,
                                           expectedType, {*source});
    TensorSemanticKind kind = call == "signed_integer_cast"
                                  ? TensorSemanticKind::SignExtend
                                  : TensorSemanticKind::ZeroExtend;
    return TensorSemanticExpr::operation(kind, expectedType, {*source});
  }
  if (call == "input_value" && arguments.empty()) {
    auto symbol = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(kSymbolIdAttr));
    if (!symbol)
      return failure();
    Attribute payload = IntegerAttr::get(IntegerType::get(mlirContext, 64),
                                         symbol.getValue());
    if (isa<ShapedType>(getResultType(context.node)))
      return TensorSemanticExpr::operation(
          TensorSemanticKind::Element, context.resultElementType,
          context.result.resultIndices, payload);
    return TensorSemanticExpr::operation(TensorSemanticKind::Symbol,
                                         context.resultElementType, {},
                                         payload);
  }
  if (call == "private_buffer_value" && arguments.empty()) {
    auto symbol = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(kSymbolIdAttr));
    if (!symbol)
      return failure();
    Builder builder(mlirContext);
    Attribute payload = builder.getDictionaryAttr(
        {builder.getNamedAttr("private_buffer", builder.getBoolAttr(true)),
         builder.getNamedAttr("symbol_id", symbol)});
    if (isa<ShapedType>(getResultType(context.node)))
      return TensorSemanticExpr::operation(
          TensorSemanticKind::Element, context.resultElementType,
          context.result.resultIndices, payload);
    return TensorSemanticExpr::operation(TensorSemanticKind::Symbol,
                                         context.resultElementType, {},
                                         payload);
  }
  if (call == "input_valid" && arguments.empty()) {
    auto symbol = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(kSymbolIdAttr));
    if (!symbol)
      return failure();
    Attribute payload = IntegerAttr::get(IntegerType::get(mlirContext, 64),
                                         symbol.getValue());
    return TensorSemanticExpr::operation(TensorSemanticKind::Defined,
                                         booleanType,
                                         context.result.resultIndices,
                                         payload);
  }
  if (call == "shift_amount_valid" && arguments.size() == 1) {
    FailureOr<TensorSemanticExpr> amount =
        evaluate(*arguments[0], {}, context);
    if (failed(amount))
      return failure();
    Builder builder(mlirContext);
    DictionaryAttr identity = builder.getDictionaryAttr({builder.getNamedAttr(
        "semantic_role", builder.getStringAttr("shift_amount_valid"))});
    return TensorSemanticExpr::operation(TensorSemanticKind::ApplyUnary,
                                         booleanType, {*amount}, identity);
  }
  if (call == "nonzero" && arguments.size() == 1) {
    FailureOr<TensorSemanticExpr> value =
        evaluate(*arguments[0], {}, context);
    if (failed(value) || !isa<IntegerType>(value->getType()))
      return failure();
    TensorSemanticExpr zero =
        TensorSemanticExpr::integer(mlirContext, 0, value->getType());
    TensorSemanticExpr equal = TensorSemanticExpr::operation(
        TensorSemanticKind::Equal, booleanType, {*value, zero});
    return TensorSemanticExpr::operation(TensorSemanticKind::Not, booleanType,
                                         {equal});
  }
  if (call == "signed_division_valid" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], {}, context);
    if (failed(lhs) || failed(rhs))
      return failure();
    auto integerType = dyn_cast<IntegerType>(lhs->getType());
    if (!integerType || rhs->getType() != lhs->getType())
      return failure();
    TensorSemanticExpr zero =
        TensorSemanticExpr::integer(mlirContext, 0, lhs->getType());
    TensorSemanticExpr minusOne =
        TensorSemanticExpr::integer(mlirContext, -1, lhs->getType());
    TensorSemanticExpr minimum = TensorSemanticExpr::integer(
        mlirContext, llvm::APInt::getSignedMinValue(integerType.getWidth())
                         .getSExtValue(),
        lhs->getType());
    auto equal = [&](TensorSemanticExpr first, TensorSemanticExpr second) {
      return TensorSemanticExpr::operation(TensorSemanticKind::Equal,
                                           booleanType, {first, second});
    };
    TensorSemanticExpr nonZero = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType, {equal(*rhs, zero)});
    TensorSemanticExpr overflow = TensorSemanticExpr::operation(
        TensorSemanticKind::And, booleanType,
        {equal(*lhs, minimum), equal(*rhs, minusOne)});
    TensorSemanticExpr noOverflow = TensorSemanticExpr::operation(
        TensorSemanticKind::Not, booleanType, {overflow});
    return TensorSemanticExpr::operation(TensorSemanticKind::And, booleanType,
                                         {nonZero, noOverflow});
  }
  if (call == "unsigned_add_overflow" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], {}, context);
    if (failed(lhs) || failed(rhs) || lhs->getType() != rhs->getType() ||
        !isa<IntegerType>(lhs->getType()))
      return failure();
    TensorSemanticExpr sum = TensorSemanticExpr::operation(
        TensorSemanticKind::Add, lhs->getType(), {*lhs, *rhs});
    return TensorSemanticExpr::operation(TensorSemanticKind::UnsignedLessThan,
                                         booleanType, {sum, *lhs});
  }
  if ((call == "signed_multiply_high" ||
       call == "unsigned_multiply_high") &&
      arguments.size() == 2 && expectedType) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], {}, context);
    if (failed(lhs) || failed(rhs))
      return failure();
    auto integerType = dyn_cast<IntegerType>(lhs->getType());
    if (!integerType ||
        lhs->getType() != rhs->getType() || expectedType != lhs->getType())
      return failure();
    TensorSemanticKind kind = call == "signed_multiply_high"
                                  ? TensorSemanticKind::SignedMultiplyHigh
                                  : TensorSemanticKind::UnsignedMultiplyHigh;
    return TensorSemanticExpr::operation(kind, expectedType, {*lhs, *rhs});
  }
  if (call == "apply_unary" && arguments.size() == 1 && expectedType) {
    FailureOr<TensorSemanticExpr> operand =
        evaluate(*arguments[0], {}, context);
    if (failed(operand))
      return failure();
    return TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyUnary, expectedType, {*operand},
        getSemanticOperationIdentity(context.node));
  }
  if (call == "apply_binary" && arguments.size() == 2 && expectedType) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], {}, context);
    if (failed(lhs) || failed(rhs))
      return failure();
    return TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyBinary, expectedType, {*lhs, *rhs},
        getSemanticOperationIdentity(context.node));
  }
  if (call == "apply_ternary" && arguments.size() == 3 && expectedType) {
    SmallVector<TensorSemanticExpr, 3> values;
    for (const FormulaPtr &argument : arguments) {
      FailureOr<TensorSemanticExpr> value = evaluate(*argument, {}, context);
      if (failed(value))
        return failure();
      values.push_back(*value);
    }
    return TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyTernary, expectedType, values,
        getSemanticOperationIdentity(context.node));
  }
  if (call == "apply_nary" && arguments.empty() && expectedType) {
    SmallVector<TensorSemanticExpr, 8> values;
    values.reserve(context.operands.size());
    for (const TensorDenotation &operand : context.operands)
      values.push_back(operand.value);
    return TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyNary, expectedType, values,
        getSemanticOperationIdentity(context.node));
  }
  if (call == "all_operands_valid" && arguments.empty()) {
    SmallVector<TensorSemanticExpr, 8> values;
    values.reserve(context.operands.size());
    for (const TensorDenotation &operand : context.operands)
      values.push_back(operand.defined);
    if (values.empty())
      return TensorSemanticExpr::boolean(mlirContext, true);
    TensorSemanticExpr result = values.front();
    for (const TensorSemanticExpr &value :
         ArrayRef<TensorSemanticExpr>(values).drop_front())
      result = TensorSemanticExpr::operation(TensorSemanticKind::And,
                                             booleanType, {result, value});
    return result;
  }
  if ((call == "operation_key" || call == "operation_key_except") &&
      (arguments.empty() ||
       (call == "operation_key_except" && arguments.size() == 1 &&
        arguments.front()->kind == Formula::Kind::Integer))) {
    std::optional<unsigned> excluded;
    if (!arguments.empty()) {
      if (arguments.front()->integer < 0 ||
          static_cast<uint64_t>(arguments.front()->integer) >=
              context.operands.size())
        return failure();
      excluded = static_cast<unsigned>(arguments.front()->integer);
    }
    SmallVector<TensorSemanticExpr, 12> values;
    for (auto [index, operand] : llvm::enumerate(context.operands))
      if (!excluded || index != *excluded)
        values.push_back(operand.value);
    auto addIndex = [&](const TensorSemanticExpr &index) {
      if (!llvm::any_of(values, [&](const TensorSemanticExpr &known) {
            return known.structurallyEquals(index);
          }))
        values.push_back(index);
    };
    for (const TensorSemanticExpr &index : context.result.resultIndices)
      addIndex(index);
    for (const TensorDenotation &operand : context.operands)
      for (const TensorSemanticExpr &index : operand.resultIndices)
        addIndex(index);
    return TensorSemanticExpr::operation(
        TensorSemanticKind::ApplyNary,
        IntegerType::get(mlirContext, 64), values,
        StringAttr::get(mlirContext, "abstract-memory-access-key"));
  }
  if (call == "program_id" && arguments.size() == 1 && expectedType &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute axis = context.node.attributes.get(arguments[0]->spelling);
    if (!axis)
      return failure();
    if (!context.inputLaunch)
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::ProgramId,
                                         expectedType,
                                         {context.inputLaunch}, axis);
  }
  if (call == "num_programs" && arguments.size() == 1 && expectedType &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute axis = context.node.attributes.get(arguments[0]->spelling);
    if (!axis || !context.inputLaunch)
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::NumPrograms,
                                         expectedType,
                                         {context.inputLaunch}, axis);
  }
  if (call == "launch_bound" && arguments.size() == 2 &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute axis = context.node.attributes.get(arguments[0]->spelling);
    FailureOr<TensorSemanticExpr> bound =
        evaluate(*arguments[1], {}, context);
    if (!axis || failed(bound) || !context.inputLaunch)
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::LaunchBound,
                                         booleanType,
                                         {context.inputLaunch, *bound}, axis);
  }
  if (call == "pointer_add" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> pointer =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> offset =
        evaluate(*arguments[1], {}, context);
    if (failed(pointer) || failed(offset))
      return failure();
    Type pointerType = expectedType ? expectedType : pointer->getType();
    if (!pointerType)
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::PointerAdd,
                                         pointerType,
                                         {*pointer, *offset});
  }
  if ((call == "memory_read" || call == "memory_valid") &&
      arguments.size() == 1) {
    FailureOr<TensorSemanticExpr> pointer =
        evaluate(*arguments[0], {}, context);
    if (failed(pointer))
      return failure();
    Type resultType = call == "memory_valid" ? booleanType : expectedType;
    if (!resultType)
      return failure();
    if (call == "memory_valid") {
      if (!context.inputMemory)
        return failure();
      return TensorSemanticExpr::operation(TensorSemanticKind::MemoryValid,
                                           resultType,
                                           {context.inputMemory, *pointer});
    }
    if (!context.inputMemory)
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::MemoryRead,
                                         resultType,
                                         {context.inputMemory, *pointer});
  }
  if (call == "input_memory" && arguments.empty() && context.inputMemory)
    return context.inputMemory;
  if (call == "partitionable_memory" && arguments.size() == 2 &&
      llvm::all_of(arguments, [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      })) {
    auto axis = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[0]->spelling));
    auto ceilDiv = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[1]->spelling));
    if (!axis || !ceilDiv || axis.getInt() < 0 || axis.getInt() > 2 ||
        (ceilDiv.getInt() != 0 && ceilDiv.getInt() != 1))
      return failure();
    Builder builder(booleanType.getContext());
    DictionaryAttr premise = builder.getDictionaryAttr(
        {builder.getNamedAttr("axis", axis),
         builder.getNamedAttr("ceil_div", ceilDiv)});
    return TensorSemanticExpr::operation(
        TensorSemanticKind::PartitionableMemory, booleanType, {}, premise);
  }
  if (call == "input_memory_valid" && arguments.empty() &&
      context.inputMemoryDefined)
    return context.inputMemoryDefined;
  if (call == "input_effect" && arguments.empty() && context.inputEffect)
    return context.inputEffect;
  if (call == "input_launch" && arguments.empty() && context.inputLaunch)
    return context.inputLaunch;
  if (call == "launch_remap" && arguments.size() == 3 &&
      context.inputLaunch && expectedType &&
      llvm::all_of(arguments, [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      })) {
    auto axis = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[0]->spelling));
    auto factor = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[1]->spelling));
    auto ceilDiv = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[2]->spelling));
    if (!axis || !factor || !ceilDiv || axis.getInt() < 0 ||
        axis.getInt() > 2 || factor.getInt() <= 0 ||
        (ceilDiv.getInt() != 0 && ceilDiv.getInt() != 1))
      return failure();
    Builder builder(expectedType.getContext());
    DictionaryAttr mapping = builder.getDictionaryAttr(
        {builder.getNamedAttr("axis", axis),
         builder.getNamedAttr("factor", factor),
         builder.getNamedAttr("ceil_div", ceilDiv)});
    return TensorSemanticExpr::operation(TensorSemanticKind::LaunchRemap,
                                         expectedType,
                                         {context.inputLaunch}, mapping);
  }
  if (call == "kernel_effect" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> memory =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> launch =
        evaluate(*arguments[1], {}, context);
    if (failed(memory) || failed(launch))
      return failure();
    return TensorSemanticExpr::operation(
        TensorSemanticKind::KernelEffect, booleanType, {*memory, *launch});
  }
  if (call == "kernel_partition" && arguments.size() == 5 &&
      llvm::all_of(arguments.drop_front(2), [](const FormulaPtr &argument) {
        return argument->kind == Formula::Kind::Identifier;
      })) {
    FailureOr<TensorSemanticExpr> memory =
        evaluate(*arguments[0], {}, context);
    FailureOr<TensorSemanticExpr> launch =
        evaluate(*arguments[1], {}, context);
    auto axis = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[2]->spelling));
    auto factor = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[3]->spelling));
    auto ceilDiv = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[4]->spelling));
    if (failed(memory) || failed(launch) || !axis || !factor || !ceilDiv ||
        axis.getInt() < 0 || axis.getInt() > 2 || factor.getInt() <= 1 ||
        (ceilDiv.getInt() != 0 && ceilDiv.getInt() != 1))
      return failure();
    Builder builder(booleanType.getContext());
    DictionaryAttr partition = builder.getDictionaryAttr(
        {builder.getNamedAttr("axis", axis),
         builder.getNamedAttr("factor", factor),
         builder.getNamedAttr("ceil_div", ceilDiv)});
    return TensorSemanticExpr::operation(TensorSemanticKind::KernelPartition,
                                         booleanType, {*memory, *launch},
                                         partition);
  }
  if (call == "input_buffer" && arguments.empty() && context.inputBuffer)
    return context.inputBuffer;
  if (call == "memory_of" && arguments.size() == 1) {
    std::optional<unsigned> operand = findOperand(context, *arguments[0]);
    if (!operand || *operand >= context.operands.size() ||
        !context.operands[*operand].memory)
      return failure();
    return context.operands[*operand].memory;
  }
  if (call == "memory_valid_of" && arguments.size() == 1) {
    std::optional<unsigned> operand = findOperand(context, *arguments[0]);
    if (!operand || *operand >= context.operands.size() ||
        !context.operands[*operand].memoryDefined)
      return failure();
    return context.operands[*operand].memoryDefined;
  }
  if (call == "buffer_of" && arguments.size() == 1) {
    std::optional<unsigned> operand = findOperand(context, *arguments[0]);
    if (!operand || *operand >= context.operands.size() ||
        !context.operands[*operand].buffer)
      return failure();
    return context.operands[*operand].buffer;
  }
  if (call == "buffer_valid_of" && arguments.size() == 1) {
    std::optional<unsigned> operand = findOperand(context, *arguments[0]);
    if (!operand || *operand >= context.operands.size() ||
        !context.operands[*operand].bufferDefined)
      return failure();
    return context.operands[*operand].bufferDefined;
  }
  if ((call == "buffer_read" || call == "buffer_valid") &&
      arguments.size() == 1 && context.inputBuffer) {
    std::optional<TensorSemanticExpr> handle;
    DictionaryAttr region;
    if (std::optional<unsigned> operand =
            findOperand(context, *arguments[0]);
        operand && *operand < context.operands.size()) {
      handle = context.operands[*operand].value;
      region = getBufferRegionDescriptor(context.operands[*operand]);
      mergeSameShapeSymbols(context.result, context.operands[*operand]);
    } else {
      FailureOr<TensorSemanticExpr> evaluated =
          evaluate(*arguments[0], {}, context);
      if (succeeded(evaluated))
        handle = *evaluated;
    }
    if (!handle)
      return failure();
    Type resultType = call == "buffer_valid" ? booleanType : expectedType;
    if (!resultType && call == "buffer_read")
      resultType = handle->getType();
    if (!resultType)
      return failure();
    return TensorSemanticExpr::operation(
        call == "buffer_read" ? TensorSemanticKind::BufferRead
                              : TensorSemanticKind::BufferValid,
        resultType, {context.inputBuffer, *handle}, region);
  }
  if (call == "buffer_view" && arguments.size() >= 1 && expectedType) {
    SmallVector<TensorSemanticExpr, 4> operands;
    for (const FormulaPtr &argument : arguments) {
      FailureOr<TensorSemanticExpr> operand =
          evaluate(*argument, {}, context);
      if (failed(operand))
        return failure();
      operands.push_back(*operand);
    }
    return TensorSemanticExpr::operation(
        TensorSemanticKind::BufferView, expectedType, operands,
        getSemanticOperationIdentity(context.node));
  }
  if (call == "buffer_fill" && arguments.size() == 2 &&
      context.inputBuffer) {
    std::optional<unsigned> target = findOperand(context, *arguments[0]);
    if (!target || *target >= context.operands.size())
      return failure();
    FailureOr<TensorSemanticExpr> fillValue =
        evaluate(*arguments[1], {}, context);
    if (failed(fillValue))
      return failure();
    const TensorDenotation &region = context.operands[*target];
    if (region.shape.size() != region.resultIndices.size())
      return failure();
    return TensorSemanticExpr::operation(
        TensorSemanticKind::BufferFill,
        IntegerType::get(mlirContext, 1),
        {context.inputBuffer, region.value, *fillValue},
        getBufferRegionDescriptor(region));
  }
  if (call == "buffer_write" &&
      (arguments.size() == 2 || arguments.size() == 3) &&
      context.inputBuffer) {
    SmallVector<TensorSemanticExpr, 4> accessOperands;
    for (const FormulaPtr &argument : arguments) {
      FailureOr<TensorSemanticExpr> operand =
          evaluate(*argument, {}, context);
      if (failed(operand))
        return failure();
      accessOperands.push_back(*operand);
    }
    FailureOr<std::optional<StaticLaneDomain>> maybeDomain =
        getStaticLaneDomain(context);
    if (failed(maybeDomain))
      return failure();
    TensorSemanticExpr state = context.inputBuffer;
    DictionaryAttr identity = getSemanticOperationIdentity(context.node);
    if (!*maybeDomain) {
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      operands.append(accessOperands.begin(), accessOperands.end());
      return TensorSemanticExpr::operation(
          TensorSemanticKind::BufferWrite,
          IntegerType::get(mlirContext, 1), operands, identity);
    }
    const StaticLaneDomain &domain = **maybeDomain;
    for (int64_t lane = 0; lane < domain.elementCount; ++lane) {
      SmallVector<TensorSemanticExpr, 4> coordinates =
          getLaneCoordinates(domain, lane);
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      for (const TensorSemanticExpr &operand : accessOperands)
        operands.push_back(substituteMany(operand, domain.indices,
                                          coordinates));
      state = TensorSemanticExpr::operation(
          TensorSemanticKind::BufferWrite,
          IntegerType::get(mlirContext, 1), operands, identity);
    }
    return state;
  }
  if (call == "memory_write" &&
      (arguments.size() == 2 || arguments.size() == 3) &&
      context.inputMemory) {
    SmallVector<TensorSemanticExpr, 4> accessOperands;
    for (const FormulaPtr &argument : arguments) {
      FailureOr<TensorSemanticExpr> operand =
          evaluate(*argument, {}, context);
      if (failed(operand))
        return failure();
      accessOperands.push_back(*operand);
    }
    FailureOr<std::optional<StaticLaneDomain>> maybeDomain =
        getStaticLaneDomain(context);
    if (failed(maybeDomain))
      return failure();
    TensorSemanticExpr state = context.inputMemory;
    if (!*maybeDomain) {
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      operands.append(accessOperands.begin(), accessOperands.end());
      return TensorSemanticExpr::operation(
          TensorSemanticKind::MemoryWrite,
          IntegerType::get(mlirContext, 1), operands);
    }
    const StaticLaneDomain &domain = **maybeDomain;
    for (int64_t lane = 0; lane < domain.elementCount; ++lane) {
      SmallVector<TensorSemanticExpr, 4> coordinates =
          getLaneCoordinates(domain, lane);
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      for (const TensorSemanticExpr &operand : accessOperands)
        operands.push_back(substituteMany(operand, domain.indices,
                                          coordinates));
      state = TensorSemanticExpr::operation(
          TensorSemanticKind::MemoryWrite,
          IntegerType::get(mlirContext, 1), operands);
    }
    return state;
  }
  if ((call == "memory_atomic" || call == "memory_atomic_masked") &&
      arguments.size() >= 2 && arguments.size() <= 3 &&
      context.inputMemory) {
    SmallVector<TensorSemanticExpr, 4> accessOperands;
    for (const FormulaPtr &argument : arguments) {
      FailureOr<TensorSemanticExpr> operand =
          evaluate(*argument, {}, context);
      if (failed(operand))
        return failure();
      accessOperands.push_back(*operand);
    }
    Builder builder(mlirContext);
    DictionaryAttr identity = getSemanticOperationIdentity(context.node);
    SmallVector<NamedAttribute, 2> payloadAttributes = {
        builder.getNamedAttr("identity", identity)};
    if (call == "memory_atomic_masked")
      payloadAttributes.push_back(
          builder.getNamedAttr("masked", builder.getBoolAttr(true)));
    DictionaryAttr payload = builder.getDictionaryAttr(payloadAttributes);
    FailureOr<std::optional<StaticLaneDomain>> maybeDomain =
        getStaticLaneDomain(context);
    if (failed(maybeDomain))
      return failure();
    TensorSemanticExpr state = context.inputMemory;
    if (!*maybeDomain) {
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      operands.append(accessOperands.begin(), accessOperands.end());
      return TensorSemanticExpr::operation(TensorSemanticKind::MemoryAtomic,
                                           IntegerType::get(mlirContext, 1),
                                           operands, payload);
    }
    const StaticLaneDomain &domain = **maybeDomain;
    for (int64_t lane = 0; lane < domain.elementCount; ++lane) {
      SmallVector<TensorSemanticExpr, 4> coordinates =
          getLaneCoordinates(domain, lane);
      SmallVector<TensorSemanticExpr, 4> operands = {state};
      for (const TensorSemanticExpr &operand : accessOperands)
        operands.push_back(substituteMany(operand, domain.indices,
                                          coordinates));
      state = TensorSemanticExpr::operation(TensorSemanticKind::MemoryAtomic,
                                            IntegerType::get(mlirContext, 1),
                                            operands, payload);
    }
    return state;
  }
  if (call == "undefined_value" && arguments.empty() && expectedType)
    return TensorSemanticExpr::operation(TensorSemanticKind::Unobserved,
                                         expectedType);
  if ((call == "constant" || call == "attr_int") &&
      arguments.size() == 1 &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    Attribute attribute =
        context.node.attributes.get(arguments[0]->spelling);
    IntegerAttr scalar = dyn_cast_or_null<IntegerAttr>(attribute);
    if (!scalar && call == "constant") {
      auto dense = dyn_cast_or_null<DenseElementsAttr>(attribute);
      if (!dense || !dense.isSplat() ||
          !isa<IntegerType>(dense.getElementType()))
        return failure();
      scalar = dense.getSplatValue<IntegerAttr>();
    }
    if (!scalar)
      return failure();
    Type type = expectedType ? expectedType : context.resultElementType;
    if (type.isInteger(1))
      return TensorSemanticExpr::boolean(mlirContext,
                                         !scalar.getValue().isZero());
    return TensorSemanticExpr::integer(mlirContext,
                                       scalar.getValue().getSExtValue(), type);
  }
  if (call == "sum")
    return evaluateBinder(formula, /*forall=*/false, context);
  if (call == "forall")
    return evaluateBinder(formula, /*forall=*/true, context);
  if (call == "not" && arguments.size() == 1) {
    FailureOr<TensorSemanticExpr> operand =
        evaluate(*arguments[0], booleanType, context);
    if (failed(operand))
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::Not, booleanType,
                                         {*operand});
  }
  if (call == "and")
    return evaluateFold(formula, TensorSemanticKind::And, booleanType,
                        booleanType, context);
  if (call == "or")
    return evaluateFold(formula, TensorSemanticKind::Or, booleanType,
                        booleanType, context);
  if (call == "xor")
    return evaluateFold(formula, TensorSemanticKind::Xor, booleanType,
                        booleanType, context);
  if (call == "bit_and")
    return evaluateFold(formula, TensorSemanticKind::BitwiseAnd,
                        expectedType, expectedType, context);
  if (call == "bit_or")
    return evaluateFold(formula, TensorSemanticKind::BitwiseOr,
                        expectedType, expectedType, context);
  if (call == "bit_xor")
    return evaluateFold(formula, TensorSemanticKind::BitwiseXor,
                        expectedType, expectedType, context);
  if (call == "add" || call == "sub" || call == "mul" ||
      call == "srem") {
    if (arguments.size() != 2)
      return failure();
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], expectedType, context);
    if (failed(lhs))
      return failure();
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], lhs->getType(), context);
    if (failed(rhs) || rhs->getType() != lhs->getType())
      return failure();
    TensorSemanticKind kind = TensorSemanticKind::Subtract;
    if (call == "add")
      kind = TensorSemanticKind::Add;
    else if (call == "mul")
      kind = TensorSemanticKind::Multiply;
    else if (call == "srem")
      kind = TensorSemanticKind::SignedRemainder;
    return TensorSemanticExpr::operation(kind, lhs->getType(), {*lhs, *rhs});
  }
  if (call == "integer_compare" && arguments.size() == 3 &&
      arguments[0]->kind == Formula::Kind::Identifier) {
    auto predicate = dyn_cast_or_null<IntegerAttr>(
        context.node.attributes.get(arguments[0]->spelling));
    if (!predicate)
      return failure();
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[1], {}, context);
    if (failed(lhs))
      return failure();
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[2], lhs->getType(), context);
    if (failed(rhs) || rhs->getType() != lhs->getType())
      return failure();
    TensorSemanticKind kind;
    if (predicate.getInt() ==
        static_cast<int64_t>(arith::CmpIPredicate::eq))
      kind = TensorSemanticKind::Equal;
    else if (predicate.getInt() ==
             static_cast<int64_t>(arith::CmpIPredicate::slt))
      kind = TensorSemanticKind::SignedLessThan;
    else
      return failure();
    return TensorSemanticExpr::operation(kind, booleanType, {*lhs, *rhs});
  }
  if (call == "signed_less_than" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    if (failed(lhs) || !isa<IntegerType>(lhs->getType()))
      return failure();
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], lhs->getType(), context);
    if (failed(rhs) || rhs->getType() != lhs->getType())
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::SignedLessThan,
                                         booleanType, {*lhs, *rhs});
  }
  if (call == "equal" && arguments.size() == 2) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], {}, context);
    if (failed(lhs))
      return failure();
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], lhs->getType(), context);
    if (failed(rhs) || rhs->getType() != lhs->getType())
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::Equal,
                                         booleanType, {*lhs, *rhs});
  }
  if (call == "select" && arguments.size() == 3) {
    FailureOr<TensorSemanticExpr> condition =
        evaluate(*arguments[0], booleanType, context);
    if (failed(condition))
      return failure();
    FailureOr<TensorSemanticExpr> trueValue =
        evaluate(*arguments[1], expectedType, context);
    if (failed(trueValue))
      return failure();
    FailureOr<TensorSemanticExpr> falseValue =
        evaluate(*arguments[2], trueValue->getType(), context);
    if (failed(falseValue) || falseValue->getType() != trueValue->getType())
      return failure();
    return TensorSemanticExpr::operation(
        TensorSemanticKind::Select, trueValue->getType(),
        {*condition, *trueValue, *falseValue});
  }
  if (call == "zext" && arguments.size() == 1 && expectedType) {
    FailureOr<TensorSemanticExpr> operand =
        evaluate(*arguments[0], {}, context);
    if (failed(operand))
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::ZeroExtend,
                                         expectedType, {*operand});
  }
  if (call == "sext" && arguments.size() == 1 && expectedType) {
    FailureOr<TensorSemanticExpr> operand =
        evaluate(*arguments[0], {}, context);
    if (failed(operand))
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::SignExtend,
                                         expectedType, {*operand});
  }
  if (call == "trunc" && arguments.size() == 1 && expectedType) {
    FailureOr<TensorSemanticExpr> operand =
        evaluate(*arguments[0], {}, context);
    if (failed(operand))
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::Truncate,
                                         expectedType, {*operand});
  }
  if (call == "udiv" && arguments.size() == 2 && expectedType) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], expectedType, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], expectedType, context);
    if (failed(lhs) || failed(rhs) || lhs->getType() != rhs->getType())
      return failure();
    return TensorSemanticExpr::operation(TensorSemanticKind::IntegerDivide,
                                         expectedType, {*lhs, *rhs});
  }
  if ((call == "shl" || call == "lshr" || call == "ashr") &&
      arguments.size() == 2 &&
      expectedType) {
    FailureOr<TensorSemanticExpr> lhs =
        evaluate(*arguments[0], expectedType, context);
    FailureOr<TensorSemanticExpr> rhs =
        evaluate(*arguments[1], expectedType, context);
    if (failed(lhs) || failed(rhs) || lhs->getType() != rhs->getType())
      return failure();
    TensorSemanticKind kind =
        call == "shl" ? TensorSemanticKind::ShiftLeft
        : call == "lshr" ? TensorSemanticKind::ShiftRightLogical
                         : TensorSemanticKind::ShiftRightArithmetic;
    return TensorSemanticExpr::operation(
        kind, expectedType, {*lhs, *rhs});
  }
  return failure();
}

static bool matchesDeclaredOperandArity(const SemanticClause &clause,
                                        unsigned operandCount) {
  if (!clause.variadicOperand)
    return clause.operands.size() == operandCount;
  const unsigned fixedOperandCount = clause.operands.size() - 1;
  const int64_t minimumOperandCount =
      clause.operandArity ? clause.operandArity->arguments[0]->integer
                          : static_cast<int64_t>(clause.operands.size());
  if (operandCount < fixedOperandCount ||
      static_cast<int64_t>(operandCount) < minimumOperandCount)
    return false;
  if (!clause.operandArity)
    return true;
  const int64_t first = minimumOperandCount;
  const int64_t step = clause.operandArity->arguments[1]->integer;
  return static_cast<int64_t>(operandCount) >= first &&
         (static_cast<int64_t>(operandCount) - first) % step == 0;
}

static bool hasImportedResultSignature(const SemanticClause &clause) {
  return clause.resultIndex && clause.operationResults &&
         clause.resultIndex->kind == Formula::Kind::Identifier &&
         clause.resultIndex->spelling == "any" &&
         clause.operationResults->kind == Formula::Kind::Identifier &&
         clause.operationResults->spelling == "imported";
}

class DeclarativeTTOpSemantics final : public TTOpSemantics {
public:
  explicit DeclarativeTTOpSemantics(StringRef operationName)
      : operationName(operationName.str()) {}

  StringRef getOperationName() const override { return operationName; }

  ArrayRef<GraphSemanticDomain> getImplementedDomains() const override {
    return implementedDomains;
  }

  TTOpSearchPolicy getSearchPolicy() const override {
    bool searchable = false;
    bool importOnly = false;
    bool regionOnly = false;
    for (const SemanticClause &clause : clauses) {
      if (clause.searchPolicy == TTOpSearchPolicy::Searchable) {
        if (!clause.candidate)
          return TTOpSearchPolicy::Incomplete;
        searchable = true;
      } else if (clause.searchPolicy == TTOpSearchPolicy::ImportOnly) {
        importOnly = true;
      } else if (clause.searchPolicy == TTOpSearchPolicy::RegionOnly) {
        regionOnly = true;
      }
    }
    unsigned kinds = static_cast<unsigned>(searchable) +
                     static_cast<unsigned>(importOnly) +
                     static_cast<unsigned>(regionOnly);
    if (kinds != 1)
      return TTOpSearchPolicy::Mixed;
    if (searchable)
      return TTOpSearchPolicy::Searchable;
    return importOnly ? TTOpSearchPolicy::ImportOnly
                      : TTOpSearchPolicy::RegionOnly;
  }

  bool supports(Operation *operation) const override {
    if (!operation)
      return false;
    const unsigned semanticResultCount =
        std::max<unsigned>(operation->getNumResults(), 1);
    for (unsigned resultIndex = 0; resultIndex < semanticResultCount;
         ++resultIndex) {
      const SemanticClause *clause =
          findClauseForOperation(operation, resultIndex);
      if (!clause)
        return false;
    }
    return true;
  }

  bool allowsSharedImport(Operation *operation) const override {
    if (!operation || operation->getNumResults() != 1)
      return false;
    const SemanticClause *clause = findClauseForOperation(operation, 0);
    if (!clause || !clause->shareable)
      return false;
    SmallVector<Type, 4> operandTypes;
    for (Value operand : operation->getOperands())
      operandTypes.push_back(operand.getType());
    ConstraintContext context{/*operation=*/operation,
                              /*resultType=*/operation->getResult(0).getType(),
                              /*operandTypes=*/operandTypes,
                              /*operandNames=*/clause->operands,
                              /*attributes=*/{},
                              /*compileOn91095=*/false,
                              /*allowUnknownOperands=*/false,
                              /*allowUnknownAttributes=*/false};
    return evaluateConstraint(*clause->shareable, context) ==
           ConstraintTruth::True;
  }

  void addClause(SemanticClause clause) {
    auto sameResultSelector = [](const SemanticClause &lhs,
                                 const SemanticClause &rhs) {
      if (!lhs.resultIndex || !rhs.resultIndex)
        return !lhs.resultIndex && !rhs.resultIndex;
      if (hasImportedResultSignature(lhs) ||
          hasImportedResultSignature(rhs))
        return hasImportedResultSignature(lhs) &&
               hasImportedResultSignature(rhs);
      return lhs.resultIndex->kind == Formula::Kind::Integer &&
             rhs.resultIndex->kind == Formula::Kind::Integer &&
             lhs.resultIndex->integer == rhs.resultIndex->integer;
    };
    for (const SemanticClause &known : clauses)
      if (known.operands.size() == clause.operands.size() &&
          known.variadicOperand == clause.variadicOperand &&
          sameResultSelector(known, clause) &&
          sameRegionDescriptor(known.region.get(), clause.region.get()) &&
          sameRegionDescriptor(known.signature.get(),
                               clause.signature.get()))
        llvm::report_fatal_error("duplicate TT semantics overload for " +
                                 Twine(operationName) + "/" +
                                 Twine(clause.operands.size()));
    clauses.push_back(std::move(clause));
  }

  FailureOr<TensorDenotation>
  denote(const GraphExprNode &node,
         ArrayRef<TensorDenotation> operands) const override {
    unsigned explicitOperandCount = operands.size();
    const TensorDenotation *bufferOperand = nullptr;
    const TensorDenotation *launchOperand = nullptr;
    const TensorDenotation *memoryOperand = nullptr;
    if (explicitOperandCount != 0 &&
        operands[explicitOperandCount - 1].isBufferStateToken)
      bufferOperand = &operands[--explicitOperandCount];
    if (explicitOperandCount != 0 &&
        operands[explicitOperandCount - 1].isLaunchStateToken)
      launchOperand = &operands[--explicitOperandCount];
    if (explicitOperandCount != 0 &&
        operands[explicitOperandCount - 1].isMemoryStateToken)
      memoryOperand = &operands[--explicitOperandCount];
    bool hasExplicitMemory = memoryOperand != nullptr;
    bool hasExplicitLaunch = launchOperand != nullptr;
    bool hasExplicitBuffer = bufferOperand != nullptr;
    unsigned resultIndex = 0;
    if (node.attributes)
      if (auto index = dyn_cast_or_null<IntegerAttr>(
              node.attributes.get(kResultIndexAttr))) {
        if (index.getInt() < 0)
          return failure();
        resultIndex = static_cast<unsigned>(index.getInt());
      }
    const SemanticClause *clause = findClauseForNode(
        node, explicitOperandCount,
        node.attributes && node.attributes.get(kResultIndexAttr)
            ? std::optional<unsigned>(resultIndex)
            : std::nullopt);
    if (!clause)
      return failure();
    bool hasMemoryEffect =
        clause->effectKind == SemanticClause::EffectKind::Read ||
        clause->effectKind == SemanticClause::EffectKind::Write ||
        clause->effectKind == SemanticClause::EffectKind::Atomic ||
        clause->effectKind == SemanticClause::EffectKind::Observable;
    bool hasBufferEffect =
        clause->effectKind == SemanticClause::EffectKind::BufferRead ||
        clause->effectKind == SemanticClause::EffectKind::BufferWrite;
    if (hasExplicitMemory && !hasMemoryEffect && !clause->memory)
      return failure();
    if (hasExplicitBuffer && !hasBufferEffect && !clause->buffer)
      return failure();

    Type resultType = getResultType(node);
    if (!resultType)
      return failure();
    TensorDenotation result = makeDenotationSkeleton(resultType, node);
    Type elementType = getElementType(resultType);
    Type boolType = IntegerType::get(resultType.getContext(), 1);
    TensorSemanticExpr initialMemory = TensorSemanticExpr::symbol(
        resultType.getContext(), "memory-state-0", boolType);
    TensorSemanticExpr inputMemory =
        hasExplicitMemory ? memoryOperand->memory : initialMemory;
    if (!inputMemory)
      return failure();
    TensorSemanticExpr initialLaunch = TensorSemanticExpr::symbol(
        resultType.getContext(), "launch-state-0", boolType);
    TensorSemanticExpr inputLaunch =
        hasExplicitLaunch ? launchOperand->launch : initialLaunch;
    if (!inputLaunch)
      return failure();
    TensorSemanticExpr initialBuffer = TensorSemanticExpr::symbol(
        resultType.getContext(), "buffer-state-0", boolType);
    TensorSemanticExpr inputBuffer =
        hasExplicitBuffer ? bufferOperand->buffer : initialBuffer;
    if (!inputBuffer)
      return failure();
    ArrayRef<TensorDenotation> explicitOperands =
        operands.take_front(explicitOperandCount);
    TensorSemanticExpr inputEffect =
        hasExplicitMemory
            ? memoryOperand->effect
            : TensorSemanticExpr::boolean(resultType.getContext(), true);
    TensorSemanticExpr inputMemoryDefined =
        hasExplicitMemory
            ? memoryOperand->memoryDefined
            : TensorSemanticExpr::boolean(resultType.getContext(), true);
    EvaluationContext context{node, explicitOperands, *clause, result,
                              elementType, {}, inputEffect, inputMemory,
                              inputMemoryDefined, inputLaunch, inputBuffer};
    FailureOr<TensorSemanticExpr> value =
        evaluate(*clause->value, elementType, context);
    FailureOr<TensorSemanticExpr> definedness =
        evaluate(*clause->definedness,
                 IntegerType::get(resultType.getContext(), 1), context);
    if (failed(value) || failed(definedness) ||
        !definedness->getType().isInteger(1))
      return failure();
    if (clause->effectKind == SemanticClause::EffectKind::Write ||
        clause->effectKind == SemanticClause::EffectKind::BufferWrite) {
      definedness = foldAllLanes(*definedness, context);
      if (failed(definedness))
        return failure();
    }
    result.value = *value;
    result.defined = *definedness;

    auto combineEffect = [&](TensorSemanticExpr lhs,
                             TensorSemanticExpr rhs) {
      return TensorSemanticExpr::operation(TensorSemanticKind::And, boolType,
                                           {lhs, rhs});
    };
    result.effect = TensorSemanticExpr::boolean(resultType.getContext(), true);
    result.memory = hasMemoryEffect ? inputMemory : initialMemory;
    result.memoryDefined =
        hasExplicitMemory
            ? memoryOperand->memoryDefined
            : TensorSemanticExpr::boolean(resultType.getContext(), true);
    if (!result.memoryDefined)
      return failure();
    // A state transition cannot erase undefined behavior from an earlier
    // transition merely because its final bytes are overwritten.  Carry the
    // incoming MemoryState validity into both the observable result and the
    // outgoing state.  Pure/read values keep their existing per-value rule.
    if (hasMemoryEffect && hasExplicitMemory)
      result.defined = combineEffect(result.memoryDefined, result.defined);
    if (hasBufferEffect && hasExplicitBuffer)
      result.defined = combineEffect(bufferOperand->bufferDefined,
                                     result.defined);
    for (const TensorDenotation &operand : operands) {
      if (!operand.effect)
        return failure();
      result.effect = combineEffect(result.effect, operand.effect);
    }
    if (clause->effect &&
        clause->effectKind != SemanticClause::EffectKind::Read) {
      FailureOr<TensorSemanticExpr> effect =
          evaluate(*clause->effect, boolType, context);
      if (failed(effect) || !effect->getType().isInteger(1))
        return failure();
      result.effect = *effect;
    }
    // Non-volatile reads are not observable transitions. Their returned value
    // still denotes memory_read(inputMemory, address), which preserves the
    // MemorySSA dependency and all value/definedness obligations when the
    // value is observed. Keeping a second read event in Effect would make an
    // otherwise dead load observable and would incorrectly reject generic
    // refinements that mask or eliminate unused lanes. Volatile reads are
    // rejected by the import/candidate clauses and therefore never reach this
    // semantic path.
    // Buffer reads are represented by the explicit BufferState threaded into
    // value/valid.  They are deliberately not appended to the global Effect
    // trace: doing both would model one access twice and would prevent a
    // private BufferState suffix from being eliminated after its value has
    // been proved equivalent.  Observable memory effects remain in Effect;
    // buffer isolation is revalidated by the online materializer.
    if (clause->memory) {
      FailureOr<TensorSemanticExpr> memory =
          evaluate(*clause->memory, boolType, context);
      if (failed(memory))
        return failure();
      result.memory = *memory;
      if (clause->effectKind == SemanticClause::EffectKind::Atomic) {
        ArrayRef<TensorSemanticExpr> transitionOperands =
            result.memory.getOperands();
        if ((result.memory.getKind() != TensorSemanticKind::MemoryWrite &&
             result.memory.getKind() != TensorSemanticKind::MemoryAtomic) ||
            transitionOperands.empty())
          return failure();
        SmallVector<TensorSemanticExpr, 4> effectOperands = {
            TensorSemanticExpr::boolean(resultType.getContext(), true)};
        effectOperands.append(transitionOperands.drop_front().begin(),
                              transitionOperands.drop_front().end());
        TensorSemanticExpr effect = TensorSemanticExpr::operation(
            result.memory.getKind(), boolType, effectOperands,
            result.memory.getPayload());
        result.effect = combineEffect(result.effect, effect);
      }
    }
    if (clause->effectKind == SemanticClause::EffectKind::Observable) {
      SmallVector<TensorSemanticExpr, 8> transitionOperands = {inputMemory};
      for (const TensorDenotation &operand : explicitOperands)
        transitionOperands.push_back(operand.value);
      TensorSemanticExpr transition = TensorSemanticExpr::operation(
          TensorSemanticKind::MemoryObservable, boolType,
          transitionOperands, getSemanticOperationIdentity(node));
      result.memory = transition;
      result.effect = combineEffect(result.effect, transition);
      FailureOr<TensorSemanticExpr> observableValid =
          foldAllLanes(result.defined, context);
      if (failed(observableValid))
        return failure();
      result.memoryDefined = combineEffect(result.memoryDefined,
                                           *observableValid);
    }
    if (clause->memoryDefinedness) {
      FailureOr<TensorSemanticExpr> memoryDefined =
          evaluate(*clause->memoryDefinedness, boolType, context);
      if (failed(memoryDefined) || !memoryDefined->getType().isInteger(1))
        return failure();
      memoryDefined = foldAllLanes(*memoryDefined, context);
      if (failed(memoryDefined))
        return failure();
      result.memoryDefined =
          hasExplicitMemory &&
                  !containsCall(*clause->memoryDefinedness,
                                "input_memory_valid")
              ? combineEffect(memoryOperand->memoryDefined, *memoryDefined)
              : *memoryDefined;
    }
    result.launch = inputLaunch;
    result.launchDefined =
        hasExplicitLaunch
            ? launchOperand->launchDefined
            : TensorSemanticExpr::boolean(resultType.getContext(), true);
    if (!result.launchDefined)
      return failure();
    if (clause->launch) {
      FailureOr<TensorSemanticExpr> launch =
          evaluate(*clause->launch, boolType, context);
      FailureOr<TensorSemanticExpr> launchDefined =
          evaluate(*clause->launchDefinedness, boolType, context);
      if (failed(launch) || failed(launchDefined) ||
          !launchDefined->getType().isInteger(1))
        return failure();
      result.launch = *launch;
      result.launchDefined = *launchDefined;
    }
    std::optional<TTIRCandidateStateDomain> producedState;
    if (clause->candidate)
      collectProducedStateDomain(*clause->candidate, producedState);
    result.isMemoryStateToken =
        isa<NoneType>(resultType) &&
        (producedState == TTIRCandidateStateDomain::Memory ||
         clause->effectKind == SemanticClause::EffectKind::Write ||
         clause->effectKind == SemanticClause::EffectKind::Observable ||
         node.getOperationName() == "ags.memory" ||
         node.getOperationName() == "ags.memory_after");
    result.isLaunchStateToken =
        isa<NoneType>(resultType) &&
        (producedState == TTIRCandidateStateDomain::Launch ||
         node.getOperationName() == "ags.launch");
    result.buffer = hasBufferEffect ? inputBuffer : initialBuffer;
    result.bufferDefined =
        hasExplicitBuffer
            ? bufferOperand->bufferDefined
            : TensorSemanticExpr::boolean(resultType.getContext(), true);
    if (!result.bufferDefined)
      return failure();
    if (clause->buffer) {
      FailureOr<TensorSemanticExpr> buffer =
          evaluate(*clause->buffer, boolType, context);
      if (failed(buffer))
        return failure();
      result.buffer = *buffer;
    }
    if (clause->bufferDefinedness) {
      FailureOr<TensorSemanticExpr> bufferDefined =
          evaluate(*clause->bufferDefinedness, boolType, context);
      if (failed(bufferDefined) || !bufferDefined->getType().isInteger(1))
        return failure();
      bufferDefined = foldAllLanes(*bufferDefined, context);
      if (failed(bufferDefined))
        return failure();
      result.bufferDefined =
          hasExplicitBuffer
              ? combineEffect(bufferOperand->bufferDefined, *bufferDefined)
              : *bufferDefined;
    }
    result.isBufferStateToken =
        isa<NoneType>(resultType) &&
        (producedState == TTIRCandidateStateDomain::Buffer ||
         node.getOperationName() == "ags.buffer" ||
         clause->effectKind == SemanticClause::EffectKind::BufferWrite);
    // Some shape-producing operations, notably scalar-to-tensor broadcast,
    // have no shaped operand from which to inherit a dynamic extent.  Their
    // unresolved extent is still the contextual result extent shared by every
    // candidate of this result type; it is not an unknowable shape conflict.
    // Give it a stable name so equivalent candidates can be compared, while
    // dimensions constrained by operands keep the more precise relation built
    // by mergeSameShapeSymbols.
    for (auto [dimension, symbol] : llvm::enumerate(result.shapeSymbols))
      if (symbol == "?")
        symbol = (llvm::Twine("result-dim-") + llvm::Twine(dimension)).str();
    return result;
  }

private:
  static bool sameRegionDescriptor(const Formula *lhs, const Formula *rhs) {
    if (!lhs || !rhs)
      return lhs == rhs;
    if (lhs->kind != rhs->kind || lhs->spelling != rhs->spelling ||
        lhs->integer != rhs->integer || lhs->boolean != rhs->boolean ||
        lhs->arguments.size() != rhs->arguments.size())
      return false;
    for (const auto &pair : llvm::zip(lhs->arguments, rhs->arguments))
      if (!sameRegionDescriptor(std::get<0>(pair).get(),
                                std::get<1>(pair).get()))
        return false;
    return true;
  }

  static bool matchesArityAndProjection(
      const SemanticClause &clause, unsigned operandCount,
      std::optional<unsigned> resultIndex) {
    return matchesDeclaredOperandArity(clause, operandCount) &&
           (hasImportedResultSignature(clause) ||
            (!resultIndex && !clause.resultIndex) ||
            (resultIndex && clause.resultIndex &&
             clause.resultIndex->integer == *resultIndex));
  }

  static bool matchesRegion(const SemanticClause &clause,
                            ArrayAttr actualRegions, Type elementType,
                            MLIRContext *context) {
    if (!clause.region)
      return !actualRegions;
    if (clause.region->kind == Formula::Kind::Identifier &&
        clause.region->spelling == "imported")
      return static_cast<bool>(actualRegions);
    if (!actualRegions || !elementType ||
        clause.region->kind != Formula::Kind::Call ||
        clause.region->spelling != "binary" ||
        clause.region->arguments.size() != 2 ||
        llvm::any_of(clause.region->arguments,
                     [](const FormulaPtr &argument) {
                       return argument->kind != Formula::Kind::Identifier;
                     }))
      return false;
    ArrayAttr expected = buildCanonicalBinaryRegion(
        context, elementType, clause.region->arguments[0]->spelling,
        clause.region->arguments[1]->spelling);
    return actualRegions == expected;
  }

  const SemanticClause *findClauseForOperation(Operation *operation,
                                                unsigned resultIndex) const {
    SmallVector<Type, 4> operandTypes;
    for (Value operand : operation->getOperands())
      operandTypes.push_back(operand.getType());
    std::optional<ArrayAttr> actualRegions;
    if (operation->getNumRegions() != 0) {
      std::string error;
      actualRegions = captureCanonicalRegions(operation, error);
      if (!actualRegions)
        return nullptr;
    }
    for (const SemanticClause &clause : clauses) {
      std::optional<unsigned> projection =
          operation->getNumResults() > 1
              ? std::optional<unsigned>(resultIndex)
              : std::nullopt;
      if (!matchesArityAndProjection(clause, operation->getNumOperands(),
                                     projection) ||
          !clause.signature)
        continue;
      if (operation->getNumResults() == 0 &&
          clause.effectKind != SemanticClause::EffectKind::Write &&
          clause.effectKind != SemanticClause::EffectKind::Observable &&
          clause.effectKind != SemanticClause::EffectKind::BufferWrite &&
          clause.searchPolicy != TTOpSearchPolicy::RegionOnly)
        continue;
      if (operation->getNumResults() != 0 &&
          (clause.effectKind == SemanticClause::EffectKind::Write ||
           clause.effectKind == SemanticClause::EffectKind::BufferWrite))
        continue;
      Type selectedType =
          operation->getNumResults() == 0
              ? NoneType::get(operation->getContext())
              : operation->getResult(resultIndex).getType();
      ConstraintContext constraintContext{
          /*operation=*/operation,
          /*resultType=*/selectedType,
          /*operandTypes=*/operandTypes,
          /*operandNames=*/clause.operands,
          /*attributes=*/{},
          /*compileOn91095=*/false,
          /*allowUnknownOperands=*/false,
          /*allowUnknownAttributes=*/false};
      constraintContext.resultIndex = resultIndex;
      if (evaluateConstraint(*clause.signature, constraintContext) !=
          ConstraintTruth::True)
        continue;
      SmallVector<Type, 2> declaredResults;
      if (!hasImportedResultSignature(clause))
        declaredResults = resolveOperationResultTypes(
            clause.operationResults.get(), constraintContext);
      if (operation->getNumResults() != 0 &&
          !hasImportedResultSignature(clause)) {
        if (operation->getNumResults() != declaredResults.size())
          continue;
        if (llvm::any_of(llvm::zip(operation->getResultTypes(),
                                  declaredResults),
                         [](const auto &pair) {
                           return std::get<0>(pair) != std::get<1>(pair);
                         }))
          continue;
      }
      Type elementType = selectedType;
      if (auto shaped = dyn_cast<ShapedType>(elementType))
        elementType = shaped.getElementType();
      if (!matchesRegion(clause, actualRegions.value_or(ArrayAttr{}),
                         elementType, operation->getContext()))
        continue;
      return &clause;
    }
    return nullptr;
  }

  const SemanticClause *findClauseForNode(
      const GraphExprNode &node, unsigned operandCount,
      std::optional<unsigned> resultIndex) const {
    Type resultType = getResultType(node);
    if (!resultType)
      return nullptr;
    Type elementType = getElementType(resultType);
    ArrayAttr actualRegions;
    if (node.attributes)
      actualRegions = dyn_cast_or_null<ArrayAttr>(
          node.attributes.get(kCanonicalRegionsAttr));
    for (const SemanticClause &clause : clauses) {
      ConstraintContext context{
          /*operation=*/nullptr,
          /*resultType=*/resultType,
          /*operandTypes=*/{},
          /*operandNames=*/clause.operands,
          /*attributes=*/node.attributes,
          /*compileOn91095=*/false,
          /*allowUnknownOperands=*/true,
          /*allowUnknownAttributes=*/false};
      context.resultIndex = resultIndex.value_or(0);
      if (!matchesArityAndProjection(clause, operandCount, resultIndex) ||
          !matchesRegion(clause, actualRegions, elementType,
                         resultType.getContext()))
        continue;

      // `import` answers whether a concrete operation may enter an online
      // island; it is not a denotation guard for graphs that were already
      // admitted by the offline candidate language.  In particular, the
      // synthetic boundary/state terminals intentionally declare
      // `import = false` but still need an executable Value + Valid meaning.
      // Prefer the import constraint when it can select the overload.  If it
      // rejects the node, fall back to the candidate constraint.  A
      // zero-operand, non-constructible clause is a declarative terminal and
      // therefore needs no candidate constraint.
      ConstraintTruth importMatch =
          evaluateConstraint(*clause.signature, context);
      if (importMatch != ConstraintTruth::False)
        return &clause;
      if (clause.candidate &&
          evaluateConstraint(*clause.candidate, context) !=
              ConstraintTruth::False)
        return &clause;
      if (!clause.candidate && clause.operands.empty())
        return &clause;
    }
    return nullptr;
  }

  std::string operationName;
  // Every parsed definition is evaluated in the common tensor-index
  // denotation. The result type supplies the output shape and canonical index
  // binders; reshape/broadcast/transpose/reduce/scan helpers substitute input
  // coordinates into Value/Definedness. Consequently Shape and Index are
  // executable domains, not coverage labels inferred by rulegen.
  SmallVector<GraphSemanticDomain, 10> implementedDomains = {
      GraphSemanticDomain::Signature, GraphSemanticDomain::Shape,
      GraphSemanticDomain::Index, GraphSemanticDomain::Layout,
      GraphSemanticDomain::Value,
      GraphSemanticDomain::Definedness, GraphSemanticDomain::Effect,
      GraphSemanticDomain::Memory, GraphSemanticDomain::Launch,
      GraphSemanticDomain::Buffer};
  SmallVector<SemanticClause, 2> clauses;
};

} // namespace

TTOpSemanticsRegistry
mlir::triton::cfg::createBuiltinTTOpSemanticsRegistry() {
  SemanticsParser parser(kBuiltinTTOpSemanticsSource);
  SmallVector<SemanticClause, 16> clauses = parser.parse();

  SmallVector<std::unique_ptr<DeclarativeTTOpSemantics>, 16> definitions;
  llvm::StringMap<unsigned> indices;
  for (SemanticClause &clause : clauses) {
    auto known = indices.find(clause.operationName);
    unsigned index;
    if (known == indices.end()) {
      index = definitions.size();
      indices.insert({clause.operationName, index});
      definitions.push_back(
          std::make_unique<DeclarativeTTOpSemantics>(clause.operationName));
    } else {
      index = known->second;
    }
    definitions[index]->addClause(std::move(clause));
  }

  TTOpSemanticsRegistry registry;
  for (std::unique_ptr<DeclarativeTTOpSemantics> &definition : definitions)
    if (failed(registry.add(std::move(definition))))
      llvm::report_fatal_error("duplicate operation in TT semantics registry");
  return registry;
}

std::string mlir::triton::cfg::getBuiltinTTOpSemanticsStableHash() {
  uint64_t hash = UINT64_C(14695981039346656037);
  // The executable interpreter is part of semantic identity just as much as
  // the declarative source. Bump this tag whenever denotation or proof
  // behavior changes so old e-class certificates cannot be reused silently.
  constexpr llvm::StringLiteral kInterpreterVersion =
      "ttsem-interpreter-v47-overflow-safe-nonnegative-domain\n";
  for (unsigned char byte : kInterpreterVersion) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  for (unsigned char byte : llvm::StringRef(kBuiltinTTOpSemanticsSource)) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}

SmallVector<TTIRCandidateOpSchema, 4>
mlir::triton::cfg::createBuiltinTTIRCandidateOpSchemas(
    const GraphOptimizationOptions &options,
    unsigned maxCandidateOperands) {
  SemanticsParser parser(kBuiltinTTOpSemanticsSource);
  SmallVector<SemanticClause, 16> clauses = parser.parse();
  SmallVector<TTIRCandidateOpSchema, 4> schemas;
  for (auto indexedClause : llvm::enumerate(clauses)) {
    const unsigned semanticClauseIndex =
        static_cast<unsigned>(indexedClause.index());
    const SemanticClause &clause = indexedClause.value();
    const unsigned fixedOperandCount =
        clause.variadicOperand ? clause.operands.size() - 1
                               : clause.operands.size();
    const unsigned minimumOperandCount =
        clause.operandArity
            ? static_cast<unsigned>(
                  clause.operandArity->arguments[0]->integer)
            : clause.operands.size();
    if (clause.variadicOperand &&
        maxCandidateOperands < minimumOperandCount)
      continue;
    const unsigned maximumOperandCount =
        clause.variadicOperand ? maxCandidateOperands : minimumOperandCount;
    for (unsigned concreteOperandCount = minimumOperandCount;
         concreteOperandCount <= maximumOperandCount;
         ++concreteOperandCount) {
      if (!matchesDeclaredOperandArity(clause, concreteOperandCount))
        continue;
      std::string operationName = clause.operationName;
      SmallVector<std::string, 4> operandNames;
      if (!clause.variadicOperand) {
        operandNames = clause.operands;
      } else {
        operandNames.append(clause.operands.begin(),
                            std::prev(clause.operands.end()));
        StringRef prefix = *clause.variadicOperand;
        while (operandNames.size() < concreteOperandCount)
          operandNames.push_back(
              (llvm::Twine(prefix) +
               llvm::Twine(operandNames.size() - fixedOperandCount))
                  .str());
      }

      TTIRCandidateOpSchema schema;
      schema.operationName = std::move(operationName);
      schema.semanticClauseIndex = semanticClauseIndex;
      schema.operandCount = operandNames.size();
      schema.resultIndex = clause.resultIndex
                               ? (hasImportedResultSignature(clause)
                                      ? 0u
                                      : static_cast<unsigned>(
                                     clause.resultIndex->integer)
                                 )
                               : 0;
      schema.operationResultCount = hasImportedResultSignature(clause)
                                        ? 0
                                    : clause.operationResults
                                        ? clause.operationResults->arguments.size()
                                        : 1;
    switch (clause.effectKind) {
    case SemanticClause::EffectKind::Pure:
      schema.effectKind = TTIRCandidateEffectKind::Pure;
      break;
    case SemanticClause::EffectKind::Read:
      schema.effectKind = TTIRCandidateEffectKind::Read;
      break;
    case SemanticClause::EffectKind::Write:
      schema.effectKind = TTIRCandidateEffectKind::Write;
      break;
    case SemanticClause::EffectKind::Atomic:
      schema.effectKind = TTIRCandidateEffectKind::Atomic;
      break;
    case SemanticClause::EffectKind::Observable:
      schema.effectKind = TTIRCandidateEffectKind::Observable;
      break;
    case SemanticClause::EffectKind::BufferRead:
      schema.effectKind = TTIRCandidateEffectKind::BufferRead;
      break;
    case SemanticClause::EffectKind::BufferWrite:
      schema.effectKind = TTIRCandidateEffectKind::BufferWrite;
      break;
    }
    if (schema.effectKind == TTIRCandidateEffectKind::Read ||
        schema.effectKind == TTIRCandidateEffectKind::Write ||
        schema.effectKind == TTIRCandidateEffectKind::Atomic ||
        schema.effectKind == TTIRCandidateEffectKind::Observable)
      schema.implicitStateDomains.push_back(
          TTIRCandidateStateDomain::Memory);
    if (schema.effectKind == TTIRCandidateEffectKind::BufferRead ||
        schema.effectKind == TTIRCandidateEffectKind::BufferWrite)
      schema.implicitStateDomains.push_back(
          TTIRCandidateStateDomain::Buffer);
    schema.constructible = static_cast<bool>(clause.candidate);
    schema.intentionalImportOnly =
        clause.searchPolicy == TTOpSearchPolicy::ImportOnly;
    schema.structuralRegionOnly =
        clause.searchPolicy == TTOpSearchPolicy::RegionOnly;
    schema.proofOnlyMaterialization = static_cast<bool>(clause.materialization);
    collectSemanticAttributeNames(*clause.signature,
                                  schema.semanticAttributeNames);
    collectSemanticAttributeNames(*clause.value,
                                  schema.semanticAttributeNames);
    collectSemanticAttributeNames(*clause.definedness,
                                  schema.semanticAttributeNames);
    schema.matchesOperation =
        [signature = clause.signature,
         operationResults = clause.operationResults,
         importedResultSignature = hasImportedResultSignature(clause),
         expectedOperationName = clause.operationName,
         expectedOperandCount = concreteOperandCount,
         expectedResultIndex = schema.resultIndex,
         operandNames](Operation *operation) {
          if (!operation || operation->getName().getStringRef() !=
                                expectedOperationName ||
              operation->getNumOperands() != expectedOperandCount)
            return false;
          if (!importedResultSignature && operation->getNumResults() != 0 &&
              expectedResultIndex >= operation->getNumResults())
            return false;
          SmallVector<Type, 4> operandTypes;
          operandTypes.append(operation->getOperandTypes().begin(),
                              operation->getOperandTypes().end());
          Type selectedType =
              operation->getNumResults() == 0
                  ? NoneType::get(operation->getContext())
                  : operation->getResult(expectedResultIndex).getType();
          ConstraintContext context{
              /*operation=*/operation,
              /*resultType=*/selectedType,
              /*operandTypes=*/operandTypes,
              /*operandNames=*/operandNames,
              /*attributes=*/{},
              /*compileOn91095=*/false,
              /*allowUnknownOperands=*/false,
              /*allowUnknownAttributes=*/false};
          context.resultIndex = expectedResultIndex;
          context.expectedOperandCount = expectedOperandCount;
          if (evaluateConstraint(*signature, context) !=
              ConstraintTruth::True)
            return false;
          if (importedResultSignature)
            return true;
          SmallVector<Type, 2> declaredResults = resolveOperationResultTypes(
              operationResults.get(), context);
          return operation->getNumResults() == 0 ||
                 (operation->getNumResults() == declaredResults.size() &&
                  llvm::equal(operation->getResultTypes(), declaredResults));
        };
    if (!clause.candidate) {
      schemas.push_back(std::move(schema));
      continue;
    }
    FormulaPtr constraint = clause.candidate;
    bool compileOn91095 = options.compileOn91095;
    schema.offlineOnly = containsCall(*constraint, "offline_only");
    schema.requiresWitnessedOperation =
        containsCall(*constraint, "witnessed_operation");
    collectAttributeBindings(*constraint, schema.attributeBindings);
    collectOperandEffectRequirements(*constraint, operandNames,
                                     schema.operandEffectRequirements);
    collectImplicitStateDomains(*constraint, schema.implicitStateDomains);
    collectProducedStateDomain(*constraint, schema.producedStateDomain);
    llvm::sort(schema.implicitStateDomains);
    if (clause.explore) {
      ConstraintContext exploreContext{/*operation=*/nullptr,
                                       /*resultType=*/{},
                                       /*operandTypes=*/{},
                                       /*operandNames=*/operandNames,
                                       /*attributes=*/{},
                                       /*compileOn91095=*/compileOn91095,
                                       /*allowUnknownOperands=*/true,
                                       /*allowUnknownAttributes=*/true};
      schema.exploratory =
          evaluateConstraint(*clause.explore, exploreContext) ==
          ConstraintTruth::True;
    }
    schema.canProduceResult =
        [constraint, operandNames, compileOn91095, concreteOperandCount,
         resultIndex = schema.resultIndex](
            Type resultType, DictionaryAttr attributes) {
          ConstraintContext context{/*operation=*/nullptr,
                                    /*resultType=*/resultType,
                                    /*operandTypes=*/{},
                                    /*operandNames=*/operandNames,
                                    /*attributes=*/attributes,
                                    /*compileOn91095=*/compileOn91095,
                                    /*allowUnknownOperands=*/true,
                                    /*allowUnknownAttributes=*/true};
          context.resultIndex = resultIndex;
          context.expectedOperandCount = concreteOperandCount;
          return evaluateConstraint(*constraint, context) !=
                 ConstraintTruth::False;
        };
    schema.mayBeWellTyped =
        [constraint, operandNames, compileOn91095, concreteOperandCount,
         resultIndex = schema.resultIndex](
            Type resultType, ArrayRef<Type> operandTypePrefix,
            DictionaryAttr attributes) {
          if (operandTypePrefix.size() > operandNames.size())
            return false;
          ConstraintContext context{/*operation=*/nullptr,
                                    /*resultType=*/resultType,
                                    /*operandTypes=*/operandTypePrefix,
                                    /*operandNames=*/operandNames,
                                    /*attributes=*/attributes,
                                    /*compileOn91095=*/compileOn91095,
                                    /*allowUnknownOperands=*/true,
                                    /*allowUnknownAttributes=*/true};
          context.resultIndex = resultIndex;
          context.expectedOperandCount = concreteOperandCount;
          return evaluateConstraint(*constraint, context) !=
                 ConstraintTruth::False;
        };
    schema.mayBeWellTypedAssignment =
        [constraint, operandNames, compileOn91095, concreteOperandCount,
         resultIndex = schema.resultIndex](
            Type resultType, ArrayRef<Type> operandTypeAssignment,
            DictionaryAttr attributes) {
          if (operandTypeAssignment.size() != operandNames.size())
            return false;
          ConstraintContext context{/*operation=*/nullptr,
                                    /*resultType=*/resultType,
                                    /*operandTypes=*/operandTypeAssignment,
                                    /*operandNames=*/operandNames,
                                    /*attributes=*/attributes,
                                    /*compileOn91095=*/compileOn91095,
                                    /*allowUnknownOperands=*/true,
                                    /*allowUnknownAttributes=*/true};
          context.resultIndex = resultIndex;
          context.expectedOperandCount = concreteOperandCount;
          return evaluateConstraint(*constraint, context) !=
                 ConstraintTruth::False;
        };
    schema.isWellTyped =
        [constraint, operandNames, compileOn91095, concreteOperandCount,
         resultIndex = schema.resultIndex](
            Type resultType, ArrayRef<Type> operandTypes,
            DictionaryAttr attributes) {
          if (operandTypes.size() != operandNames.size())
            return false;
          ConstraintContext context{/*operation=*/nullptr,
                                    /*resultType=*/resultType,
                                    /*operandTypes=*/operandTypes,
                                    /*operandNames=*/operandNames,
                                    /*attributes=*/attributes,
                                    /*compileOn91095=*/compileOn91095,
                                    /*allowUnknownOperands=*/false,
                                    /*allowUnknownAttributes=*/true};
          context.resultIndex = resultIndex;
          context.expectedOperandCount = concreteOperandCount;
          ConstraintTruth truth = evaluateConstraint(*constraint, context);
          // The legacy online beam calls this before it has copied attributes
          // from a witnessed operation, so a null dictionary keeps Unknown
          // provisionally legal. Offline enumeration always supplies a
          // concrete (possibly empty) dictionary and therefore requires a
          // fully proved typing result.
          return attributes ? truth == ConstraintTruth::True
                            : truth != ConstraintTruth::False;
        };
      FormulaPtr operationResults = clause.operationResults;
      schema.buildOperationResultTypes =
          [operationResults, operandNames, concreteOperandCount,
           resultIndex = schema.resultIndex](
              Type resultType, ArrayRef<Type> operandTypes,
              DictionaryAttr attributes) -> SmallVector<Type, 2> {
        ConstraintContext context{/*operation=*/nullptr,
                                  /*resultType=*/resultType,
                                  /*operandTypes=*/operandTypes,
                                  /*operandNames=*/operandNames,
                                  /*attributes=*/attributes,
                                  /*compileOn91095=*/false,
                                  /*allowUnknownOperands=*/false,
                                  /*allowUnknownAttributes=*/false};
        context.resultIndex = resultIndex;
        context.expectedOperandCount = concreteOperandCount;
        SmallVector<Type, 2> types = resolveOperationResultTypes(
            operationResults.get(), context);
        if (resultIndex >= types.size() || types[resultIndex] != resultType)
          return {};
        return types;
      };
      if (clause.action) {
        FormulaPtr action = clause.action;
        schema.buildModuleAttributePatch =
            [action](DictionaryAttr attributes) {
          return buildModuleAttributePatch(*action, attributes);
        };
        schema.buildCompilerAction = [action](DictionaryAttr attributes) {
          return buildCompilerAction(*action, attributes);
        };
      }
      if (clause.region) {
        FormulaPtr region = clause.region;
        if (region->kind != Formula::Kind::Call ||
            region->spelling != "binary" || region->arguments.size() != 2 ||
            llvm::any_of(region->arguments, [](const FormulaPtr &argument) {
              return argument->kind != Formula::Kind::Identifier;
            }))
          llvm::report_fatal_error(
              "region currently requires binary(combiner, terminator)");
        std::string combiner = region->arguments[0]->spelling;
        std::string terminator = region->arguments[1]->spelling;
        schema.buildRegions =
            [combiner = std::move(combiner),
             terminator = std::move(terminator)](
                Type resultType, ArrayRef<Type> operandTypes,
                DictionaryAttr) -> ArrayAttr {
          if (operandTypes.empty())
            return {};
          Type elementType = operandTypes.front();
          if (auto shaped = dyn_cast<ShapedType>(elementType))
            elementType = shaped.getElementType();
          if (!elementType)
            return {};
          return buildCanonicalBinaryRegion(resultType.getContext(),
                                            elementType, combiner, terminator);
        };
      }
      schemas.push_back(std::move(schema));
    }
  }
  return schemas;
}

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H
#define TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace triton {
namespace cfg {

// The identity of every rule is one bit of the pass rule mask, so the mask
using GraphOptimizationRuleMask = uint32_t;

enum class GraphOptimizationRuleId : GraphOptimizationRuleMask {
  LoadStoreTranspose = 1,
  TransposePointwiseReorder = 2,
  StoreCoalescing = 4,
  // RowCoalescing is a pure-SIMT-only graph rule. Its provider contributes to
  // the same per-function candidate pool as the other TTIR graph rules; the
  // compile-mode guard prevents it from entering non-SIMT searches.
  RowCoalescing = 8,
  // DiagonalMaskRemoval rewrites compute logic rather than memory access, so
  // it is architecture and template independent.
  DiagonalMaskRemoval = 128,
  // ConvertModuloToMask linearizes wrapped tile addresses.  It runs before the
  // memory-access rules so they see the linear form.
  ConvertModuloToMask = 256,
  // Reserved for the NarrowUnsignedTensor Manual-12 baseline while its
  // semantics are migrated into AutoGraphSolver.
  NarrowUnsignedTensor = 512,
  // The following identities run at later IR stages. They share the automatic
  // solver's control plane, but their candidates can only be analyzed and
  // materialized after TritonToLinalg has exposed the required layout/memory
  // facts.
  StridedAxisCoalescing = 16,
  ChunkCoalescing = 32,
  StridedLoadStoreRewrite = 64,
  // Post-TritonToLinalg buffer lifetime/coverage optimization. It shares the
  // solver rule mask even though its materializer runs on Linalg/MemRef IR.
  MergeConcatLoadBuffer = 1024,
  // Enables the generic AutoGraphSolver provider. Individual executable-op
  // schemas carry their own target/type guards inside the solver language;
  // this identity is not tied to a source graph pattern.
  AutoGraphSolver = 2048,
};

inline constexpr const char *kGraphOptimizationRuleMaskAttrName =
    "hacc.auto_graph_solver_rule_mask";
inline constexpr const char *kAutoGraphRuleDatabaseAttrName =
    "hacc.auto_graph_solver_rule_database";
inline constexpr const char *kAutoGraphOnlineBudgetAttrName =
    "hacc.auto_graph_solver_online_budget_ms";
inline constexpr const char *kAutoGraphRewriteBudgetAttrName =
    "hacc.auto_graph_solver_rewrite_budget";
inline constexpr const char *kAutoGraphTarget91095AttrName =
    "hacc.auto_graph_solver_target_91095";

// Independently measurable NarrowUnsigned semantic families owned by the
// automatic solver. The mask exists only for cold-compile experiments and
// per-family performance attribution.
enum class NarrowUnsignedRuleId : uint8_t {
  Select = 1,
  Bitwise = 2,
  ShiftRight = 4,
  CompareEqual = 8,
};

constexpr uint8_t getNarrowUnsignedRuleMask(NarrowUnsignedRuleId rule) {
  return static_cast<uint8_t>(rule);
}

constexpr uint8_t kAllNarrowUnsignedRuleMask =
    getNarrowUnsignedRuleMask(NarrowUnsignedRuleId::Select) |
    getNarrowUnsignedRuleMask(NarrowUnsignedRuleId::Bitwise) |
    getNarrowUnsignedRuleMask(NarrowUnsignedRuleId::ShiftRight) |
    getNarrowUnsignedRuleMask(NarrowUnsignedRuleId::CompareEqual);

constexpr bool isValidNarrowUnsignedRuleMask(uint8_t ruleMask) {
  constexpr uint8_t unknownRuleBits =
      static_cast<uint8_t>(~kAllNarrowUnsignedRuleMask);
  return (ruleMask & unknownRuleBits) == 0;
}

constexpr const char *
getGraphOptimizationRuleName(GraphOptimizationRuleId rule) {
  switch (rule) {
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return "LoadStoreTranspose";
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return "TransposePointwiseReorder";
  case GraphOptimizationRuleId::StoreCoalescing:
    return "StoreCoalescing";
  case GraphOptimizationRuleId::RowCoalescing:
    return "RowCoalescing";
  case GraphOptimizationRuleId::DiagonalMaskRemoval:
    return "DiagonalMaskRemoval";
  case GraphOptimizationRuleId::ConvertModuloToMask:
    return "ConvertModuloToMask";
  case GraphOptimizationRuleId::NarrowUnsignedTensor:
    return "NarrowUnsignedTensor";
  case GraphOptimizationRuleId::StridedAxisCoalescing:
    return "StridedAxisCoalescing";
  case GraphOptimizationRuleId::ChunkCoalescing:
    return "ChunkCoalescing";
  case GraphOptimizationRuleId::StridedLoadStoreRewrite:
    return "StridedLoadStoreRewrite";
  case GraphOptimizationRuleId::MergeConcatLoadBuffer:
    return "MergeConcatLoadBuffer";
  case GraphOptimizationRuleId::AutoGraphSolver:
    return "AutoGraphSolver";
  }
  return "Unknown";
}

constexpr GraphOptimizationRuleMask
getGraphOptimizationRuleMask(GraphOptimizationRuleId rule) {
  return static_cast<GraphOptimizationRuleMask>(rule);
}

constexpr GraphOptimizationRulePhase
getGraphOptimizationRulePhase(GraphOptimizationRuleId rule) {
  switch (rule) {
  case GraphOptimizationRuleId::DiagonalMaskRemoval:
    return GraphOptimizationRulePhase::DiagonalMaskRemoval;
  case GraphOptimizationRuleId::ConvertModuloToMask:
    return GraphOptimizationRulePhase::ConvertModuloToMask;
  case GraphOptimizationRuleId::IndependentAxisTensorize:
    return GraphOptimizationRulePhase::ProgramMapping;
  case GraphOptimizationRuleId::PersistentTaskStripMining:
    return GraphOptimizationRulePhase::PersistentTaskMapping;
  case GraphOptimizationRuleId::LoadStoreTranspose:
    return GraphOptimizationRulePhase::LoadStoreTranspose;
  case GraphOptimizationRuleId::TransposePointwiseReorder:
    return GraphOptimizationRulePhase::TransposePointwiseReorder;
  case GraphOptimizationRuleId::ResidentLoadForwarding:
    return GraphOptimizationRulePhase::ResidentLoadForwarding;
  case GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision:
    return GraphOptimizationRulePhase::IntermediatePrecisionBoundaryElision;
  case GraphOptimizationRuleId::StoreCoveragePlanning:
    return GraphOptimizationRulePhase::StoreCoveragePlanning;
  case GraphOptimizationRuleId::StoreCoalescing:
    return GraphOptimizationRulePhase::StoreCoalescing;
  case GraphOptimizationRuleId::ContiguousBlockAccessFormation:
    return GraphOptimizationRulePhase::ContiguousBlockAccessFormation;
  case GraphOptimizationRuleId::SelectedBlockScan:
    return GraphOptimizationRulePhase::SelectedBlockScan;
  case GraphOptimizationRuleId::QueryRowBatching:
    return GraphOptimizationRulePhase::ProgramMapping;
  case GraphOptimizationRuleId::RowCoalescing:
    return GraphOptimizationRulePhase::RowCoalescing;
  case GraphOptimizationRuleId::StridedAxisCoalescing:
  case GraphOptimizationRuleId::ChunkCoalescing:
  case GraphOptimizationRuleId::StridedLoadStoreRewrite:
    return GraphOptimizationRulePhase::Compatibility;
  }
  return GraphOptimizationRulePhase::Compatibility;
}

constexpr std::array<GraphOptimizationRuleId, 17>
    kGraphOptimizationRuleRegistry = {
        GraphOptimizationRuleId::LoadStoreTranspose,
        GraphOptimizationRuleId::TransposePointwiseReorder,
        GraphOptimizationRuleId::StoreCoalescing,
        GraphOptimizationRuleId::RowCoalescing,
        GraphOptimizationRuleId::DiagonalMaskRemoval,
        GraphOptimizationRuleId::ConvertModuloToMask,
        GraphOptimizationRuleId::StridedAxisCoalescing,
        GraphOptimizationRuleId::ChunkCoalescing,
        GraphOptimizationRuleId::StridedLoadStoreRewrite,
        GraphOptimizationRuleId::IndependentAxisTensorize,
        GraphOptimizationRuleId::PersistentTaskStripMining,
        GraphOptimizationRuleId::ResidentLoadForwarding,
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision,
        GraphOptimizationRuleId::StoreCoveragePlanning,
        GraphOptimizationRuleId::ContiguousBlockAccessFormation,
        GraphOptimizationRuleId::SelectedBlockScan,
        GraphOptimizationRuleId::QueryRowBatching,
};

constexpr bool hasUniqueSingleBitGraphOptimizationRuleIds() {
  for (std::size_t index = 0; index < kGraphOptimizationRuleRegistry.size();
       ++index) {
    const GraphOptimizationRuleMask mask =
        getGraphOptimizationRuleMask(kGraphOptimizationRuleRegistry[index]);
    if (mask == 0 || (mask & (mask - 1)) != 0)
      return false;
    for (std::size_t other = index + 1;
         other < kGraphOptimizationRuleRegistry.size(); ++other) {
      if (mask ==
          getGraphOptimizationRuleMask(kGraphOptimizationRuleRegistry[other]))
        return false;
    }
  }
  return true;
}

static_assert(hasUniqueSingleBitGraphOptimizationRuleIds(),
              "GraphOptimizationRuleId values must be unique single bits");

constexpr GraphOptimizationRuleMask kLegacyGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::LoadStoreTranspose) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::TransposePointwiseReorder) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::StoreCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::RowCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::DiagonalMaskRemoval) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ConvertModuloToMask) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::NarrowUnsignedTensor) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedAxisCoalescing) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::ChunkCoalescing) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StridedLoadStoreRewrite) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::MergeConcatLoadBuffer) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::AutoGraphSolver);

constexpr GraphOptimizationRuleMask kKnownGraphOptimizationRuleMask =
    kLegacyGraphOptimizationRuleMask |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IndependentAxisTensorize) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::PersistentTaskStripMining) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ResidentLoadForwarding) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StoreCoveragePlanning) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ContiguousBlockAccessFormation) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::SelectedBlockScan) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::QueryRowBatching);

constexpr GraphOptimizationRuleMask kDefaultEligibleGraphOptimizationRuleMask =
    kLegacyGraphOptimizationRuleMask |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IndependentAxisTensorize) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::PersistentTaskStripMining);

// Fixed default-off: these bits are known and can be enabled through the rule
// mask, but they are not part of the default bundle.  RFC #2686 keeps
// SelectedBlockScanRule and QueryRowBatchingRule opt-in for that reason: the
// scan rewrite reorders the floating-point reduction (the fwd scan tile is
// wider than one block) and the row lift merges R query rows into one program,
// and both pay off only for some shapes.
constexpr GraphOptimizationRuleMask kFixedDefaultOffGraphOptimizationRuleMask =
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ResidentLoadForwarding) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::IntermediatePrecisionBoundaryElision) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::StoreCoveragePlanning) |
    getGraphOptimizationRuleMask(
        GraphOptimizationRuleId::ContiguousBlockAccessFormation) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::SelectedBlockScan) |
    getGraphOptimizationRuleMask(GraphOptimizationRuleId::QueryRowBatching);

constexpr GraphOptimizationRuleMask kDefaultGraphOptimizationRuleMask =
    kDefaultEligibleGraphOptimizationRuleMask;

constexpr GraphOptimizationRuleMask kAllGraphOptimizationRuleMask =
    kKnownGraphOptimizationRuleMask;

static_assert(kLegacyGraphOptimizationRuleMask == 511,
              "legacy graph optimization rule mask is an ABI contract");
static_assert(kDefaultGraphOptimizationRuleMask ==
                  kDefaultEligibleGraphOptimizationRuleMask,
              "the default set must stay equal to the default-eligible set");
static_assert(
    kDefaultGraphOptimizationRuleMask == 3071,
    "default graph rule mask stays 3071: the #2686 SelectedBlockScan and "
    "QueryRowBatching bits are fixed default-off (RFC #2686)");
static_assert(kKnownGraphOptimizationRuleMask ==
                  (kDefaultEligibleGraphOptimizationRuleMask |
                   kFixedDefaultOffGraphOptimizationRuleMask),
              "known graph rule identities must retain their scope partition");
static_assert((kDefaultEligibleGraphOptimizationRuleMask &
               kFixedDefaultOffGraphOptimizationRuleMask) == 0,
              "fixed-default-off rules must not become default candidates");
static_assert((kDefaultGraphOptimizationRuleMask &
               ~kDefaultEligibleGraphOptimizationRuleMask) == 0,
              "default graph rules must stay within the approved candidates");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::ResidentLoadForwarding)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::StoreCoveragePlanning)),
              "resident forwarding must run before store planning");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::StoreCoveragePlanning)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::ContiguousBlockAccessFormation)),
              "store planning must run before access formation");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::ConvertModuloToMask)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::SelectedBlockScan)),
              "modulo-to-mask must run before the selected-block scan");
static_assert(static_cast<unsigned>(getGraphOptimizationRulePhase(
                  GraphOptimizationRuleId::SelectedBlockScan)) <
                  static_cast<unsigned>(getGraphOptimizationRulePhase(
                      GraphOptimizationRuleId::IndependentAxisTensorize)),
              "selected-block scan must run before program mapping");

constexpr bool
isValidGraphOptimizationRuleMask(GraphOptimizationRuleMask ruleMask) {
  constexpr GraphOptimizationRuleMask unknownRuleBits =
      ~kKnownGraphOptimizationRuleMask;
  return (ruleMask & unknownRuleBits) == 0;
}

// GraphOptimize records the validated mask on the module so the later
// TritonToLinalg and Linalg/MemRef solver stages make decisions from the same
// experiment configuration. A module that did not run GraphOptimize retains
// the historical all-rules-enabled behavior.
inline uint16_t getModuleGraphOptimizationRuleMask(ModuleOp moduleOp) {
  auto maskAttr = moduleOp->getAttrOfType<IntegerAttr>(
      kGraphOptimizationRuleMaskAttrName);
  if (!maskAttr)
    return kAllGraphOptimizationRuleMask;
  const uint16_t mask = static_cast<uint16_t>(maskAttr.getInt());
  return isValidGraphOptimizationRuleMask(mask) ? mask : 0;
}

inline bool isGraphOptimizationRuleEnabled(ModuleOp moduleOp,
                                           GraphOptimizationRuleId rule) {
  return (getModuleGraphOptimizationRuleMask(moduleOp) &
          getGraphOptimizationRuleMask(rule)) != 0;
}

struct GraphOptimizationOptions {
  // A zero mask intentionally disables every automatic graph-optimization
  // stage, including the later layout/memory and buffer materializers.
  uint16_t enabledRuleMask = kAllGraphOptimizationRuleMask;
  uint8_t enabledNarrowUnsignedRuleMask = kAllNarrowUnsignedRuleMask;
  unsigned maxRewritesPerFunction = 64;
  unsigned ubCapacityBytes = 0;
  unsigned mappingUBCapacityBytes = 0;
  unsigned storeCoalescingUBBudgetBytes = 0;
  unsigned ubSafetyPercent = 80;
  unsigned reservedUBBytes = 0;
  // Cube on-chip buffer capacities for program-mapping resource checks.
  // Zero capacities keep the L0A/L0B/L0C candidate checks disabled.
  unsigned mappingL0ACapacityBytes = 0;
  unsigned mappingL0BCapacityBytes = 0;
  unsigned mappingL0CCapacityBytes = 0;
  unsigned l0SafetyPercent = 80;
  // Physical AI-core blocks the launcher schedules (AIV for pure-SIMT kernels,
  // AIC otherwise).  The row-count policy keeps one full wave after R-way
  // coalescing; zero disables the gridX/R >= coreNum check.
  unsigned coreNum = 0;
  // RowCoalescing changes the launch grid and is valid only for
  // compile_mode="simt_only".  Keep the source selector rather than a
  // second derived force flag so every consumer follows one mode contract.
  std::string compileMode = "simd_simt_template";
  bool compileOn91095 = false;
  // Per-function online AGS budget. Zero disables the wall-time gate. Offline
  // database lookup should use a small finite value in production/CI.
  unsigned maxAutoGraphOnlineMilliseconds = 0;
  // Versioned artifact emitted by triton-ags-rulegen. Production AGS performs
  // canonical lookup only. An empty path leaves AGS disabled instead of
  // silently moving the offline candidate search into user compilation.
  std::string autoGraphRuleDatabase;
  // Test-only compatibility path for the original in-process POC. Production
  // compilation must resolve a RuleDB and perform lookup only; it must never
  // turn a missing artifact into an unbounded online candidate search.
  bool allowOnlineSemanticEnumeration = false;
};

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options = {});

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_H

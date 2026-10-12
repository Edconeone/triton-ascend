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

#include "TritonToGraph/GraphOptimizationContext.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/Passes.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "Utils/Utils.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

namespace mlir {
namespace triton {
namespace cfg {
#define GEN_PASS_DEF_GRAPHOPTIMIZE
#include "ascend/include/TritonToGraph/Passes.h.inc"
} // namespace cfg
} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {
namespace cfg {
namespace {

using ProgramOrderMap = llvm::DenseMap<Operation *, unsigned>;

// AscendSemantic emits this private prefix through the common create_assert
// API.  Materializing the AGS provenance attribute here keeps the marker and
// all of its interpretation inside the Ascend backend.
constexpr llvm::StringLiteral kAutoOverflowAssertPrefix =
    "__triton_ascend_auto_overflow__: ";

void materializeAutoOverflowAssertMarkers(ModuleOp module) {
  MLIRContext *context = module.getContext();
  module.walk([&](triton::AssertOp assertOp) {
    auto message = dyn_cast<StringAttr>(assertOp.getMessageAttr());
    if (!message || !message.getValue().starts_with(kAutoOverflowAssertPrefix))
      return;

    assertOp->setAttr("tt.auto_overflow_assert",
                      UnitAttr::get(context));
    assertOp->setAttr(
        "message",
        StringAttr::get(context,
                        message.getValue().drop_front(
                            kAutoOverflowAssertPrefix.size())));
  });
}

ProgramOrderMap buildProgramOrderMap(triton::FuncOp function) {
  ProgramOrderMap programOrder;
  unsigned nextOrder = 0;
  function.walk(
      [&](Operation *operation) { programOrder[operation] = nextOrder++; });
  return programOrder;
}

unsigned getProgramOrder(const RewritePlan &plan,
                         const ProgramOrderMap &programOrder) {
  Operation *anchor = plan.getAnchor();
  if (!anchor)
    return std::numeric_limits<unsigned>::max();

  auto it = programOrder.find(anchor);
  if (it == programOrder.end())
    return std::numeric_limits<unsigned>::max();
  return it->second;
}

bool isRuleEnabled(GraphOptimizationRuleMask ruleMask,
                   GraphOptimizationRuleId ruleId) {
  return (ruleMask & getGraphOptimizationRuleMask(ruleId)) != 0;
}

constexpr bool requiresPostRewriteCleanup(GraphOptimizationRuleId ruleId) {
  return ruleId == GraphOptimizationRuleId::IndependentAxisTensorize ||
         ruleId == GraphOptimizationRuleId::PersistentTaskStripMining ||
         ruleId == GraphOptimizationRuleId::SelectedBlockScan ||
         ruleId == GraphOptimizationRuleId::QueryRowBatching;
}

// Keep the pre-graph canonicalization limited to operations that its two
// patterns can rewrite. A module-wide greedy driver also folds and erases
// unrelated dead operations before graph analyses observe them. Some graph
// rules intentionally use that visibility to reject incompatible IR.
bool isNarrowUnsignedTensorCandidate(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() < 2 ||
      !isa<arith::SelectOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
           arith::ShRSIOp, arith::ShRUIOp, arith::CmpIOp>(op))
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (isa<arith::SelectOp>(op))
    type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return type && type.getElementType().isInteger(64);
}

bool isFoldHistogramParkingCandidate(Operation *op) {
  auto sub = dyn_cast<arith::SubIOp>(op);
  return sub && isa_and_nonnull<HistogramOp>(sub.getLhs().getDefiningOp());
}

constexpr bool isPlanHigherPriority(unsigned lhsBenefit, unsigned lhsOrder,
                                    GraphOptimizationRuleId lhsRuleId,
                                    unsigned rhsBenefit, unsigned rhsOrder,
                                    GraphOptimizationRuleId rhsRuleId) {
  if (lhsBenefit != rhsBenefit)
    return lhsBenefit > rhsBenefit;
  if (lhsOrder != rhsOrder)
    return lhsOrder < rhsOrder;
  return getGraphOptimizationRuleMask(lhsRuleId) <
         getGraphOptimizationRuleMask(rhsRuleId);
}

class GraphOptimizePass final
    : public impl::GraphOptimizeBase<GraphOptimizePass> {
public:
  GraphOptimizePass() = default;

  explicit GraphOptimizePass(const GraphOptimizationOptions &options) {
    this->ruleMask = options.enabledRuleMask;
    this->narrowUnsignedRuleMask = options.enabledNarrowUnsignedRuleMask;
    this->maxRewritesPerFunction = options.maxRewritesPerFunction;
    this->ubCapacityBytes = options.ubCapacityBytes;
    this->mappingUBCapacityBytes = options.mappingUBCapacityBytes;
    this->storeCoalescingUBBudgetBytes = options.storeCoalescingUBBudgetBytes;
    this->ubSafetyPercent = options.ubSafetyPercent;
    this->reservedUBBytes = options.reservedUBBytes;
    this->mappingL0ACapacityBytes = options.mappingL0ACapacityBytes;
    this->mappingL0BCapacityBytes = options.mappingL0BCapacityBytes;
    this->mappingL0CCapacityBytes = options.mappingL0CCapacityBytes;
    this->l0SafetyPercent = options.l0SafetyPercent;
    this->coreNum = options.coreNum;
    this->compileMode = options.compileMode;
    this->compileOn91095 = options.compileOn91095;
    this->maxAutoGraphOnlineMilliseconds =
        options.maxAutoGraphOnlineMilliseconds;
    this->autoGraphRuleDatabase = options.autoGraphRuleDatabase;
    this->allowOnlineSemanticEnumeration =
        options.allowOnlineSemanticEnumeration;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, arith::ArithDialect,
                    triton::TritonDialect,
                    triton::ascend::TritonAscendDialect>();
  }

  void runOnOperation() override;

private:
  LogicalResult getStableOptions(GraphOptimizationOptions &options);
  LogicalResult runStructuralCleanup();
};

LogicalResult GraphOptimizePass::runStructuralCleanup() {
  PassManager cleanup(&getContext(), getOperation().getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return runPipeline(cleanup, getOperation());
}

LogicalResult
GraphOptimizePass::getStableOptions(GraphOptimizationOptions &options) {
  const uint64_t cliRuleMask = this->ruleMask;
  const uint64_t cliNarrowUnsignedRuleMask = this->narrowUnsignedRuleMask;
  const uint64_t cliMaxRewrites = this->maxRewritesPerFunction;
  const uint64_t cliUBCapacityBytes = this->ubCapacityBytes;
  const uint64_t cliMaxAutoGraphOnlineMilliseconds =
      this->maxAutoGraphOnlineMilliseconds;

  if (cliRuleMask > std::numeric_limits<GraphOptimizationRuleMask>::max() ||
      !isValidGraphOptimizationRuleMask(
          static_cast<GraphOptimizationRuleMask>(cliRuleMask))) {
    getOperation().emitError()
        << "graph-optimize rule-mask contains unknown or out-of-range bits: "
        << cliRuleMask;
    return failure();
  }

  if (cliMaxRewrites > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize max-rewrites-per-function is out of range: "
        << cliMaxRewrites;
    return failure();
  }

  if (cliNarrowUnsignedRuleMask > std::numeric_limits<uint8_t>::max() ||
      !isValidNarrowUnsignedRuleMask(
          static_cast<uint8_t>(cliNarrowUnsignedRuleMask))) {
    getOperation().emitError()
        << "graph-optimize narrow-unsigned-rule-mask contains unknown or "
           "out-of-range bits: "
        << cliNarrowUnsignedRuleMask;
    return failure();
  }

  if (cliUBCapacityBytes > std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize ub-capacity-bytes is out of range: "
        << cliUBCapacityBytes;
    return failure();
  }

  if (cliMaxAutoGraphOnlineMilliseconds >
      std::numeric_limits<unsigned>::max()) {
    getOperation().emitError()
        << "graph-optimize max-auto-graph-online-ms is out of range: "
        << cliMaxAutoGraphOnlineMilliseconds;
    return failure();
  }

  if (!triton::ascend::parseCompileMode(this->compileMode)) {
    getOperation().emitError()
        << "graph-optimize compile-mode is invalid: " << this->compileMode;
    return failure();
  }

  options.enabledRuleMask = static_cast<uint16_t>(cliRuleMask);
  options.enabledNarrowUnsignedRuleMask =
      static_cast<uint8_t>(cliNarrowUnsignedRuleMask);
  options.maxRewritesPerFunction = static_cast<unsigned>(cliMaxRewrites);
  options.ubCapacityBytes =
      static_cast<unsigned>(effectiveStoreCoalescingUBBudget);
  options.mappingUBCapacityBytes =
      static_cast<unsigned>(effectiveMappingUBCapacity);
  options.storeCoalescingUBBudgetBytes =
      static_cast<unsigned>(effectiveStoreCoalescingUBBudget);
  options.ubSafetyPercent = static_cast<unsigned>(cliUBSafetyPercent);
  options.reservedUBBytes = static_cast<unsigned>(cliReservedUBBytes);
  options.mappingL0ACapacityBytes =
      static_cast<unsigned>(cliMappingL0ACapacityBytes);
  options.mappingL0BCapacityBytes =
      static_cast<unsigned>(cliMappingL0BCapacityBytes);
  options.mappingL0CCapacityBytes =
      static_cast<unsigned>(cliMappingL0CCapacityBytes);
  options.l0SafetyPercent = static_cast<unsigned>(cliL0SafetyPercent);
  options.coreNum = static_cast<unsigned>(cliCoreNum);
  options.compileMode = this->compileMode;
  options.compileOn91095 = this->compileOn91095;
  options.maxAutoGraphOnlineMilliseconds =
      static_cast<unsigned>(cliMaxAutoGraphOnlineMilliseconds);
  options.autoGraphRuleDatabase = this->autoGraphRuleDatabase;
  options.allowOnlineSemanticEnumeration =
      this->allowOnlineSemanticEnumeration;
  return success();
}

void GraphOptimizePass::runOnOperation() {
  GraphOptimizationOptions options;
  if (failed(getStableOptions(options))) {
    signalPassFailure();
    return;
  }

  ModuleOp module = getOperation();
  module->setAttr(
      kGraphOptimizationRuleMaskAttrName,
      IntegerAttr::get(IntegerType::get(module.getContext(), 16),
                       options.enabledRuleMask));
  Builder moduleBuilder(module.getContext());
  if (!options.autoGraphRuleDatabase.empty())
    module->setAttr(kAutoGraphRuleDatabaseAttrName,
                    moduleBuilder.getStringAttr(
                        options.autoGraphRuleDatabase));
  module->setAttr(kAutoGraphOnlineBudgetAttrName,
                  moduleBuilder.getI64IntegerAttr(
                      options.maxAutoGraphOnlineMilliseconds));
  module->setAttr(kAutoGraphRewriteBudgetAttrName,
                  moduleBuilder.getI64IntegerAttr(
                      options.maxRewritesPerFunction));
  module->setAttr(kAutoGraphTarget91095AttrName,
                  moduleBuilder.getBoolAttr(options.compileOn91095));

  materializeAutoOverflowAssertMarkers(module);

  SmallVector<std::unique_ptr<GraphOptimizationProvider>> ownedProviders;
  populateBuiltinGraphOptimizationProviders(options, ownedProviders);

  SmallVector<GraphOptimizationProvider *> enabledProviders;
  for (const std::unique_ptr<GraphOptimizationProvider> &provider :
       ownedProviders)
    if (provider)
      enabledProviders.push_back(provider.get());

  for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
    ResourceSnapshot resources = ResourceSnapshot::fromExplicit(
        options.mappingUBCapacityBytes, options.ubSafetyPercent,
        options.reservedUBBytes, options.mappingL0ACapacityBytes,
        options.mappingL0BCapacityBytes, options.mappingL0CCapacityBytes,
        options.l0SafetyPercent);
    // The physical core count comes only from the pass option (the backend
    // injects it per kernel execution mode).  The snapshot stores it so the
    // row-count policy reads it from there and the rules need no separate
    // query interface.
    // The physical core count comes from the pass option (the backend injects
    // it per kernel execution mode); the snapshot carries it so the row-count
    // policy reads one resource interface instead of a rule-private query.
    resources.coreNum = options.coreNum;
    GraphOptimizationContext context(function, resources);
    unsigned rewriteCount = 0;

    while (rewriteCount < options.maxRewritesPerFunction) {
      ProgramOrderMap programOrder = buildProgramOrderMap(function);
      SmallVector<std::unique_ptr<RewritePlan>> plans;

      // Every enabled domain contributes alternatives to one candidate pool.
      // The driver chooses one globally best valid plan for this IR epoch,
      // applies it transactionally, invalidates facts, and solves again.
      for (GraphOptimizationProvider *provider : enabledProviders) {
        if (failed(context.ensure(provider->getAnalysisRequirements()))) {
          function.emitError() << "graph-optimize failed to build analyses";
          signalPassFailure();
          return;
        }
        if (failed(provider->findCandidates(context, plans))) {
          function.emitError()
              << "graph-optimize failed to discover rewrite candidates";
          signalPassFailure();
          return;
        }
      }

      plans.erase(
          std::remove_if(
              plans.begin(), plans.end(),
              [&options](const std::unique_ptr<RewritePlan> &plan) {
                return !plan || !isRuleEnabled(options.enabledRuleMask,
                                                plan->getRuleId());
              }),
          plans.end());
      if (plans.empty())
        break;

      std::stable_sort(
          plans.begin(), plans.end(),
          [&programOrder](const std::unique_ptr<RewritePlan> &lhs,
                          const std::unique_ptr<RewritePlan> &rhs) {
            if (lhs->getBenefit() != rhs->getBenefit())
              return lhs->getBenefit() > rhs->getBenefit();

            const unsigned lhsOrder = getProgramOrder(*lhs, programOrder);
            const unsigned rhsOrder = getProgramOrder(*rhs, programOrder);
            if (lhsOrder != rhsOrder)
              return lhsOrder < rhsOrder;

            return static_cast<unsigned>(lhs->getRuleId()) <
                   static_cast<unsigned>(rhs->getRuleId());
          });

      std::unique_ptr<RewritePlan> selectedPlan;
      for (std::unique_ptr<RewritePlan> &plan : plans) {
        if (plan->getCreationEpoch() != context.getEpoch())
          continue;
        if (failed(plan->revalidate(context))) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "[" DEBUG_TYPE "] dropped stale graph optimization rule "
              << static_cast<unsigned>(plan->getRuleId()) << " ("
              << getGraphOptimizationRuleName(plan->getRuleId()) << ")\n");
          continue;
        }
        selectedPlan = std::move(plan);
        break;
      }
      if (!selectedPlan)
        break;

      const GraphOptimizationRuleId appliedRuleId = selectedPlan->getRuleId();
      IRRewriter rewriter(&getContext());
      if (failed(selectedPlan->apply(rewriter))) {
        selectedPlan.reset();
        plans.clear();
        function.emitError() << "graph-optimize failed to apply rewrite";
        signalPassFailure();
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] applied graph optimization rule "
                 << static_cast<unsigned>(appliedRuleId) << " ("
                 << getGraphOptimizationRuleName(appliedRuleId) << ")\n");

      // Plans can retain pointers into analysis results, so destroy all of
      // them before invalidating the context for the next IR epoch.
      selectedPlan.reset();
      plans.clear();
      context.invalidate();
      ++rewriteCount;
    }

    // The frontend attaches these only as proof inputs for the whole-kernel
    // AGS key.  They must not leak into TritonAdapter or target IR after this
    // pass has consumed them.
    function->removeAttr("ags.launch_grid_bound_axes");
    function->removeAttr("ags.launch_grid_bound_arguments");
    function->removeAttr("ags.launch_grid_bound_axis");
    function->removeAttr("ags.launch_grid_bound_argument");
  }
}

} // namespace

void populateBuiltinGraphOptimizationProviders(
    const GraphOptimizationOptions &options,
    SmallVectorImpl<std::unique_ptr<GraphOptimizationProvider>> &providers) {
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::AutoGraphSolver)) {
    providers.push_back(createAutoGraphSolverAdapter(options));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::DiagonalMaskRemoval)) {
    providers.push_back(createDiagonalMaskRemovalRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::ConvertModuloToMask)) {
    providers.push_back(createConvertModuloToMaskRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::IndependentAxisTensorize)) {
    rules.push_back(
        createIndependentAxisTensorizeRule(options.independentAxisTensorize));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::PersistentTaskStripMining)) {
    rules.push_back(
        createPersistentTaskStripMiningRule(options.persistentTaskStripMining));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::LoadStoreTranspose)) {
    providers.push_back(createLoadStoreTransposeRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::TransposePointwiseReorder)) {
    providers.push_back(createTransposePointwiseReorderRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::StoreCoalescing)) {
    providers.push_back(createStoreCoalescingRule(options.ubCapacityBytes));
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::SelectedBlockScan)) {
    rules.push_back(createSelectedBlockScanRule());
  }
  if (isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::QueryRowBatching)) {
    rules.push_back(createQueryRowBatchingRule());
  }
  const auto compileMode =
      triton::ascend::parseCompileMode(options.compileMode);
  if (compileMode && *compileMode == triton::ascend::CompileMode::SimtOnly &&
      isRuleEnabled(options.enabledRuleMask,
                    GraphOptimizationRuleId::RowCoalescing)) {
    providers.push_back(createRowCoalescingRule());
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createGraphOptimizePass(GraphOptimizationOptions options) {
  return std::make_unique<GraphOptimizePass>(options);
}

} // namespace cfg
} // namespace triton
} // namespace mlir

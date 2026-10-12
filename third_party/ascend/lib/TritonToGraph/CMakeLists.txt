set(TT_OP_SEMANTICS_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/BuiltinTTOpSemantics.ttsem")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${TT_OP_SEMANTICS_FILE}")
file(READ "${TT_OP_SEMANTICS_FILE}" TT_OP_SEMANTICS_SOURCE)
configure_file(BuiltinTTOpSemantics.inc.in
               "${CMAKE_CURRENT_BINARY_DIR}/BuiltinTTOpSemantics.inc" @ONLY)

add_triton_library(TritonGraphUniverse
  GraphUniverseSpec.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
)

add_triton_library(TritonGraphSemantics
  AutoGraphSolver.cpp
  CanonicalRegion.cpp
  TTOpSemanticsDSL.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  TritonIR
  TritonGraphUniverse
  LLVMSupport
)

target_include_directories(TritonGraphSemantics PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
)

add_triton_library(TritonGraphOperationCatalog
  GraphOperationCatalog.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TritonGraphSemantics
  TritonGraphUniverse
  LLVMSupport
)

add_triton_library(TritonGraphEnumeration
  CanonicalGraphKey.cpp
  EquivalenceClassStore.cpp
  OfflineEquivalenceProver.cpp
  OfflineGraphEnumerator.cpp
  ParetoExtractor.cpp
  RuleDatabase.cpp
  SemanticFingerprint.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  TritonGraphSemantics
  TritonGraphUniverse
  LLVMSupport
)

add_triton_library(TritonGraphStageCostModel
  StageModelGraphCostModel.cpp

  LINK_LIBS PUBLIC
  AscendModelIR
  AscendModelTransforms
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  TritonGraphEnumeration
  TritonGraphSemantics
  TritonIR
  TritonGPUIR
  LLVMSupport
)

# The cost-model subdirectories are configured by the Ascend parent after the
# TritonToGraph libraries. Keep the bridge's source and generated include
# paths explicit instead of relying on directory-scoped include propagation.
target_include_directories(TritonGraphStageCostModel PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../costmodel/include"
  "${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include"
  "${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel"
)

add_triton_library(TritonToGraph
  AliasAnalysis.cpp
  AutoGraphSolverPipeline.cpp
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
  DataflowGraph.cpp
  EntryArgPointerAliasAnalysis.cpp
  GraphOptimizationContext.cpp
  ResourceCostModel.cpp
  GraphOptimizationPass.cpp
  LayoutMemoryCompatibilityPass.cpp
  MemorySsaBuilder.cpp
  MergeConcatLoadBuffer.cpp
  PermutationAnalysis.cpp
  Passes.cpp
  RowLifting.cpp
  LegacyMemoryAccess/ChunkCoalescing.cpp
  LegacyMemoryAccess/RowCoalescing.cpp
  LegacyMemoryAccess/StridedAxisCoalescing.cpp
  LegacyMemoryAccess/StridedLoadStoreRewrite.cpp
  AutoGraphSolverAdapter.cpp
  Rules/ConvertModuloToMaskRule.cpp
  Rules/ReservedGraphOptimizationRule.cpp
  Rules/DiagonalMaskRemovalRule.cpp
  Rules/IndependentAxisTensorizeRule.cpp
  Rules/PersistentTaskStripMiningRule.cpp
  Rules/ResidentLoadForwardingRule.cpp
  Rules/IntermediatePrecisionBoundaryElisionRule.cpp
  Rules/StoreCoveragePlanningRule.cpp
  Rules/ContiguousBlockAccessFormationRule.cpp
  Rules/LoadStoreTransposeRule.cpp
  Rules/RowCoalescingRule.cpp
  Rules/DotRowCoalescing.cpp
  Rules/QueryRowBatchingRule.cpp
  Rules/SelectedBlockScanBuild.cpp
  Rules/SelectedBlockScanMatch.cpp
  Rules/SelectedBlockScanRule.cpp
  Rules/TransposePointwiseReorderRule.cpp
  Rules/StoreCoalescingRule.cpp

  DEPENDS
  TritonToGraphPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRHACCUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  MLIRTritonNPUUtils
  TritonGraphOperationCatalog
  TritonGraphEnumeration
  TritonGraphSemantics
  TritonGraphUniverse
  TritonAscendIR
  TritonIR
  TritonMemoryAccessAnalysis
  TritonToStructured
  TritonTransforms
)

target_include_directories(TritonToGraph PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
)

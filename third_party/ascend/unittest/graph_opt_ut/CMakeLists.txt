include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME GraphUniverseSpec
  SRCS
    CanonicalRegionTest.cpp
    CanonicalGraphKeyTest.cpp
    EquivalenceClassStoreTest.cpp
    GraphUniverseSpecTest.cpp
    IndexLayoutSemanticsTest.cpp
    OfflineGraphEnumeratorTest.cpp
    OnlineRuleDatabasePipelineTest.cpp
    ParetoExtractorTest.cpp
    RuleDatabaseTest.cpp
    SemanticEquivalenceOracleTest.cpp
    StageModelGraphCostModelTest.cpp
    TTOpSemanticsDSLTest.cpp
    Z3SemanticSolverTest.cpp
  LIBS
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    TritonGraphOperationCatalog
    TritonGraphEnumeration
    TritonGraphSemantics
    TritonGraphStageCostModel
    TritonGraphUniverse
    TritonToGraph
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

target_compile_definitions(GraphUniverseSpec PRIVATE
  AGS_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Compile the optional SMT adapter only into its tests.  Using
# add_triton_library here would register it in the global Triton library set
# and accidentally add libz3 to the online compiler runtime.
find_path(AGS_Z3_INCLUDE_DIR z3++.h)
find_library(AGS_Z3_LIBRARY NAMES z3)
if(AGS_Z3_INCLUDE_DIR AND AGS_Z3_LIBRARY)
  target_sources(GraphUniverseSpec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/TritonToGraph/Z3SemanticSolver.cpp)
  target_include_directories(GraphUniverseSpec PRIVATE
    "${AGS_Z3_INCLUDE_DIR}")
  target_link_libraries(GraphUniverseSpec PRIVATE "${AGS_Z3_LIBRARY}")
  target_compile_definitions(GraphUniverseSpec PRIVATE
    TRITON_GRAPH_HAVE_Z3=1)
endif()

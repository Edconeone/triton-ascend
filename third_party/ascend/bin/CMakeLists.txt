get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# triton-mlir-opt: Triton-Ascend optimizer combining MLIR and BishengIR dialects
add_llvm_executable(triton-mlir-opt triton-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-mlir-opt)

target_link_libraries(triton-mlir-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

mlir_check_all_link_libraries(triton-mlir-opt)

# triton-ags-rulegen: offline AutoGraphSolver universe explorer and rule DB
# builder. The first milestone validates and fingerprints GraphUniverseSpec;
# later milestones add enumeration, proof, extraction and database emission.
add_llvm_executable(triton-ags-rulegen triton-ags-rulegen.cpp
  PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-ags-rulegen)

target_link_libraries(triton-ags-rulegen PRIVATE
  ${triton_libs}
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  TritonGraphOperationCatalog
  TritonGraphEnumeration
  TritonGraphSemantics
  TritonGraphStageCostModel
  TritonGraphUniverse
  TritonAscendIR
  TritonGPUIR
  TritonIR
  LLVMSupport
)

# The bin directory is configured before TritonToGraph libraries in the
# top-level project, so discover and compile the small offline adapter here as
# well instead of relying on target creation order.
find_path(AGS_Z3_INCLUDE_DIR z3++.h)
find_library(AGS_Z3_LIBRARY NAMES z3)
if(AGS_Z3_INCLUDE_DIR AND AGS_Z3_LIBRARY)
  target_sources(triton-ags-rulegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TritonToGraph/Z3SemanticSolver.cpp)
  target_include_directories(triton-ags-rulegen PRIVATE
    "${AGS_Z3_INCLUDE_DIR}")
  target_link_libraries(triton-ags-rulegen PRIVATE "${AGS_Z3_LIBRARY}")
  target_compile_definitions(triton-ags-rulegen PRIVATE
    TRITON_GRAPH_HAVE_Z3=1)
endif()

# Every production RuleDB shard must carry one identical target-cost identity
# so the databases can be merged safely.  This configuration belongs to the
# production targets below and must not disappear when TRITON_BUILD_UT is OFF.
# The generic Pareto extractor calls StageModel only for e-classes that actually
# contain a cost choice.
set(AGS_ASCEND_950DT_STAGE_MODEL_CONFIG
  ${CMAKE_SOURCE_DIR}/third_party/ascend/costmodel/configs/ascend_davidv100.json)
set(AGS_ASCEND_950DT_STAGE_MODEL_ARGS
  --stage-model
  --stage-model-hardware-config=${AGS_ASCEND_950DT_STAGE_MODEL_CONFIG})

if(TRITON_BUILD_UT)

  # Automatically expands every `dialect.*` selector from the declarative
  # semantics registry.  Completeness additionally requires at least one
  # well-typed graph for every selected OP, so adding a .ttsem definition with
  # a missing type/attribute domain fails this test without editing C++.
  add_test(
    NAME AGSRulegenAllSemanticsReachability
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinAllSemanticsSmokeUniverse.json
      --enumerate
      --schema-reachability-only
      --max-enumerated-graphs=1000000
      --max-rulegen-wall-ms=120000
      --max-rulegen-observed-rss-bytes=4294967296
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-all-semantics-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-all-semantics-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-all-semantics-enumeration.json)
  set_tests_properties(AGSRulegenAllSemanticsReachability PROPERTIES
    TIMEOUT 120
    LABELS "ags;offline-rulegen;all-semantics")

  add_test(
    NAME AGSRulegenScalarClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinScalarEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-scalar-ci-resources.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-scalar-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-scalar-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-scalar-ci.agsdb)
  set_tests_properties(AGSRulegenScalarClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen")

  add_test(
    NAME AGSRulegenMultiResultClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinMultiResultEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-multi-result-ci.agsdb)
  set_tests_properties(AGSRulegenMultiResultClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen;multi-result")

  add_test(
    NAME AGSRulegenRegionClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinRegionEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=32
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-region-ci.agsdb)
  set_tests_properties(AGSRulegenRegionClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen;region")

  add_test(
    NAME AGSRulegenMemoryReadClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinMemoryEnumerationUniverse.json
      --enumerate
      --quotient-equivalent-operands
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=10000
      --max-rule-db-bytes=8388608
      --max-rulegen-wall-ms=120000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-resources.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-ci.agsdb)
  set_tests_properties(AGSRulegenMemoryReadClosedUniverse PROPERTIES
    TIMEOUT 120
    LABELS "ags;offline-rulegen;memory")

  add_test(
    NAME AGSRulegenAtomicStateClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinAtomicEnumerationUniverse.json
      --enumerate
      --quotient-equivalent-operands
      --fingerprint
      --fingerprint-models=64
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=120000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-resources.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-ci.agsdb)
  set_tests_properties(AGSRulegenAtomicStateClosedUniverse PROPERTIES
    TIMEOUT 120
    LABELS "ags;offline-rulegen;memory;atomic")

  add_test(
    NAME AGSRulegenLayoutClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinLayoutEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-layout-ci.agsdb)
  set_tests_properties(AGSRulegenLayoutClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen;layout")

  add_test(
    NAME AGSRulegenIndexClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinIndexEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=1000
      --max-rule-db-bytes=1048576
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-index-ci.agsdb)
  set_tests_properties(AGSRulegenIndexClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen;index")

  add_test(
    NAME AGSRulegenLaunchStateClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinLaunchEnumerationUniverse.json
      --enumerate
      --quotient-equivalent-operands
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=2000
      --max-rule-db-bytes=4194304
      --max-rulegen-wall-ms=60000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-ci.agsdb)
  set_tests_properties(AGSRulegenLaunchStateClosedUniverse PROPERTIES
    TIMEOUT 60
    LABELS "ags;offline-rulegen;launch")

  add_test(
    NAME AGSRulegenDynamicSliceFillClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinDynamicBufferEnumerationUniverse.json
      --enumerate
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=10000
      --max-rule-db-bytes=33554432
      --max-rulegen-wall-ms=120000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-dynamic-buffer-ci.agsdb)
  set_tests_properties(AGSRulegenDynamicSliceFillClosedUniverse PROPERTIES
    TIMEOUT 120
    LABELS "ags;offline-rulegen;buffer;dynamic-shape")

  add_test(
    NAME AGSRulegenBufferStateClosedUniverse
    COMMAND $<TARGET_FILE:triton-ags-rulegen>
      --universe
      ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinBufferStateEnumerationUniverse.json
      --enumerate
      --quotient-equivalent-operands
      --fingerprint
      --fingerprint-models=16
      --prove
      --max-enumerated-graphs=2000
      --max-rule-db-bytes=16777216
      --max-rulegen-wall-ms=120000
      --max-rulegen-observed-rss-bytes=2147483648
      --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-resources.json
      --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-coverage.json
      --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-enumeration.json
      --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-fingerprints.json
      --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-proofs.json
      --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-eclasses.json
      --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci-pareto.json
      --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-ci.agsdb)
  set_tests_properties(AGSRulegenBufferStateClosedUniverse PROPERTIES
    TIMEOUT 120
    LABELS "ags;offline-rulegen;buffer")
endif()

# Explicit, non-default target for a broad, bounded depth-1 value audit. It is
# kept out of normal developer builds because completeness, rather than a
# silent graph cap, is the contract: hitting the graph/time/RSS/DB budget fails
# the target and leaves the stage JSON files for the gap report. Cross-family
# depth-2 closure is independently built by the exact 64-way production
# partition target; raising this monolithic universe's depth would merely
# recreate the unbounded Cartesian product that partitioning is meant to avoid.
add_custom_target(ags-large-universe-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinGraphUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=32
    --fingerprint-jobs=32
    --prove
    --max-enumerated-graphs=1000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=1800000
    --max-rulegen-observed-rss-bytes=8589934592
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-large.agsdb
  DEPENDS triton-ags-rulegen
  USES_TERMINAL
  COMMENT "Run the bounded-complete AGS large-universe audit")

# Non-default denominator audit. Unlike schema reachability, this checks every
# registered operation in the TTIR-stage configured dialects (arith/math/tt),
# including operations that do not yet have a .ttsem definition. TTG and the
# Linalg/Buffer stage execute after GraphOptimize and have independent finite
# universes; mixing their complete dialect catalogs into this denominator
# would turn a pipeline boundary into a false semantic gap.
add_custom_target(ags-full-catalog-coverage-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinAllSemanticsSmokeUniverse.json
    --require-complete-catalog
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-full-catalog-coverage.json
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-full-catalog-resources.json
  DEPENDS triton-ags-rulegen
  USES_TERMINAL
  COMMENT "Audit every registered TTIR-stage OP against declarative AGS semantics")

# Rank-1 linear masked-memory candidate closure.  The wrapped-address source
# graph is absent from this universe; C04 is accepted only when its full
# MemoryState denotation independently normalizes to one of these cheaper
# stateful candidates.
add_custom_target(ags-linear-masked-memory-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinLinearMaskedMemoryEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --fingerprint-jobs=32
    --prove
    --z3-proof-timeout-ms=100
    --z3-proof-retry-timeout-ms=10000
    --z3-proof-bitblast-timeout-ms=0
    --max-enumerated-graphs=10000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=7200000
    --max-rulegen-observed-rss-bytes=34359738368
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-proofs.json
    --resolved-proof-cache-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-proof-cache.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-modulo-mask.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-modulo-mask.mlir
    --check-prefix=DB-RANK1
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover a rank-1 masked linear memory candidate and run C04 blind acceptance")

# The same semantic family at rank 2, kept in its own finite type/shape shard
# so bottom-up closure does not pay the Cartesian product of unrelated ranks.
add_custom_target(ags-rank2-linear-masked-memory-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinRank2LinearMaskedMemoryEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --fingerprint-jobs=32
    --prove
    --z3-proof-timeout-ms=100
    --z3-proof-retry-timeout-ms=10000
    --z3-proof-bitblast-timeout-ms=0
    --max-enumerated-graphs=10000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=7200000
    --max-rulegen-observed-rss-bytes=34359738368
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-proofs.json
    --resolved-proof-cache-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-proof-cache.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-modulo-mask.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-modulo-mask.mlir
    --check-prefix=DB-RANK2
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover a rank-2 masked linear memory candidate and run C04 blind acceptance")

# C07 is intentionally an audit target rather than a handwritten matcher or a
# normal-build test.  The universe contains only operation semantics and
# finite domains; success means the generic enumerator/prover discovered the
# split-store and wide-store equivalence without being told either topology.
add_custom_target(ags-store-coalescing-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinStoreCoalescingEnumerationUniverse.json
    --enumerate
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-store-coalescing.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-store-coalescing.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and certify the C07 store-coalescing equivalence")

# C11 blind acceptance.  The source topology below is not provided to
# rulegen: the generic enumerator independently constructs TTIR pointer graphs
# and Ascend indirect-memory graphs from their declarative semantics, proves
# the common MemoryState denotation, and lets the cost model select the smaller
# target form.
add_custom_target(ags-indirect-memory-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinIndirectMemoryEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-indirect-memory.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-indirect-memory.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and certify target indirect memory lowering without a matcher")

# Rank-1 stride lowering is a second blind C11 gate.  The expanded TTIR
# address/mask topology is only present in the FileCheck input below; rulegen
# receives the foundational arithmetic and rectangular-address semantics.
# Higher ranks reuse the same two semantic combinators and are type-checked by
# the rank 1..5 schema unit test instead of introducing rank-specific C++.
add_custom_target(ags-stride-memory-rank1-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinStrideMemoryRank1EnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=10000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=7200000
    --max-rulegen-observed-rss-bytes=34359738368
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-stride-memory.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-stride-memory.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and certify rank-1 strided memory lowering without a matcher")

# Generic whole-kernel launch partition audit.  No Row/Axis/Chunk topology is
# encoded here: the finite language contains program-id arithmetic, memory
# transitions and the proof-only launch action.  The normalizer proves only a
# complete residue partition; the cost model then compares amortized work per
# logical program and the online path commits graph+module action atomically.
add_custom_target(ags-launch-partition-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinLaunchPartitionEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-launch-partition.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-launch-partition.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover, prove, cost, and apply an exact launch partition")

add_custom_target(ags-guarded-launch-partition-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinGuardedLaunchPartitionEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=10000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=7200000
    --max-rulegen-observed-rss-bytes=34359738368
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-guarded-launch-partition.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-guarded-launch-partition.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and certify a guarded tail launch partition")

# End-to-end BufferState audit: enumerate and prove the value equivalence from
# declarative stage semantics, compile it into a RuleDB, then exercise the
# generic stage importer/materializer with both a private positive case and an
# escaping-buffer rejection.  The source topology is not an input to rulegen.
add_custom_target(ags-buffer-transaction-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinBufferStateEnumerationUniverse.json
    --enumerate
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=1000
    --max-rule-db-bytes=33554432
    --max-rulegen-wall-ms=120000
    --max-rulegen-observed-rss-bytes=2147483648
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-buffer-copy.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit.agsdb"
    --merge-concat-load-buffer
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-buffer-copy.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover, certify, and apply a private BufferState transaction")

# Static C12 blind acceptance.  The universe contains only types, finite slice
# attributes and declarative OP semantics.  The two-buffer source topology and
# one-buffer candidate topology are both discovered by generic enumeration.
add_custom_target(ags-buffer-merge-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinBufferMergeEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-buffer-merge.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge.agsdb"
    --merge-concat-load-buffer
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-buffer-merge.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover, certify, and apply the static C12 buffer merge")

# Exercise the whole stateful pipeline for an unused atomic result: discover
# the MemoryState equivalence offline, compile it into the database, match the
# operation-root canonical graph online, and materialize the replacement as a
# transaction.  No atomic topology is encoded in C++.
add_custom_target(ags-atomic-transaction-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/graph_opt_ut/testdata/universes/BuiltinAtomicEnumerationUniverse.json
    --enumerate
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=20000
    --max-rule-db-bytes=33554432
    --max-rulegen-wall-ms=120000
    --max-rulegen-observed-rss-bytes=2147483648
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-atomic.mlir
    "--graph-optimize=rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-atomic.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-atomic-audit-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and certify an atomic MemoryState transaction")

# Bounded library of inexpensive integer candidates. Complex online source
# graphs are normalized once and query this database by semantic key; rulegen
# does not need to enumerate every larger source topology in order to offer a
# certified smaller representative. The exact max-node layer is partitioned
# after one shared complete prefix because its serial Cartesian product exceeds
# the one-hour production gate; aggregate validation still requires the union
# of every shard before the C01 blind test can run.
add_custom_target(ags-integer-value-candidate-audit
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/.codex-bench/ags_partitioned_rule_db_build.py
    --rulegen=$<TARGET_FILE:triton-ags-rulegen>
    --completeness-reporter=${CMAKE_SOURCE_DIR}/.codex-bench/ags_completeness_report.py
    --partition-reporter=${CMAKE_SOURCE_DIR}/.codex-bench/ags_partition_completeness_report.py
    --universe=${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinIntegerValueEnumerationUniverse.json
    --partition-count=256
    # Fingerprinting retains all model signatures until proof construction.
    # 256 simultaneous C01 shards exceed the 950DT container's 364 GiB cgroup
    # even though every shard remains far below its independent 16 GiB gate.
    # Two 128-worker waves preserve the exact partition family and hard shard
    # budgets while leaving headroom for the compiler, page cache and zstd.
    --jobs=128
    --compress-intermediates
    --compression-jobs=1
    --merge-batch-size=4
    --rulegen-extra-arg=--stage-model
    --rulegen-extra-arg=--stage-model-hardware-config=${AGS_ASCEND_950DT_STAGE_MODEL_CONFIG}
    --rulegen-extra-arg=--fingerprint-jobs=1
    --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    --output-prefix=ags-integer-value
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value.agsdb
    --aggregate-output=${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value-partitions.json
    --fingerprint-models=16
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-narrow-unsigned.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 narrow-unsigned-rule-mask=0 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-narrow-unsigned.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Build the bounded integer-value candidate library and run C01 blind acceptance")

# Generic reduction/scan/histogram candidate library.  The source correction
# and diagonal topologies remain visible only to the blind tests; the universe
# contains primitive OP semantics and small candidate DAGs.
add_custom_target(ags-reduction-candidate-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinReductionCandidateEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction.agsdb
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/.codex-bench/ags_stage_cost_coverage_audit.py
    --eclasses=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-eclasses.json
    --pareto=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-pareto.json
    --require-operation=tt.histogram
    --output=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-stage-cost-audit.json
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-histogram.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-histogram-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-histogram.mlir
    --check-prefix=AGS
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-histogram-optimized.mlir
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-diagonal.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-diagonal-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-diagonal.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction-diagonal-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Build reduction candidates and run C02/C03 blind acceptance")

# Generic closure of rank-2 index permutations with the registered unary
# floating-point elementwise language.  This is one semantic-family shard, not
# a TransposePointwiseReorder matcher: every listed pointwise operation is
# explored on both sides of every finite permutation in the universe.
add_custom_target(ags-pointwise-permutation-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinPointwisePermutationEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-transpose-pointwise.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-transpose-pointwise.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover pointwise/layout permutation equalities from OP semantics")

# Joint Value/Index/Memory closure for loads, stores and rank-2 permutations.
# The same generic enumerator explores every legal operand combination; this
# shard exists to keep the stateful Cartesian product auditable and bounded.
add_custom_target(ags-memory-permutation-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinMemoryPermutationEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover load/store/layout equalities from OP semantics")

# Symbolic masked load/store permutations at the production C06 tile. The
# finite universe contains no fused-SwiGLU topology: it proves rank-2 memory
# equalities over typed pointer/value/mask holes, and the online matcher binds
# those holes to arbitrary already-computed TTIR subgraphs.
add_custom_target(ags-masked-memory-permutation-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinMaskedMemoryPermutationEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    --check-prefix=DB-C06-TEMPLATE
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and bind masked rank-2 memory permutation templates")

# Exhaustive closure of one finite rank-2 affine memory family.  The universe
# contains only OP semantics, types, shapes and attributes; the C06 topology is
# supplied afterwards as a blind acceptance input.  Unlike the small
# load/trans round-trip shard above, this gate requires a complete MemoryState
# rewrite and exercises the generic innermost-stride cost dimension.
add_custom_target(ags-rank2-memory-layout-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinRank2MemoryLayoutEnumerationUniverse.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --prove
    --max-enumerated-graphs=20000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=7200000
    --max-rulegen-observed-rss-bytes=34359738368
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout.agsdb
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-load-store-transpose.mlir
    --check-prefix=DB-C06
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Discover and blind-test a complete rank-2 affine memory layout rewrite")

# A production, exhaustive depth-1 layer over every constructible declarative
# semantic.  Family shards below/above explore deeper graphs, but this shard
# prevents an OP from being present only in a reachability smoke test and never
# entering the shipped RuleDB.  The online input is deliberately not part of
# the universe and exercises semantic-normal-form lookup rather than an exact
# topology entry.
add_custom_target(ags-production-all-ops-depth1-audit
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    ${AGS_ASCEND_950DT_STAGE_MODEL_ARGS}
    --universe
    ${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinProductionAllOpsDepth1Universe.json
    --enumerate
    --quotient-equivalent-operands
    --fingerprint
    --fingerprint-models=16
    --fingerprint-jobs=32
    --prove
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=1073741824
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
    --resource-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-resources.json
    --coverage-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-coverage.json
    --enumeration-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-enumeration.json
    --fingerprint-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-fingerprints.json
    --proof-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-proofs.json
    --eclass-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-eclasses.json
    --pareto-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-pareto.json
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1.agsdb
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/.codex-bench/ags_completeness_report.py
    --coverage=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-coverage.json
    --enumeration=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-enumeration.json
    --fingerprints=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-fingerprints.json
    --proofs=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-proofs.json
    --eclasses=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-eclasses.json
    --pareto=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-pareto.json
    --rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1.agsdb
    --resources=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-resources.json
    --output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-completeness.json
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-catalog-identity.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-catalog-identity.mlir
    --check-prefix=CATALOG
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Exhaustively compile every constructible OP semantic at depth one")

# Cross-family depth-2 closure over every constructible declarative semantic.
# The final layer is split by a stable candidate-prefix hash. One standalone
# depth-1 build emits a hash-bound prefix for the depth-2 partition family;
# every shard re-denotes the proof-certified prefix frontier and enumerates
# only its final layer. Rule databases are merged hierarchically with bounded
# fan-in so the 64-way closure never materializes every uncompressed shard at
# once.
# The aggregate report rejects missing/duplicate indices, divergent prefixes,
# unrealized schema unions and any resource/completeness failure before merge.
add_custom_target(ags-production-all-ops-depth2-audit
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/.codex-bench/ags_partitioned_rule_db_build.py
    --rulegen=$<TARGET_FILE:triton-ags-rulegen>
    --completeness-reporter=${CMAKE_SOURCE_DIR}/.codex-bench/ags_completeness_report.py
    --partition-reporter=${CMAKE_SOURCE_DIR}/.codex-bench/ags_partition_completeness_report.py
    --universe=${CMAKE_SOURCE_DIR}/third_party/ascend/configs/ags/universes/BuiltinProductionAllOpsDepth2Universe.json
    --partition-count=64
    --jobs=8
    --compress-intermediates
    --compression-jobs=2
    --merge-batch-size=4
    --rulegen-extra-arg=--stage-model
    --rulegen-extra-arg=--stage-model-hardware-config=${AGS_ASCEND_950DT_STAGE_MODEL_CONFIG}
    --rulegen-extra-arg=--fingerprint-jobs=48
    --output-dir=${CMAKE_CURRENT_BINARY_DIR}
    --output-prefix=ags-production-all-ops-depth2
    --rule-db-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2.agsdb
    --aggregate-output=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2-partitions.json
    --fingerprint-models=16
    --max-enumerated-graphs=5000000
    --max-rule-db-bytes=2147483648
    --max-rulegen-wall-ms=3600000
    --max-rulegen-observed-rss-bytes=17179869184
  COMMAND $<TARGET_FILE:triton-opt>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-catalog-identity.mlir
    "--graph-optimize=compile-on-910-95=true rule-mask=2048 auto-graph-rule-database=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2.agsdb"
    -o ${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2-optimized.mlir
  COMMAND $<TARGET_FILE:FileCheck>
    ${CMAKE_SOURCE_DIR}/third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-auto-solver-catalog-identity.mlir
    --check-prefix=CATALOG
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2-optimized.mlir
  DEPENDS triton-ags-rulegen triton-opt FileCheck
  USES_TERMINAL
  COMMENT "Exhaustively compile the partitioned all-semantics depth-two layer")

# Assemble every Ascend950DT shard into the single immutable artifact consumed
# by the online compiler.  Keep this target explicit: exhaustive discovery is
# an offline release/CI job, not work that belongs on an application compile
# path.  Generic-target CI shards and the atomic audit are intentionally not
# mixed in because RuleDatabase::merge requires one identical target cost
# context for every member of a bundle.
set(AGS_ASCEND_RULE_DB_BUNDLE
  ${CMAKE_CURRENT_BINARY_DIR}/auto_graph_rules.agsdb)
add_custom_target(ags-rule-db-bundle-config-audit
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/.codex-bench/ags_rule_db_bundle_audit.py
    --manifest=${CMAKE_SOURCE_DIR}/.codex-bench/ags_rule_db_bundle_manifest.json
    --source-root=${CMAKE_SOURCE_DIR}
    --output=${CMAKE_CURRENT_BINARY_DIR}/ags-rule-db-bundle-audit.json
  USES_TERMINAL
  COMMENT "Audit AGS RuleDB shard, merge, and blind-test coverage")
add_custom_target(ags-rule-db-bundle
  COMMAND $<TARGET_FILE:triton-ags-rulegen>
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-integer-value.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-reduction.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-linear-masked-memory.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-linear-memory.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-store-coalescing.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-indirect-memory.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-stride-memory-rank1.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-launch-partition.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-guarded-launch.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-audit.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-buffer-merge.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-pointwise-permutation.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-memory-permutation.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-masked-memory-permutation.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-rank2-memory-layout.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth1.agsdb
    --merge-rule-db=${CMAKE_CURRENT_BINARY_DIR}/ags-production-all-ops-depth2.agsdb
    --rule-db-output=${AGS_ASCEND_RULE_DB_BUNDLE}
  DEPENDS
    ags-rule-db-bundle-config-audit
    ags-integer-value-candidate-audit
    ags-reduction-candidate-audit
    ags-linear-masked-memory-audit
    ags-rank2-linear-masked-memory-audit
    ags-store-coalescing-audit
    ags-indirect-memory-audit
    ags-stride-memory-rank1-audit
    ags-launch-partition-audit
    ags-guarded-launch-partition-audit
    ags-buffer-transaction-audit
    ags-buffer-merge-audit
    ags-pointwise-permutation-audit
    ags-memory-permutation-audit
    ags-masked-memory-permutation-audit
    ags-rank2-memory-layout-audit
    ags-production-all-ops-depth1-audit
    ags-production-all-ops-depth2-audit
  USES_TERMINAL
  COMMENT "Build the merged offline Ascend950DT AutoGraphSolver RuleDB")

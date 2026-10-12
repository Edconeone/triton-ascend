//===- triton-ags-rulegen.cpp - Offline AGS database builder -------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "TritonToGraph/EquivalenceClassStore.h"
#include "TritonToGraph/GraphUniverseSpec.h"
#include "TritonToGraph/GraphOperationCatalog.h"
#include "TritonToGraph/OfflineEquivalenceProver.h"
#include "TritonToGraph/OfflineGraphEnumerator.h"
#include "TritonToGraph/ParetoExtractor.h"
#include "TritonToGraph/RuleDatabase.h"
#include "TritonToGraph/SemanticFingerprint.h"
#include "TritonToGraph/StageModelGraphCostModel.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"
#if defined(TRITON_GRAPH_HAVE_Z3)
#include "TritonToGraph/Z3SemanticSolver.h"
#endif

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace mlir::triton::cfg;

namespace {

template <typename Function>
static void parallelForWork(unsigned count, unsigned requestedJobs,
                            Function &&function) {
  const unsigned jobs = std::min(count, std::max(1u, requestedJobs));
  if (jobs <= 1 || count <= 1) {
    for (unsigned index = 0; index < count; ++index)
      function(index);
    return;
  }

  std::atomic<unsigned> next{0};
  std::vector<std::thread> workers;
  workers.reserve(jobs);
  for (unsigned worker = 0; worker < jobs; ++worker) {
    workers.emplace_back([&] {
      while (true) {
        unsigned index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count)
          return;
        function(index);
      }
    });
  }
  for (std::thread &worker : workers)
    worker.join();
}

static uint64_t getObservedRssBytes() {
#if defined(__linux__)
  struct rusage usage {};
  if (::getrusage(RUSAGE_SELF, &usage) == 0)
    return static_cast<uint64_t>(usage.ru_maxrss) * UINT64_C(1024);
#endif
  return static_cast<uint64_t>(llvm::sys::Process::GetMallocUsage());
}

static void releaseUnusedHeapPages() {
#if defined(__GLIBC__)
  // Full-catalog rule generation deliberately has disjoint, memory-heavy
  // phases.  glibc keeps freed pages in its per-thread arenas after the
  // parallel certificate phase, which otherwise makes those dead pages
  // overlap the following e-class store in RSS.  Return them before crossing
  // the phase boundary; this does not change the 16 GiB peak-RSS gate.
  (void)::malloc_trim(0);
#endif
}

llvm::cl::opt<std::string> universePath(
    "universe", llvm::cl::desc("Path to a finite AGS universe JSON file"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::list<std::string> mergeRuleDatabaseInputs(
    "merge-rule-db",
    llvm::cl::desc("Load one compatible .agsdb shard; repeat for every finite "
                   "universe to compile into one deterministic RuleDB"),
    llvm::cl::value_desc("filename"), llvm::cl::ZeroOrMore);

llvm::cl::opt<bool> reproveMergedRuleDatabase(
    "reprove-merged-rule-db",
    llvm::cl::desc(
        "Reconstruct the merged graph catalog and rerun fingerprint/proof "
        "before compiling a final RuleDB; requires --universe and shard "
        "inputs"),
    llvm::cl::init(false));

llvm::cl::opt<bool> semanticQuotientMergedRuleDatabase(
    "semantic-quotient-merged-rule-db",
    llvm::cl::desc(
        "After cross-shard reproof, remove replay-certified duplicate "
        "maximal roots; requires --reprove-merged-rule-db"),
    llvm::cl::init(false));

llvm::cl::opt<bool> dumpCanonical(
    "dump-canonical",
    llvm::cl::desc("Print the canonical universe JSON after validation"),
    llvm::cl::init(false));

llvm::cl::opt<bool> dumpCoverage(
    "dump-coverage",
    llvm::cl::desc("Print the registered-operation semantic coverage report"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> coverageOutput(
    "coverage-output",
    llvm::cl::desc("Write a pure JSON semantic coverage report"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<bool> requireCompleteCatalog(
    "require-complete-catalog",
    llvm::cl::desc("Fail unless every registered operation in the selected "
                   "catalog dialects has complete declarative semantics"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enumerateGraphs(
    "enumerate",
    llvm::cl::desc("Run the current bounded offline typed-DAG enumerator"),
    llvm::cl::init(false));

llvm::cl::opt<unsigned> maxEnumeratedGraphs(
    "max-enumerated-graphs",
    llvm::cl::desc("Safety limit; hitting it makes enumeration incomplete"),
    llvm::cl::init(1000000));

llvm::cl::opt<bool> quotientEquivalentOperands(
    "quotient-equivalent-operands",
    llvm::cl::desc("Retain every candidate graph but expand larger DAGs only "
                   "from proof-certified semantic e-class representatives; "
                   "depth-1 also quotients typed input alpha-renamings"),
    llvm::cl::init(false));

llvm::cl::opt<bool> finalLayerSemanticQuotient(
    "final-layer-semantic-quotient",
    llvm::cl::desc(
        "Proof-safe catalog mode: retain one exact semantic representative "
        "for each final-layer root meaning; graph catalog is compressed"),
    llvm::cl::init(false));

llvm::cl::opt<bool> semanticFrontierCatalogOnly(
    "semantic-frontier-catalog-only",
    llvm::cl::desc(
        "Proof-safe recursive catalog mode: omit pure intermediate graphs "
        "already proved equivalent to an expansion-frontier representative; "
        "reports semantic coverage, not complete syntactic enumeration"),
    llvm::cl::init(false));

llvm::cl::opt<bool> schemaReachabilityOnly(
    "schema-reachability-only",
    llvm::cl::desc(
        "Forward-chain only until every declarative candidate schema has one "
        "well-typed realization; this is not exhaustive enumeration and "
        "cannot emit proof or RuleDB artifacts"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> enumerationOutput(
    "enumeration-output",
    llvm::cl::desc("Write a pure JSON typed-DAG enumeration report"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> enumerationPrefixInput(
    "enumeration-prefix-input",
    llvm::cl::desc("Load a hash-bound complete lower-layer graph prefix and "
                   "enumerate only the remaining layers"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> enumerationPrefixOutput(
    "enumeration-prefix-output",
    llvm::cl::desc("Stream the complete graph prefix below the final layer "
                   "for reuse by compatible exact partitions"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> enumerationPrefixFamilyUniverse(
    "enumeration-prefix-family-universe",
    llvm::cl::desc("Bind a standalone lower-depth prefix build to this "
                   "compatible final-layer partition universe"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<bool> fingerprintGraphs(
    "fingerprint",
    llvm::cl::desc("Compute sound semantic pre-filter buckets after enumeration"),
    llvm::cl::init(false));

llvm::cl::opt<unsigned> fingerprintModels(
    "fingerprint-models",
    llvm::cl::desc("Number of deterministic concrete models per fingerprint"),
    llvm::cl::init(4));

llvm::cl::opt<unsigned> fingerprintJobs(
    "fingerprint-jobs",
    llvm::cl::desc("Maximum concurrent workers for offline denotation and "
                   "concrete-model fingerprint evaluation"),
    llvm::cl::init(1));

llvm::cl::opt<unsigned> counterexampleModels(
    "counterexample-models",
    llvm::cl::desc("Maximum additional deterministic models used to refine "
                   "primary fingerprint collisions before symbolic proof"),
    // Models 0..79 reach the complete GF(5) covering array. The extra sixteen
    // enter its deterministic de-correlation tail, which is needed when more
    // than six independent typed inputs or private buffers share a candidate.
    llvm::cl::init(96));

llvm::cl::opt<std::string> fingerprintOutput(
    "fingerprint-output",
    llvm::cl::desc("Write a pure JSON semantic fingerprint report"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<bool> proveGraphs(
    "prove",
    llvm::cl::desc("Prove or classify every fingerprint candidate pair"),
    llvm::cl::init(false));

llvm::cl::opt<bool> useZ3Proof(
    "z3-proof",
    llvm::cl::desc("Use the offline Z3 bit-vector backend for normalizer "
                   "Unknown results"),
#if defined(TRITON_GRAPH_HAVE_Z3)
    llvm::cl::init(true));
#else
    llvm::cl::init(false));
#endif

llvm::cl::opt<unsigned> z3ProofTimeoutMilliseconds(
    "z3-proof-timeout-ms",
    llvm::cl::desc("Per-query timeout for the offline Z3 proof backend"),
    llvm::cl::init(20));

llvm::cl::opt<unsigned> z3ContextQueryLimit(
    "z3-context-query-limit",
    llvm::cl::desc("Recycle each offline Z3 worker context after this many "
                   "queries to bound retained AST/solver state"),
    llvm::cl::init(4096));

llvm::cl::opt<unsigned> z3ProofRetryTimeoutMilliseconds(
    "z3-proof-retry-timeout-ms",
    llvm::cl::desc("Retry a timed-out offline Z3 query once with this larger "
                   "timeout; zero or a value no larger than the primary "
                   "timeout disables retry"),
    llvm::cl::init(0));

llvm::cl::opt<unsigned> z3ProofBitBlastTimeoutMilliseconds(
    "z3-proof-bitblast-timeout-ms",
    llvm::cl::desc("After timed SMT attempts, retry with an explicit "
                   "simplify/bit-blast/SAT pipeline; zero disables it"),
    llvm::cl::init(0));

llvm::cl::opt<unsigned> z3ProofBitBlastJobs(
    "z3-proof-bitblast-jobs",
    llvm::cl::desc("Maximum concurrent explicit bit-blast/SAT retries"),
    llvm::cl::init(8));

llvm::cl::opt<std::string> proofOutput(
    "proof-output",
    llvm::cl::desc("Write a pure JSON tri-state equivalence proof report"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> resolvedProofCacheInput(
    "resolved-proof-cache-input",
    llvm::cl::desc("Reuse terminal proofs and prioritize the pending Unknown "
                   "frontier from an earlier interrupted proof run"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> resolvedProofCacheOutput(
    "resolved-proof-cache-output",
    llvm::cl::desc("Write restartable terminal results plus an Unknown "
                   "frontier that is scheduling state, not proof evidence"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> eclassInput(
    "eclass-input",
    llvm::cl::desc("Load and incrementally extend a prior AGS e-class store"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> eclassOutput(
    "eclass-output",
    llvm::cl::desc("Write a versioned, replay-validated AGS e-class store"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string> paretoOutput(
    "pareto-output",
    llvm::cl::desc("Write Pareto candidates from the replay-validated e-class "
                   "store; uses explicit structural fallback until a complete "
                   "StageModel evaluator is configured"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<bool> useStageModel(
    "stage-model",
    llvm::cl::desc("Materialize every candidate DAG and run the in-process "
                   "Ascend StageCostModel; incomplete classes fall back as a "
                   "unit to structural cost"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> stageModelHardwareConfig(
    "stage-model-hardware-config",
    llvm::cl::desc("Hardware JSON for the in-process StageCostModel; providing "
                   "it also enables --stage-model"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<bool> requireCompleteStageModel(
    "require-complete-stage-model",
    llvm::cl::desc("Fail rule generation if any equivalence class cannot be "
                   "scored completely by the StageCostModel"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> ruleDatabaseOutput(
    "rule-db-output",
    llvm::cl::desc("Compile replay-validated e-classes and Pareto candidates "
                   "into a versioned .agsdb JSON artifact"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<uint64_t> maxRuleDatabaseBytes(
    "max-rule-db-bytes",
    llvm::cl::desc("Fail if the serialized .agsdb exceeds this budget; zero "
                   "means unlimited"),
    llvm::cl::init(0));

llvm::cl::opt<uint64_t> maxRulegenWallMilliseconds(
    "max-rulegen-wall-ms",
    llvm::cl::desc("Fail at a stage boundary when total rule-generation wall "
                   "time exceeds this budget; zero means unlimited"),
    llvm::cl::init(0));

llvm::cl::opt<uint64_t> maxRulegenObservedRssBytes(
    "max-rulegen-observed-rss-bytes",
    llvm::cl::desc("Fail when RSS sampled at a stage boundary exceeds this "
                   "budget; zero means unlimited"),
    llvm::cl::init(0));

llvm::cl::opt<std::string> resourceOutput(
    "resource-output",
    llvm::cl::desc("Write machine-readable wall-time and observed-RSS samples"),
    llvm::cl::value_desc("filename"), llvm::cl::init(""));

static bool writeReport(llvm::StringRef path, llvm::StringRef report,
                        llvm::StringRef reportName) {
  std::error_code errorCode;
  llvm::raw_fd_ostream output(path, errorCode);
  if (errorCode) {
    llvm::errs() << "cannot write " << reportName << " report '" << path
                 << "': " << errorCode.message() << '\n';
    return false;
  }
  output << report;
  return true;
}

template <typename Writer>
static bool writeStreamingReport(llvm::StringRef path,
                                 llvm::StringRef reportName, Writer &&writer,
                                 uint64_t *writtenBytes = nullptr) {
  std::error_code errorCode;
  llvm::raw_fd_ostream output(path, errorCode);
  if (errorCode) {
    llvm::errs() << "cannot write " << reportName << " report '" << path
                 << "': " << errorCode.message() << '\n';
    return false;
  }
  writer(output);
  output.flush();
  if (output.has_error()) {
    llvm::errs() << "cannot finish " << reportName << " report '" << path
                 << "'\n";
    return false;
  }
  if (writtenBytes)
    *writtenBytes = output.tell();
  return true;
}

static bool writeFingerprintReport(
    llvm::StringRef path,
    const OfflineSemanticFingerprintResult &fingerprints,
    llvm::StringRef universeHash, bool enumerationComplete) {
  std::error_code errorCode;
  llvm::raw_fd_ostream output(path, errorCode);
  if (errorCode) {
    llvm::errs() << "cannot write fingerprint report '" << path
                 << "': " << errorCode.message() << '\n';
    return false;
  }
  fingerprints.writeJSON(output, universeHash, enumerationComplete);
  output.flush();
  if (output.has_error()) {
    llvm::errs() << "cannot finish fingerprint report '" << path << "'\n";
    return false;
  }
  return true;
}

static bool writeProofReport(llvm::StringRef path,
                             const OfflineEquivalenceProofResult &proofs,
                             llvm::StringRef universeHash,
                             bool enumerationComplete,
                             bool fingerprintComplete) {
  std::error_code errorCode;
  llvm::raw_fd_ostream output(path, errorCode);
  if (errorCode) {
    llvm::errs() << "cannot write proof report '" << path
                 << "': " << errorCode.message() << '\n';
    return false;
  }
  proofs.writeJSON(output, universeHash, enumerationComplete,
                   fingerprintComplete);
  output.flush();
  if (output.has_error()) {
    llvm::errs() << "cannot finish proof report '" << path << "'\n";
    return false;
  }
  return true;
}

static bool writeResolvedProofCache(
    llvm::StringRef path, const OfflineResolvedProofCache &cache) {
  return writeStreamingReport(
      path, "resolved proof cache",
      [&](llvm::raw_ostream &output) { cache.writeJSON(output); });
}

// A partitioned RuleDB is not complete merely because every shard proved its
// local pairs. Reconstruct the deduplicated graph catalog and run the generic
// fingerprint/proof/Pareto pipeline once more so cross-shard equivalences are
// visible. This path is deliberately independent of operation names; the only
// semantic source is the versioned TTOpSemantics registry.
static std::optional<RuleDatabase> reproveMergedDatabase(
    const RuleDatabase &merged, mlir::MLIRContext &context,
    llvm::StringRef outputUniverseHash,
    const TTOpSemanticsRegistry &semantics, unsigned fingerprintModelCount,
    unsigned fingerprintJobCount, bool enableZ3, unsigned z3TimeoutMs,
    unsigned z3ContextLimit, unsigned z3RetryTimeoutMs,
    unsigned z3BitBlastTimeoutMs, unsigned z3BitBlastJobs,
    bool enableStageModel, llvm::StringRef stageModelConfig,
    bool semanticQuotient, std::string &error) {
  std::optional<std::vector<OfflineTypedGraph>> reconstructed =
      reconstructOfflineGraphs(merged.graphs, context, error);
  if (!reconstructed)
    return std::nullopt;

  TensorSemanticNormalizer normalizer;
  OfflineSemanticFingerprinter fingerprinter(fingerprintModelCount,
                                              fingerprintJobCount);
  OfflineSemanticFingerprintResult fingerprints =
      fingerprinter.fingerprint(*reconstructed, semantics, normalizer);
  if (!fingerprints.isComplete()) {
    error = "cross-shard reproof produced wildcard semantic fingerprints";
    return std::nullopt;
  }

  const SemanticEquivalenceBackend *proofBackend = nullptr;
#if defined(TRITON_GRAPH_HAVE_Z3)
  std::unique_ptr<Z3SemanticSolver> z3ProofBackend;
  if (enableZ3) {
    z3ProofBackend = std::make_unique<Z3SemanticSolver>(
        z3TimeoutMs, z3ContextLimit, z3RetryTimeoutMs,
        z3BitBlastTimeoutMs, z3BitBlastJobs);
    proofBackend = z3ProofBackend.get();
  }
#else
  if (enableZ3) {
    error = "cross-shard reproof requested Z3 but this build has no Z3";
    return std::nullopt;
  }
#endif
  SemanticEquivalenceOracle oracle(normalizer, proofBackend);
  OfflineEquivalenceProofOptions proofOptions;
  proofOptions.retainRefutedPairs = false;
  proofOptions.retainGraphKeys = false;
  proofOptions.pruneTransitivelyImpliedPairs = true;
  proofOptions.additionalCounterexampleModels = counterexampleModels;
  proofOptions.counterexampleJobs = fingerprintJobCount;
  OfflineEquivalenceProofResult proofs = OfflineEquivalenceProver().prove(
      *reconstructed, fingerprints, oracle, proofOptions);
  if (!proofs.isComplete()) {
    error = "cross-shard semantic proof is incomplete or contains Unknown";
    return std::nullopt;
  }

  std::vector<std::optional<CertifiedSemanticGraph>> certificates(
      fingerprints.denotations.size());
  for (auto [index, denotation] : llvm::enumerate(fingerprints.denotations)) {
    if (!denotation)
      continue;
    std::optional<SemanticEquivalenceProof> reflexive =
        oracle.prove(*denotation, *denotation);
    if (!reflexive || index >= reconstructed->size()) {
      error = "cross-shard reproof failed to certify a graph denotation";
      return std::nullopt;
    }
    certificates[index] = CertifiedSemanticGraph{
        (*reconstructed)[index].canonicalKey,
        getRuleDatabaseSemanticKeyDigest(
            getStableTensorDenotationKey(*denotation)),
        reflexive->getStableHash()};
  }
  std::vector<CertifiedSemanticGraph> certifiedGraphs;
  certifiedGraphs.reserve(certificates.size());
  for (auto &certificate : certificates)
    if (certificate)
      certifiedGraphs.push_back(std::move(*certificate));

  std::optional<PersistentEquivalenceClassStore> store =
      PersistentEquivalenceClassStore::build(
          *reconstructed, proofs, outputUniverseHash,
          getBuiltinTTOpSemanticsStableHash(), /*enumerationComplete=*/true,
          /*fingerprintComplete=*/true, /*base=*/nullptr, error);
  if (!store)
    return std::nullopt;

  GraphCostContext costContext;
  costContext.target = "cross-shard-merged";
  std::unique_ptr<OfflineGraphCostModel> costModel;
  if (enableStageModel)
    costModel = std::make_unique<AscendStageModelGraphCostModel>(
        stageModelConfig.str());
  else
    costModel = std::make_unique<StructuralFallbackCostModel>();
  ParetoExtractionResult pareto = ParetoExtractor().extract(
      *store, *costModel, costContext, store->getStableHash());
  if (pareto.stageModelUnknownReasons.size() != 0 && enableStageModel) {
    error = "cross-shard reproof StageCostModel coverage is incomplete";
    return std::nullopt;
  }
  std::optional<RuleDatabase> database =
      RuleDatabase::compile(*store, pareto, error, certifiedGraphs,
                            store->getStableHash());
  if (!database)
    return std::nullopt;
  if (semanticQuotient) {
    uint64_t prunedGraphs = 0;
    uint64_t prunedEntries = 0;
    std::optional<RuleDatabase> quotiented =
        RuleDatabase::quotientFinalLayerBySemanticKey(
            *database, certifiedGraphs, prunedGraphs, prunedEntries, error);
    if (!quotiented)
      return std::nullopt;
    llvm::outs() << "cross_shard_semantic_quotient_pruned_graphs="
                 << prunedGraphs << '\n'
                 << "cross_shard_semantic_quotient_pruned_entries="
                 << prunedEntries << '\n';
    database = std::move(quotiented);
  }
  llvm::outs() << "cross_shard_reproof_graph_count=" << reconstructed->size()
               << '\n'
               << "cross_shard_reproof_proved_pairs=" << proofs.provedPairs
               << '\n'
               << "cross_shard_reproof_transitive_pairs="
               << proofs.transitivelyProvedPairs << '\n'
               << "cross_shard_reproof_unknown_pairs=" << proofs.unknownPairs
               << '\n'
               << "cross_shard_reproof_eclass_count=" << store->classes.size()
               << '\n'
               << "cross_shard_reproof_rule_db_entries="
               << database->entries.size() << '\n'
               << "cross_shard_reproof_rule_db_templates="
               << database->templates.size() << '\n';
  return database;
}

} // namespace

int main(int argc, char **argv) {
  const auto processStart = std::chrono::steady_clock::now();
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Offline AutoGraphSolver rule database builder\n");

  if (semanticQuotientMergedRuleDatabase && !reproveMergedRuleDatabase) {
    llvm::errs() << "--semantic-quotient-merged-rule-db requires "
                    "--reprove-merged-rule-db\n";
    return 1;
  }

  uint64_t maxObservedRss = 0;
  struct ResourceSample {
    std::string stage;
    uint64_t wallMilliseconds = 0;
    uint64_t observedRssBytes = 0;
  };
  std::vector<ResourceSample> resourceSamples;
  auto writeResourceSamples = [&](bool withinBudget) {
    if (resourceOutput.empty())
      return true;
    llvm::json::Array samples;
    for (const ResourceSample &sample : resourceSamples) {
      llvm::json::Object object;
      object["stage"] = sample.stage;
      object["wall_ms"] = static_cast<int64_t>(sample.wallMilliseconds);
      object["observed_rss_bytes"] =
          static_cast<int64_t>(sample.observedRssBytes);
      samples.push_back(std::move(object));
    }
    llvm::json::Object report;
    report["schema_version"] = 1;
    report["within_budget"] = withinBudget;
    report["max_wall_ms"] =
        static_cast<int64_t>(maxRulegenWallMilliseconds);
    report["max_observed_rss_bytes"] =
        static_cast<int64_t>(maxRulegenObservedRssBytes);
    report["observed_peak_rss_bytes"] =
        static_cast<int64_t>(maxObservedRss);
    report["samples"] = std::move(samples);
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << llvm::formatv("{0:2}\n",
                            llvm::json::Value(std::move(report)));
    return writeReport(resourceOutput, serialized, "resource");
  };
  auto checkResourceBudget = [&](llvm::StringRef stage) {
    uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - processStart)
            .count());
    uint64_t rss = getObservedRssBytes();
    maxObservedRss = std::max(maxObservedRss, rss);
    resourceSamples.push_back({stage.str(), elapsed, rss});
    llvm::outs() << "resource_stage=" << stage << '\n'
                 << "resource_wall_ms=" << elapsed << '\n'
                 << "resource_observed_rss_bytes=" << rss << '\n'
                 << "resource_max_observed_rss_bytes=" << maxObservedRss
                 << '\n';
    if (maxRulegenWallMilliseconds != 0 &&
        elapsed > maxRulegenWallMilliseconds) {
      llvm::errs() << "rule generation exceeds --max-rulegen-wall-ms at "
                   << stage << ": " << elapsed << " > "
                   << maxRulegenWallMilliseconds << '\n';
      (void)writeResourceSamples(false);
      return false;
    }
    if (maxRulegenObservedRssBytes != 0 &&
        maxObservedRss > maxRulegenObservedRssBytes) {
      llvm::errs()
          << "rule generation exceeds --max-rulegen-observed-rss-bytes at "
          << stage << ": " << maxObservedRss << " > "
          << maxRulegenObservedRssBytes << '\n';
      (void)writeResourceSamples(false);
      return false;
    }
    return writeResourceSamples(true);
  };

  if (!mergeRuleDatabaseInputs.empty() && !reproveMergedRuleDatabase) {
    if (!universePath.empty() || enumerateGraphs || fingerprintGraphs ||
        proveGraphs || dumpCanonical || dumpCoverage ||
        !coverageOutput.empty() || !enumerationOutput.empty() ||
        !enumerationPrefixInput.empty() || !enumerationPrefixOutput.empty() ||
        !enumerationPrefixFamilyUniverse.empty() ||
        !fingerprintOutput.empty() || !proofOutput.empty() ||
        !resolvedProofCacheInput.empty() ||
        !resolvedProofCacheOutput.empty() ||
        !eclassInput.empty() || !eclassOutput.empty() ||
        !paretoOutput.empty() || ruleDatabaseOutput.empty()) {
      llvm::errs()
          << "--merge-rule-db is a standalone mode and requires only one "
             "or more input shards plus --rule-db-output\n";
      return 1;
    }
    std::vector<RuleDatabase> shards;
    shards.reserve(mergeRuleDatabaseInputs.size());
    for (llvm::StringRef shardPath : mergeRuleDatabaseInputs) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
          llvm::MemoryBuffer::getFile(shardPath);
      if (!input) {
        llvm::errs() << "cannot read rule database shard '" << shardPath
                     << "': " << input.getError().message() << '\n';
        return 1;
      }
      std::string loadError;
      std::optional<RuleDatabase> shard = RuleDatabase::fromJSON(
          input.get()->getBuffer(), loadError);
      if (!shard) {
        llvm::errs() << "invalid rule database shard '" << shardPath
                     << "': " << loadError << '\n';
        return 1;
      }
      if (shard->semanticsHash != getBuiltinTTOpSemanticsStableHash()) {
        llvm::errs() << "rule database shard '" << shardPath
                     << "' was generated from a different semantics version\n";
        return 1;
      }
      shards.push_back(std::move(*shard));
    }
    std::string mergeError;
    std::optional<RuleDatabase> merged =
        RuleDatabase::merge(shards, mergeError);
    if (!merged) {
      llvm::errs() << "cannot merge rule database shards: " << mergeError
                   << '\n';
      return 1;
    }
    std::string serialized = merged->toJSON();
    if (maxRuleDatabaseBytes != 0 &&
        serialized.size() > maxRuleDatabaseBytes) {
      llvm::errs() << "merged rule database exceeds --max-rule-db-bytes: "
                   << serialized.size() << " > " << maxRuleDatabaseBytes
                   << '\n';
      return 4;
    }
    llvm::outs() << "merged_rule_db_shard_count=" << shards.size() << '\n'
                 << "rule_db_hash=" << merged->getStableHash() << '\n'
                 << "rule_db_graph_count=" << merged->graphs.size() << '\n'
                 << "rule_db_entry_count=" << merged->entries.size() << '\n'
                 << "rule_db_template_count=" << merged->templates.size()
                 << '\n'
                 << "rule_db_semantic_entry_count="
                 << merged->semanticEntries.size()
                 << '\n'
                 << "rule_db_bytes=" << serialized.size() << '\n';
    if (!writeReport(ruleDatabaseOutput, serialized, "rule database"))
      return 1;
    if (!checkResourceBudget("rule-database-merge"))
      return 6;
    return 0;
  }

  if (universePath.empty()) {
    llvm::errs() << "--universe is required outside --merge-rule-db mode\n";
    return 1;
  }

  std::string error;
  std::optional<GraphUniverseSpec> spec =
      GraphUniverseSpec::loadFromFile(universePath, error);
  if (!spec) {
    llvm::errs() << "invalid AGS universe: " << error << '\n';
    return 1;
  }
  std::optional<GraphUniverseSpec> prefixFamilySpec;
  if (!enumerationPrefixFamilyUniverse.empty()) {
    prefixFamilySpec = GraphUniverseSpec::loadFromFile(
        enumerationPrefixFamilyUniverse, error);
    if (!prefixFamilySpec) {
      llvm::errs() << "invalid AGS prefix family universe: " << error << '\n';
      return 1;
    }
    if (enumerationPrefixOutput.empty() ||
        !enumerationPrefixInput.empty() || spec->bounds.maxNodes == 0 ||
        spec->bounds.maxNodes + 1 != prefixFamilySpec->bounds.maxNodes) {
      llvm::errs() << "--enumeration-prefix-family-universe requires a "
                      "standalone prefix output whose max_nodes is exactly "
                      "one below the family universe\n";
      return 1;
    }
    GraphUniverseSpec normalizedPrefix = *spec;
    GraphUniverseSpec normalizedFamily = *prefixFamilySpec;
    normalizedPrefix.name = "ags-prefix-family-compatibility";
    normalizedFamily.name = "ags-prefix-family-compatibility";
    normalizedPrefix.bounds.maxNodes = normalizedFamily.bounds.maxNodes;
    normalizedPrefix.bounds.finalLayerPartitionCount = 1;
    normalizedPrefix.bounds.finalLayerPartitionIndex = 0;
    normalizedFamily.bounds.finalLayerPartitionCount = 1;
    normalizedFamily.bounds.finalLayerPartitionIndex = 0;
    if (normalizedPrefix.getStableHash() !=
        normalizedFamily.getStableHash()) {
      llvm::errs() << "prefix and family universes differ outside name, "
                      "depth, or final-layer partition fields\n";
      return 1;
    }
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  context.getOrLoadDialect<mlir::triton::ascend::TritonAscendDialect>();
  context.getOrLoadDialect<mlir::triton::gpu::TritonGPUDialect>();

  TTOpSemanticsRegistry semantics = createBuiltinTTOpSemanticsRegistry();
  if (!spec->resolveOperationPatterns(semantics.getOperationNames(), error)) {
    llvm::errs() << "invalid AGS universe operation selection: " << error
                 << '\n';
    return 1;
  }
  if (prefixFamilySpec &&
      !prefixFamilySpec->resolveOperationPatterns(
          semantics.getOperationNames(), error)) {
    llvm::errs() << "invalid AGS prefix-family operation selection: "
                 << error << '\n';
    return 1;
  }
  GraphOperationCatalog catalog =
      GraphOperationCatalog::build(context, *spec, semantics);
  if (catalog.unregisteredSelectedCount != 0) {
    llvm::errs() << "invalid AGS universe: " << catalog.unregisteredSelectedCount
                 << " selected operation(s) are not registered\n";
    return 1;
  }

  llvm::outs() << "universe=" << spec->name << '\n'
               << "target=" << spec->target << '\n'
               << "universe_hash=" << spec->getStableHash() << '\n'
               << "operation_count=" << spec->operations.size() << '\n'
               << "semantic_domain_count=" << spec->semanticDomains.size()
               << '\n'
               << "tensor_shape_count=" << spec->tensorShapes.size() << '\n'
               << "attribute_domain_count=" << spec->attributeDomains.size()
               << '\n'
               << "catalog_registered_operation_count="
               << catalog.registeredOperationCount << '\n'
               << "semantic_definition_count="
               << catalog.semanticDefinitionCount << '\n'
               << "orphan_semantic_definition_count="
               << catalog.orphanSemanticDefinitionCount << '\n'
               << "coverage_missing=" << catalog.missingCount << '\n'
               << "coverage_partial=" << catalog.partialCount << '\n'
               << "coverage_complete=" << catalog.completeCount
               << '\n'
               << "searchable_semantics=" << catalog.searchableCount << '\n'
               << "import_only_semantics=" << catalog.importOnlyCount << '\n'
               << "region_only_semantics=" << catalog.regionOnlyCount << '\n'
               << "mixed_search_policy_semantics="
               << catalog.mixedSearchPolicyCount << '\n'
               << "incomplete_search_policy_semantics="
               << catalog.incompleteSearchPolicyCount << '\n'
               << "selected_coverage_missing="
               << catalog.selectedMissingCount << '\n'
               << "selected_coverage_partial="
               << catalog.selectedPartialCount << '\n'
               << "selected_coverage_complete="
               << catalog.selectedCompleteCount
               << '\n';
  if (dumpCanonical)
    llvm::outs() << spec->toCanonicalJSON();
  if (dumpCoverage)
    llvm::outs() << catalog.toJSON(spec->getStableHash());
  if (!coverageOutput.empty()) {
    if (!writeReport(coverageOutput, catalog.toJSON(spec->getStableHash()),
                     "coverage"))
      return 1;
  }
  if (requireCompleteCatalog &&
      (catalog.missingCount != 0 || catalog.partialCount != 0 ||
       catalog.orphanSemanticDefinitionCount != 0)) {
    llvm::errs() << "AGS operation catalog is incomplete: "
                 << catalog.missingCount << " missing, "
                 << catalog.partialCount << " partial, "
                 << catalog.orphanSemanticDefinitionCount
                 << " orphan semantic definitions\n";
    return 7;
  }
  if (!checkResourceBudget("catalog"))
    return 6;

  const bool runEclassStore = !eclassInput.empty() || !eclassOutput.empty() ||
                              !paretoOutput.empty() ||
                              !ruleDatabaseOutput.empty();
  const bool runProof = proveGraphs || !proofOutput.empty() ||
                        !resolvedProofCacheInput.empty() ||
                        !resolvedProofCacheOutput.empty() || runEclassStore;
  const bool runFingerprint =
      fingerprintGraphs || !fingerprintOutput.empty() || runProof;
  if (runFingerprint && !enumerateGraphs && !reproveMergedRuleDatabase) {
    llvm::errs() << "--fingerprint/--prove requires --enumerate\n";
    return 1;
  }
  if (schemaReachabilityOnly && !reproveMergedRuleDatabase &&
      (!enumerateGraphs || runFingerprint)) {
    llvm::errs() << "--schema-reachability-only requires --enumerate and "
                    "cannot be combined with fingerprint/proof/RuleDB output\n";
    return 1;
  }
  if ((!enumerationPrefixInput.empty() ||
       !enumerationPrefixOutput.empty()) &&
      (!enumerateGraphs || schemaReachabilityOnly ||
       (spec->bounds.maxNodes < 2 && !prefixFamilySpec))) {
    llvm::errs() << "enumeration prefix input/output requires exhaustive "
                    "--enumerate and max_nodes >= 2 unless a compatible "
                    "family universe is supplied\n";
    return 1;
  }

  // Coverage is a generation precondition, not merely a diagnostic report.
  // Otherwise a universe that requests (for example) Launch or Buffer could
  // still publish a database whose selected operations never interpreted that
  // domain.  Every command path that creates proof-bearing artifacts therefore
  // fails closed before enumeration.
  if (enumerateGraphs &&
      (catalog.selectedMissingCount != 0 ||
       catalog.selectedPartialCount != 0)) {
    llvm::errs() << "selected AGS operation semantics are incomplete: "
                 << catalog.selectedMissingCount << " missing, "
                 << catalog.selectedPartialCount << " partial\n";
    return 7;
  }

  if (reproveMergedRuleDatabase) {
    if (mergeRuleDatabaseInputs.empty() || ruleDatabaseOutput.empty() ||
        enumerateGraphs || fingerprintGraphs || proveGraphs ||
        !eclassInput.empty() || !eclassOutput.empty() ||
        !paretoOutput.empty() || !proofOutput.empty() ||
        !fingerprintOutput.empty() || !resolvedProofCacheInput.empty() ||
        !resolvedProofCacheOutput.empty()) {
      llvm::errs()
          << "--reprove-merged-rule-db requires shard inputs, --universe, "
             "and --rule-db-output only\n";
      return 1;
    }
    std::vector<RuleDatabase> shards;
    shards.reserve(mergeRuleDatabaseInputs.size());
    for (llvm::StringRef shardPath : mergeRuleDatabaseInputs) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
          llvm::MemoryBuffer::getFile(shardPath);
      if (!input) {
        llvm::errs() << "cannot read rule database shard '" << shardPath
                     << "': " << input.getError().message() << '\n';
        return 1;
      }
      std::string loadError;
      std::optional<RuleDatabase> shard = RuleDatabase::fromJSON(
          input.get()->getBuffer(), loadError);
      if (!shard) {
        llvm::errs() << "invalid rule database shard '" << shardPath
                     << "': " << loadError << '\n';
        return 1;
      }
      if (shard->semanticsHash != getBuiltinTTOpSemanticsStableHash()) {
        llvm::errs() << "rule database shard '" << shardPath
                     << "' was generated from a different semantics version\n";
        return 1;
      }
      shards.push_back(std::move(*shard));
    }
    std::string mergeError;
    std::optional<RuleDatabase> merged =
        RuleDatabase::merge(shards, mergeError);
    if (!merged) {
      llvm::errs() << "cannot merge rule database shards: " << mergeError
                   << '\n';
      return 1;
    }
    std::string reproofError;
    std::optional<RuleDatabase> reproved = reproveMergedDatabase(
        *merged, context, spec->getStableHash(), semantics, fingerprintModels,
        fingerprintJobs,
        useZ3Proof, z3ProofTimeoutMilliseconds, z3ContextQueryLimit,
        z3ProofRetryTimeoutMilliseconds, z3ProofBitBlastTimeoutMilliseconds,
        z3ProofBitBlastJobs,
        useStageModel || !stageModelHardwareConfig.empty(),
        stageModelHardwareConfig, semanticQuotientMergedRuleDatabase,
        reproofError);
    if (!reproved) {
      llvm::errs() << "cannot reprove merged rule database: " << reproofError
                   << '\n';
      return 8;
    }
    std::string serialized = reproved->toJSON();
    if (maxRuleDatabaseBytes != 0 &&
        serialized.size() > maxRuleDatabaseBytes) {
      llvm::errs() << "reproved rule database exceeds --max-rule-db-bytes: "
                   << serialized.size() << " > " << maxRuleDatabaseBytes
                   << '\n';
      return 4;
    }
    llvm::outs() << "reproved_rule_db_shard_count=" << shards.size() << '\n'
                 << "rule_db_hash=" << reproved->getStableHash() << '\n'
                 << "rule_db_graph_count=" << reproved->graphs.size() << '\n'
                 << "rule_db_entry_count=" << reproved->entries.size()
                 << '\n'
                 << "rule_db_template_count=" << reproved->templates.size()
                 << '\n'
                 << "rule_db_semantic_entry_count="
                 << reproved->semanticEntries.size() << '\n'
                 << "rule_db_bytes=" << serialized.size() << '\n';
    if (!writeReport(ruleDatabaseOutput, serialized, "reproved rule database"))
      return 1;
    if (!checkResourceBudget("cross-shard-reproof"))
      return 6;
    return 0;
  }

  if (enumerateGraphs) {
    GraphOptimizationOptions options;
    options.compileOn91095 = spec->target == "ascend910_95" ||
                             spec->target == "ascend950dt";
    llvm::SmallVector<TTIRCandidateOpSchema, 4> schemas =
        createBuiltinTTIRCandidateOpSchemas(options,
                                            spec->bounds.maxOperands);
    OfflineTypeDomain typeDomain = buildOfflineTypeDomain(context, *spec);
    OfflineEnumerationOptions enumerationOptions;
    enumerationOptions.maxGraphs = maxEnumeratedGraphs;
    enumerationOptions.semantics = &semantics;
    TensorSemanticNormalizer quotientNormalizer;
    enumerationOptions.normalizer = &quotientNormalizer;
    enumerationOptions.quotientEquivalentOperands =
        quotientEquivalentOperands;
    enumerationOptions.finalLayerSemanticQuotient =
        finalLayerSemanticQuotient;
    enumerationOptions.semanticFrontierCatalogOnly =
        semanticFrontierCatalogOnly;
    enumerationOptions.alphaNormalizeFinalLayerInputs = !prefixFamilySpec;
    enumerationOptions.semanticValidationJobs = fingerprintJobs;
    enumerationOptions.recordFinalLayerOperandFrontier =
        prefixFamilySpec.has_value();
    enumerationOptions.schemaReachabilityOnly = schemaReachabilityOnly;
    std::optional<OfflineEnumerationPrefixCache> prefixCache;
    if (!enumerationPrefixInput.empty()) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
          llvm::MemoryBuffer::getFile(enumerationPrefixInput);
      if (!input) {
        llvm::errs() << "cannot read enumeration prefix cache '"
                     << enumerationPrefixInput
                     << "': " << input.getError().message() << '\n';
        return 1;
      }
      std::string cacheError;
      prefixCache = OfflineEnumerationPrefixCache::fromJSON(
          input.get()->getBuffer(), context, cacheError,
          maxEnumeratedGraphs);
      if (!prefixCache) {
        llvm::errs() << "invalid enumeration prefix cache '"
                     << enumerationPrefixInput << "': " << cacheError
                     << '\n';
        return 1;
      }
      GraphUniverseSpec partitionFamily = *spec;
      partitionFamily.name = "ags-final-layer-partition-family";
      partitionFamily.bounds.finalLayerPartitionIndex = 0;
      const std::string expectedFamilyHash =
          partitionFamily.getStableHash();
      const std::string expectedSemanticsHash =
          getBuiltinTTOpSemanticsStableHash();
      if (prefixCache->partitionFamilyHash != expectedFamilyHash ||
          prefixCache->semanticsHash != expectedSemanticsHash ||
          prefixCache->maximumOperatorSize + 1 != spec->bounds.maxNodes ||
          prefixCache->quotientEquivalentOperands !=
              quotientEquivalentOperands) {
        llvm::errs()
            << "enumeration prefix cache is incompatible: family="
            << prefixCache->partitionFamilyHash << " expected="
            << expectedFamilyHash << ", semantics="
            << prefixCache->semanticsHash << " expected="
            << expectedSemanticsHash << ", maximum_operator_size="
            << prefixCache->maximumOperatorSize << " expected="
            << (spec->bounds.maxNodes - 1) << ", quotient="
            << prefixCache->quotientEquivalentOperands << " expected="
            << quotientEquivalentOperands << '\n';
        return 1;
      }
      // ru_maxrss records the true parser peak even if the streaming reader
      // has already released its per-record JSON object.  Gate the load phase
      // explicitly; otherwise an oversized cache parser could escape the
      // resource policy before enumeration starts polling it.
      if (!checkResourceBudget("enumeration-prefix-load"))
        return 6;
      enumerationOptions.prefixCache = &*prefixCache;
      enumerationOptions.consumePrefixCache = true;
      const std::string &cacheHash =
          prefixCache->validatedStableHash.empty()
              ? (prefixCache->validatedStableHash =
                     prefixCache->getStableHash())
              : prefixCache->validatedStableHash;
      llvm::outs() << "enumeration_prefix_cache_hash="
                   << cacheHash << '\n'
                   << "enumeration_prefix_graph_count="
                   << prefixCache->graphs.size() << '\n';
    }
    enumerationOptions.shouldStop = [&]() {
      uint64_t elapsed = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - processStart)
              .count());
      if (maxRulegenWallMilliseconds != 0 &&
          elapsed > maxRulegenWallMilliseconds)
        return true;
      if (maxRulegenObservedRssBytes == 0)
        return false;
      uint64_t rss = getObservedRssBytes();
      maxObservedRss = std::max(maxObservedRss, rss);
      return maxObservedRss > maxRulegenObservedRssBytes;
    };
    OfflineTypedGraphEnumerator enumerator(enumerationOptions);
    OfflineEnumerationResult enumeration =
        enumerator.enumerate(*spec, schemas, typeDomain.types);
    enumeration.unsupportedUniverseFeatures.insert(
        enumeration.unsupportedUniverseFeatures.end(),
        typeDomain.unsupportedUniverseFeatures.begin(),
        typeDomain.unsupportedUniverseFeatures.end());

    llvm::outs() << "enumeration_complete="
                 << (enumeration.isComplete() ? "true" : "false") << '\n'
                 << "schema_reachability_complete="
                 << (enumeration.isSchemaReachabilityComplete() ? "true"
                                                                  : "false")
                 << '\n'
                 << "finite_type_count=" << typeDomain.types.size() << '\n'
                 << "enumerated_graph_count=" << enumeration.graphs.size()
                 << '\n'
                 << "enumeration_attempted_candidates="
                 << enumeration.attemptedCandidates << '\n'
                 << "enumeration_rejected_by_type="
                 << enumeration.rejectedByType << '\n'
                 << "enumeration_rejected_by_semantics="
                 << enumeration.rejectedBySemantics << '\n'
                 << "enumeration_semantic_validation_graph_count="
                 << enumeration.semanticValidationGraphCount << '\n'
                 << "enumeration_semantic_validation_jobs="
                 << enumeration.semanticValidationJobs << '\n'
                 << "schema_typing_assignments_tested="
                  << enumeration.schemaTypingAssignmentsTested << '\n'
                 << "enumeration_operand_quotient_pruned="
                 << enumeration.operandQuotientPruned << '\n'
                 << "enumeration_final_layer_semantic_quotient="
                 << (enumeration.finalLayerSemanticQuotient ? "true"
                                                            : "false")
                 << '\n'
                 << "enumeration_semantic_frontier_catalog_only="
                 << (enumeration.semanticFrontierCatalogOnly ? "true"
                                                              : "false")
                 << '\n'
                 << "enumeration_final_layer_semantic_quotient_pruned="
                 << enumeration.finalLayerSemanticQuotientPruned << '\n'
                 << "enumeration_final_layer_abstract_signature_pruned="
                 << enumeration.finalLayerAbstractSignaturePruned << '\n'
                 << "enumeration_final_layer_normalized_structural_pruned="
                 << enumeration.finalLayerNormalizedStructuralPruned << '\n'
                 << "enumeration_semantic_frontier_catalog_pruned="
                 << enumeration.semanticFrontierCatalogPruned << '\n'
                 << "enumeration_semantic_coverage_complete="
                 << (enumeration.isSemanticCoverageComplete() ? "true"
                                                               : "false")
                 << '\n'
                 << "enumeration_saturated_reachability_domains="
                 << enumeration.saturatedReachabilityDomains << '\n'
                 << "enumeration_saturated_reachability_candidates_skipped="
                 << enumeration.saturatedReachabilityCandidatesSkipped << '\n'
                  << "enumeration_input_renaming_quotient_pruned="
                  << enumeration.inputRenamingQuotientPruned << '\n'
                 << "enumeration_reused_prefix_graph_count="
                 << enumeration.reusedPrefixGraphCount << '\n'
                 << "enumeration_reused_prefix_operand_frontier_graph_count="
                 << enumeration.reusedPrefixOperandFrontierGraphCount << '\n'
                 << "enumeration_reused_prefix_cache_hash="
                 << enumeration.reusedPrefixCacheHash << '\n'
                 << "enumeration_hit_graph_limit="
                 << (enumeration.hitGraphLimit ? "true" : "false") << '\n'
                 << "enumeration_interrupted="
                 << (enumeration.interrupted ? "true" : "false") << '\n';
    std::string enumerationJSON =
        enumeration.toJSON(spec->getStableHash());
    if (!enumerationOutput.empty() &&
        !writeReport(enumerationOutput, enumerationJSON, "enumeration"))
      return 1;
    if (!enumerationPrefixOutput.empty()) {
      unsigned maximumPrefixOperatorSize = spec->bounds.maxNodes - 1;
      if (prefixFamilySpec) {
        maximumPrefixOperatorSize = spec->bounds.maxNodes;
        GraphUniverseSpec partitionFamily = *prefixFamilySpec;
        partitionFamily.name = "ags-final-layer-partition-family";
        partitionFamily.bounds.finalLayerPartitionIndex = 0;
        enumeration.finalLayerPartitionFamilyHash =
            partitionFamily.getStableHash();
      }
      std::string cacheError;
      std::error_code outputError;
      llvm::raw_fd_ostream output(enumerationPrefixOutput, outputError);
      if (outputError ||
          !OfflineEnumerationPrefixCache::writeJSONFromEnumeration(
              output, enumeration, getBuiltinTTOpSemanticsStableHash(),
              maximumPrefixOperatorSize, cacheError)) {
        if (!cacheError.empty())
          llvm::errs() << "cannot build enumeration prefix cache: "
                       << cacheError << '\n';
        else
          llvm::errs() << "cannot write enumeration prefix cache '"
                       << enumerationPrefixOutput
                       << "': " << outputError.message() << '\n';
        return 1;
      }
      output.flush();
      if (output.has_error()) {
        llvm::errs() << "cannot finish enumeration prefix cache '"
                     << enumerationPrefixOutput << "'\n";
        return 1;
      }
    }
    if (!checkResourceBudget("enumeration"))
      return 6;

    bool fingerprintComplete = true;
    if (runFingerprint) {
      TensorSemanticNormalizer normalizer;
      OfflineSemanticFingerprinter fingerprinter(fingerprintModels,
                                                  fingerprintJobs);
      OfflineSemanticFingerprintResult fingerprints =
          fingerprinter.fingerprint(enumeration.graphs, semantics, normalizer);
      fingerprintComplete = fingerprints.isComplete();
      llvm::outs() << "fingerprint_complete="
                   << (fingerprintComplete ? "true" : "false") << '\n'
                   << "fingerprinted_graph_count="
                   << fingerprints.getFingerprintedGraphCount() << '\n'
                   << "fingerprint_wildcard_graph_count="
                   << fingerprints.wildcardGraphIds.size() << '\n'
                   << "fingerprint_bucket_count="
                   << fingerprints.buckets.size() << '\n'
                   << "proof_pairs_before_fingerprint="
                   << fingerprints.comparablePairsBefore << '\n'
                   << "proof_pairs_after_fingerprint="
                   << fingerprints.proofCandidatePairsAfter << '\n';
      if (!fingerprintOutput.empty() &&
          !writeFingerprintReport(fingerprintOutput, fingerprints,
                                  spec->getStableHash(),
                                  enumeration.isSemanticCoverageComplete()))
        return 1;
      if (!checkResourceBudget("fingerprint"))
        return 6;

      if (runProof) {
        std::optional<OfflineResolvedProofCache> priorResolvedProofCache;
        if (!resolvedProofCacheInput.empty()) {
          llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
              llvm::MemoryBuffer::getFile(resolvedProofCacheInput);
          if (!input) {
            llvm::errs() << "cannot read resolved proof cache '"
                         << resolvedProofCacheInput
                         << "': " << input.getError().message() << '\n';
            return 1;
          }
          std::string cacheError;
          priorResolvedProofCache = OfflineResolvedProofCache::fromJSON(
              input.get()->getBuffer(), cacheError);
          if (!priorResolvedProofCache ||
              priorResolvedProofCache->universeHash !=
                  spec->getStableHash() ||
              priorResolvedProofCache->graphCount !=
                  enumeration.graphs.size()) {
            llvm::errs() << "invalid resolved proof cache '"
                         << resolvedProofCacheInput << "': ";
            if (!priorResolvedProofCache)
              llvm::errs() << cacheError;
            else
              llvm::errs() << "universe hash or graph count mismatch";
            llvm::errs() << '\n';
            return 1;
          }
        }
        const SemanticEquivalenceBackend *proofBackend = nullptr;
#if defined(TRITON_GRAPH_HAVE_Z3)
        std::unique_ptr<Z3SemanticSolver> z3ProofBackend;
        if (useZ3Proof) {
          z3ProofBackend = std::make_unique<Z3SemanticSolver>(
              z3ProofTimeoutMilliseconds, z3ContextQueryLimit,
              z3ProofRetryTimeoutMilliseconds,
              z3ProofBitBlastTimeoutMilliseconds, z3ProofBitBlastJobs);
          proofBackend = z3ProofBackend.get();
        }
#else
        if (useZ3Proof) {
          llvm::errs() << "--z3-proof was requested, but this build has no "
                          "Z3 development library\n";
          return 1;
        }
#endif
        SemanticEquivalenceOracle oracle(normalizer, proofBackend);
        OfflineEquivalenceProofOptions proofOptions;
        proofOptions.retainRefutedPairs = false;
        proofOptions.retainGraphKeys = false;
        proofOptions.pruneTransitivelyImpliedPairs = true;
        proofOptions.additionalCounterexampleModels = counterexampleModels;
        proofOptions.counterexampleJobs = fingerprintJobs;
        proofOptions.shouldStop = enumerationOptions.shouldStop;
        proofOptions.resolvedProofCache =
            priorResolvedProofCache ? &*priorResolvedProofCache : nullptr;
        OfflineEquivalenceProofResult proofs =
            OfflineEquivalenceProver().prove(enumeration.graphs, fingerprints,
                                             oracle, proofOptions);
        llvm::outs() << "proof_complete="
                     << (proofs.isComplete() ? "true" : "false") << '\n'
                     << "proof_interrupted="
                     << (proofs.interrupted ? "true" : "false") << '\n'
                     << "proof_prefilter_refuted_pair_count="
                     << proofs.prefilterRefutedPairs << '\n'
                     << "proof_candidate_pair_count=" << proofs.candidatePairs
                     << '\n'
                     << "proof_processed_pair_count=" << proofs.processedPairs
                     << '\n'
                     << "proof_proved_pair_count=" << proofs.provedPairs << '\n'
                     << "proof_transitively_proved_pair_count="
                     << proofs.transitivelyProvedPairs << '\n'
                     << "proof_refuted_pair_count=" << proofs.refutedPairs
                     << '\n'
                     << "proof_counterexample_refuted_pair_count="
                     << proofs.counterexampleRefutedPairs << '\n'
                     << "proof_counterexample_model_evaluations="
                     << proofs.counterexampleModelEvaluations << '\n'
                     << "proof_counterexample_model_evaluations_avoided="
                     << proofs.counterexampleModelEvaluationsAvoided << '\n'
                     << "proof_symbolic_premerge_comparisons="
                     << proofs.symbolicPremergeComparisons << '\n'
                     << "proof_symbolic_premerge_merged_groups="
                     << proofs.symbolicPremergeMergedGroups << '\n'
                     << "proof_exact_post_refinement_comparisons="
                     << proofs.exactPostRefinementComparisons << '\n'
                     << "proof_exact_post_refinement_proved="
                     << proofs.exactPostRefinementProved << '\n'
                     << "proof_resolved_cache_hits="
                     << proofs.resolvedProofCacheHits << '\n'
                     << "proof_resolved_cache_rejected="
                     << proofs.resolvedProofCacheRejected << '\n'
                     << "proof_resolved_cache_entry_count="
                     << proofs.resolvedProofCache.pairs.size() << '\n'
                     << "proof_pending_cache_attempts="
                     << proofs.pendingProofCacheAttempts << '\n'
                     << "proof_pending_cache_resolved="
                     << proofs.pendingProofCacheResolved << '\n'
                     << "proof_pending_cache_remaining="
                     << proofs.pendingProofCacheRemaining << '\n'
                     << "proof_pending_cache_rejected="
                     << proofs.pendingProofCacheRejected << '\n'
                     << "proof_unknown_pair_count=" << proofs.unknownPairs
                     << '\n'
                     << "proof_oracle_comparison_count="
                     << proofs.oracleComparisons << '\n'
                     << "proof_semantic_quotient_group_count="
                     << proofs.semanticQuotientGroups << '\n'
                     << "proof_semantic_quotient_class_count="
                     << proofs.semanticQuotientClasses << '\n'
                     << "proof_counterexample_jobs="
                     << proofs.counterexampleJobs
                     << '\n';
#if defined(TRITON_GRAPH_HAVE_Z3)
        if (z3ProofBackend) {
          Z3SemanticSolverStatistics statistics =
              z3ProofBackend->getStatistics();
          llvm::outs() << "z3_query_count=" << statistics.queries << '\n'
                       << "z3_proved_count=" << statistics.proved << '\n'
                       << "z3_refuted_count=" << statistics.refuted << '\n'
                       << "z3_unknown_count=" << statistics.unknown << '\n'
                       << "z3_unsupported_count=" << statistics.unsupported
                       << '\n'
                       << "z3_timeout_count=" << statistics.timedOut << '\n'
                       << "z3_other_unknown_count="
                       << statistics.otherUnknown << '\n'
                       << "z3_simplifier_proved_count="
                       << statistics.simplifierProved << '\n'
                       << "z3_simplifier_refuted_count="
                       << statistics.simplifierRefuted << '\n'
                       << "z3_retry_check_count=" << statistics.retryChecks
                       << '\n'
                       << "z3_retry_unsat_check_count="
                       << statistics.retryUnsatChecks
                       << '\n'
                       << "z3_retry_sat_check_count="
                       << statistics.retrySatChecks
                       << '\n'
                       << "z3_retry_unknown_count=" << statistics.retryUnknown
                       << '\n'
                       << "z3_bitblast_check_count="
                       << statistics.bitBlastChecks << '\n'
                       << "z3_bitblast_unsat_check_count="
                       << statistics.bitBlastUnsatChecks << '\n'
                       << "z3_bitblast_sat_check_count="
                       << statistics.bitBlastSatChecks << '\n'
                       << "z3_bitblast_unknown_count="
                       << statistics.bitBlastUnknown << '\n'
                       << "z3_finite_shift_domain_check_count="
                       << statistics.finiteShiftDomainChecks << '\n'
                       << "z3_finite_shift_domain_unsat_check_count="
                       << statistics.finiteShiftDomainUnsatChecks << '\n'
                       << "z3_finite_shift_domain_sat_check_count="
                       << statistics.finiteShiftDomainSatChecks << '\n'
                       << "z3_finite_shift_domain_unknown_check_count="
                       << statistics.finiteShiftDomainUnknownChecks << '\n'
                       << "z3_worker_context_count="
                       << statistics.workerContexts
                       << '\n'
                       << "z3_worker_context_creation_count="
                       << statistics.workerContextCreations
                       << '\n'
                       << "z3_worker_context_recycle_count="
                       << statistics.workerContextRecycles
                       << '\n';
        }
#endif
        proofs.resolvedProofCache.universeHash = spec->getStableHash();
        proofs.resolvedProofCache.graphCount = enumeration.graphs.size();
        if (!resolvedProofCacheOutput.empty() &&
            !writeResolvedProofCache(resolvedProofCacheOutput,
                                     proofs.resolvedProofCache))
          return 1;
        if (!proofOutput.empty() &&
            !writeProofReport(proofOutput, proofs, spec->getStableHash(),
                              enumeration.isSemanticCoverageComplete(),
                              fingerprintComplete))
          return 1;
        if (!checkResourceBudget("proof"))
          return 6;
        if (!proofs.isComplete()) {
          llvm::errs()
              << "semantic equivalence proof is incomplete; interrupted or "
                 "unknown candidate pairs remain\n";
          return 8;
        }

        std::vector<CertifiedSemanticGraph> certifiedSemanticGraphs;
        if (runEclassStore && !ruleDatabaseOutput.empty()) {
          std::string databaseError;
          std::optional<std::vector<PersistentGraphRecord>> persistentGraphs =
              buildPersistentGraphRecords(enumeration.graphs, databaseError);
          if (!persistentGraphs) {
            llvm::errs() << "cannot certify semantic graph identities: "
                         << databaseError << '\n';
            return 1;
          }
          if (!checkResourceBudget("rule-database-graph-identities"))
            return 6;
          std::vector<std::optional<CertifiedSemanticGraph>> certificates(
              fingerprints.denotations.size());
          std::atomic<bool> certificateFailure{false};
          parallelForWork(
              fingerprints.denotations.size(), fingerprintJobs,
              [&](unsigned index) {
                if (certificateFailure.load(std::memory_order_relaxed))
                  return;
                const std::optional<TensorDenotation> &denotation =
                    fingerprints.denotations[index];
                if (!denotation)
                  return;
                std::optional<SemanticEquivalenceProof> reflexive =
                    oracle.prove(*denotation, *denotation);
                if (!reflexive || index >= persistentGraphs->size()) {
                  certificateFailure.store(true, std::memory_order_relaxed);
                  return;
                }
                certificates[index] = CertifiedSemanticGraph{
                    (*persistentGraphs)[index].graphHash,
                     getRuleDatabaseSemanticKeyDigest(
                         getStableTensorDenotationKey(*denotation)),
                    reflexive->getStableHash()};
              });
          if (certificateFailure.load(std::memory_order_relaxed)) {
            llvm::errs()
                << "cannot produce reflexive semantic certificate for one or "
                   "more enumerated graphs\n";
            return 1;
          }
          certifiedSemanticGraphs.reserve(fingerprints.denotations.size());
          for (std::optional<CertifiedSemanticGraph> &certificate :
               certificates)
            if (certificate)
              certifiedSemanticGraphs.push_back(std::move(*certificate));
          if (!checkResourceBudget("rule-database-certificates"))
            return 6;
        }

        // Fingerprint buckets and normalized denotations are only needed by
        // the prover and the reflexive RuleDB certification above.  Releasing
        // them before materializing the persistent e-class/Pareto snapshots
        // prevents the complete depth-one catalog from retaining two full
        // semantic representations at once.
        fingerprints = OfflineSemanticFingerprintResult();
        releaseUnusedHeapPages();
        if (!checkResourceBudget("proof-state-release"))
          return 6;

        if (runEclassStore) {
          std::optional<PersistentEquivalenceClassStore> base;
          if (!eclassInput.empty()) {
            llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
                llvm::MemoryBuffer::getFile(eclassInput);
            if (!input) {
              llvm::errs() << "cannot read e-class store '" << eclassInput
                           << "': " << input.getError().message() << '\n';
              return 1;
            }
            std::string loadError;
            base = PersistentEquivalenceClassStore::fromJSON(
                input.get()->getBuffer(), context, oracle, loadError);
            if (!base) {
              llvm::errs() << "invalid e-class store '" << eclassInput
                           << "': " << loadError << '\n';
              return 1;
            }
          }

          std::string storeError;
          std::optional<PersistentEquivalenceClassStore> store =
              PersistentEquivalenceClassStore::build(
                  enumeration.graphs, proofs, spec->getStableHash(),
                  getBuiltinTTOpSemanticsStableHash(),
                  enumeration.isSemanticCoverageComplete(), fingerprintComplete,
                  base ? &*base : nullptr, storeError);
          if (!store) {
            llvm::errs() << "cannot build e-class store: " << storeError
                         << '\n';
            return 1;
          }
          const std::string eclassStoreHash = store->getStableHash();
          llvm::outs() << "eclass_store_complete="
                       << (store->isComplete() ? "true" : "false") << '\n'
                       << "eclass_store_hash=" << eclassStoreHash
                       << '\n'
                       << "eclass_graph_count=" << store->graphs.size()
                       << '\n'
                       << "eclass_proof_edge_count="
                       << store->proofEdges.size() << '\n'
                       << "eclass_count=" << store->classes.size() << '\n';
          std::vector<OfflineSemanticProofPair>().swap(proofs.pairs);
          std::vector<OfflineTypedGraph>().swap(enumeration.graphs);
          releaseUnusedHeapPages();
          if (!eclassOutput.empty() &&
              !writeStreamingReport(
                  eclassOutput, "e-class store",
                  [&](llvm::raw_ostream &output) {
                    store->writeJSON(output, eclassStoreHash);
                  }))
            return 1;
          if (!checkResourceBudget("eclass-store"))
            return 6;

          if (!paretoOutput.empty() || !ruleDatabaseOutput.empty()) {
            GraphCostContext costContext;
            costContext.target = spec->target;
            std::unique_ptr<OfflineGraphCostModel> costModel;
            if (useStageModel || !stageModelHardwareConfig.empty())
              costModel = std::make_unique<AscendStageModelGraphCostModel>(
                  stageModelHardwareConfig);
            else
              costModel = std::make_unique<StructuralFallbackCostModel>();
            ParetoExtractionResult pareto = ParetoExtractor().extract(
                *store, *costModel, costContext, eclassStoreHash);
            llvm::outs()
                << "pareto_cost_model=" << pareto.costModelName << '\n'
                << "pareto_candidate_count_before="
                << pareto.candidateCountBefore << '\n'
                << "pareto_candidate_count_after="
                << pareto.candidateCountAfter << '\n'
                << "pareto_eclass_count=" << pareto.classes.size() << '\n'
                << "pareto_structural_fallback_class_count="
                << pareto.structuralFallbackClassCount << '\n'
                << "pareto_stage_model_complete="
                << (pareto.usedStageModelForEveryClass() ? "true" : "false")
                << '\n';
            for (llvm::StringRef reason : pareto.stageModelUnknownReasons)
              llvm::outs() << "pareto_stage_model_unknown=" << reason << '\n';
            if (requireCompleteStageModel &&
                !pareto.usedStageModelForEveryClass()) {
              llvm::errs()
                  << "StageModel coverage is incomplete; refusing to emit a "
                     "mixed/fallback cost database\n";
              return 5;
            }
            if (!paretoOutput.empty() &&
                !writeStreamingReport(
                    paretoOutput, "Pareto", [&](llvm::raw_ostream &output) {
                      pareto.writeJSON(output);
                    }))
              return 1;
            if (!checkResourceBudget("pareto"))
              return 6;
            if (!ruleDatabaseOutput.empty()) {
              std::string databaseError;
              std::optional<RuleDatabase> database =
                  RuleDatabase::compile(*store, pareto, databaseError,
                                        certifiedSemanticGraphs,
                                        eclassStoreHash);
              if (!database) {
                llvm::errs() << "cannot compile rule database: "
                             << databaseError << '\n';
                return 1;
              }
              const uint64_t certifiedSemanticGraphCount =
                  certifiedSemanticGraphs.size();
              std::vector<CertifiedSemanticGraph>().swap(
                  certifiedSemanticGraphs);
              pareto = ParetoExtractionResult();
              store.reset();
              if (!checkResourceBudget("rule-database-compile"))
                return 6;
              llvm::outs() << "rule_db_hash="
                           << database->getStableHash() << '\n'
                           << "rule_db_graph_count="
                           << database->graphs.size() << '\n'
                           << "rule_db_entry_count="
                           << database->entries.size() << '\n'
                           << "rule_db_template_count="
                           << database->templates.size() << '\n'
                           << "rule_db_semantic_entry_count="
                           << database->semanticEntries.size() << '\n';
              llvm::outs() << "rule_db_certified_semantic_graph_count="
                            << certifiedSemanticGraphCount << '\n';
              uint64_t serializedDatabaseBytes = 0;
              if (!writeStreamingReport(
                      ruleDatabaseOutput, "rule database",
                      [&](llvm::raw_ostream &output) {
                        database->writeJSON(output);
                      },
                      &serializedDatabaseBytes))
                return 1;
              if (!checkResourceBudget("rule-database-serialize"))
                return 6;
              llvm::outs() << "rule_db_bytes=" << serializedDatabaseBytes
                           << '\n';
              if (maxRuleDatabaseBytes != 0 &&
                  serializedDatabaseBytes > maxRuleDatabaseBytes) {
                llvm::errs() << "rule database exceeds --max-rule-db-bytes: "
                             << serializedDatabaseBytes << " > "
                             << maxRuleDatabaseBytes << '\n';
                return 4;
              }
              if (!checkResourceBudget("rule-database"))
                return 6;
            }
          }
        }
      }
    }
    if (schemaReachabilityOnly &&
        !enumeration.isSchemaReachabilityComplete()) {
      llvm::errs() << "declarative candidate-schema reachability is "
                      "incomplete; inspect the JSON gap fields\n";
      return 2;
    }
    if (!schemaReachabilityOnly &&
        !enumeration.isSemanticCoverageComplete()) {
      llvm::errs() << "offline enumeration is incomplete; inspect the JSON "
                      "gap fields\n";
      return 2;
    }
    if (!fingerprintComplete) {
      llvm::errs() << "semantic fingerprinting is incomplete; wildcard graphs "
                      "remain comparable by type\n";
      return 3;
    }
  }

  return 0;
}

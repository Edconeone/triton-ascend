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

#ifndef TRITON_TO_GRAPH_CANONICAL_GRAPH_KEY_H
#define TRITON_TO_GRAPH_CANONICAL_GRAPH_KEY_H

#include "TritonToGraph/EquivalenceClassStore.h"
#include "TritonToGraph/TTOpSemanticsDSL.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::triton::cfg {

struct CanonicalGraphBuildResult {
  std::string rootGraphHash;
  // Byte-stable key of the normalized multi-domain denotation.  This permits
  // database lookup for an unseen source topology without enumerating any
  // candidate online. Empty means the imported fragment was not executable
  // in the current semantic language; exact graph lookup remains available.
  std::string normalizedSemanticKey;
  // Canonical LaunchState used by launch-sensitive imported operations and,
  // for whole-kernel roots, by ags.kernel. Empty for value/effect imports that
  // never request launch state.
  std::string launchStateHash;
  std::vector<PersistentGraphRecord> graphs;
  std::vector<Value> boundaryInputs;
  // Online-only provenance for every imported SSA result. Template matching
  // binds a symbolic offline input to a canonical online graph hash; this map
  // turns that hash back into the already-computed SSA value so a certified
  // RHS can reuse it without re-enumerating or re-proving anything online.
  std::vector<std::pair<std::string, Value>> graphValues;
  // Source operations that produced canonical MemoryState nodes. This is
  // online-only provenance used to apply a certified state-suffix rewrite as
  // one transaction; it is intentionally absent from persistent graph IDs.
  std::vector<std::pair<std::string, Operation *>> stateOperations;
};

// Imports one pure semantic island into the exact persistent graph identity
// used by offline enumeration. Unsupported/shared producers become numbered
// ags.input terminals. Values produced by a private allocation become
// ags.private_buffer terminals so generic BufferState proofs may use their
// non-aliasing identity. Attribute selection is derived from candidate schemas.
class CanonicalTTIRGraphBuilder {
public:
  CanonicalTTIRGraphBuilder(const TTOpSemanticsRegistry &semantics,
                            llvm::ArrayRef<TTIRCandidateOpSchema> schemas,
                            std::function<bool()> shouldStop = {});

  std::optional<CanonicalGraphBuildResult> build(Value root,
                                                  std::string &error) const;

  // Canonicalizes a state transition: a zero-result write, or an atomic whose
  // SSA result is unused. Its persistent root type is `none`: this is the
  // resulting MemoryState, not a synthetic SSA result in the user's TTIR.
  std::optional<CanonicalGraphBuildResult> build(Operation *root,
                                                  std::string &error) const;

  // Canonicalizes the complete straight-line memory behavior of one kernel
  // scope and binds its final MemoryState to the initial LaunchState through
  // the declarative proof-only ags.kernel root. This is the online identity
  // used by whole-kernel launch transformations.
  std::optional<CanonicalGraphBuildResult>
  buildKernel(Operation *scope, std::string &error) const;

private:
  std::optional<CanonicalGraphBuildResult>
  buildImpl(Value valueRoot, Operation *operationRoot,
            std::string &error, bool includeLaunchState = false) const;
  bool shouldInterrupt(std::string &error) const;

  const TTOpSemanticsRegistry &semantics;
  llvm::ArrayRef<TTIRCandidateOpSchema> schemas;
  std::function<bool()> shouldStop;
  mutable bool interrupted = false;
};

} // namespace mlir::triton::cfg

#endif // TRITON_TO_GRAPH_CANONICAL_GRAPH_KEY_H

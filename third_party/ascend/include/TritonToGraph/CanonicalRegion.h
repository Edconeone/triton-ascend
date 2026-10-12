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

#ifndef TRITON_TO_GRAPH_CANONICAL_REGION_H
#define TRITON_TO_GRAPH_CANONICAL_REGION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {

class Operation;
class RewriterBase;

namespace triton::cfg {

inline constexpr llvm::StringLiteral kCanonicalRegionsAttr =
    "ags.canonical_regions";

// Recursively captures operations in every block into a stable,
// location-independent CFG/SSA recipe. Nested regions use the same encoding;
// block arguments use block-local
// negative references (-1-index); operation results use a region-wide
// flattened non-negative ordinal. Successor block ordinals are explicit. The
// function rejects unsupported structure instead of dropping it from the
// graph identity.
std::optional<ArrayAttr> captureCanonicalRegions(Operation *operation,
                                                 std::string &error);

// Validates a serialized recipe without mutating IR.
LogicalResult validateCanonicalRegions(ArrayAttr regions,
                                       std::string &error);

// Populates the already-created empty regions of `operation`. The caller owns
// rollback and must erase the parent operation if this function fails.
LogicalResult materializeCanonicalRegions(Operation *operation,
                                          ArrayAttr regions,
                                          RewriterBase &rewriter,
                                          std::string &error);

// Creates the canonical recipe for the common binary-combiner region shape.
// Operation/terminator names are data supplied by TTOpSemantics, not switches
// in the region engine.
ArrayAttr buildCanonicalBinaryRegion(MLIRContext *context, Type elementType,
                                     StringRef combinerOperation,
                                     StringRef terminatorOperation);

} // namespace triton::cfg
} // namespace mlir

#endif // TRITON_TO_GRAPH_CANONICAL_REGION_H

// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H
#define TRITON_ASCEND_GRAPH_NARROW_UNSIGNED_TENSOR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <cstdint>

namespace mlir::triton {
namespace narrow_unsigned_tensor {

inline bool constant(Value value, uint64_t &bits) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return false;
  bits = integer.getZExtValue();
  return true;
}

// Only values proven to have zero upper bits may change signed right shift
// into unsigned right shift. In particular, an arbitrary i64 is NOT a u32.
inline bool isU32(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    return getElementTypeOrSelf(ext.getIn().getType()).isInteger(32);
  uint64_t bits;
  return constant(value, bits) && bits <= UINT32_MAX;
}

inline Type i32Type(Type type, RewriterBase &rewriter) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.clone(rewriter.getI32Type());
  return rewriter.getI32Type();
}

// Build a narrowed bitwise node while eliminating identities that generic
// arith folding does not currently recognize for dense tensor splats.
inline Value createBitwise(Operation *source, Value lhs, Value rhs,
                           RewriterBase &rewriter, Location location) {
  uint64_t lhsBits = 0;
  uint64_t rhsBits = 0;
  const bool lhsConstant = constant(lhs, lhsBits);
  const bool rhsConstant = constant(rhs, rhsBits);

  if (isa<arith::AndIOp>(source)) {
    if (lhsConstant && lhsBits == UINT32_MAX)
      return rhs;
    if (rhsConstant && rhsBits == UINT32_MAX)
      return lhs;
    if (lhsConstant && lhsBits == 0)
      return lhs;
    if (rhsConstant && rhsBits == 0)
      return rhs;
    return rewriter.createOrFold<arith::AndIOp>(location, lhs, rhs);
  }
  if (isa<arith::OrIOp>(source)) {
    if (lhsConstant && lhsBits == 0)
      return rhs;
    if (rhsConstant && rhsBits == 0)
      return lhs;
    if (lhsConstant && lhsBits == UINT32_MAX)
      return lhs;
    if (rhsConstant && rhsBits == UINT32_MAX)
      return rhs;
    return rewriter.createOrFold<arith::OrIOp>(location, lhs, rhs);
  }
  if (lhsConstant && lhsBits == 0)
    return rhs;
  if (rhsConstant && rhsBits == 0)
    return lhs;
  return rewriter.createOrFold<arith::XOrIOp>(location, lhs, rhs);
}

// Compute the low 32 bits without materializing wide bitwise temporaries.
// Truncation commutes with these operations, but not with right shift.
inline Value lowBits(Value value, RewriterBase &rewriter,
                     unsigned depth = 0) {
  Location loc = value.getLoc();
  Type type = i32Type(value.getType(), rewriter);
  if (value.getType() == type)
    return value;
  if (auto ext = value.getDefiningOp<arith::ExtUIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType() == type)
      return ext.getIn();
  uint64_t bits;
  if (constant(value, bits)) {
    auto attr =
        rewriter.getIntegerAttr(rewriter.getI32Type(), bits & UINT32_MAX);
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return rewriter.create<arith::ConstantOp>(
          loc, type, DenseElementsAttr::get(tensor, attr));
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }
  if (depth < 8) {
    if (auto op = value.getDefiningOp<SplatOp>())
      return rewriter.create<SplatOp>(
          loc, type, lowBits(op.getSrc(), rewriter, depth + 1));
    Operation *op = value.getDefiningOp();
    if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      uint64_t lhsBits = 0;
      uint64_t rhsBits = 0;
      const bool lhsConstant = constant(op->getOperand(0), lhsBits);
      const bool rhsConstant = constant(op->getOperand(1), rhsBits);
      const uint64_t lhsLowBits = lhsBits & UINT32_MAX;
      const uint64_t rhsLowBits = rhsBits & UINT32_MAX;
      if (isa<arith::AndIOp>(op)) {
        if (lhsConstant && lhsLowBits == UINT32_MAX)
          return lowBits(op->getOperand(1), rewriter, depth + 1);
        if (rhsConstant && rhsLowBits == UINT32_MAX)
          return lowBits(op->getOperand(0), rewriter, depth + 1);
      } else if (isa<arith::OrIOp>(op)) {
        if (lhsConstant && lhsLowBits == 0)
          return lowBits(op->getOperand(1), rewriter, depth + 1);
        if (rhsConstant && rhsLowBits == 0)
          return lowBits(op->getOperand(0), rewriter, depth + 1);
      } else {
        if (lhsConstant && lhsLowBits == 0)
          return lowBits(op->getOperand(1), rewriter, depth + 1);
        if (rhsConstant && rhsLowBits == 0)
          return lowBits(op->getOperand(0), rewriter, depth + 1);
      }
      Value lhs = lowBits(op->getOperand(0), rewriter, depth + 1);
      Value rhs = lowBits(op->getOperand(1), rewriter, depth + 1);
      return createBitwise(op, lhs, rhs, rewriter, loc);
    }
  }
  return rewriter.create<arith::TruncIOp>(loc, type, value);
}

} // namespace narrow_unsigned_tensor
} // namespace mlir::triton
#endif

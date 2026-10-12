# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

import numbers
from typing import Optional, Tuple

from triton._C.libtriton import ir
from triton.language import core as tl
from triton.language.semantic import TritonSemantic, TensorTy


# Private frontend-to-Ascend provenance channel.  The common Triton builder
# only knows how to create a normal ``tt.assert``.  The Ascend graph pass
# consumes this prefix before AGS matching, restores the user-facing message,
# and attaches its private provenance attribute entirely inside the backend.
_AUTO_OVERFLOW_ASSERT_PREFIX = "__triton_ascend_auto_overflow__: "


class AscendSemantic(TritonSemantic):
    """Ascend-only semantic extensions kept out of Triton's common layer."""

    def normalize_math_dtype(self, value):
        if getattr(value, "was_bool_to_int8", False):
            return "int1"
        return super().normalize_math_dtype(value)

    def accepts_math_dtype(self, fn, dtype, dtypes):
        # Ascend's libdevice accepts fp16 for exponential.
        return getattr(fn, "__name__", None) == "exp" and dtype == "fp16"

    def dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
        # Keep the common Triton IR vocabulary.  Ascend's lowering maps TF32
        # to its HF32 implementation after the public IR is built.
        if input_precision == "hf32":
            input_precision = "tf32"

        # Ascend does not support imprecise accumulation controls yet.
        if max_num_imprecise_acc is not None:
            print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                  "Thus it is ignored.")
            max_num_imprecise_acc = None

        return super().dot(lhs, rhs, acc, input_precision,
                           max_num_imprecise_acc, out_dtype)

    def binary_op_sanitize_overflow_impl(self, lhs: TensorTy, rhs: TensorTy, binary_op: callable):
        if lhs.type.scalar.int_bitwidth >= 64 or not self.builder.options.sanitize_overflow:
            return
        lhs_sca_ty = lhs.type.scalar
        rhs_sca_ty = rhs.type.scalar
        assert lhs_sca_ty == rhs_sca_ty
        assert lhs_sca_ty.is_int()
        lhs = self.cast(lhs, tl.int64)
        rhs = self.cast(rhs, tl.int64)
        ret = binary_op(lhs, rhs, False)
        max_value = lhs_sca_ty.get_int_max_value()
        max_value = self.scalar_constant(max_value, tl.int64)
        min_value = lhs_sca_ty.get_int_min_value()
        min_value = self.scalar_constant(min_value, tl.int64)
        cond = self.and_(self.less_equal(ret, max_value), self.greater_equal(ret, min_value))
        msg = f"int{lhs_sca_ty.int_bitwidth} overflow detected for operation {binary_op.__name__}"
        self.device_assert(cond, msg, None, auto_overflow=True)


    def mod(self, input: TensorTy | numbers.Number, other: TensorTy | numbers.Number) -> TensorTy:
        input, other = self.binary_op_type_checking_impl(input, other, False, False, True, True)
        scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
        # float % float
        if scalar_ty.is_floating():
            return self.tensor(self.builder.create_frem(input.handle, other.handle), input.type)
        # % int
        elif scalar_ty.is_int():
            if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
                raise TypeError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                                "because they have different signedness;"
                                "this is unlikely to result in a useful answer. Cast them to the same signedness.")
            if hasattr(input, 'was_bool_to_int8'):
                false_val = self.builder.get_int1(False)
                return self.tensor(false_val, tl.int1)
            if scalar_ty.is_int_signed():
                return self.tensor(self.builder.create_srem(input.handle, other.handle), input.type)
            else:
                return self.tensor(self.builder.create_urem(input.handle, other.handle), input.type)
        raise TypeError(f"unexpected type {scalar_ty}")

##############
# other arithmetic ops
##############


    def logical_and(self, input: TensorTy, other: TensorTy) -> TensorTy:
        dst_sca_ty = tl.dtype("int1")
        dst_bits = dst_sca_ty.primitive_bitwidth
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if not input.type.is_int1():
            src_sca_ty = input.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                input = self.bitcast(input, tl.int1)
            else:
                input = self.not_equal(input, 0)
        if hasattr(other, 'was_bool_to_int8'):
            assert other.type.scalar.is_int8(), "Other input wat bool to int8. However, other input.type is not int8."
            other = self.cast(other, tl.int1)
        if not other.type.is_int1():
            src_sca_ty = other.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                other = self.bitcast(other, tl.int1)
            else:
                other = self.not_equal(other, 0)
        return self.and_(input, other)


    def logical_or(self, input: TensorTy, other: TensorTy) -> TensorTy:
        dst_sca_ty = tl.dtype("int1")
        dst_bits = dst_sca_ty.primitive_bitwidth
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if not input.type.is_int1():
            src_sca_ty = input.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                input = self.bitcast(input, tl.int1)
            else:
                input = self.not_equal(input, 0)
        if hasattr(other, 'was_bool_to_int8'):
            assert other.type.scalar.is_int8(), "Other wat bool to int8. However, other.type is not int8."
            other = self.cast(other, tl.int1)
        if not other.type.is_int1():
            src_sca_ty = other.type.scalar
            src_bits = src_sca_ty.primitive_bitwidth
            if src_bits == dst_bits or src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
                other = self.bitcast(other, tl.int1)
            else:
                other = self.not_equal(other, 0)
        return self.or_(input, other)


    def not_(self, input: TensorTy):
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        if input.type.scalar.is_floating():
            raise TypeError(f"unexpected type {input.type.scalar}")
        return self.invert(input)


    def minus(self, input: TensorTy) -> TensorTy:
        input_sca_ty = input.type.scalar
        if hasattr(input, 'was_bool_to_int8'):
            if input.type.scalar.is_int8():
                raise TypeError("unexpected type bool")
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
        _0 = self.tensor(self.builder.get_null_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.sub(_0, input, True)


    def invert(self, input: TensorTy) -> TensorTy:
        if hasattr(input, 'was_bool_to_int8'):
            assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
            input = self.cast(input, tl.int1)
        input_sca_ty = input.type.scalar
        if input_sca_ty.is_floating():
            raise TypeError(f"unexpected type {input_sca_ty}")
        if input_sca_ty.is_ptr():
            raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
        _1 = self.tensor(self.builder.get_all_ones_value(input_sca_ty.to_ir(self.builder)), input_sca_ty)
        return self.xor_(input, _1)


    def arange(self, start: int, end: int, *, ret_ty: tl.block_type = None) -> TensorTy:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("arange's arguments must be of type tl.constexpr")
        is_start_int64 = bool(start >> 32)
        is_end_int64 = bool(end >> 32)
        if is_start_int64 or is_end_int64:
            raise ValueError("arange must fit in int32")
        if end <= start:
            raise ValueError("arange's end argument must be greater than the start argument")
        range = end - start
        # Check if compile_mode is simt, then range must be a power of 2
        if self.builder.is_simt_mode():
            # Check if range is a power of 2
            if (range & (range - 1)) != 0:
                raise ValueError("arange's range must be a power of 2")
        shape = [range]
        if ret_ty is None:
            ret_ty = tl.block_type(tl.int32, shape)
        ret_ty_ir = ret_ty.to_ir(self.builder)
        return self.tensor(self.builder.create_make_range(ret_ty_ir, start, end), ret_ty)


    def _load_block_pointer(self, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
        # Load by a block pointer: `pointer_type<block_type<>>`
        # Block pointer can not have `mask` and `other` arguments
        if mask is not None or other is not None:
            raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

        elt_ty = ptr.type.element_ty.element_ty
        assert elt_ty != tl.int1, "`tl.int1` should be rewritten in `tl.make_block_ptr`"
        if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
            raise ValueError("Padding option `nan` is not supported for integer block pointers")

        # `dst_ty` is de-referenced type of the pointer type
        dst_ty = ptr.type.element_ty

        # Check `boundary_check` argument
        boundary_check = self._canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

        if boundary_check and padding is None:
            padding = ir.PADDING_OPTION.PAD_ZERO
        # Build IR
        return self.tensor(
            self.builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction, is_volatile),
            dst_ty)


    def _load_legacy(self, ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile):
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        if not ptr.type.scalar.is_ptr():
            raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

        # Check `mask`, `other`, `boundary_check`, and `padding` arguments
        if mask is None and other is not None:
            raise ValueError("`other` cannot be provided without `mask`")
        if padding or boundary_check:
            raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                             "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                             "use block pointers (defined by `make_block_ptr`) instead")
        if mask is not None and other is None:
            # Get element type to determine default padding value
            elt_ty = ptr.type.scalar.element_ty
            # Use 0.0 for floating point types, 0 for integer types
            default_value = 0.0 if elt_ty.is_floating() else 0
            other = self.to_tensor(default_value)
        # For a pointer of scalar, check the type of `mask` and `other`
        if not ptr.type.is_block():
            if mask and mask.type.is_block():
                raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
            if other and other.type.is_block():
                raise ValueError("Other argument cannot be block type if pointer argument is not a block")

        # Make `mask` and `other` into the same shape as `ptr`
        if ptr.type.is_block():
            if mask is not None:
                ptr, mask = self.broadcast_impl_value(ptr, mask)
            if other is not None:
                ptr, other = self.broadcast_impl_value(ptr, other)

        # Get `pointer_type<elt_ty>` and `elt_ty`
        ptr_ty = ptr.type.scalar
        elt_ty = ptr_ty.element_ty

        # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
        is_bool = elt_ty == tl.int1
        if is_bool:
            elt_ty = tl.int8
            ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
            ptr = self.cast(ptr, ptr_ty)

        # Cast `other` into `elt_ty` type
        if other is not None:
            other = self.cast(other, elt_ty)

        # Create loaded result type `dst_ty`
        if ptr.type.is_block():
            dst_ty = ptr.type.with_element_ty(elt_ty)
        else:
            # Load by de-referencing the pointer of scalar
            dst_ty = elt_ty

        # Build IR
        if mask is None:
            load_handle = self.builder.create_load(ptr.handle, cache, eviction, is_volatile)
        else:
            load_handle = self.builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None,
                                                          cache, eviction, is_volatile)

        if is_bool:
            load_handle.set_attr("was_bool_to_int8", self.builder.get_bool_attr(True))

        ret = self.tensor(load_handle, dst_ty)
        # Do not cast back to int1 when is_bool=true. We directly use the int8 tensor given by tl.load
        if is_bool:
            ret.was_bool_to_int8 = True

        return ret


    def device_assert(self, cond: TensorTy, msg: str, mask: Optional[TensorTy],
                      auto_overflow: bool = False) -> TensorTy:
        if not self.builder.options.debug:
            return
        if mask is not None:
            cond = self.or_(cond, self.not_(mask))
        if auto_overflow:
            self.builder.create_assert(
                cond.handle, _AUTO_OVERFLOW_ASSERT_PREFIX + msg)
        else:
            self.builder.create_assert(cond.handle, msg)
        return self.tensor(None, tl.void)


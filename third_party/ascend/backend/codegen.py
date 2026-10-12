# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Ascend-specific hooks for Triton's common Python-to-TTIR code generator.
# Keeping these hooks in the backend prevents the public compiler from
# importing CANN, BiShengIR, or Ascend builder modules.

import ast
import logging

from triton import language
from triton._C.libtriton import buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
try:
    from triton._C.libtriton import distributed
except ImportError:
    distributed = None

from triton.backends.compiler import CodegenHooks
from triton._utils import (
    reset_allow_non_power_of_two_block_shape,
    set_allow_non_power_of_two_block_shape,
)
from triton.extension.buffer.language.builder import setup_unified_builder_with_buffer_builder
from triton.language.extra.cann.extension import int64, is_builtin, parallel
from triton.language.extra.cann.extension.builder import setup_unified_builder
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH


class AscendCodegenHooks(CodegenHooks):
    """Install Ascend builders and syntax extensions for one kernel compile."""

    def create_semantic(self, builder):
        from .semantic import AscendSemantic
        return AscendSemantic(builder)

    def begin_codegen(self):
        return set_allow_non_power_of_two_block_shape(True)

    def end_codegen(self, token):
        if token is not None:
            reset_allow_non_power_of_two_block_shape(token)

    def configure(self, generator, context, options, file_name, begin_line):
        if generator.is_gluon:
            return

        # DistributedOpBuilder is an Ascend build-time extension.  Keep the
        # fallback to the common Triton builder for installations where the
        # optional distributed binding is not compiled.
        if distributed is not None:
            generator.builder = distributed.ir.DistributedOpBuilder(context)
            generator.builder.set_loc(file_name, begin_line, 0)

        compile_mode = "simt" if getattr(options, "compile_mode", "simd") == "simt_only" else "simd"
        generator.ascend_builder = ascend_ir.ascendnpu_ir_builder(
            context, getattr(options, "arch", ""), compile_mode=compile_mode)
        generator.ascend_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder(generator.builder, generator.ascend_builder)

        generator.buffer_builder = buffer_ir.buffer_builder(context)
        generator.buffer_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder_with_buffer_builder(generator.builder, generator.buffer_builder)

    def finalize_module(self, generator):
        if generator.is_gluon:
            return
        options = generator.builder.options
        if not getattr(options, "arch", ""):
            return
        try:
            builder = ascend_ir.ascendnpu_ir_builder(generator.context, options.arch)
            target_attr = builder.parse_attr(f'#hacc.target<"{options.arch}">')
            generator.module.set_attr("hacc.target", target_attr)
            generator.module.set_attr("tt.allow_non_power_of_two", generator.builder.get_unit_attr())
        except Exception as exc:
            logging.warning("[Ascend] failed to set hacc.target: %s", exc)

    def handle_with(self, generator, node):
        if len(node.items) != 1:
            return False, None
        context = node.items[0].context_expr
        if not isinstance(context, ast.Call):
            return False, None
        with_item_class = generator.visit(context.func)
        handler = ASCEND_WITH_DISPATCH.get(with_item_class)
        if handler is None:
            return False, None
        return True, handler(generator, node)

    def is_parallel_iterator(self, iterator_class) -> bool:
        return iterator_class is parallel

    def configure_loop(self, generator, for_op, iterator_class, disable_licm):
        if disable_licm:
            for_op.set_attr("tt.disable_licm", generator.builder.get_unit_attr())
        if iterator_class is parallel:
            for_op.set_attr("hivm.parallel_loop", generator.builder.get_unit_attr())

    def is_constexpr_argument(self, value) -> bool:
        return isinstance(value, language.PropagateNan)

    def builder_for_call(self, fn, default_builder):
        return getattr(default_builder, "_ascend_builder", default_builder) if is_builtin(fn) else default_builder

    def get_static_implementation(self, fn, static_executor):
        if fn is int64:
            return static_executor(int64)
        return None


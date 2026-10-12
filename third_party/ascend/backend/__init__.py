# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
__version__ = '3.6.0'

from .testing import do_bench_npu


def _make_default_codegen_hooks():
    from .codegen import AscendCodegenHooks
    return AscendCodegenHooks()


from triton.backends.compiler import set_default_codegen_hooks
set_default_codegen_hooks(_make_default_codegen_hooks)


def _apply_ascend_patch():
    """Register Ascend hooks for legacy direct ``ast_to_ttir`` callers.

    Normal compilation obtains hooks from ``AscendBackend``.  This explicit
    compatibility entry point replaces the old global monkeypatch behavior
    without modifying Triton's common language implementation.
    """
    set_default_codegen_hooks(_make_default_codegen_hooks)

    # Install the unified, non-intrusive dtype interception for operators
    # whose dtypes are unsupported on Ascend. The rules live in
    # triton.backends.ascend.op_dtype_config, next to this backend.
    if not getattr(CodeGenerator, "_ascend_dtype_guard_applied", False):
        try:
            from triton.backends.ascend.dtype_guard import install_dtype_guard

            install_dtype_guard()
            CodeGenerator._ascend_dtype_guard_applied = True
        except Exception as e:
            logging.warning(f"[Ascend Patch] Failed to install dtype guard: {e}")


__all__ = ["do_bench_npu", "_apply_ascend_patch"]

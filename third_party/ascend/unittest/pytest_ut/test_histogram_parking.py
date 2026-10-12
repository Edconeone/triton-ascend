# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fold parked bin-zero counts without losing genuine zeros or other users."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _sum_without_overflow_guard(lhs, rhs):
    # Every operand is 0 or 1 and N is a compile-time tensor extent, so this
    # reduction cannot overflow i32 in the covered cases.  Spell that fact at
    # the operation which otherwise gets an overflow-preserving expansion.
    return tl.add(lhs, rhs, sanitize_overflow=False)


@triton.jit
def _parking(X, M, O, N: tl.constexpr, BINS: tl.constexpr, PARK: tl.constexpr, WRONG_MASK: tl.constexpr,
             KEEP_RAW: tl.constexpr):
    i = tl.arange(0, N)
    mask = tl.load(M + i) != 0
    x = tl.load(X + i)
    hist = tl.histogram(tl.where(mask, x, PARK), BINS)
    counted_mask = mask if not WRONG_MASK else (i % 3 == 0)
    count = tl.reduce((~counted_mask).to(tl.int32), axis=0,
                      combine_fn=_sum_without_overflow_guard)
    bins = tl.arange(0, BINS)
    corrected = tl.sub(hist, tl.where(bins == 0, count, 0),
                       sanitize_overflow=False)
    tl.store(O + bins, corrected)
    tl.store(O + BINS + bins, hist if KEEP_RAW else corrected)


def _check_parking(size, bins, park, wrong_mask, mask_kind, keep_raw=False):
    torch.manual_seed(37)
    x = torch.randint(-8, bins + 8, (size, ), dtype=torch.int32)
    x[::7] = 0
    mask = torch.randint(0, 2, (size, ), dtype=torch.int32)
    if mask_kind != "random":
        mask.fill_(int(mask_kind == "all"))
    device_x, device_mask = x.npu(), mask.npu()
    output = torch.empty((2, bins), dtype=torch.int32, device=device_x.device)
    compiled = _parking[(1, )](device_x, device_mask, output, size, bins, park,
                               wrong_mask, keep_raw,
                               enable_graph_optimize=True,
                               sanitize_overflow=False)
    parked = torch.where(mask.bool(), x, park)
    hist = torch.bincount(parked[(parked >= 0) & (parked < bins)].long(), minlength=bins)
    count_mask = torch.arange(size) % 3 == 0 if wrong_mask else mask.bool()
    expected = hist.clone()
    expected[0] -= (~count_mask).sum()
    second = hist if keep_raw else expected
    torch.testing.assert_close(output.cpu().long(), torch.stack((expected, second)), rtol=0, atol=0)
    return compiled.asm["ttir"]


@pytest.mark.parametrize("size", [256, 8192])
@pytest.mark.parametrize("bins", [16, 256])
@pytest.mark.parametrize("park,wrong_mask", [(0, False), (1, False), (0, True)])
@pytest.mark.parametrize("mask_kind", ["random", "none", "all"])
def test_histogram_parking(size, bins, park, wrong_mask, mask_kind):
    ttir = _check_parking(size, bins, park, wrong_mask, mask_kind)
    should_fold = park == 0 and not wrong_mask
    if should_fold:
        assert "tt.reduce" not in ttir
        assert "arith.subi" not in ttir
    else:
        # AGS may still simplify the independent invalid-lane reduction.  The
        # complete parking fold is rejected unless the parked and corrected
        # masks agree, which is witnessed by the surviving subtraction.
        assert "arith.subi" in ttir


def test_histogram_parking_other_users():
    ttir = _check_parking(8192, 256, 0, False, "random", keep_raw=True)
    assert "arith.subi" in ttir


@triton.jit
def _shared_compare_parking(X, M, O, P, N: tl.constexpr,
                            BINS: tl.constexpr):
    i = tl.arange(0, N)
    live = tl.load(M + i) != 0
    x = tl.load(X + i)
    histogram = tl.histogram(tl.where(live, x, 0), BINS)
    invalid_count = tl.reduce((~live).to(tl.int32), axis=0,
                              combine_fn=_sum_without_overflow_guard)
    bins = tl.arange(0, BINS)
    is_parking_bin = bins == 0
    corrected = tl.sub(histogram,
                       tl.where(is_parking_bin, invalid_count, 0),
                       sanitize_overflow=False)
    tl.store(O + bins, corrected)
    # Keep the equality predicate shared, as in radix_threshold where all four
    # histogram corrections reuse one `bins == 0` value.
    tl.store(P + bins, is_parking_bin)


def test_histogram_parking_shared_compare():
    """A shared equality fact must remain visible to semantic proof."""
    size = 8192
    bins = 256
    torch.manual_seed(101)
    x = torch.randint(-8, bins + 8, (size, ), dtype=torch.int32)
    live = torch.randint(0, 2, (size, ), dtype=torch.int32)
    device_x, device_live = x.npu(), live.npu()
    output = torch.empty((bins, ), dtype=torch.int32, device=device_x.device)
    predicate = torch.empty((bins, ), dtype=torch.bool,
                            device=device_x.device)
    compiled = _shared_compare_parking[(1, )](
        device_x, device_live, output, predicate, size, bins,
        enable_graph_optimize=True, sanitize_overflow=False)

    selected = x[live.bool()]
    selected = selected[(selected >= 0) & (selected < bins)]
    expected = torch.bincount(selected.long(), minlength=bins)
    torch.testing.assert_close(output.cpu().long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        predicate.cpu(), torch.arange(bins) == 0, rtol=0, atol=0)
    assert "tt.reduce" not in compiled.asm["ttir"]
    assert "arith.subi" not in compiled.asm["ttir"]


@triton.jit
def _semantic_parking_holdout(X, M, O, N: tl.constexpr,
                              BINS: tl.constexpr, PARK: tl.constexpr):
    i = tl.arange(0, N)
    live = tl.load(M + i) != 0
    x = tl.load(X + i)
    parked = tl.where(live, x, PARK)
    histogram = tl.histogram(parked, BINS)
    invalid_count = tl.reduce((~live).to(tl.int32), axis=0,
                              combine_fn=_sum_without_overflow_guard)
    bins = tl.arange(0, BINS)
    result = tl.sub(histogram, tl.where(bins == PARK, invalid_count, 0),
                    sanitize_overflow=False)
    tl.store(O + bins, result)


def test_histogram_parking_semantic_holdout():
    """A nonzero parking bin is equivalent but outside the old matcher."""
    size = 8192
    bins = 256
    park = 7
    torch.manual_seed(73)
    x = torch.randint(-8, bins + 8, (size, ), dtype=torch.int32)
    x[::11] = park
    live = torch.randint(0, 2, (size, ), dtype=torch.int32)
    device_x, device_live = x.npu(), live.npu()
    output = torch.empty((bins, ), dtype=torch.int32, device=device_x.device)
    compiled = _semantic_parking_holdout[(1, )](
        device_x, device_live, output, size, bins, park,
        enable_graph_optimize=True, sanitize_overflow=False)
    selected = x[live.bool()]
    selected = selected[(selected >= 0) & (selected < bins)]
    expected = torch.bincount(selected.long(), minlength=bins)
    torch.testing.assert_close(output.cpu().long(), expected, rtol=0, atol=0)
    assert "tt.reduce" not in compiled.asm["ttir"]
    assert "arith.subi" not in compiled.asm["ttir"]

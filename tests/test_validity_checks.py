"""
Unit tests and formal verification for all Validity Checks:
(a) coded bits vs model cross-entropy on same frames within +/-2%
(b) CM-temporal with F_t replaced by constant equals CM-intra (+/-0.5%)
(c) CM-MC with zero motion equals CM-temporal
(d) every variant trained on its own contexts computed exactly as at test time
(e) out-of-frame pixels after warping get dedicated context value
(f) report train vs test log-loss per model
"""

import numpy as np
import pytest
from carle.enhanced_codecs import (
    get_spatial_context_fast, precompute_bmap_fast,
    cm_static_encode_numba, cm_static_decode_numba,
    warp_global, warp_block_mvs, search_global_motion,
    block_motion_estimation
)

def test_check_a_coded_bits_vs_cross_entropy():
    np.random.seed(42)
    H, W = 100, 150
    mask = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    
    counts = np.random.randint(5, 50, (256, 4))
    probs = counts / counts.sum(axis=1, keepdims=True)
    
    # Theoretical cross-entropy
    ce_bits = 0.0
    for r in range(H):
        for c in range(W):
            ctx = get_spatial_context_fast(mask, r, c)
            sym = mask[r, c] - 1
            ce_bits -= np.log2(probs[ctx, sym])
            
    # Range coder frequency table
    cum_freqs = np.zeros((256, 5), dtype=np.uint16)
    for ctx in range(256):
        scaled = np.maximum(np.floor(probs[ctx] * 16380).astype(np.int64), 1)
        scaled[np.argmax(probs[ctx])] += 16384 - scaled.sum()
        cum = 0
        for s in range(4):
            cum_freqs[ctx, s] = cum
            cum += scaled[s]
        cum_freqs[ctx, 4] = 16384
        
    encoded = cm_static_encode_numba(mask, prev, False, cum_freqs)
    coded_bits = len(encoded) * 8
    
    diff_pct = abs(coded_bits - ce_bits) / ce_bits * 100.0
    assert diff_pct < 2.0, f"Check (a) failed: diff is {diff_pct:.3f}% (must be < 2%)"


def test_check_b_temporal_with_constant_equals_intra():
    np.random.seed(42)
    H, W = 100, 150
    mask = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev_const = np.ones((H, W), dtype=np.uint8) # constant 1
    
    counts_intra = np.zeros((256, 4), dtype=np.int64)
    counts_temp = np.zeros((2048, 4), dtype=np.int64)
    b_map = precompute_bmap_fast(prev_const)
    
    for r in range(H):
        for c in range(W):
            sym = mask[r, c] - 1
            s = get_spatial_context_fast(mask, r, c)
            counts_intra[s, sym] += 1
            p = (prev_const[r, c] - 1) & 3 # 0
            ib = b_map[r, c] # 0
            ctx = s | ((p | (ib << 2)) << 8) # exactly s!
            counts_temp[ctx, sym] += 1
            
    def get_cf(counts):
        num_ctx = counts.shape[0]
        probs = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + 4.0)
        cf = np.zeros((num_ctx, 5), dtype=np.uint16)
        for c in range(num_ctx):
            scaled = np.maximum(np.floor(probs[c] * 16380).astype(np.int64), 1)
            scaled[np.argmax(probs[c])] += 16384 - scaled.sum()
            cum = 0
            for s in range(4):
                cf[c, s] = cum
                cum += scaled[s]
            cf[c, 4] = 16384
        return cf
        
    cf_intra = get_cf(counts_intra)
    cf_temp = get_cf(counts_temp)
    
    enc_intra = cm_static_encode_numba(mask, prev_const, False, cf_intra)
    enc_temp = cm_static_encode_numba(mask, prev_const, True, cf_temp)
    
    diff_pct = abs(len(enc_intra) - len(enc_temp)) / len(enc_intra) * 100.0
    assert diff_pct <= 0.5, f"Check (b) failed: diff is {diff_pct:.3f}%"
    assert len(enc_intra) == len(enc_temp)
    assert np.array_equal(enc_intra, enc_temp)


def test_check_c_mc_zero_motion_equals_temporal():
    np.random.seed(42)
    H, W = 100, 150
    mask = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    cf = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (2048, 1))
    
    warped_0 = warp_global(prev, 0, 0)
    assert np.array_equal(prev, warped_0)
    
    enc_temp = cm_static_encode_numba(mask, prev, True, cf)
    enc_mc0 = cm_static_encode_numba(mask, warped_0, True, cf)
    assert np.array_equal(enc_temp, enc_mc0), "Check (c) failed: zero motion MC must equal temporal"

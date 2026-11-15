import numpy as np
import pytest
from carle.enhanced_codecs import (
    cm_static_encode_numba, cm_static_decode_numba,
    cm_adaptive_encode_numba, cm_adaptive_decode_numba,
    search_global_motion, warp_global,
    block_motion_estimation,
    cm_flicker_encode_numba, cm_flicker_decode_numba
)

def test_static_cm_bit_exact():
    np.random.seed(42)
    H, W = 100, 150
    curr = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    
    # 1. Intra
    cf_intra = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (256, 1))
    enc_intra = cm_static_encode_numba(curr, prev, False, cf_intra)
    dec_intra = cm_static_decode_numba(enc_intra, H, W, prev, False, cf_intra)
    assert np.array_equal(curr, dec_intra), "Static CM-Intra decode mismatch"
    
    # 2. Temporal
    cf_temp = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (2048, 1))
    enc_temp = cm_static_encode_numba(curr, prev, True, cf_temp)
    dec_temp = cm_static_decode_numba(enc_temp, H, W, prev, True, cf_temp)
    assert np.array_equal(curr, dec_temp), "Static CM-Temporal decode mismatch"


def test_adaptive_cm_bit_exact():
    np.random.seed(42)
    H, W = 100, 150
    curr = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    
    # 1. Adaptive Intra
    enc_ad_intra = cm_adaptive_encode_numba(curr, prev, False)
    dec_ad_intra = cm_adaptive_decode_numba(enc_ad_intra, H, W, prev, False)
    assert np.array_equal(curr, dec_ad_intra), "Adaptive CM-Intra decode mismatch"
    
    # 2. Adaptive Temporal
    enc_ad_temp = cm_adaptive_encode_numba(curr, prev, True)
    dec_ad_temp = cm_adaptive_decode_numba(enc_ad_temp, H, W, prev, True)
    assert np.array_equal(curr, dec_ad_temp), "Adaptive CM-Temporal decode mismatch"


def test_motion_compensation_bit_exact():
    np.random.seed(42)
    H, W = 100, 150
    curr = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    
    # Global Motion
    dx, dy = search_global_motion(curr, prev, max_shift=16)
    warped_global = warp_global(prev, dx, dy)
    assert warped_global.shape == curr.shape
    
    # Block Motion
    mvs, warped_block = block_motion_estimation(curr, prev, block_size=16, max_range=4)
    assert warped_block.shape == curr.shape
    
    # CM on warped reference
    cf_temp = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (2048, 1))
    enc = cm_static_encode_numba(curr, warped_block, True, cf_temp)
    dec = cm_static_decode_numba(enc, H, W, warped_block, True, cf_temp)
    assert np.array_equal(curr, dec), "Motion compensated CM decode mismatch"


def test_flicker_context_bit_exact():
    np.random.seed(42)
    H, W = 100, 150
    curr = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    prev2 = np.random.randint(1, 5, (H, W), dtype=np.uint8)
    
    cf_flicker = np.tile(np.array([0, 4096, 8192, 12288, 16384], dtype=np.uint16), (2048, 1))
    enc = cm_flicker_encode_numba(curr, prev, prev2, cf_flicker)
    dec = cm_flicker_decode_numba(enc, H, W, prev, prev2, cf_flicker)
    assert np.array_equal(curr, dec), "Flicker CM decode mismatch"

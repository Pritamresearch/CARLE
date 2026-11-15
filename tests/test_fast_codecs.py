import numpy as np
import pytest
from carle.fast_codecs import (
    CARLECanonicalNumba,
    CARLEv2Numba,
    ContextModelCoder,
    extract_diff_runs_numba,
    apply_diff_runs_numba
)

def test_extract_apply_runs():
    H, W = 100, 150
    m1 = np.ones((H, W), dtype=np.uint8)
    m2 = m1.copy()
    m2[20:40, 30:60] = 2
    m2[50:60, 80:100] = 3
    
    starts, lengths, classes = extract_diff_runs_numba(m2, m1)
    rec = apply_diff_runs_numba(m1, starts, lengths, classes)
    assert np.array_equal(rec, m2), "Diff runs failed reconstruction"

def test_carle_canonical_numba():
    codec = CARLECanonicalNumba()
    H, W = 100, 150
    m1 = np.random.randint(1, 5, size=(H, W), dtype=np.uint8)
    m2 = m1.copy()
    m2[10:30, 10:30] = 2
    
    # Frame 1
    p1 = codec.encode(m1)
    d1 = codec.decode(p1)
    assert np.array_equal(d1, m1), "Canonical Numba Frame 1 mismatch"
    
    # Frame 2
    p2 = codec.encode(m2)
    d2 = codec.decode(p2)
    assert np.array_equal(d2, m2), "Canonical Numba Frame 2 mismatch"

def test_carle_v2_numba():
    codec = CARLEv2Numba()
    H, W = 100, 150
    m1 = np.random.randint(1, 5, size=(H, W), dtype=np.uint8)
    m2 = m1.copy()
    m2[10:30, 10:30] = 2
    
    p1 = codec.encode(m1)
    d1 = codec.decode(p1)
    assert np.array_equal(d1, m1), "v2 Numba Frame 1 mismatch"
    
    p2 = codec.encode(m2)
    d2 = codec.decode(p2)
    assert np.array_equal(d2, m2), "v2 Numba Frame 2 mismatch"

def test_cm_intra_and_temporal():
    H, W = 50, 60
    m1 = np.random.randint(1, 5, size=(H, W), dtype=np.uint8)
    m2 = m1.copy()
    m2[10:25, 10:25] = 3
    
    for mode in ["intra", "temporal"]:
        codec = ContextModelCoder(mode=mode)
        p1 = codec.encode(m1)
        d1 = codec.decode(p1)
        assert np.array_equal(d1, m1), f"CM-{mode} Frame 1 mismatch"
        
        p2 = codec.encode(m2)
        d2 = codec.decode(p2)
        assert np.array_equal(d2, m2), f"CM-{mode} Frame 2 mismatch"

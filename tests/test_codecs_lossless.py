"""
Pytest unit tests for all CARLE and baseline codecs.
Ensures decode(encode(F)) == F on every frame, 0 failures allowed.
"""

import numpy as np
import pytest
from carle.codecs import (
    FullMapPNG,
    FullMapZstd,
    DiffMapPNG,
    DiffMapZstd,
    CARLECanonical,
    CARLEv2_rANS,
    CARLEv2_IP,
)

@pytest.fixture
def test_masks():
    H, W = 100, 150
    # Frame 0: mostly road (1) with some vehicles (2) and pedestrians (3)
    f0 = np.ones((H, W), dtype=np.uint8)
    f0[30:50, 40:80] = 2
    f0[60:70, 90:100] = 3
    
    # Frame 1: vehicle moves right, new pedestrian appears
    f1 = f0.copy()
    f1[30:50, 40:50] = 1 # vacated becomes road
    f1[30:50, 80:90] = 2 # extended
    f1[10:20, 10:20] = 3 # new pedestrian
    
    # Frame 2: identical to Frame 1 (zero change)
    f2 = f1.copy()
    
    # Frame 3: large scene change (valid classes 1..4, where 0 is reserved for null)
    f3 = np.random.choice([1, 2, 3, 4], size=(H, W), p=[0.6, 0.2, 0.15, 0.05]).astype(np.uint8)
    
    # Frame 4: single pixel change
    f4 = f3.copy()
    f4[50, 50] = 4 if f3[50, 50] != 4 else 2
    
    # Frame 5: entire frame changes
    f5 = np.where(f4 == 1, 2, 1).astype(np.uint8)

    return [f0, f1, f2, f3, f4, f5]


def test_full_map_png(test_masks):
    codec = FullMapPNG(level=9)
    for m in test_masks:
        enc = codec.encode(m)
        dec = codec.decode(enc)
        assert np.array_equal(dec, m), "FullMapPNG failed exact equality"


def test_full_map_zstd(test_masks):
    for lvl in [3, 19]:
        codec = FullMapZstd(level=lvl)
        for m in test_masks:
            enc = codec.encode(m)
            dec = codec.decode(enc)
            assert np.array_equal(dec, m), f"FullMapZstd (lvl {lvl}) failed exact equality"


def test_diff_map_png(test_masks):
    codec = DiffMapPNG(level=9)
    prev = None
    for m in test_masks:
        enc = codec.encode(m, prev)
        dec = codec.decode(enc, prev)
        assert np.array_equal(dec, m), "DiffMapPNG failed exact equality"
        prev = dec


def test_diff_map_zstd(test_masks):
    for lvl in [3, 19]:
        codec = DiffMapZstd(level=lvl)
        prev = None
        for m in test_masks:
            enc = codec.encode(m, prev)
            dec = codec.decode(enc, prev)
            assert np.array_equal(dec, m), f"DiffMapZstd (lvl {lvl}) failed exact equality"
            prev = dec


def test_carle_canonical(test_masks):
    codec = CARLECanonical()
    prev = None
    for m in test_masks:
        enc = codec.encode(m, prev)
        dec = codec.decode(enc, prev)
        assert np.array_equal(dec, m), "CARLECanonical failed exact equality"
        prev = dec


def test_carle_v2_rans(test_masks):
    codec = CARLEv2_rANS()
    prev = None
    for m in test_masks:
        enc = codec.encode(m, prev)
        dec = codec.decode(enc, prev)
        assert np.array_equal(dec, m), "CARLEv2_rANS failed exact equality"
        prev = dec


def test_carle_v2_ip(test_masks):
    for n in [2, 10, float("inf")]:
        codec = CARLEv2_IP(refresh_N=n)
        prev = None
        for m in test_masks:
            enc = codec.encode(m, prev)
            dec = codec.decode(enc, prev)
            assert np.array_equal(dec, m), f"CARLEv2_IP (N={n}) failed exact equality"
            prev = dec


def test_full_resolution_lossless():
    """Test on 900x1600 realistic masks."""
    H, W = 900, 1600
    m0 = np.ones((H, W), dtype=np.uint8) # road
    m0[400:600, 700:1000] = 2 # vehicle
    m0[350:450, 300:350] = 3 # pedestrian
    
    m1 = m0.copy()
    m1[400:600, 750:1050] = 2 # vehicle moves right
    m1[400:600, 700:750] = 1 # vacated road
    
    codecs = [
        FullMapPNG(level=9),
        FullMapZstd(level=3),
        DiffMapPNG(level=9),
        DiffMapZstd(level=3),
        CARLECanonical(),
        CARLEv2_rANS(),
        CARLEv2_IP(refresh_N=10),
    ]
    for c in codecs:
        enc0 = c.encode(m0)
        dec0 = c.decode(enc0)
        assert np.array_equal(dec0, m0), f"{c.__class__.__name__} failed on frame 0"
        
        enc1 = c.encode(m1, dec0)
        dec1 = c.decode(enc1, dec0)
        assert np.array_equal(dec1, m1), f"{c.__class__.__name__} failed on frame 1"

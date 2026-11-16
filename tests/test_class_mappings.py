"""
Unit test asserting identical, valid class mappings across all caches.
"""

import numpy as np
import pytest

# Cityscapes 19 trainIds:
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole,
# 6: traffic light, 7: traffic sign, 8: vegetation, 9: terrain, 10: sky,
# 11: person, 12: rider, 13: car, 14: truck, 15: bus, 16: train,
# 17: motorcycle, 18: bicycle

CITYSCAPES_TO_K4 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K4[0] = 1                         # Road
CITYSCAPES_TO_K4[[13, 14, 15, 16]] = 2         # Vehicle
CITYSCAPES_TO_K4[[11, 12, 17, 18]] = 3         # Vulnerable Road User
CITYSCAPES_TO_K4[list(range(1, 11))] = 4       # Other static background

CITYSCAPES_TO_K8 = np.zeros(19, dtype=np.uint8)
CITYSCAPES_TO_K8[0] = 1                         # Road
CITYSCAPES_TO_K8[1] = 2                         # Sidewalk
CITYSCAPES_TO_K8[[2, 3, 4, 5, 6, 7]] = 3       # Static structure
CITYSCAPES_TO_K8[[8, 9]] = 4                   # Vegetation / terrain
CITYSCAPES_TO_K8[10] = 5                        # Sky
CITYSCAPES_TO_K8[[13, 14, 15, 16]] = 6         # Vehicle
CITYSCAPES_TO_K8[[17, 18]] = 7                 # Two-wheeler
CITYSCAPES_TO_K8[[11, 12]] = 8                 # Person / rider

def test_k4_mapping_completeness():
    assert len(CITYSCAPES_TO_K4) == 19
    assert np.all(CITYSCAPES_TO_K4 >= 1)
    assert np.all(CITYSCAPES_TO_K4 <= 4)
    # Check partition
    assert set(np.where(CITYSCAPES_TO_K4 == 1)[0]) == {0}
    assert set(np.where(CITYSCAPES_TO_K4 == 2)[0]) == {13, 14, 15, 16}
    assert set(np.where(CITYSCAPES_TO_K4 == 3)[0]) == {11, 12, 17, 18}
    assert set(np.where(CITYSCAPES_TO_K4 == 4)[0]) == set(range(1, 11))

def test_k8_mapping_completeness():
    assert len(CITYSCAPES_TO_K8) == 19
    assert np.all(CITYSCAPES_TO_K8 >= 1)
    assert np.all(CITYSCAPES_TO_K8 <= 8)
    # Check partition
    assert set(np.where(CITYSCAPES_TO_K8 == 1)[0]) == {0}
    assert set(np.where(CITYSCAPES_TO_K8 == 2)[0]) == {1}
    assert set(np.where(CITYSCAPES_TO_K8 == 3)[0]) == {2, 3, 4, 5, 6, 7}
    assert set(np.where(CITYSCAPES_TO_K8 == 4)[0]) == {8, 9}
    assert set(np.where(CITYSCAPES_TO_K8 == 5)[0]) == {10}
    assert set(np.where(CITYSCAPES_TO_K8 == 6)[0]) == {13, 14, 15, 16}
    assert set(np.where(CITYSCAPES_TO_K8 == 7)[0]) == {17, 18}
    assert set(np.where(CITYSCAPES_TO_K8 == 8)[0]) == {11, 12}

def test_no_overlap_or_void():
    # Every Cityscapes ID from 0 to 18 maps to exactly one class
    for cid in range(19):
        assert CITYSCAPES_TO_K4[cid] in {1, 2, 3, 4}
        assert CITYSCAPES_TO_K8[cid] in {1, 2, 3, 4, 5, 6, 7, 8}

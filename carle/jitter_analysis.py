"""
Numba-accelerated Boundary Jitter and Flicker Analysis.

Computes:
1. Pixel-level difference delta_pix
2. Class boundary mask of F_t (4-connected)
3. Boundary distance dilation (within 1 px, within 2 px)
4. Fraction of changed pixels within 1 px and 2 px of class boundaries
5. 3-frame flicker rate (pixels that flip and flip back: F_{t+2} == F_t != F_{t+1})
"""

import numpy as np
import numba
from numba import njit, uint8, int64, float64

@njit(fastmath=True)
def compute_boundary_mask_numba(mask: np.ndarray) -> np.ndarray:
    """Computes binary mask where 1 indicates pixel is adjacent to a class boundary."""
    H, W = mask.shape
    boundary = np.zeros((H, W), dtype=np.uint8)
    
    for r in range(H):
        for c in range(W):
            val = mask[r, c]
            is_b = False
            if r > 0 and mask[r - 1, c] != val:
                is_b = True
            elif r < H - 1 and mask[r + 1, c] != val:
                is_b = True
            elif c > 0 and mask[r, c - 1] != val:
                is_b = True
            elif c < W - 1 and mask[r, c + 1] != val:
                is_b = True
                
            if is_b:
                boundary[r, c] = 1
                
    return boundary

@njit(fastmath=True)
def dilate_mask_numba(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilates a binary mask with Manhattan/Chebyshev neighborhood of given radius."""
    H, W = mask.shape
    out = mask.copy()
    
    for r in range(H):
        for c in range(W):
            if mask[r, c] == 1:
                r_min = max(0, r - radius)
                r_max = min(H - 1, r + radius)
                c_min = max(0, c - radius)
                c_max = min(W - 1, c + radius)
                for rr in range(r_min, r_max + 1):
                    for cc in range(c_min, c_max + 1):
                        out[rr, cc] = 1
                        
    return out

@njit(fastmath=True)
def analyze_jitter_pair_numba(mask_t: np.ndarray, mask_t1: np.ndarray):
    """
    Analyzes jitter between mask_t and mask_t1.
    Returns:
        total_changed: int
        changed_within_1px: int
        changed_within_2px: int
    """
    H, W = mask_t.shape
    boundary_t = compute_boundary_mask_numba(mask_t)
    bound_1px = dilate_mask_numba(boundary_t, 1)
    bound_2px = dilate_mask_numba(boundary_t, 2)
    
    total_changed = 0
    changed_within_1px = 0
    changed_within_2px = 0
    
    for r in range(H):
        for c in range(W):
            if mask_t1[r, c] != mask_t[r, c]:
                total_changed += 1
                if bound_1px[r, c] == 1:
                    changed_within_1px += 1
                if bound_2px[r, c] == 1:
                    changed_within_2px += 1
                    
    return total_changed, changed_within_1px, changed_within_2px

@njit(fastmath=True)
def analyze_flicker_triplet_numba(mask_t: np.ndarray, mask_t1: np.ndarray, mask_t2: np.ndarray):
    """
    Analyzes 3-frame flicker:
    Pixel flips at t+1 and flips back at t+2: mask_t2[r, c] == mask_t[r, c] and mask_t1[r, c] != mask_t[r, c].
    """
    H, W = mask_t.shape
    flicker_count = 0
    changed_count = 0
    
    for r in range(H):
        for c in range(W):
            v0 = mask_t[r, c]
            v1 = mask_t1[r, c]
            v2 = mask_t2[r, c]
            
            if v1 != v0:
                changed_count += 1
                if v2 == v0:
                    flicker_count += 1
                    
    return flicker_count, changed_count

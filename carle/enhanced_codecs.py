"""
Enhanced Lossless Codecs for Autonomous Vehicle Semantic Label Maps.
Contains:
1. Fast Numba CM-Intra and CM-Temporal (precomputed boundary map, ~3-5 ms)
2. Adaptive Range Coders: CM-Intra-Adaptive and CM-Temporal-Adaptive (0 byte table)
3. Global Motion Compensation (GMC) + CM
4. Block Motion Compensation (BMC, 32x32 and 16x16) + CM
5. Flicker-Aware Context (2-frame history F_t, F_{t-1})
6. Baselines: JPEG-XL (libjxl), WebP (libwebp), JPEG-LS (CharLS), Zstandard
"""

import struct
import io
import time
import numpy as np
from numba import njit
from PIL import Image
import imagecodecs
import zstandard as zstd

# Range Coder Constants (32-bit range, 16-bit cumulative frequencies)
RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)


# -----------------------------------------------------------------------------
# 1. Spatial and Fast Temporal Context Extraction
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def get_spatial_context_fast(mask: np.ndarray, r: int, c: int) -> int:
    H, W = mask.shape
    w_val = (mask[r, c - 1] - 1) if c > 0 else 0
    n_val = (mask[r - 1, c] - 1) if r > 0 else 0
    nw_val = (mask[r - 1, c - 1] - 1) if (r > 0 and c > 0) else 0
    ne_val = (mask[r - 1, c + 1] - 1) if (r > 0 and c < W - 1) else 0
    return (w_val & 3) | ((n_val & 3) << 2) | ((nw_val & 3) << 4) | ((ne_val & 3) << 6)


@njit(fastmath=True)
def precompute_bmap_fast(prev: np.ndarray) -> np.ndarray:
    H, W = prev.shape
    b_map = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        r_p = r - 1 if r > 0 else 0
        r_n = r + 1 if r < H - 1 else H - 1
        for c in range(W):
            c_p = c - 1 if c > 0 else 0
            c_n = c + 1 if c < W - 1 else W - 1
            v = prev[r, c]
            if (prev[r_p, c] != v or prev[r_n, c] != v or
                prev[r, c_p] != v or prev[r, c_n] != v):
                b_map[r, c] = 1
    return b_map


# -----------------------------------------------------------------------------
# 2. Optimized Static CM Encode & Decode (Pretrained Table)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_static_encode_numba(mask_curr: np.ndarray, mask_prev: np.ndarray,
                           use_temporal: bool, cum_freqs: np.ndarray) -> np.ndarray:
    H, W = mask_curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    b_map = precompute_bmap_fast(mask_prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(mask_curr, r, c)
            if use_temporal:
                p_val = (mask_prev[r, c] - 1) & 3
                is_b = b_map[r, c]
                ctx = spat | ((p_val | (is_b << 2)) << 8)
            else:
                ctx = spat
                
            sym = int(mask_curr[r, c] - 1)
            c_low = np.int64(cum_freqs[ctx, sym])
            c_high = np.int64(cum_freqs[ctx, sym + 1])
            
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    for _ in range(4):
        out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
        out_ptr += 1
        low = (low << 8) & np.int64(0xFFFFFFFF)
        
    return out_buf[:out_ptr]


@njit(fastmath=True)
def cm_static_decode_numba(payload: np.ndarray, H: int, W: int,
                           mask_prev: np.ndarray, use_temporal: bool,
                           cum_freqs: np.ndarray) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    b_map = precompute_bmap_fast(mask_prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
        code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
        in_ptr += 1
        
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(decoded, r, c)
            if use_temporal:
                p_val = (mask_prev[r, c] - 1) & 3
                is_b = b_map[r, c]
                ctx = spat | ((p_val | (is_b << 2)) << 8)
            else:
                ctx = spat
                
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            # Find symbol
            sym = 0
            for s in range(4):
                if val >= cum_freqs[ctx, s] and val < cum_freqs[ctx, s + 1]:
                    sym = s
                    break
                    
            decoded[r, c] = np.uint8(sym + 1)
            
            c_low = np.int64(cum_freqs[ctx, sym])
            c_high = np.int64(cum_freqs[ctx, sym + 1])
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    return decoded


# -----------------------------------------------------------------------------
# 3. Adaptive CM Encode & Decode (Zero Pretrained Table)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_adaptive_encode_numba(mask_curr: np.ndarray, mask_prev: np.ndarray,
                             use_temporal: bool) -> np.ndarray:
    H, W = mask_curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    num_contexts = 2048 if use_temporal else 256
    counts = np.ones((num_contexts, 4), dtype=np.uint16)
    totals = np.full(num_contexts, 4, dtype=np.uint16)
    
    b_map = precompute_bmap_fast(mask_prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(mask_curr, r, c)
            if use_temporal:
                p_val = (mask_prev[r, c] - 1) & 3
                is_b = b_map[r, c]
                ctx = spat | ((p_val | (is_b << 2)) << 8)
            else:
                ctx = spat
                
            sym = int(mask_curr[r, c] - 1)
            tot = int(totals[ctx])
            
            c_low_count = 0
            for s in range(sym):
                c_low_count += int(counts[ctx, s])
            c_high_count = c_low_count + int(counts[ctx, sym])
            
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low:
                c_high = c_low + 1
            if c_high > 16384:
                c_high = 16384
                
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            # Update counts
            counts[ctx, sym] += 1
            totals[ctx] += 1
            if totals[ctx] >= 2048:
                new_tot = 0
                for s in range(4):
                    counts[ctx, s] = (counts[ctx, s] + 1) >> 1
                    new_tot += counts[ctx, s]
                totals[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    for _ in range(4):
        out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
        out_ptr += 1
        low = (low << 8) & np.int64(0xFFFFFFFF)
        
    return out_buf[:out_ptr]


@njit(fastmath=True)
def cm_adaptive_decode_numba(payload: np.ndarray, H: int, W: int,
                             mask_prev: np.ndarray, use_temporal: bool) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    num_contexts = 2048 if use_temporal else 256
    counts = np.ones((num_contexts, 4), dtype=np.uint16)
    totals = np.full(num_contexts, 4, dtype=np.uint16)
    
    b_map = precompute_bmap_fast(mask_prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
        code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
        in_ptr += 1
        
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(decoded, r, c)
            if use_temporal:
                p_val = (mask_prev[r, c] - 1) & 3
                is_b = b_map[r, c]
                ctx = spat | ((p_val | (is_b << 2)) << 8)
            else:
                ctx = spat
                
            tot = int(totals[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            # Find symbol from current counts
            cum = 0
            sym = 3
            c_low_count = 0
            for s in range(4):
                next_cum = ( (cum + int(counts[ctx, s])) * 16384 ) // tot
                cur_low = ( cum * 16384 ) // tot
                if val >= cur_low and val < next_cum:
                    sym = s
                    c_low_count = cum
                    break
                cum += int(counts[ctx, s])
                
            decoded[r, c] = np.uint8(sym + 1)
            
            c_high_count = c_low_count + int(counts[ctx, sym])
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            # Update counts
            counts[ctx, sym] += 1
            totals[ctx] += 1
            if totals[ctx] >= 2048:
                new_tot = 0
                for s in range(4):
                    counts[ctx, s] = (counts[ctx, s] + 1) >> 1
                    new_tot += counts[ctx, s]
                totals[ctx] = new_tot
                
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    return decoded


# -----------------------------------------------------------------------------
# 4. Motion Estimation and Compensation
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def search_global_motion(curr: np.ndarray, prev: np.ndarray, max_shift: int = 32) -> tuple:
    H, W = curr.shape
    best_dx = 0
    best_dy = 0
    
    # 0,0 baseline difference
    d0 = 0
    for r in range(0, H, 2):
        for c in range(0, W, 2):
            if curr[r, c] != prev[r, c]:
                d0 += 1
    best_diff = d0
    
    # Coarse search
    step = 4
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            diff = 0
            for r in range(max_shift, H - max_shift, 4):
                for c in range(max_shift, W - max_shift, 4):
                    if curr[r, c] != prev[r - dy, c - dx]:
                        diff += 1
            if diff < best_diff:
                best_diff = diff
                best_dx = dx
                best_dy = dy
                
    # Fine search around best
    fine_dx = best_dx
    fine_dy = best_dy
    best_fine = best_diff
    for dy in range(best_dy - step + 1, best_dy + step):
        if dy < -max_shift or dy > max_shift: continue
        for dx in range(best_dx - step + 1, best_dx + step):
            if dx < -max_shift or dx > max_shift: continue
            diff = 0
            for r in range(max_shift, H - max_shift, 2):
                for c in range(max_shift, W - max_shift, 2):
                    if curr[r, c] != prev[r - dy, c - dx]:
                        diff += 1
            if diff < best_fine:
                best_fine = diff
                fine_dx = dx
                fine_dy = dy
                
    return fine_dx, fine_dy


@njit(fastmath=True)
def warp_global(prev: np.ndarray, dx: int, dy: int) -> np.ndarray:
    H, W = prev.shape
    warped = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        src_r = r - dy
        if src_r < 0 or src_r >= H: continue
        for c in range(W):
            src_c = c - dx
            if src_c < 0 or src_c >= W: continue
            warped[r, c] = prev[src_r, src_c]
    return warped


@njit(fastmath=True)
def block_motion_estimation(curr: np.ndarray, prev: np.ndarray,
                            block_size: int = 32, max_range: int = 8):
    H, W = curr.shape
    n_blocks_r = (H + block_size - 1) // block_size
    n_blocks_c = (W + block_size - 1) // block_size
    
    mvs = np.zeros((n_blocks_r, n_blocks_c, 2), dtype=np.int8)
    warped = np.zeros_like(prev)
    
    for br in range(n_blocks_r):
        r0 = br * block_size
        r1 = min(H, r0 + block_size)
        for bc in range(n_blocks_c):
            c0 = bc * block_size
            c1 = min(W, c0 + block_size)
            
            # Baseline diff with no motion (0,0)
            d0 = 0
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if curr[r, c] != prev[r, c]: d0 += 1
            best_diff = d0
            best_dx = 0
            best_dy = 0
            
            if d0 > 0: # only search if block changed
                for dy in range(-max_range, max_range + 1):
                    if r0 + dy < 0 or r1 + dy > H: continue
                    for dx in range(-max_range, max_range + 1):
                        if c0 + dx < 0 or c1 + dx > W: continue
                        diff = 0
                        for r in range(r0, r1, 2):
                            for c in range(c0, c1, 2):
                                if curr[r, c] != prev[r + dy, c + dx]:
                                    diff += 1
                        if diff * 4 < best_diff:
                            # Full check
                            full_diff = 0
                            for r in range(r0, r1):
                                for c in range(c0, c1):
                                    if curr[r, c] != prev[r + dy, c + dx]:
                                        full_diff += 1
                            if full_diff < best_diff:
                                best_diff = full_diff
                                best_dx = dx
                                best_dy = dy
                                
            mvs[br, bc, 0] = best_dx
            mvs[br, bc, 1] = best_dy
            for r in range(r0, r1):
                for c in range(c0, c1):
                    warped[r, c] = prev[r + best_dy, c + best_dx]
                    
    return mvs, warped


# -----------------------------------------------------------------------------
# 5. Flicker-Aware Context (Two-Frame History: F_t and F_{t-1})
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_flicker_encode_numba(curr: np.ndarray, prev: np.ndarray,
                            prev2: np.ndarray, cum_freqs: np.ndarray) -> np.ndarray:
    """
    Context includes:
    - Causal spatial (W, N, NW, NE): 256 states
    - Prev value F_t(r, c): 4 states
    - Flicker flag (F_t(r, c) != F_{t-1}(r, c)): 2 states
    Total contexts = 256 * 4 * 2 = 2048 states.
    """
    H, W = curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(curr, r, c)
            p_val = (prev[r, c] - 1) & 3
            is_flicker = 1 if prev[r, c] != prev2[r, c] else 0
            ctx = spat | ((p_val | (is_flicker << 2)) << 8)
            
            sym = int(curr[r, c] - 1)
            c_low = np.int64(cum_freqs[ctx, sym])
            c_high = np.int64(cum_freqs[ctx, sym + 1])
            
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
                    out_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    for _ in range(4):
        out_buf[out_ptr] = np.uint8((low >> 24) & 0xFF)
        out_ptr += 1
        low = (low << 8) & np.int64(0xFFFFFFFF)
        
    return out_buf[:out_ptr]


@njit(fastmath=True)
def cm_flicker_decode_numba(payload: np.ndarray, H: int, W: int,
                            prev: np.ndarray, prev2: np.ndarray,
                            cum_freqs: np.ndarray) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
        code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
        in_ptr += 1
        
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            spat = get_spatial_context_fast(decoded, r, c)
            p_val = (prev[r, c] - 1) & 3
            is_flicker = 1 if prev[r, c] != prev2[r, c] else 0
            ctx = spat | ((p_val | (is_flicker << 2)) << 8)
            
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            sym = 0
            for s in range(4):
                if val >= cum_freqs[ctx, s] and val < cum_freqs[ctx, s + 1]:
                    sym = s
                    break
            decoded[r, c] = np.uint8(sym + 1)
            
            c_low = np.int64(cum_freqs[ctx, sym])
            c_high = np.int64(cum_freqs[ctx, sym + 1])
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
                    code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
                    in_ptr += 1
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    return decoded


@njit(fastmath=True)
def warp_block_mvs(prev: np.ndarray, mvs: np.ndarray, block_size: int = 32) -> np.ndarray:
    H, W = prev.shape
    warped = np.zeros_like(prev)
    n_br, n_bc, _ = mvs.shape
    for br in range(n_br):
        r0 = br * block_size
        r1 = min(H, r0 + block_size)
        for bc in range(n_bc):
            c0 = bc * block_size
            c1 = min(W, c0 + block_size)
            b_dx = int(mvs[br, bc, 0])
            b_dy = int(mvs[br, bc, 1])
            for r in range(r0, r1):
                for c in range(c0, c1):
                    warped[r, c] = prev[r + b_dy, c + b_dx]
    return warped

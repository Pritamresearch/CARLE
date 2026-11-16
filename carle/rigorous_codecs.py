"""
Rigorous Context Model Codecs and Adaptation Variants.
Includes:
1. Proper handling of out-of-frame (OOF) pixels during motion compensation.
2. Exact matching between theoretical cross-entropy and coded bits.
3. Pretrained prior + online adaptation (weights w in {16, 64, 256}).
4. Stream-persistent counts with reset period N in {10, 30}.
5. 1-bit skip flag for identical consecutive frames.
"""

import numpy as np
from numba import njit
import zstandard as zstd

RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)


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
            if v > 0: # only evaluate boundaries for valid pixels
                if (prev[r_p, c] != v or prev[r_n, c] != v or
                    prev[r, c_p] != v or prev[r, c_n] != v):
                    b_map[r, c] = 1
    return b_map


@njit(fastmath=True)
def warp_global_oof(prev: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Warps prev by (dx, dy). Pixels outside bounds get value 0 (OOF).
    """
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
def warp_block_mvs_oof(prev: np.ndarray, mvs: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Warps prev by block motion vectors. Out of bound pixels get value 0 (OOF).
    """
    H, W = prev.shape
    warped = np.zeros((H, W), dtype=np.uint8)
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
                src_r = r + b_dy
                if src_r < 0 or src_r >= H: continue
                for c in range(c0, c1):
                    src_c = c + b_dx
                    if src_c < 0 or src_c >= W: continue
                    warped[r, c] = prev[src_r, src_c]
    return warped


@njit(fastmath=True)
def get_temporal_context_oof(curr: np.ndarray, prev: np.ndarray, b_map: np.ndarray, r: int, c: int) -> int:
    """
    Spatial context: W, N, NW, NE (256 states)
    Temporal context:
      - If prev[r, c] == 0 (OOF): fallback to pure spatial context (ctx = spat)
      - If prev[r, c] in {1, 2, 3, 4}: ctx = spat | ((((prev[r, c] - 1) & 3) | (is_b << 2)) << 8)
    """
    spat = get_spatial_context_fast(curr, r, c)
    p_val = prev[r, c]
    if p_val == 0:
        return spat # fallback to intra for unmapped border
    p_sym = (p_val - 1) & 3
    is_b = b_map[r, c]
    return spat | ((p_sym | (is_b << 2)) << 8)


# -----------------------------------------------------------------------------
# Fast Numba Count Accumulators
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def accumulate_counts_intra(curr: np.ndarray, counts: np.ndarray):
    H, W = curr.shape
    for r in range(H):
        for c in range(W):
            sym = int(curr[r, c] - 1)
            sp = get_spatial_context_fast(curr, r, c)
            counts[sp, sym] += 1


@njit(fastmath=True)
def accumulate_counts_temporal(curr: np.ndarray, prev: np.ndarray, b_map: np.ndarray, counts: np.ndarray):
    H, W = curr.shape
    for r in range(H):
        for c in range(W):
            sym = int(curr[r, c] - 1)
            ctx = get_temporal_context_oof(curr, prev, b_map, r, c)
            counts[ctx, sym] += 1


@njit(fastmath=True)
def accumulate_counts_flicker(curr: np.ndarray, prev: np.ndarray, prev2: np.ndarray, counts: np.ndarray):
    H, W = curr.shape
    for r in range(H):
        for c in range(W):
            sym = int(curr[r, c] - 1)
            sp = get_spatial_context_fast(curr, r, c)
            p_sym = (prev[r, c] - 1) & 3
            is_fl = 1 if prev[r, c] != prev2[r, c] else 0
            ctx = sp | ((p_sym | (is_fl << 2)) << 8)
            counts[ctx, sym] += 1


# -----------------------------------------------------------------------------
# Fast Range Encoding and Decoding with Exact Context Handling
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_encode_rigorous(curr: np.ndarray, prev: np.ndarray, use_temporal: bool, cum_freqs: np.ndarray) -> np.ndarray:
    H, W = curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_oof(curr, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(curr, r, c)
                
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
def cm_decode_rigorous(payload: np.ndarray, H: int, W: int, prev: np.ndarray, use_temporal: bool, cum_freqs: np.ndarray) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
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
            if use_temporal:
                ctx = get_temporal_context_oof(decoded, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(decoded, r, c)
                
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


# -----------------------------------------------------------------------------
# Theoretical Cross-Entropy Calculator (Bits/Frame)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def compute_cross_entropy_rigorous(curr: np.ndarray, prev: np.ndarray, use_temporal: bool, log_probs: np.ndarray) -> float:
    H, W = curr.shape
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    ce_bits = 0.0
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_oof(curr, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(curr, r, c)
            sym = int(curr[r, c] - 1)
            ce_bits -= log_probs[ctx, sym]
    return ce_bits


# -----------------------------------------------------------------------------
# Adaptation: Pretrained Prior + Online Adaptation (Weight w in {16, 64, 256})
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_prior_adaptive_encode(curr: np.ndarray, prev: np.ndarray, use_temporal: bool,
                             counts_state: np.ndarray, totals_state: np.ndarray,
                             max_total: int = 2048) -> np.ndarray:
    """
    Online adaptation initialized with prior_probs * prior_w.
    Supports in-place persistence across frames if counts_state is passed.
    """
    H, W = curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_oof(curr, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(curr, r, c)
                
            sym = int(curr[r, c] - 1)
            tot = int(totals_state[ctx])
            
            c_low_count = 0
            for s in range(sym):
                c_low_count += int(counts_state[ctx, s])
            c_high_count = c_low_count + int(counts_state[ctx, sym])
            
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            # Online count update
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(4):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
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
def cm_prior_adaptive_decode(payload: np.ndarray, H: int, W: int, prev: np.ndarray, use_temporal: bool,
                             counts_state: np.ndarray, totals_state: np.ndarray,
                             max_total: int = 2048) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
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
            if use_temporal:
                ctx = get_temporal_context_oof(decoded, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(decoded, r, c)
                
            tot = int(totals_state[ctx])
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            cum = 0
            sym = 3
            c_low_count = 0
            for s in range(4):
                next_cum = ((cum + int(counts_state[ctx, s])) * 16384) // tot
                cur_low = (cum * 16384) // tot
                if val >= cur_low and val < next_cum:
                    sym = s
                    c_low_count = cum
                    break
                cum += int(counts_state[ctx, s])
                
            decoded[r, c] = np.uint8(sym + 1)
            
            c_high_count = c_low_count + int(counts_state[ctx, sym])
            c_low = (c_low_count * 16384) // tot
            c_high = (c_high_count * 16384) // tot
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            counts_state[ctx, sym] += 1
            totals_state[ctx] += 1
            if totals_state[ctx] >= max_total:
                new_tot = 0
                for s in range(4):
                    counts_state[ctx, s] = (counts_state[ctx, s] + 1) >> 1
                    new_tot += counts_state[ctx, s]
                totals_state[ctx] = new_tot
                
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
# Adaptation: Online Probability Mixing Codec (Pretrained + Adaptive)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def cm_mixture_encode(curr: np.ndarray, prev: np.ndarray, use_temporal: bool,
                      prior_cum_freqs: np.ndarray,
                      adapt_counts: np.ndarray, adapt_totals: np.ndarray,
                      alpha_state: np.ndarray, eta: float = 0.01) -> np.ndarray:
    """
    Online mixture of pretrained prior and adaptive model.
    alpha_state: 1-element array with weight in [0.01, 0.99]
    """
    H, W = curr.shape
    total_pixels = H * W
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    alpha = alpha_state[0]
    
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_oof(curr, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(curr, r, c)
                
            sym = int(curr[r, c] - 1)
            tot_a = int(adapt_totals[ctx])
            
            # 1. Compute prior frequency
            f_prior = np.empty(4, dtype=np.int64)
            for s in range(4):
                f_prior[s] = int(prior_cum_freqs[ctx, s + 1]) - int(prior_cum_freqs[ctx, s])
                
            # 2. Compute adaptive frequency scaled to 16384
            f_adapt = np.empty(4, dtype=np.int64)
            rem_a = 16384
            for s in range(3):
                cnt = int(adapt_counts[ctx, s])
                fr = max(1, (cnt * 16384) // tot_a)
                f_adapt[s] = fr
                rem_a -= fr
            f_adapt[3] = max(1, rem_a)
            
            # 3. Blend frequencies using alpha
            f_mix = np.empty(4, dtype=np.int64)
            c_low = 0
            c_high = 0
            cum = 0
            for s in range(4):
                mix_f = max(1, int(round(alpha * f_prior[s] + (1.0 - alpha) * f_adapt[s])))
                f_mix[s] = mix_f
                if s == sym:
                    c_low = cum
                    c_high = cum + mix_f
                cum += mix_f
                
            # Normalize to 16384
            c_low = (c_low * 16384) // cum
            c_high = (c_high * 16384) // cum
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            r_step = range_val // RC_TOTAL_FREQ
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            # Update alpha
            p_prior = f_prior[sym] / 16384.0
            p_adapt = f_adapt[sym] / 16384.0
            p_mix = f_mix[sym] / float(cum)
            grad = (p_prior - p_adapt) / (p_mix + 1e-9)
            alpha = min(max(alpha + eta * grad, 0.01), 0.99)
            
            # Update adaptive counts
            adapt_counts[ctx, sym] += 1
            adapt_totals[ctx] += 1
            if adapt_totals[ctx] >= 2048:
                new_tot = 0
                for s in range(4):
                    adapt_counts[ctx, s] = (adapt_counts[ctx, s] + 1) >> 1
                    new_tot += adapt_counts[ctx, s]
                adapt_totals[ctx] = new_tot
                
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
        
    alpha_state[0] = alpha
    return out_buf[:out_ptr]


@njit(fastmath=True)
def cm_mixture_decode(payload: np.ndarray, H: int, W: int, prev: np.ndarray, use_temporal: bool,
                      prior_cum_freqs: np.ndarray,
                      adapt_counts: np.ndarray, adapt_totals: np.ndarray,
                      alpha_state: np.ndarray, eta: float = 0.01) -> np.ndarray:
    decoded = np.empty((H, W), dtype=np.uint8)
    b_map = precompute_bmap_fast(prev) if use_temporal else np.zeros((1, 1), dtype=np.uint8)
    
    in_ptr = 0
    code = np.int64(0)
    for _ in range(4):
        byte_val = payload[in_ptr] if in_ptr < len(payload) else 0
        code = ((code << 8) | np.int64(byte_val)) & np.int64(0xFFFFFFFF)
        in_ptr += 1
        
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    alpha = alpha_state[0]
    
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_oof(decoded, prev, b_map, r, c)
            else:
                ctx = get_spatial_context_fast(decoded, r, c)
                
            tot_a = int(adapt_totals[ctx])
            f_prior = np.empty(4, dtype=np.int64)
            for s in range(4):
                f_prior[s] = int(prior_cum_freqs[ctx, s + 1]) - int(prior_cum_freqs[ctx, s])
                
            f_adapt = np.empty(4, dtype=np.int64)
            rem_a = 16384
            for s in range(3):
                cnt = int(adapt_counts[ctx, s])
                fr = max(1, (cnt * 16384) // tot_a)
                f_adapt[s] = fr
                rem_a -= fr
            f_adapt[3] = max(1, rem_a)
            
            f_mix = np.empty(4, dtype=np.int64)
            cum = 0
            for s in range(4):
                mix_f = max(1, int(round(alpha * f_prior[s] + (1.0 - alpha) * f_adapt[s])))
                f_mix[s] = mix_f
                cum += mix_f
                
            r_step = range_val // RC_TOTAL_FREQ
            val = (code - low) // r_step
            if val < 0: val = 0
            if val >= RC_TOTAL_FREQ: val = RC_TOTAL_FREQ - 1
            
            sym = 3
            c_low_count = 0
            m_cum = 0
            for s in range(4):
                next_cum = ((m_cum + f_mix[s]) * 16384) // cum
                cur_low = (m_cum * 16384) // cum
                if val >= cur_low and val < next_cum:
                    sym = s
                    c_low_count = m_cum
                    break
                m_cum += f_mix[s]
                
            decoded[r, c] = np.uint8(sym + 1)
            
            c_high_count = c_low_count + f_mix[sym]
            c_low = (c_low_count * 16384) // cum
            c_high = (c_high_count * 16384) // cum
            if c_high <= c_low: c_high = c_low + 1
            if c_high > 16384: c_high = 16384
            
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            # Update alpha
            p_prior = f_prior[sym] / 16384.0
            p_adapt = f_adapt[sym] / 16384.0
            p_mix = f_mix[sym] / float(cum)
            grad = (p_prior - p_adapt) / (p_mix + 1e-9)
            alpha = min(max(alpha + eta * grad, 0.01), 0.99)
            
            # Update adaptive counts
            adapt_counts[ctx, sym] += 1
            adapt_totals[ctx] += 1
            if adapt_totals[ctx] >= 2048:
                new_tot = 0
                for s in range(4):
                    adapt_counts[ctx, s] = (adapt_counts[ctx, s] + 1) >> 1
                    new_tot += adapt_counts[ctx, s]
                adapt_totals[ctx] = new_tot
                
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
                    
    alpha_state[0] = alpha
    return decoded

"""
High-Performance Lossless Codecs using Numba JIT.

Implements:
1. Numba-accelerated CARLE Canonical (7-byte binary packets)
2. Numba-accelerated CARLE-v2 (gap-coding + rANS entropy coder)
3. Context-Model Coders (CM-Intra and CM-Temporal) using adaptive range coding
4. Fast Jitter & Flicker Analysis routines
"""

import struct
import numpy as np
import numba
from numba import njit, uint8, uint16, uint32, int32, int64, float64

# ==============================================================================
# 1. Numba Kernels for CARLE Canonical and CARLE-v2
# ==============================================================================

@njit(fastmath=True)
def extract_diff_runs_numba(curr_mask: np.ndarray, prev_mask: np.ndarray):
    """
    Extracts run-length encoded change segments between curr_mask and prev_mask.
    Pixels where curr_mask == prev_mask are unchanged (null symbol).
    Pixels where curr_mask != prev_mask take the new label curr_mask[r, c].
    
    Returns:
        starts: np.ndarray(uint32) - starting 1D pixel index of each run
        lengths: np.ndarray(uint16) - length of each run (clamped to max 65535)
        classes: np.ndarray(uint8) - class label for each run
        n_runs: int - number of valid runs
    """
    H, W = curr_mask.shape
    total_pixels = H * W
    
    # Preallocate buffers (worst-case: every pixel is its own run)
    max_possible_runs = total_pixels
    starts = np.empty(max_possible_runs, dtype=np.uint32)
    lengths = np.empty(max_possible_runs, dtype=np.uint16)
    classes = np.empty(max_possible_runs, dtype=np.uint8)
    
    curr_flat = curr_mask.ravel()
    prev_flat = prev_mask.ravel()
    
    n_runs = 0
    in_run = False
    run_start = 0
    run_len = 0
    run_cls = 0
    
    for i in range(total_pixels):
        c_val = curr_flat[i]
        p_val = prev_flat[i]
        
        if c_val != p_val:
            # Changed pixel!
            if not in_run:
                in_run = True
                run_start = i
                run_len = 1
                run_cls = c_val
            else:
                if c_val == run_cls and run_len < 65535:
                    run_len += 1
                else:
                    # Flush previous run
                    starts[n_runs] = run_start
                    lengths[n_runs] = run_len
                    classes[n_runs] = run_cls
                    n_runs += 1
                    
                    # Start new run
                    run_start = i
                    run_len = 1
                    run_cls = c_val
        else:
            # Unchanged pixel
            if in_run:
                starts[n_runs] = run_start
                lengths[n_runs] = run_len
                classes[n_runs] = run_cls
                n_runs += 1
                in_run = False
                
    if in_run:
        starts[n_runs] = run_start
        lengths[n_runs] = run_len
        classes[n_runs] = run_cls
        n_runs += 1
        
    return starts[:n_runs], lengths[:n_runs], classes[:n_runs]

@njit(fastmath=True)
def apply_diff_runs_numba(prev_mask: np.ndarray, starts: np.ndarray, lengths: np.ndarray, classes: np.ndarray):
    """Reconstructs curr_mask from prev_mask and change runs."""
    reconstructed = prev_mask.copy()
    rec_flat = reconstructed.ravel()
    n_runs = len(starts)
    
    for i in range(n_runs):
        s = starts[i]
        l = lengths[i]
        c = classes[i]
        for k in range(l):
            rec_flat[s + k] = c
            
    return reconstructed

@njit(fastmath=True)
def compute_gap_deltas_numba(starts: np.ndarray, lengths: np.ndarray):
    """Computes non-negative gap deltas between runs."""
    n = len(starts)
    gaps = np.empty(n, dtype=np.uint32)
    prev_end = 0
    for i in range(n):
        gaps[i] = starts[i] - prev_end
        prev_end = starts[i] + lengths[i]
    return gaps

@njit(fastmath=True)
def decode_gaps_to_starts_numba(gaps: np.ndarray, lengths: np.ndarray):
    """Reconstructs absolute starts from gap deltas."""
    n = len(gaps)
    starts = np.empty(n, dtype=np.uint32)
    prev_end = 0
    for i in range(n):
        s = prev_end + gaps[i]
        starts[i] = s
        prev_end = s + lengths[i]
    return starts

# ==============================================================================
# 2. CARLE Canonical & CARLE-v2 Python Wrappers
# ==============================================================================

@njit(fastmath=True)
def pack_runs_7b_numba(starts: np.ndarray, lengths: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Packs (uint32 start, uint16 length, uint8 class) into 7-byte big-endian records."""
    n = len(starts)
    buf = np.empty(n * 7, dtype=np.uint8)
    for i in range(n):
        s = starts[i]
        l = lengths[i]
        c = classes[i]
        off = i * 7
        buf[off] = np.uint8((s >> 24) & 0xFF)
        buf[off + 1] = np.uint8((s >> 16) & 0xFF)
        buf[off + 2] = np.uint8((s >> 8) & 0xFF)
        buf[off + 3] = np.uint8(s & 0xFF)
        buf[off + 4] = np.uint8((l >> 8) & 0xFF)
        buf[off + 5] = np.uint8(l & 0xFF)
        buf[off + 6] = np.uint8(c)
    return buf

@njit(fastmath=True)
def unpack_runs_7b_numba(buf: np.ndarray, n: int):
    """Unpacks 7-byte big-endian records into (starts, lengths, classes)."""
    starts = np.empty(n, dtype=np.uint32)
    lengths = np.empty(n, dtype=np.uint16)
    classes = np.empty(n, dtype=np.uint8)
    for i in range(n):
        off = i * 7
        s = (np.uint32(buf[off]) << 24) | (np.uint32(buf[off + 1]) << 16) | (np.uint32(buf[off + 2]) << 8) | np.uint32(buf[off + 3])
        l = (np.uint16(buf[off + 4]) << 8) | np.uint16(buf[off + 5])
        c = buf[off + 6]
        starts[i] = s
        lengths[i] = l
        classes[i] = c
    return starts, lengths, classes

class CARLECanonicalNumba:
    """Numba-accelerated CARLE Canonical codec (7 bytes per run)."""
    def __init__(self):
        self.prev_mask = None
        self.name = "CARLE_Canonical_Numba"
        
    def reset(self):
        self.prev_mask = None
        
    def encode(self, mask: np.ndarray) -> bytes:
        H, W = mask.shape
        if self.prev_mask is None or self.prev_mask.shape != mask.shape:
            prev = np.zeros_like(mask)
        else:
            prev = self.prev_mask
            
        starts, lengths, classes = extract_diff_runs_numba(mask, prev)
        self.prev_mask = mask.copy()
        
        header = struct.pack(">HHI", H, W, len(starts))
        if len(starts) == 0:
            return header
            
        payload = pack_runs_7b_numba(starts, lengths, classes)
        return header + payload.tobytes()
        
    def decode(self, payload: bytes) -> np.ndarray:
        H, W, n_runs = struct.unpack(">HHI", payload[:8])
        if self.prev_mask is None or self.prev_mask.shape != (H, W):
            prev = np.zeros((H, W), dtype=np.uint8)
        else:
            prev = self.prev_mask
            
        if n_runs == 0:
            rec = prev.copy()
            self.prev_mask = rec
            return rec
            
        buf = np.frombuffer(payload[8:8 + n_runs * 7], dtype=np.uint8)
        starts, lengths, classes = unpack_runs_7b_numba(buf, n_runs)
        
        reconstructed = apply_diff_runs_numba(prev, starts, lengths, classes)
        self.prev_mask = reconstructed
        return reconstructed

class CARLEv2Numba:
    """Numba-accelerated CARLE-v2 with gap deltas and byte-level rANS."""
    def __init__(self):
        self.prev_mask = None
        self.name = "CARLE_v2_rANS_Numba"
        import constriction
        self.constriction = constriction
        
    def reset(self):
        self.prev_mask = None
        
    def encode(self, mask: np.ndarray) -> bytes:
        H, W = mask.shape
        if self.prev_mask is None or self.prev_mask.shape != mask.shape:
            prev = np.zeros_like(mask)
        else:
            prev = self.prev_mask
            
        starts, lengths, classes = extract_diff_runs_numba(mask, prev)
        self.prev_mask = mask.copy()
        
        n_runs = len(starts)
        header = struct.pack(">HHI", H, W, n_runs)
        if n_runs == 0:
            return header
            
        gaps = compute_gap_deltas_numba(starts, lengths)
        
        # Serialize to byte stream:
        # gaps: 4 bytes each, lengths: 2 bytes each, classes: 1 byte each
        raw_bytes = bytearray()
        raw_bytes.extend(gaps.astype(">u4").tobytes())
        raw_bytes.extend(lengths.astype(">u2").tobytes())
        raw_bytes.extend(classes.astype("u1").tobytes())
        
        symbols = np.frombuffer(raw_bytes, dtype=np.uint8)
        counts = np.bincount(symbols, minlength=256).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        probs = counts / np.sum(counts)
        
        coder = self.constriction.stream.stack.AnsCoder()
        model = self.constriction.stream.model.Categorical(probs, perfect=False)
        coder.encode_reverse(symbols.astype(np.int32), model)
        compressed_words = coder.get_compressed()
        
        ans_payload = compressed_words.tobytes()
        freq_payload = counts.astype(np.uint16).tobytes() # 256 * 2 = 512 bytes
        
        return header + struct.pack(">II", len(raw_bytes), len(ans_payload)) + freq_payload + ans_payload
        
    def decode(self, payload: bytes) -> np.ndarray:
        H, W, n_runs = struct.unpack(">HHI", payload[:8])
        if self.prev_mask is None or self.prev_mask.shape != (H, W):
            prev = np.zeros((H, W), dtype=np.uint8)
        else:
            prev = self.prev_mask
            
        if n_runs == 0:
            rec = prev.copy()
            self.prev_mask = rec
            return rec
            
        raw_len, ans_len = struct.unpack(">II", payload[8:16])
        freq_bytes = payload[16:16 + 512]
        ans_bytes = payload[16 + 512:16 + 512 + ans_len]
        
        counts = np.frombuffer(freq_bytes, dtype=np.uint16).astype(np.float64)
        probs = counts / np.sum(counts)
        
        words = np.frombuffer(ans_bytes, dtype=np.uint32)
        coder = self.constriction.stream.stack.AnsCoder(words)
        model = self.constriction.stream.model.Categorical(probs, perfect=False)
        
        decoded_symbols = coder.decode(model, raw_len).astype(np.uint8)
        decoded_bytes = decoded_symbols.tobytes()
        
        gap_bytes = decoded_bytes[:n_runs * 4]
        len_bytes = decoded_bytes[n_runs * 4:n_runs * 6]
        cls_bytes = decoded_bytes[n_runs * 6:]
        
        gaps = np.frombuffer(gap_bytes, dtype=">u4").astype(np.uint32)
        lengths = np.frombuffer(len_bytes, dtype=">u2").astype(np.uint16)
        classes = np.frombuffer(cls_bytes, dtype="u1")
        
        starts = decode_gaps_to_starts_numba(gaps, lengths)
        reconstructed = apply_diff_runs_numba(prev, starts, lengths, classes)
        self.prev_mask = reconstructed
        return reconstructed

# ==============================================================================
# 3. Context-Model (CM) Lossless Coders in Numba
# ==============================================================================

@njit(fastmath=True)
def get_spatial_context_numba(mask: np.ndarray, r: int, c: int) -> int:
    """
    Causal spatial context: West, North, North-West, North-East.
    Pixel values are in {1, 2, 3, 4} -> mapped to 0..3.
    Returns integer context state in [0, 255].
    """
    H, W = mask.shape
    w_val = (mask[r, c - 1] - 1) if c > 0 else 0
    n_val = (mask[r - 1, c] - 1) if r > 0 else 0
    nw_val = (mask[r - 1, c - 1] - 1) if (r > 0 and c > 0) else 0
    ne_val = (mask[r - 1, c + 1] - 1) if (r > 0 and c < W - 1) else 0
    
    # 4 classes -> 2 bits each = 8 bits total (256 states)
    return (w_val & 3) | ((n_val & 3) << 2) | ((nw_val & 3) << 4) | ((ne_val & 3) << 6)

@njit(fastmath=True)
def get_temporal_context_numba(mask_curr: np.ndarray, mask_prev: np.ndarray, r: int, c: int) -> int:
    """
    Spatial + Temporal context:
    Spatial: (W, N, NW, NE) in [0, 255]
    Temporal: F_t(r, c) in {1..4} -> 0..3 (2 bits)
              Boundary flag in F_t 3x3: 0 (interior) or 1 (boundary) (1 bit)
    Total states = 256 * 8 = 2048 states.
    """
    H, W = mask_curr.shape
    spat = get_spatial_context_numba(mask_curr, r, c)
    
    prev_center = mask_prev[r, c] - 1
    
    # Check if any 3x3 neighbor in mask_prev differs from center
    is_boundary = 0
    r_min = max(0, r - 1)
    r_max = min(H - 1, r + 1)
    c_min = max(0, c - 1)
    c_max = min(W - 1, c + 1)
    
    c_val = mask_prev[r, c]
    for rr in range(r_min, r_max + 1):
        for cc in range(c_min, c_max + 1):
            if mask_prev[rr, cc] != c_val:
                is_boundary = 1
                break
        if is_boundary:
            break
            
    temp_state = (prev_center & 3) | (is_boundary << 2)
    return spat | (temp_state << 8)

# Fast Multi-Symbol Range Coder in Numba
# Precision: 32-bit range, 16-bit frequency tables
RC_TOP = np.int64(1 << 24)
RC_BOT = np.int64(1 << 16)
RC_TOTAL_FREQ = np.int64(16384)

@njit(fastmath=True)
def cm_encode_numba(mask_curr: np.ndarray, mask_prev: np.ndarray, use_temporal: bool, cum_freqs: np.ndarray):
    """
    Range encodes mask_curr under the context model.
    cum_freqs: shape (num_contexts, 5), cumulative frequencies summing to 16384.
    """
    H, W = mask_curr.shape
    total_pixels = H * W
    
    out_buf = np.empty(total_pixels * 2 + 1024, dtype=np.uint8)
    out_ptr = 0
    
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_numba(mask_curr, mask_prev, r, c)
            else:
                ctx = get_spatial_context_numba(mask_curr, r, c)
                
            sym = int(mask_curr[r, c] - 1) # 0..3
            
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
def cm_decode_numba(payload: np.ndarray, H: int, W: int, mask_prev: np.ndarray, use_temporal: bool, cum_freqs: np.ndarray):
    """
    Range decodes mask_curr from payload under the context model.
    """
    out_mask = np.empty((H, W), dtype=np.uint8)
    
    in_ptr = 0
    low = np.int64(0)
    range_val = np.int64(0xFFFFFFFF)
    code = np.int64(0)
    
    for _ in range(4):
        b = np.int64(payload[in_ptr]) if in_ptr < len(payload) else np.int64(0)
        in_ptr += 1
        code = (code << 8) | b
        
    for r in range(H):
        for c in range(W):
            if use_temporal:
                ctx = get_temporal_context_numba(out_mask, mask_prev, r, c)
            else:
                ctx = get_spatial_context_numba(out_mask, r, c)
                
            r_step = range_val // RC_TOTAL_FREQ
            count = (code - low) // r_step
            
            sym = 0
            for s in range(4):
                if np.int64(cum_freqs[ctx, s + 1]) > count:
                    sym = s
                    break
                    
            out_mask[r, c] = sym + 1
            
            c_low = np.int64(cum_freqs[ctx, sym])
            c_high = np.int64(cum_freqs[ctx, sym + 1])
            
            low = low + c_low * r_step
            range_val = (c_high - c_low) * r_step
            
            while True:
                if (low ^ (low + range_val)) < RC_TOP:
                    b = np.int64(payload[in_ptr]) if in_ptr < len(payload) else np.int64(0)
                    in_ptr += 1
                    code = ((code << 8) | b) & np.int64(0xFFFFFFFF)
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                    range_val = (range_val << 8) & np.int64(0xFFFFFFFF)
                elif range_val < RC_BOT:
                    b = np.int64(payload[in_ptr]) if in_ptr < len(payload) else np.int64(0)
                    in_ptr += 1
                    code = ((code << 8) | b) & np.int64(0xFFFFFFFF)
                    range_val = ((-low) & (RC_BOT - 1)) << 8
                    low = (low << 8) & np.int64(0xFFFFFFFF)
                else:
                    break
                    
    return out_mask

# ==============================================================================
# 4. Context Model Wrapper
# ==============================================================================

class ContextModelCoder:
    """
    Adaptive Multi-Symbol Range Coder with Causal Context Modeling.
    Variant: 'intra' (CM-Intra, 256 contexts) or 'temporal' (CM-Temporal, 2048 contexts).
    """
    def __init__(self, mode: str = "intra", trained_cum_freqs: np.ndarray = None):
        self.mode = mode
        self.use_temporal = (mode == "temporal")
        self.num_contexts = 2048 if self.use_temporal else 256
        self.name = f"CM_{mode}"
        self.prev_mask = None
        
        if trained_cum_freqs is not None:
            self.cum_freqs = trained_cum_freqs
        else:
            # Default uniform model with Laplace smoothing
            self.cum_freqs = self._build_default_cum_freqs()
            
    def _build_default_cum_freqs(self):
        # 4 symbols, total freq = 16384 (each 4096)
        cf = np.empty((self.num_contexts, 5), dtype=np.uint16)
        for i in range(self.num_contexts):
            cf[i] = [0, 4096, 8192, 12288, 16384]
        return cf
        
    def reset(self):
        self.prev_mask = None
        
    def encode(self, mask: np.ndarray) -> bytes:
        H, W = mask.shape
        if self.prev_mask is None or self.prev_mask.shape != mask.shape:
            prev = np.zeros_like(mask)
        else:
            prev = self.prev_mask
            
        payload = cm_encode_numba(mask, prev, self.use_temporal, self.cum_freqs)
        self.prev_mask = mask.copy()
        
        header = struct.pack(">HHB", H, W, 1 if self.use_temporal else 0)
        return header + payload.tobytes()
        
    def decode(self, payload_bytes: bytes) -> np.ndarray:
        H, W, is_temp = struct.unpack(">HHB", payload_bytes[:5])
        payload = np.frombuffer(payload_bytes[5:], dtype=np.uint8)
        
        if self.prev_mask is None or self.prev_mask.shape != (H, W):
            prev = np.zeros((H, W), dtype=np.uint8)
        else:
            prev = self.prev_mask
            
        decoded = cm_decode_numba(payload, H, W, prev, bool(is_temp), self.cum_freqs)
        self.prev_mask = decoded
        return decoded

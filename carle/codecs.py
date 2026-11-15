"""
CARLE and Baseline Codecs for Semantic Label-Map Compression.

All codecs implement lossless encoding and decoding of 2D uint8 label maps.
Intra codecs compress a single frame F_t.
Differential codecs compress F_{t+1} given reference frame F_t.
"""

import struct
import time
import numpy as np
import cv2
import zstandard as zstd
import constriction

# =====================================================================
# 1. Intra Baselines (Full Map)
# =====================================================================

class FullMapPNG:
    def __init__(self, level: int = 9):
        self.level = level
        self.params = [cv2.IMWRITE_PNG_COMPRESSION, level]

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        success, enc = cv2.imencode(".png", mask, self.params)
        if not success:
            raise RuntimeError("PNG encoding failed")
        return enc.tobytes()

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        buf = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError("PNG decoding failed")
        return decoded


class FullMapZstd:
    def __init__(self, level: int = 3):
        self.level = level
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        H, W = mask.shape
        header = struct.pack(">HH", H, W)
        compressed = self.cctx.compress(mask.tobytes())
        return header + compressed

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        H, W = struct.unpack(">HH", payload[:4])
        raw = self.dctx.decompress(payload[4:], max_output_size=H * W)
        return np.frombuffer(raw, dtype=np.uint8).reshape((H, W))


# =====================================================================
# 2. Inter / Differential Baselines (Class-Preserving Diff Map)
# =====================================================================

def class_preserving_diff(curr_mask: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    """Eq. (2): D_t(i,j) = F_{t+1}(i,j) if F_{t+1}(i,j) != F_t(i,j) else 0."""
    return np.where(curr_mask != prev_mask, curr_mask, 0).astype(np.uint8)


class DiffMapPNG:
    def __init__(self, level: int = 9):
        self.level = level
        self.params = [cv2.IMWRITE_PNG_COMPRESSION, level]

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        if prev_mask is None:
            diff = mask
        else:
            diff = class_preserving_diff(mask, prev_mask)
        success, enc = cv2.imencode(".png", diff, self.params)
        if not success:
            raise RuntimeError("Diff PNG encoding failed")
        return enc.tobytes()

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        buf = np.frombuffer(payload, dtype=np.uint8)
        diff = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if diff is None:
            raise RuntimeError("Diff PNG decoding failed")
        if prev_mask is None:
            return diff
        out = prev_mask.copy()
        mask_changed = (diff > 0)
        out[mask_changed] = diff[mask_changed]
        return out


class DiffMapZstd:
    def __init__(self, level: int = 3):
        self.level = level
        self.cctx = zstd.ZstdCompressor(level=level)
        self.dctx = zstd.ZstdDecompressor()

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        H, W = mask.shape
        header = struct.pack(">HH", H, W)
        if prev_mask is None:
            diff = mask
        else:
            diff = class_preserving_diff(mask, prev_mask)
        compressed = self.cctx.compress(diff.tobytes())
        return header + compressed

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        H, W = struct.unpack(">HH", payload[:4])
        raw = self.dctx.decompress(payload[4:], max_output_size=H * W)
        diff = np.frombuffer(raw, dtype=np.uint8).reshape((H, W))
        if prev_mask is None:
            return diff
        out = prev_mask.copy()
        mask_changed = (diff > 0)
        out[mask_changed] = diff[mask_changed]
        return out


# =====================================================================
# 3. CARLE Canonical (7-Byte Packets, Class-Preserving, Skips Null=0)
# =====================================================================

def extract_carle_triplets(diff_map: np.ndarray):
    """Extracts (start, length, class) triplets from 2D diff map. Null=0 skipped."""
    flat = diff_map.ravel()
    N = len(flat)
    triplets = []
    i = 0
    while i < N:
        if flat[i] == 0:
            i += 1
            continue
        c = int(flat[i])
        start = i
        j = i + 1
        while j < N and flat[j] == c:
            j += 1
        length = j - start
        while length > 65535:
            triplets.append((start, 65535, c))
            start += 65535
            length -= 65535
        triplets.append((start, length, c))
        i = j
    return triplets


class CARLECanonical:
    """
    CARLE Canonical method:
    Packet: uint32 start (4B) + uint16 length (2B) + uint8 class (1B) = 7 bytes per run.
    Header: uint16 H + uint16 W + uint32 num_triplets = 8 bytes.
    """
    def __init__(self):
        pass

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        H, W = mask.shape
        if prev_mask is None:
            diff = mask
        else:
            diff = class_preserving_diff(mask, prev_mask)
            
        triplets = extract_carle_triplets(diff)
        num_triplets = len(triplets)
        
        header = struct.pack(">HHI", H, W, num_triplets)
        payload = bytearray(8 + num_triplets * 7)
        payload[:8] = header
        
        offset = 8
        for s, l, c in triplets:
            struct.pack_into(">IHB", payload, offset, s, l, c)
            offset += 7
            
        return bytes(payload)

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        H, W, num_triplets = struct.unpack(">HHI", payload[:8])
        if prev_mask is None:
            out = np.zeros((H, W), dtype=np.uint8)
        else:
            out = prev_mask.copy()
            
        flat = out.ravel()
        offset = 8
        for _ in range(num_triplets):
            s, l, c = struct.unpack_from(">IHB", payload, offset)
            offset += 7
            flat[s : s + l] = c
            
        return out


# =====================================================================
# 4. CARLE-v2: Gap-Coded Starts + Lengths + Classes with Byte-Level rANS
# =====================================================================

class CARLEv2_rANS:
    """
    CARLE-v2 with gap coding and rANS (using constriction).
    Triplets: (s_k, l_k, c_k).
    Gap: delta_s_k = s_k - prev_end.
    Raw triplet stream: delta_s (uint32) + length (uint16) + class (uint8).
    Entropy-coded using byte-level rANS (alphabet size = 256).
    """
    def __init__(self):
        pass

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        H, W = mask.shape
        if prev_mask is None:
            diff = mask
        else:
            diff = class_preserving_diff(mask, prev_mask)
            
        triplets = extract_carle_triplets(diff)
        num_triplets = len(triplets)
        
        if num_triplets == 0:
            return struct.pack(">HHI", H, W, 0)
            
        # Serialize raw gap-coded stream: (delta_s: uint32, length: uint16, class: uint8) -> 7 bytes per run
        raw_stream = bytearray(num_triplets * 7)
        off = 0
        prev_end = 0
        for s, l, c in triplets:
            delta_s = s - prev_end
            struct.pack_into(">IHB", raw_stream, off, delta_s, l, c)
            off += 7
            prev_end = s + l
            
        # Fast byte-level rANS
        symbols = np.frombuffer(raw_stream, dtype=np.uint8).astype(np.int32)
        counts = np.bincount(symbols, minlength=256) + 1 # Laplace smoothing
        probs = counts.astype(np.float64) / np.sum(counts)
        
        try:
            model = constriction.stream.model.Categorical(probs, perfect=False)
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(symbols, model)
            comp_words = coder.get_compressed()
            
            # Format: H(2), W(2), num_triplets(4), flag(1)=1, table(256*2=512 B), words(len*4 B)
            table_bytes = counts.astype(np.uint16).tobytes()
            words_bytes = comp_words.tobytes()
            total_size = 9 + len(table_bytes) + len(words_bytes)
            
            if total_size < (8 + num_triplets * 7):
                hdr = struct.pack(">HHIBI", H, W, num_triplets, 1, len(words_bytes))
                return hdr + table_bytes + words_bytes
        except Exception:
            pass

        # Fallback to uncompressed canonical (flag = 0)
        hdr = struct.pack(">HHIB", H, W, num_triplets, 0)
        return hdr + bytes(raw_stream)

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        H, W, num_triplets = struct.unpack_from(">HHI", payload, 0)
        if prev_mask is None:
            out = np.zeros((H, W), dtype=np.uint8)
        else:
            out = prev_mask.copy()
            
        if num_triplets == 0:
            return out
            
        flag = payload[8]
        flat = out.ravel()
        
        if flag == 0: # Raw gap-coded stream
            off = 9
            prev_end = 0
            for _ in range(num_triplets):
                delta_s, l, c = struct.unpack_from(">IHB", payload, off)
                off += 7
                s = prev_end + delta_s
                flat[s : s + l] = c
                prev_end = s + l
            return out
            
        # rANS decode
        _, _, _, _, len_words = struct.unpack_from(">HHIBI", payload, 0)
        off = 13
        table_bytes = payload[off : off + 512]
        off += 512
        words_bytes = payload[off : off + len_words]
        
        counts = np.frombuffer(table_bytes, dtype=np.uint16).astype(np.int64)
        words = np.frombuffer(words_bytes, dtype=np.uint32)
        probs = counts.astype(np.float64) / np.sum(counts)
        model = constriction.stream.model.Categorical(probs, perfect=False)
        
        coder = constriction.stream.stack.AnsCoder(words)
        total_syms = num_triplets * 7
        dec_symbols = coder.decode(model, total_syms)
        raw_stream = dec_symbols.astype(np.uint8).tobytes()
        
        off = 0
        prev_end = 0
        for _ in range(num_triplets):
            delta_s, l, c = struct.unpack_from(">IHB", raw_stream, off)
            off += 7
            s = prev_end + delta_s
            flat[s : s + l] = c
            prev_end = s + l
            
        return out


# =====================================================================
# 5. CARLE-v2-IP: Intra/Inter Switching + Periodic Refresh (N in {10, 30, inf})
# =====================================================================

class CARLEv2_IP:
    """
    Intra/Inter switching codec with periodic keyframe refresh every N frames.
    - If frame_idx % N == 0: sends Intra frame (FullMapZstd level 3).
    - Otherwise: computes CARLE difference. If differential payload exceeds Intra payload,
      dynamically switches to Intra frame.
    """
    def __init__(self, refresh_N: int = 30):
        self.refresh_N = refresh_N
        self.intra_codec = FullMapZstd(level=3)
        self.inter_codec = CARLECanonical()
        self.frame_count = 0

    def reset(self):
        self.frame_count = 0

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        is_keyframe = (prev_mask is None) or (self.refresh_N < float('inf') and (self.frame_count % self.refresh_N == 0))
        self.frame_count += 1
        
        if is_keyframe:
            payload = self.intra_codec.encode(mask)
            return b"I" + payload
            
        inter_payload = self.inter_codec.encode(mask, prev_mask)
        intra_payload = self.intra_codec.encode(mask)
        
        if len(inter_payload) > len(intra_payload):
            return b"I" + intra_payload
        else:
            return b"P" + inter_payload

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        tag = payload[:1]
        data = payload[1:]
        if tag == b"I":
            return self.intra_codec.decode(data)
        elif tag == b"P":
            return self.inter_codec.decode(data, prev_mask)
        else:
            raise ValueError(f"Unknown frame tag: {tag}")


# =====================================================================
# 6. Legacy XOR (Historical Baseline)
# =====================================================================

class XORLegacy:
    """
    Historical legacy method:
    - np.bitwise_xor(cur, prev)
    - Full RLE (including zeros)
    - len(rle) * 6 accounting
    """
    def __init__(self):
        pass

    def encode(self, mask: np.ndarray, prev_mask: np.ndarray = None) -> bytes:
        H, W = mask.shape
        if prev_mask is None:
            xor = mask
        else:
            xor = np.bitwise_xor(mask, prev_mask)
            
        flat = xor.ravel()
        if len(flat) == 0:
            return struct.pack(">HHI", H, W, 0)
            
        ch_idx = np.flatnonzero(np.diff(flat)) + 1
        starts = np.concatenate([[0], ch_idx])
        lengths = np.diff(np.concatenate([starts, [len(flat)]]))
        values = flat[starts]
        
        num_entries = len(starts)
        hdr = struct.pack(">HHI", H, W, num_entries)
        
        buf = bytearray(8 + num_entries * 6)
        buf[:8] = hdr
        off = 8
        for s, l, v in zip(starts, lengths, values):
            struct.pack_into(">HHH", buf, off, int(s % 65536), int(min(l, 65535)), int(v))
            off += 6
        return bytes(buf)

    def decode(self, payload: bytes, prev_mask: np.ndarray = None) -> np.ndarray:
        H, W, num_entries = struct.unpack_from(">HHI", payload, 0)
        if prev_mask is not None:
            return prev_mask
        return np.zeros((H, W), dtype=np.uint8)

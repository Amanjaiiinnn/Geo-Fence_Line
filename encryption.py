"""
crypto_manager.py
─────────────────
Real-time AES-256-GCM per-frame encryption / decryption for recorded video.

Integrates with SmartRecorder via  encrypt=True:
  SmartRecorder writes BOTH the .avi (unchanged) AND a .enc file
  in real-time, on the same write thread — zero impact on inference.

Standalone helpers
──────────────────
  from crypto_manager import encrypt_file, decrypt_to_avi

  encrypt_file("recordings/rec.avi", "recordings/rec.enc")
  decrypt_to_avi("recordings/rec.enc", "out.avi", fps=15.0)

Playback
────────
  dec = StreamDecryptor("recordings/rec.enc")
  dec.play_with_ffplay()          # pipe MJPEG → ffplay, no temp file
  for frame in dec.frames():      # OR iterate BGR frames directly
      cv2.imshow("Playback", frame)
  dec.close()

File format  (ENCV2)
────────────────────
  Header    : MAGIC(5 B) + VERSION(1 B)
  Per chunk : ciphertext_len(4 B, big-endian)
              nonce(12 B)
              chunk_id(8 B, big-endian)
              ciphertext  (AES-256-GCM; 16-byte tag appended by AESGCM)
  One chunk = one JPEG-encoded video frame.

Key
───
  Loaded once from  config.ENCRYPTION_KEY_PATH  (same key for all videos).
  Must be exactly 32 bytes (AES-256).
  Generate once:
      python -c "import os; open('keys/master.key','wb').write(os.urandom(32))"
"""

import os
import struct
import subprocess
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from config import ENCRYPTION_KEY_PATH, RECORDING_QUALITY

MAGIC   = b"ENCV2"
VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
#  Key loader (internal)
# ─────────────────────────────────────────────────────────────────────────────
def _load_key() -> bytes:
    """
    Load the 32-byte AES-256 key from config.ENCRYPTION_KEY_PATH.
    Raises FileNotFoundError with a generation hint if missing.
    """
    key = None
    path = ENCRYPTION_KEY_PATH
    if not os.path.exists(path):
        # Auto-create the keys/ directory and generate a fresh 32-byte key
        print(f"[Crypto] Key file not found: '{path}'")
        print(f"  Generating New master.key")
        key_dir = os.path.dirname(path)
        if key_dir:
            os.makedirs(key_dir, exist_ok=True)
        key = os.urandom(32)
        with open(path, "wb") as f:
            f.write(key)
        print(f"[Crypto] Key auto-generated and saved → '{path}'")
        print(f"[Crypto] ⚠️  Keep this file safe — losing it means losing access to all recordings.")
    
    with open(path, "rb") as f:
        key = f.read()
    if len(key) != 32:
        raise ValueError(
            f"[Crypto] Key must be exactly 32 bytes (AES-256), got {len(key)} bytes."
        )
    return key

# ─────────────────────────────────────────────────────────────────────────────
#  STREAM ENCRYPTOR
# ─────────────────────────────────────────────────────────────────────────────
class StreamEncryptor:
    """
    Real-time per-frame encryptor — used internally by SmartRecorder.

    Each BGR frame is JPEG-encoded then written as one AES-256-GCM chunk.
    The key is loaded once at construction from config.ENCRYPTION_KEY_PATH.

    Direct usage example (if you need to encrypt outside SmartRecorder):
        enc = StreamEncryptor("recordings/rec_20250101_120000.enc")
        enc.write_frame(bgr_frame)   # one call per frame
        enc.close()
    """

    def __init__(self, output_path: str, jpeg_quality: int = RECORDING_QUALITY):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self.aes          = AESGCM(_load_key())
        self.jpeg_quality = jpeg_quality
        self.chunk_id     = 0
        self.output_path  = output_path

        self._f = open(output_path, "wb")
        self._f.write(MAGIC)
        self._f.write(struct.pack("B", VERSION))
        print(f"[Encryptor] Opened  : {output_path}")

    # ── Public ────────────────────────────────────────────────────────────────

    def write_frame(self, frame: np.ndarray) -> None:
        """
        JPEG-encode a BGR uint8 frame and write it as one encrypted chunk.
        Called from SmartRecorder's write thread — never blocks the inference thread.
        """
        ret, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ret:
            return
        self._write_chunk(buf.tobytes())

    def close(self) -> None:
        """Flush and close the .enc file."""
        self._f.close()
        print(
            f"[Encryptor] Saved   : {self.output_path}  "
            f"({self.chunk_id} frames encrypted)"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_chunk(self, data: bytes) -> None:
        """Pack one chunk:  len(4) | nonce(12) | chunk_id(8) | ciphertext."""
        nonce          = os.urandom(12)
        chunk_id_bytes = struct.pack(">Q", self.chunk_id)
        encrypted      = self.aes.encrypt(nonce, data, chunk_id_bytes)

        self._f.write(struct.pack(">I", len(encrypted)))
        self._f.write(nonce)
        self._f.write(chunk_id_bytes)
        self._f.write(encrypted)
        self.chunk_id += 1


# ─────────────────────────────────────────────────────────────────────────────
#  STREAM DECRYPTOR
# ─────────────────────────────────────────────────────────────────────────────
class StreamDecryptor:
    """
    Reads a .enc file written by StreamEncryptor.

    Two playback modes
    ──────────────────
    1. frames()           — generator that yields BGR numpy arrays.
                            Use this when you want to process or display frames
                            with OpenCV directly.

    2. play_with_ffplay() — pipes raw MJPEG bytes to ffplay.
                            No temp file needed.  Requires ffplay installed:
                              sudo apt-get install ffmpeg

    Example — cv2 window:
        dec = StreamDecryptor("recordings/rec.enc")
        for frame in dec.frames():
            cv2.imshow("Playback", frame)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                break
        dec.close()
        cv2.destroyAllWindows()

    Example — ffplay:
        dec = StreamDecryptor("recordings/rec.enc")
        dec.play_with_ffplay()
        dec.close()
    """

    def __init__(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[Decryptor] File not found: '{input_path}'")

        self.aes        = AESGCM(_load_key())
        self._f         = open(input_path, "rb")
        self.input_path = input_path

        # Validate header
        magic = self._f.read(5)
        if magic != MAGIC:
            self._f.close()
            raise ValueError(
                f"[Decryptor] Invalid magic bytes — expected {MAGIC!r}, got {magic!r}.\n"
                f"  Is this really an ENCV2 file?"
            )
        ver = struct.unpack("B", self._f.read(1))[0]
        if ver != VERSION:
            self._f.close()
            raise ValueError(
                f"[Decryptor] Unsupported file version: {ver} (expected {VERSION})."
            )

        print(f"[Decryptor] Opened  : {input_path}")

    # ── Public ────────────────────────────────────────────────────────────────

    def frames(self):
        """
        Generator — yields decrypted BGR frames as uint8 numpy arrays.
        Skips corrupt or truncated chunks with a warning rather than crashing.
        """
        for idx, jpeg_bytes in self._raw_chunks():
            arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame
            else:
                print(f"[Decryptor] ⚠ Frame {idx}: JPEG decode returned None")

    def play_with_ffplay(self) -> None:
        """
        Pipe raw JPEG chunks to ffplay as an MJPEG stream.
        No intermediate file is created.
        Blocks until playback is complete or ffplay is closed.
        """
        proc = subprocess.Popen(
            ["ffplay", "-autoexit", "-f", "mjpeg", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _, jpeg_bytes in self._raw_chunks():
                if proc.poll() is not None:   # user closed ffplay window
                    break
                proc.stdin.write(jpeg_bytes)
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait()
        print("[Decryptor] Playback done.")

    def close(self) -> None:
        """Close the file handle."""
        self._f.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raw_chunks(self):
        """
        Generator — yields (chunk_index, plaintext_bytes).
        Skips and warns on authentication failures.
        """
        idx = 0
        while True:
            header = self._f.read(4)
            if not header:
                break                                   # clean EOF
            if len(header) < 4:
                print(f"[Decryptor] ⚠ Truncated at chunk {idx} — file may be incomplete")
                break

            size      = struct.unpack(">I", header)[0]
            nonce     = self._f.read(12)
            chunk_id  = self._f.read(8)
            encrypted = self._f.read(size)

            try:
                plaintext = self.aes.decrypt(nonce, encrypted, chunk_id)
                yield idx, plaintext
            except Exception as exc:
                print(f"[Decryptor] ⚠ Chunk {idx} auth/decrypt failed: {exc}")

            idx += 1


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def encrypt_file(avi_path: str, enc_path: str, jpeg_quality: int = 75) -> None:
    """
    Post-process an existing .avi file into a .enc file.

    Handy for recordings made before  encrypt=True  was set on SmartRecorder.

    Example:
        from crypto_manager import encrypt_file
        encrypt_file("recordings/rec_20250101.avi", "recordings/rec_20250101.enc")
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise RuntimeError(f"[Crypto] Cannot open source file: '{avi_path}'")

    enc   = StreamEncryptor(enc_path, jpeg_quality=jpeg_quality)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        enc.write_frame(frame)
        count += 1

    cap.release()
    enc.close()
    print(f"[Crypto] encrypt_file complete: {count} frames → '{enc_path}'")


def decrypt_to_avi(enc_path: str, avi_path: str, fps: float = 15.0) -> None:
    """
    Decrypt a .enc file back to a standard MJPEG .avi.

    fps should match the original recording framerate (config.RECORDING_FPS).

    Example:
        from crypto_manager import decrypt_to_avi
        decrypt_to_avi("recordings/rec_20250101.enc", "out.avi", fps=15.0)
    """
    dec    = StreamDecryptor(enc_path)
    writer = None
    count  = 0

    for frame in dec.frames():
        if writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h), True)
            if not writer.isOpened():
                dec.close()
                raise RuntimeError(f"[Crypto] Cannot open writer: '{avi_path}'")
        writer.write(frame)
        count += 1

    dec.close()
    if writer:
        writer.release()
    print(f"[Crypto] decrypt_to_avi complete: {count} frames → '{avi_path}'")

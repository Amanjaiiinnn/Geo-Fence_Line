"""
decryptor.py
────────────
AES-256-GCM per-frame decryption for .enc video files.

Reads .enc files written by encryptor.py / StreamEncryptor.
Encryption is handled separately in encryptor.py.

Playback:
  dec = StreamDecryptor("recordings/rec.enc")
  dec.play_with_ffplay()          # pipe MJPEG → ffplay, no temp file
  for frame in dec.frames():      # OR iterate BGR frames directly
      cv2.imshow("Playback", frame)
  dec.close()

Standalone helper:
  from decryptor import decrypt_to_avi
  decrypt_to_avi("recordings/rec.enc", "out.avi", fps=15.0)

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
  Loaded once from config.ENCRYPTION_KEY_PATH.
  Must be exactly 32 bytes (AES-256).
  Must be the SAME key used during encryption — decryption is impossible otherwise.
"""

import os
import struct
import subprocess
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from config import ENCRYPTION_KEY_PATH

MAGIC   = b"ENCV2"
VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
#  Key loader (internal)
# ─────────────────────────────────────────────────────────────────────────────
def _load_key(key_path: str = None) -> bytes:
    """
    Load the 32-byte AES-256 key.
    key_path : explicit path to key file (overrides config.ENCRYPTION_KEY_PATH).
               If None, falls back to config.ENCRYPTION_KEY_PATH.
    Raises FileNotFoundError if missing — decryption is impossible without
    the original key that was used to encrypt.
    """
    path = key_path if key_path else ENCRYPTION_KEY_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Crypto] Key file not found: '{path}'\n"
            f"  Decryption is impossible without the original encryption key.\n"
            f"  Make sure '{path}' is the same key used when the file was recorded."
        )

    with open(path, "rb") as f:
        key = f.read()

    if len(key) != 32:
        raise ValueError(
            f"[Crypto] Key file '{path}' must be exactly 32 bytes (AES-256), "
            f"got {len(key)} bytes."
        )
    print(f"[Decryptor] Key    : {path}")
    return key


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
                              Windows : winget install ffmpeg
                              Linux   : sudo apt-get install ffmpeg

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

    def __init__(self, input_path: str, key_path: str = None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"[Decryptor] File not found: '{input_path}'")

        self.aes        = AESGCM(_load_key(key_path))
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
#  CONVENIENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def decrypt_to_avi(enc_path: str, avi_path: str, key_path: str = None) -> None:
    """
    Decrypt a .enc file back to a standard MJPEG .avi.

    fps      : should match the original recording framerate (config.RECORDING_FPS).
    key_path : optional path to key file. Defaults to config.ENCRYPTION_KEY_PATH.

    Example:
        from decryptor import decrypt_to_avi
        decrypt_to_avi("recordings/rec.enc", "out.avi", fps=15.0)
        decrypt_to_avi("recordings/rec.enc", "out.avi", fps=15.0, key_path="D:/backup/master.key")
    """
    dec    = StreamDecryptor(enc_path, key_path=key_path)
    writer = None
    count  = 0

    for frame in dec.frames():
        if writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(avi_path, fourcc, 15, (w, h), True)
            if not writer.isOpened():
                dec.close()
                raise RuntimeError(f"[Crypto] Cannot open writer: '{avi_path}'")
        writer.write(frame)
        count += 1

    dec.close()
    if writer:
        writer.release()
    print(f"[Crypto] decrypt_to_avi complete: {count} frames → '{avi_path}'")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Decrypt AES-encrypted video recordings back to AVI format"
    )
    parser.add_argument(
        "--enc_file",
        type=str,
        help="Path to the encrypted .enc file to decrypt"
    )
    parser.add_argument(
        "--key_path",
        type=str,
        default=None,
        help="Path to the master.key file (optional, uses default location if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for decrypted AVI file (default: <input_name>_decrypted.avi)"
    )

    args = parser.parse_args()

    # Auto-build output path if not specified
    if args.output is None:
        enc_file = Path(args.enc_file)
        output_path = str(enc_file.with_name(enc_file.stem + "_decrypted.avi"))
    else:
        output_path = args.output

    print(f"[Decryptor] Input  : {args.enc_file}")
    print(f"[Decryptor] Output : {output_path}")

    try:
        decrypt_to_avi(args.enc_file, output_path, key_path=args.key_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Decryption failed: {e}")
        sys.exit(1)
import os, threading, datetime, queue
from collections import deque
import numpy as np
import cv2

from encryption import StreamEncryptor
from config import  RECORDING_QUALITY

# ═════════════════════════════════════════════════════════════════════════════
#  SMART RECORDER
#  Runs on its own daemon thread — zero impact on inference or display.
# ═════════════════════════════════════════════════════════════════════════════
class SmartRecorder:
    """
    Trigger-based video recorder with three save modes:

      encrypt=False, encrypt_only=False  → .avi only         (default)
      encrypt_only=True                  → .enc only, no .avi (--encrypt-only)

    CLI flags in main.py
    ─────────────────────
      (none)             saves plain .avi
      --encrypt          saves .enc only  (no .avi written to disk)

    Thread model
    ────────────
      _write_thread() drains _write_deque via an Event.
      Caller (inference thread) pushes frames via push() — never blocks.
    """

    def __init__(
        self,
        record_dir:     str,
        frame_width:    int,
        frame_height:   int,
        framerate:      int,
        encrypt:   bool = False,        # save .enc only (no .avi)
    ):

        self.record_dir     = record_dir
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.framerate      = framerate
        self._encrypt       = encrypt

        os.makedirs(record_dir, exist_ok=True)

        # State — modified only in the inference thread (single writer)
        self.is_recording = False   # read by Cairo / draw (GIL-safe)

        # OPT-5: replace queue.Queue with deque + Event for lower overhead.
        # deque(maxlen=62): auto-drops oldest frame if writer falls behind —
        # no try/except needed on every push.  Two control slots (OPEN/CLOSE)
        # + 60 frame slots → maxlen=62 gives same headroom as the old maxsize=60.
        self._write_deque: deque = deque(maxlen=62)
        self._write_event: threading.Event = threading.Event()

        # VideoWriter lives only in the write thread
        self._writer:    cv2.VideoWriter | None = None
        self._encryptor: StreamEncryptor | None = None
        self._current_file: str = ""

        self._stop_event  = threading.Event()
        self._write_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="RecordWriter"
        )
        self._write_thread.start()

        if encrypt:
            mode_label = "ON"
        else:
            mode_label = "OFF"

        print(f"[Recorder] Dir        : {os.path.abspath(record_dir)}")
        print(f"[Recorder] Encryption : {mode_label}")

    # ── PUBLIC — called from inference thread ──────────────────────────────────

    def start(self) -> None:
        """Start recording immediately (without waiting for detections)."""
        if not self.is_recording:
            self.is_recording = True
            self._write_deque.append(("OPEN", None))
            self._write_event.set()
            print("[Recorder] ▶ Started")

    def push(self, frame: np.ndarray) -> None:
        """
        Feed latest frame + detections.  Updates state machine, then queues
        frame for disk write if recording.  Non-blocking.
        """

        # ── Queue frame ───────────────────────────────────────────────────────
        if self.is_recording:
            # OPT-5: deque auto-drops oldest if full — no try/except needed
            self._write_deque.append(("FRAME", frame))
            self._write_event.set()

    def stop(self) -> None:
        """Graceful shutdown — flush queue then close writer."""
        self._stop_event.set()
        self._write_deque.append(("CLOSE", None))
        self._write_deque.append(("STOP",  None))
        self._write_event.set()
        self._write_thread.join(timeout=5.0)

    # ── PRIVATE — write thread only ───────────────────────────────────────────

    def _new_filepath(self) -> str:
        """Return base path WITHOUT extension, e.g. recordings/rec_20250101_120000"""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.record_dir, f"rec_{ts}")

    def _open_writer(self) -> None:
        """
        Open a new VideoWriter (MJPEG / AVI).

        Why MJPEG + AVI
        ────────────────
        OpenCV on Linux defaults to a GStreamer back-end for MP4 output,
        which conflicts with the existing libcamerasrc pipeline.
        MJPEG uses OpenCV's own built-in JPEG encoder — zero GStreamer
        involvement, no pipeline conflicts, files are I-frame-only and
        robust to truncation on power-loss.
        """
        if self._writer is not None or self._encryptor is not None:
            self._close_writer()

        path   = self._new_filepath()
        fourcc = None

        # ── AVI writer (skipped in encrypt-only mode) ─────────────────────────
        if not self._encrypt:
            path = path + ".avi"
            fourcc   = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(
            path, fourcc, self.framerate,
            (self.frame_width, self.frame_height),
            isColor=True,
        )

        # Guard: force native OpenCV encoder if GStreamer crept in
        backend = int(writer.get(cv2.CAP_PROP_BACKEND)) if writer.isOpened() else -1
        if writer.isOpened() and backend != 200:
            writer.release()
            writer = cv2.VideoWriter(
                path, cv2.CAP_OPENCV_MJPEG, fourcc, self.framerate,
                (self.frame_width, self.frame_height), isColor=True,
            )

        if writer.isOpened():
            # OPT-6: quality 75 instead of 95 — frees CPU on write thread
            # with barely visible quality difference; helps inference FPS.
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, RECORDING_QUALITY)
            self._writer       = writer
            self._current_file = path
            print(f"[Recorder] Opened  : {path}  "
                  f"[{self.frame_width}×{self.frame_height} "
                  f"@ {self.framerate} fps  MJPG/AVI]")
        else:
            print("[Recorder] ⚠ VideoWriter failed — check OpenCV MJPEG support")

        # ── Encryptor (.enc) ──────────────────────────────────────────────────
        if self._encrypt:
            path = path + ".enc"
            # In encrypt-only mode open regardless of _writer status;
            # in dual mode open only if the AVI writer succeeded.
            if self._encrypt or self._writer is not None:
                try:
                    self._encryptor = StreamEncryptor(path, jpeg_quality=75)
                except Exception as exc:
                    print(f"[Recorder] ⚠ Encryptor init failed: {exc}")
                    self._encryptor = None

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"[Recorder] Saved   : {self._current_file}")
            self._writer       = None
            self._current_file = ""

        if self._encryptor is not None:
            self._encryptor.close()
            self._encryptor = None

    def _writer_loop(self) -> None:
        # OPT-5: drain the deque on each wake instead of one item per iteration
        while not self._stop_event.is_set():
            self._write_event.wait(timeout=1.0)
            self._write_event.clear()

            while self._write_deque:
                cmd, payload = self._write_deque.popleft()

                if cmd == "OPEN":
                    self._open_writer()

                elif cmd == "CLOSE":
                    self._close_writer()

                elif cmd == "STOP":
                    self._close_writer()
                    return

                elif cmd == "FRAME" and payload is not None:
                    # Write plain .avi
                    if self._writer is not None:
                        self._writer.write(payload)

                    # Write encrypted .enc
                    if self._encryptor is not None:
                        try:
                            self._encryptor.write_frame(payload)
                        except Exception as exc:
                            print(f"[Recorder] ⚠ Encrypt write failed: {exc}")
                            self._encryptor = None
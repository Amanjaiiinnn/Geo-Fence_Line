import os, time, threading, queue
import numpy as np
import cv2
from config import (CLASS_COLORS_BGR, _DEFAULT_COLOR_BGR, _REC_COLOR_BGR,INFER_EVERY_N_FRAMES,RECORDING,
                    REC_DOT_RADIUS,REC_TEXT_SCALE_CV,REC_TEXT_THICKNESS_CV,REC_DOT_W_POS,REC_DOT_H_POS,
                    REC_TEXT_W_POS,REC_TEXT_H_POS,
                    _LABEL_SIZE_CACHE, _model_label, _pin_thread)
from recorder import SmartRecorder
from detection import Detector
from line_detector import ProximityDetector

# ═════════════════════════════════════════════════════════════════════════════
#  VIDEO FILE MODE — pure OpenCV, no GStreamer
# ═════════════════════════════════════════════════════════════════════════════
class VideoFileWidget:
    """
    Runs detection on a video file.

    Threading
    ─────────
      _read_thread  : reads frames from disk at native FPS → _frame_queue
      _infer_thread : pulls frames, runs detector, draws overlay,
                      feeds recorder, calls cv2.imshow  (owns the window)

    cv2.imshow + cv2.waitKey must run on the same thread (OpenCV Linux req).
    SmartRecorder write thread handles all disk I/O independently.

    Changes
    ───────
      Change 1 – Inference runs on every 3rd frame only; the last known
                 detections are reused and re-drawn on the skipped frames so
                 the overlay is always visible.
      Change 2 – ProximityDetector is told the actual video resolution so
                 p1/p2 are clamped on load.
      Change 3 – Distance measured from nearest box edge (handled in
                 line_detector.py).
      Change 4 – Alert colour stays red until object leaves frame boundary
                 (handled in line_detector.py).
      Headless  – When no display is available (GUI_AVAILABLE = False on app),
                  all cv2.namedWindow / cv2.imshow / cv2.waitKey calls are
                  skipped. Detection, recording, and encryption continue
                  normally. Stop with Ctrl+C.
    """

    _EOF = object()   # sentinel to signal end of video

    def __init__(self, app):
        self.app = app

        # Inherit display availability from the Application instance so we
        # don't need to re-probe or import GUI_AVAILABLE here directly.
        self._gui = getattr(app, "gui_available", False)

        if not app.video_path:
            raise ValueError(
                "--source Video requires --video_path /path/to/file or camera index (e.g., 0)"
            )
        
        try:
            app.video_path = int(app.video_path)
        except (ValueError, TypeError):
            pass

        # A camera index or a network stream URL has no defined "end" —
        # a failed read there is a transient glitch and should be retried.
        # An actual video FILE does have an end: ret=False there means EOF,
        # not a glitch. (This was previously hardcoded to True, so a
        # finished file was never recognised as finished — see
        # _read_loop, which both skips the EOF signal AND skips the
        # native-FPS pacing sleep when this is True.)
        self._is_webcam = (
            isinstance(app.video_path, int)
            or str(app.video_path).startswith(("http://", "https://"))
        )

        if not self._is_webcam and not os.path.exists(app.video_path):
            raise FileNotFoundError(f"Video file not found: {app.video_path}")

        self.cap = cv2.VideoCapture(app.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {app.video_path}")

        self._vid_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._vid_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"[Video] File       : {app.video_path}")
        print(f"[Video] Resolution : {self._vid_w}×{self._vid_h} "
              f"@ {self._vid_fps:.2f} fps")

        if not self._gui:
            print("[Video] Headless — preview window disabled, recording only.")

        self.detector = Detector(
            model_path     = app.model_path,
            conf_threshold = app.conf_threshold,
            iou_threshold  = app.iou_threshold,
        )

        # Change 2 — pass actual resolution so config points are clamped correctly
        self.proximity_detector = ProximityDetector(
            frame_width  = self._vid_w,
            frame_height = self._vid_h,
        )

        self.recorder = SmartRecorder(
            record_dir   = app.record_dir,
            frame_width  = self._vid_w,
            frame_height = self._vid_h,
            framerate    = self.app.record_fps,
            encrypt      = app.encrypt_video_without_avi, 
        )

        self._frame_queue = queue.Queue(maxsize=4)
        self._stop_event  = threading.Event()

        self._read_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="VideoReader"
        )
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="VideoInfer"
        )

        # Metrics
        self.inference_fps    = 0.0
        self._inf_fps_counter = 0
        self._inf_fps_time    = time.time()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def run(self):
        """Block until video ends or user presses q / Esc."""
        self._read_thread.start()
        self._infer_thread.start()
        self._infer_thread.join()
        self.recorder.stop()
        if self._gui:
            cv2.destroyAllWindows()
        print("[Video] Finished.")

    def stop(self):
        self._stop_event.set()

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _read_loop(self):
        """Read frames from disk or webcam and push to queue."""
        interval = 1.0 / self._vid_fps

        while not self._stop_event.is_set():
            t0         = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if not self._is_webcam:
                    self._frame_queue.put(self._EOF)
                    break
                else:
                    time.sleep(0.01)
                    continue

            try:
                self._frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass   # inference fell behind — drop frame

            if not self._is_webcam:
                sleep = interval - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        self.cap.release()

    def _infer_loop(self):
        """
        Pull frames, run inference (every INFER_EVERY_N_FRAMES), draw overlay
        with OpenCV, feed recorder.
        When a display is available, owns the imshow window (show + waitKey
        must be on the same thread).
        When headless, skips all GUI calls and runs until EOF or Ctrl+C.
        """
        _pin_thread(1)

        label    = _model_label(self.app.model_path)

        if self._gui:
            win_name = f"{label} - Video  (press q to quit)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, self._vid_w, self._vid_h)

        # Change 1 — state for frame-skip logic
        frame_counter            = 0
        last_detections          = []          # raw detections from last infer frame
        last_detections_w_alerts = []          # (det, alert_flag) pairs for drawing
        _last_rec_time = 0.0

        # Start recording immediately from frame 1
        if RECORDING:
            self.recorder.start()
            _last_rec_time = time.time()
        
        # _rec_interval = max(1, round(self._vid_fps / self.app.record_fps))
        _rec_interval = 1.0 / self.app.record_fps   # e.g. 1/24 = 0.0417 s

        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is self._EOF:
                break

            frame_counter += 1
            # Update inference FPS counter only on infer frames
            self._inf_fps_counter += 1
            now = time.time()
            if now - self._inf_fps_time >= 1.0:
                # Reported FPS reflects true inference rate
                self.inference_fps    = self._inf_fps_counter / (now - self._inf_fps_time)
                self._inf_fps_counter = 0
                self._inf_fps_time    = now

            # Change 1 — only run the detector, and only recompute proximity
            # alerts, on every INFER_EVERY-th frame.  Feeding update() the
            # exact same detections list again on skipped frames would
            # always produce the exact same alert flags (cx/cy unchanged,
            # so it just re-matches the tracks it itself stored last time)
            # while still paying for the full track-match loop plus the
            # os.path.getmtime() config check inside update()'s load().
            # On skipped frames we simply keep last_detections_w_alerts.
            if frame_counter % INFER_EVERY_N_FRAMES == 0:
                last_detections = self.detector.detect(frame)
                alert_flags = self.proximity_detector.update(last_detections)
                last_detections_w_alerts = list(zip(last_detections, alert_flags))

            # Always draw overlay (using last known detections)
            self._draw_overlay(frame, last_detections_w_alerts)

            # Push the ANNOTATED frame (with boxes/line/colors) to recorder.
            # No .copy() needed: `frame` is a fresh array from cap.read()
            # for this iteration only — nothing reuses or mutates it after
            # this point, so the recorder's async writer thread can safely
            # hold this exact object.
            if RECORDING and (now - _last_rec_time) >= _rec_interval:
                self.recorder.push(frame)
                _last_rec_time = now
            
            # Display — skipped entirely when headless
            if self._gui:
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):      # q or Esc
                    self._stop_event.set()
                    break

    def _draw_overlay(self, frame: np.ndarray, detections_with_alerts: list) -> None:
        """Draw boxes, labels, FPS, and REC indicator onto the BGR frame."""
        w            = frame.shape[1]
        is_recording = self.recorder.is_recording

        # FPS HUD
        cv2.putText(
            frame, f"Inf: {self.inference_fps:.1f} fps",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 242, 26), 2,
        )

        # REC indicator
        if is_recording:
            cv2.circle(frame, (w - REC_DOT_W_POS, REC_DOT_H_POS), REC_DOT_RADIUS, _REC_COLOR_BGR, -1)
            cv2.putText(frame, "REC", (w - REC_TEXT_W_POS, REC_TEXT_H_POS),
                cv2.FONT_HERSHEY_SIMPLEX, REC_TEXT_SCALE_CV, _REC_COLOR_BGR, REC_TEXT_THICKNESS_CV)

        # Draw proximity line
        self.proximity_detector.draw_line_opencv(frame)

        # Bounding boxes + labels
        for item in detections_with_alerts:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            (x1, y1, x2, y2, label, conf), is_alert = item
            # Change 4 — is_alert already carries the "stays red" latch from
            #            line_detector.py; just use it directly here
            color = (0, 0, 255) if is_alert else CLASS_COLORS_BGR.get(label, _DEFAULT_COLOR_BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            ty   = max(y1 - 6, 16)

            if label not in _LABEL_SIZE_CACHE:
                _LABEL_SIZE_CACHE[label] = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )[0]
            tw, th = _LABEL_SIZE_CACHE[label]
            tw += 40   # pixel budget for " 0.99" confidence suffix

            # Dark background strip for readability
            cv2.rectangle(
                frame,
                (x1, ty - th - 4), (x1 + tw + 4, ty + 2),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, text, (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )
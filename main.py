"""
main.py
───────
Unified CSI camera surveillance — YOLOv8n or YOLOv9t INT8.
Includes continuous video recording and a Video file mode.

Architecture
────────────
  libcamerasrc (dual-pad)
      │
      ├─► [RGB16] queue_disp ──► cairooverlay ──► videoconvert ──► fpsdisplaysink
      │                                │                                  │
      │                          draws boxes                       gtkwaylandsink
      │                          on every frame
      │
      └─► [BGR] videorate ──► queue_nn ──► appsink
                                                │
                                       _on_new_sample()
                                       copy frame into a pool slot + unmap
                                                │
                                       _latest_frame (shared var + Event)
                                                │
                                       inference thread (core 1)
                                                │
                                       detector.detect(frame)   ← detection.py
                                                │
                                       self.detections  ← Cairo reads this
                                                │
                                       SmartRecorder   ← recording logic

Recording Logic
───────────────
  - Recording starts immediately whenever RECORDING (config.py) is True —
    there is no detection-count trigger or grace period.
  - Frames are pushed to SmartRecorder at --record_fps, annotated with the
    most recent detections regardless of whether that exact frame was the
    one just run through the detector.
  - Each recording session → new timestamped .avi/.enc file under
    --record_dir.
  - Recording I/O runs on a dedicated background thread (no inference
    impact).

CPU pinning
───────────
  Main process (GStreamer / GTK / Cairo)  → core 0
  Inference thread (NPU kickoff + postprocess) → core 1

Usage
─────
  # CSI camera — YOLOv9t NPU
  python3 main.py --model_path models/yolov9t_full_integer_quant.nb

  # CSI camera — YOLOv8n NPU
  python3 main.py --model_path models/yolov8n_full_integer_quant.nb

  # Video file — YOLOv9t TFLite (CPU)
  python3 main.py --source Video \
                  --video_path person.avi \
                  --model_path models/yolov9t_full_integer_quant.tflite

  # Full options
  python3 main.py \
      --model_path            models/yolov9t_full_integer_quant.nb \
      --source                Camera \

Keys: Esc / q → quit
"""

import os
import re
import time
import signal
import argparse
import threading
import math
import subprocess

import numpy as np
import cv2
has_gui_libs = False
try:
    import gi
    import cairo
    gi.require_version("Gst", "1.0")
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Gdk, Gst
    has_gui_libs = True
except ImportError:
    class DummyBase:
        def __init__(self, *args, **kwargs):
            pass
    Gtk = DummyBase
    Gtk.Box = DummyBase
    Gtk.Window = DummyBase
    Gdk = DummyBase
    Gdk.KEY_Escape = None
    cairo = DummyBase

# ── CHANGED: unified import — works for both YOLOv8n and YOLOv9t ─────────────
from config import (DETECTION_ON, MODEL_PATH, VIDEO_PATH,
                    RECORDING_SAVE_DIR, RECORDING_FPS,FRAMERATE,
                    _model_label, CLASS_COLORS, _DEFAULT_COLOR, _REC_COLOR,
                    CLASS_COLORS_BGR, _DEFAULT_COLOR_BGR, _REC_COLOR_BGR,INFER_EVERY_N_FRAMES,
                    FRAME_WIDTH,FRAME_HEIGHT,CONFIDENCE_TH,IOU_TH,ENCRYPTION,RECORDING,
                    REC_TEXT_SIZE_CAIRO,REC_DOT_RADIUS,REC_TEXT_SCALE_CV,REC_TEXT_THICKNESS_CV,
                    REC_DOT_W_POS,REC_DOT_H_POS,REC_TEXT_W_POS,REC_TEXT_H_POS,
                    _LABEL_SIZE_CACHE, _pin_thread)
from detection import Detector
from recorder import SmartRecorder
from video_widget import VideoFileWidget
from line_detector import ProximityDetector

# GStreamer / GTK init must happen before any element is created
GUI_AVAILABLE = False

if has_gui_libs:
    Gst.init(None)
    try:
        ok, _ = Gtk.init_check(None)
        GUI_AVAILABLE = bool(ok)
    except Exception as exc:
        GUI_AVAILABLE = False
    if not GUI_AVAILABLE:
        print("[GUI] No display found — running headless (no preview window).")

# ═════════════════════════════════════════════════════════════════════════════
#  GSTREAMER WIDGET  (Camera mode)
# ═════════════════════════════════════════════════════════════════════════════
# A real Gtk.Box (or any Gtk.Widget) cannot be instantiated without a live
# display connection in GTK3 — even when it's never shown. If there's no
# display, GstWidget must fall back to a plain object instead of Gtk.Box,
# or the constructor itself aborts the process.
_GstWidgetBase = Gtk.Box if GUI_AVAILABLE else object


class GstWidget(_GstWidgetBase):

    def __init__(self, app):
        if GUI_AVAILABLE:
            super().__init__()
        self.app = app
        if GUI_AVAILABLE:
            self.connect("realize", self._on_realize)

        self.inited      = False

        # Camera-rate FPS (measured at appsink)
        self.fps_counter    = 0
        self.fps_start_time = time.time()
        self.current_fps    = 0.0

        # Inference FPS
        self.inference_fps    = 0.0
        self._inf_fps_counter = 0
        self._inf_fps_time    = time.time()

        # Detections — written by inference thread, read by Cairo.
        # CPython list assignment is atomic under the GIL — no lock needed.
        self.detections: list = []

        # OPT-2: replace queue.Queue(maxsize=1) with a shared variable +
        # lock + Event.  Inference always processes the LATEST frame; no
        # queue overhead, no put_nowait/get round-trip, and — unlike a
        # bare shared variable — no busy-polling: the inference thread
        # sleeps at zero CPU cost until a frame is actually ready.
        self._latest_frame      = None
        self._latest_frame_lock = threading.Lock()
        self._frame_ready       = threading.Event()

        # OPT-4 (fixed): a small ring of pre-allocated buffers instead of
        # one.  _on_new_sample copies straight into a slot and hands that
        # SAME array to the inference thread — no follow-up .copy(), so
        # every camera frame now costs one memcpy instead of two.  Pool
        # size 3 leaves ~2 frame-intervals of headroom before a slot the
        # inference thread might still be reading gets reused.
        self._FRAME_POOL_SIZE = 3
        self._frame_pool = [
            np.empty((app.frame_height, app.frame_width, 3), dtype=np.uint8)
            for _ in range(self._FRAME_POOL_SIZE)
        ]
        self._pool_idx = 0

        # OPT-3: Cairo font / line-width are set once per draw call — guard
        # flag ensures select_font_face is called only on the first draw.
        self._cairo_inited = False

        # Load model before pipeline starts (avoid startup race)
        self.detector = Detector(
            model_path     = self.app.model_path,
            conf_threshold = self.app.conf_threshold,
            iou_threshold  = self.app.iou_threshold,
        )

        # Change 2 — pass actual resolution so config points are clamped correctly
        self.proximity_detector = ProximityDetector(
            frame_width  = self.app.frame_width,
            frame_height = self.app.frame_height,
        )

        self.recorder = SmartRecorder(
            record_dir   = self.app.record_dir,
            frame_width  = self.app.frame_width,
            frame_height = self.app.frame_height,
            framerate    = self.app.record_fps,
            encrypt      = self.app.encrypt_video_without_avi,
        )

        # Daemon thread — dies automatically when main exits
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )

    # ── realize ───────────────────────────────────────────────────────────────
    def _on_realize(self, widget):
        self.start()

    def start(self):
        self._build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.inited = True
        self._inference_thread.start()
        print("[Pipeline] Running — Esc to quit.")

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE  — kept identical to the working reference build
    # ═════════════════════════════════════════════════════════════════════════
    def _build_pipeline(self):
        w   = self.app.frame_width
        h   = self.app.frame_height
        fps = self.app.framerate

        self.pipeline = Gst.Pipeline.new("pipeline")

        # ── source ────────────────────────────────────────────────────────────
        src = Gst.ElementFactory.make("libcamerasrc", "libcamera")
        if not src:
            raise RuntimeError("libcamerasrc plugin not found — "
                               "is libcamera-gst installed?")

        # ── caps ──────────────────────────────────────────────────────────────
        caps_disp = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=RGB16"
        )
        caps_nn = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=BGR,framerate={fps}/1"
        )

        # ── display branch ────────────────────────────────────────────────────
        queue_disp = Gst.ElementFactory.make("queue", "queue_disp")
        queue_disp.set_property("leaky",            2)
        queue_disp.set_property("max-size-buffers", 2)

        self.cairooverlay = Gst.ElementFactory.make("cairooverlay", "overlay")
        if not self.cairooverlay:
            raise RuntimeError("cairooverlay plugin not found")
        self.cairooverlay.connect("draw", self._draw)

        vconv_disp = Gst.ElementFactory.make("videoconvert", "vconv_disp")

        if GUI_AVAILABLE:
            gtkwaylandsink = Gst.ElementFactory.make("gtkwaylandsink", "sink_disp")
            if not gtkwaylandsink:
                raise RuntimeError("gtkwaylandsink plugin not found")
            self.pack_start(gtkwaylandsink.props.widget, True, True, 0)
            gtkwaylandsink.props.widget.show()
            video_sink = gtkwaylandsink
        else:
            video_sink = Gst.ElementFactory.make("fakesink", "sink_disp")
            video_sink.set_property("sync", False)

        fps_sink = Gst.ElementFactory.make("fpsdisplaysink", "fps_sink")
        fps_sink.set_property("signal-fps-measurements", True)
        fps_sink.set_property("fps-update-interval",     1000)
        fps_sink.set_property("text-overlay",            False)
        fps_sink.set_property("video-sink",              video_sink)

        # ── AI branch ─────────────────────────────────────────────────────────
        vrate_nn = Gst.ElementFactory.make("videorate", "vrate_nn")
        queue_nn = Gst.ElementFactory.make("queue",     "queue_nn")
        queue_nn.set_property("leaky",            2)
        queue_nn.set_property("max-size-buffers", 1)

        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals",       True)
        self.appsink.set_property("sync",               False)
        self.appsink.set_property("max-buffers",        1)
        self.appsink.set_property("drop",               True)
        self.appsink.set_property("enable-last-sample", False)
        self.appsink.connect("new-sample", self._on_new_sample)

        # ── add all to pipeline ───────────────────────────────────────────────
        for el in [src,
                   queue_disp, self.cairooverlay, vconv_disp, fps_sink,
                   vrate_nn, queue_nn, self.appsink]:
            self.pipeline.add(el)

        # ── link display branch ───────────────────────────────────────────────
        src.link(queue_disp)
        queue_disp.link_filtered(self.cairooverlay, caps_disp)
        self.cairooverlay.link(vconv_disp)
        vconv_disp.link(fps_sink)

        # view-finder pad (display)
        static_src = src.get_static_pad("src")
        static_src.set_property("stream-role", 3)

        # ── link AI branch ────────────────────────────────────────────────────
        pad_tmpl   = src.get_pad_template("src_%u")
        src_pad_nn = src.request_pad(pad_tmpl, None, None)
        src_pad_nn.set_property("stream-role", 1)   # still-capture
        src_pad_nn.link(vrate_nn.get_static_pad("sink"))
        vrate_nn.link(queue_nn)
        queue_nn.link_filtered(self.appsink, caps_nn)

        # ── bus ───────────────────────────────────────────────────────────────
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error",         self._on_error)
        bus.connect("message::eos",           self._on_eos)
        bus.connect("message::state-changed", self._on_state_changed)

    # ── Bus callbacks ─────────────────────────────────────────────────────────
    def _on_eos(self, bus, msg):
        print("[GST] End-of-stream")

    def _on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print(f"[GST] Error  : {err}\n      {dbg}")

    def _on_state_changed(self, bus, msg):
        old, new, _ = msg.parse_state_changed()
        if old == Gst.State.NULL and new == Gst.State.READY:
            Gst.debug_bin_to_dot_file(
                self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline"
            )

    def stop(self):
        self.recorder.stop()
        self.pipeline.set_state(Gst.State.NULL)

    # ═════════════════════════════════════════════════════════════════════════
    #  APPSINK CALLBACK — GStreamer streaming thread
    # ═════════════════════════════════════════════════════════════════════════
    def _on_new_sample(self, appsink):
        """Must return in <1 ms.  Maps buffer, copies bytes, unmaps."""
        sample = appsink.emit("pull-sample")
        if sample is None or not self.inited:
            return Gst.FlowReturn.OK

        # Camera-rate FPS counter
        self.fps_counter += 1
        now     = time.time()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps    = self.fps_counter / elapsed
            self.fps_counter    = 0
            self.fps_start_time = now

        # Map → copy → unmap immediately so pipeline buffer is freed
        buf          = sample.get_buffer()
        caps         = sample.get_caps()
        ret, mem_buf = buf.map(Gst.MapFlags.READ)
        if not ret:
            return Gst.FlowReturn.OK

        sh = caps.get_structure(0)
        h  = sh.get_value("height")
        w  = sh.get_value("width")

        # OPT-4 (fixed): ONE copy, straight from the GStreamer buffer into
        # the next pool slot — no second .copy(), so this is the only
        # memcpy that happens per camera frame now.
        dest = self._frame_pool[self._pool_idx]
        self._pool_idx = (self._pool_idx + 1) % self._FRAME_POOL_SIZE
        np.copyto(dest, np.ndarray((h, w, 3), dtype=np.uint8, buffer=mem_buf.data))

        buf.unmap(mem_buf)     # pipeline buffer freed before inference

        # OPT-2: publish the frame and wake the inference thread.  Both
        # happen under the same lock so the two threads can never disagree
        # about which frame is "current".
        with self._latest_frame_lock:
            self._latest_frame = dest
            self._frame_ready.set()

        return Gst.FlowReturn.OK

    # ═════════════════════════════════════════════════════════════════════════
    #  INFERENCE THREAD — pinned to core 1
    # ═════════════════════════════════════════════════════════════════════════
    def _inference_loop(self):
        _pin_thread(1)        # NPU kickoff + postprocess on core 1
        # OPT-2 (fixed): block on an Event instead of polling the shared
        # variable.  The thread costs zero CPU while idle and wakes the
        # instant _on_new_sample signals a frame — no more 1 kHz
        # wake-up-and-check-a-lock loop.

        _last_rec_time = 0.0
        
        if RECORDING:
            self.recorder.start()
            _last_rec_time = time.time()

        # _rec_interval = max(1, round(FRAMERATE / RECORDING_FPS))  # 30/15 = 2
        _rec_interval = 1.0 / self.app.record_fps   # e.g. 1/24 = 0.0417 s

        _frame_counter = 0
        
        while True:
            # Timeout just keeps the loop alive if the camera stalls; it is
            # NOT a polling interval — wait() returns immediately once
            # _on_new_sample calls set().
            self._frame_ready.wait(timeout=1.0)

            with self._latest_frame_lock:
                frame = self._latest_frame
                self._latest_frame = None   # consume it
                self._frame_ready.clear()

            if frame is None:
                continue
            
            _frame_counter += 1
            # Inference FPS
            self._inf_fps_counter += 1
            now     = time.time()
            elapsed = now - self._inf_fps_time
            if elapsed >= 1.0:
                self.inference_fps    = self._inf_fps_counter / elapsed
                self._inf_fps_counter = 0
                self._inf_fps_time    = now

            if _frame_counter % INFER_EVERY_N_FRAMES == 0:
                self._run_inference(frame)

            # recording every 2nd frame  ← NEW, independent
            if RECORDING and (now - _last_rec_time) >= _rec_interval:
                record_frame = frame.copy()
                self._draw_boxes_on_frame(record_frame, self.detections)
                self.recorder.push(record_frame)
                _last_rec_time = now

    def _draw_boxes_on_frame(self, frame: np.ndarray, detections_with_alerts: list) -> None:
        """Draw bounding boxes onto a BGR frame for recording."""
        w = frame.shape[1]

        if self.recorder.is_recording:
            cv2.circle(frame, (w - REC_DOT_W_POS, REC_DOT_H_POS), REC_DOT_RADIUS, _REC_COLOR_BGR, -1)
            cv2.putText(frame, "REC", (w - REC_TEXT_W_POS, REC_TEXT_H_POS),
                cv2.FONT_HERSHEY_SIMPLEX, REC_TEXT_SCALE_CV, _REC_COLOR_BGR, REC_TEXT_THICKNESS_CV)

        # Draw proximity line on recorded frame
        self.proximity_detector.draw_line_opencv(frame)

        for item in detections_with_alerts:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            (x1, y1, x2, y2, label, conf), is_alert = item
            color = (0, 0, 255) if is_alert else CLASS_COLORS_BGR.get(label, _DEFAULT_COLOR_BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            ty   = max(y1 - 6, 16)
            # OPT-7: use cached text size — getTextSize is skipped for
            # labels already seen; fixed +40 px budget covers conf suffix.
            if label not in _LABEL_SIZE_CACHE:
                _LABEL_SIZE_CACHE[label] = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )[0]
            tw, th = _LABEL_SIZE_CACHE[label]
            tw += 40   # pixel budget for " 0.99" confidence suffix
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(frame, text, (x1 + 2, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    def _run_inference(self, frame: np.ndarray):
        detections = self.detector.detect(frame)
        alert_flags = self.proximity_detector.update(detections)
        detections_with_alerts = list(zip(detections, alert_flags))

        self.detections = detections_with_alerts            # atomic list write (GIL)

    # ═════════════════════════════════════════════════════════════════════════
    #  CAIRO DRAW CALLBACK — display thread, independent of inference
    # ═════════════════════════════════════════════════════════════════════════
    def _draw(self, overlay, context, timestamp, duration):
        """
        Called on every display frame.  Reads self.detections atomically.
        No lock needed — CPython GIL makes list assignment atomic.
        """
        detections   = self.detections          # local snapshot
        is_recording = self.recorder.is_recording

        # OPT-3: select_font_face and set_line_width are called only once;
        # subsequent frames skip these calls — reduces per-frame Cairo overhead.
        if not self._cairo_inited:
            context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            context.set_line_width(2)
            self._cairo_inited = True

        # ── FPS HUD ───────────────────────────────────────────────────────────
        context.set_source_rgb(0.2, 1.0, 0.2)
        context.set_font_size(20)
        context.move_to(10, 28)
        context.show_text(f"FPS: {self.current_fps:.1f}")
        context.move_to(10, 52)
        context.show_text(f"Inf: {self.inference_fps:.1f} fps")

        # ── REC indicator — top-right corner ──────────────────────────────────
        if is_recording:
            r, g, b = _REC_COLOR
            context.set_source_rgb(r, g, b)
            context.arc(
                self.app.frame_width - REC_DOT_W_POS, REC_DOT_H_POS, REC_DOT_RADIUS,   # cx, cy, radius
                0, 2 * math.pi,
            )
            context.fill()
            context.set_font_size(REC_TEXT_SIZE_CAIRO)
            context.move_to(self.app.frame_width - REC_TEXT_W_POS, REC_TEXT_H_POS)
            context.show_text("REC")

        # Draw virtual line on Cairo overlay
        self.proximity_detector.draw_line_cairo(context)

        # ── Bounding boxes ────────────────────────────────────────────────────
        for item in detections:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            (x1, y1, x2, y2, label, conf), is_alert = item
            if is_alert:
                r, g, b = (1.0, 0.0, 0.0)  # Red
            else:
                r, g, b = CLASS_COLORS.get(label, _DEFAULT_COLOR)
            context.set_source_rgb(r, g, b)
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()
            context.set_font_size(14)
            context.move_to(x1, max(y1 - 5, 14))
            context.show_text(f"{label} {conf:.2f}")

# ═════════════════════════════════════════════════════════════════════════════
#  WINDOW  (Camera mode only)
# ═════════════════════════════════════════════════════════════════════════════
class MainWindow(Gtk.Window):

    def __init__(self, app):
        Gtk.Window.__init__(self)
        self.app = app

        # ── CHANGED: derive title from model filename ─────────────────────────
        label = _model_label(app.model_path)
        self.set_title(f"{label} Surveillance")

        self.set_decorated(False)
        self.maximize()
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("destroy",         Gtk.main_quit)
        self.connect("key-press-event", self._on_key)
        self.add(app.gst_widget)

    def _on_key(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            print("[Window] Esc — shutting down.")
            self.app.gst_widget.stop()
            Gtk.main_quit()


# ═════════════════════════════════════════════════════════════════════════════
#  CAMERA SENSOR PRESENCE CHECK
# ═════════════════════════════════════════════════════════════════════════════
def _check_camera_sensor():
    """
    Scan dmesg for a known camera-sensor probe failure (e.g. an imx219 not
    responding on I2C because nothing is plugged into the CSI connector).

    This matters because on this board the DCMIPP/CSI capture pipeline does
    NOT raise a GStreamer error when the sensor is missing — libcamerasrc
    still reaches PLAYING and the capture engine keeps producing blank or
    garbage frames forever, with nothing ever showing up on the GStreamer
    bus. Without this check the app would silently "work" and record
    meaningless video with no camera attached.

    Returns (ok, sensor_name, detail):
        ok          - False if a sensor probe failure was found in dmesg
        sensor_name - driver name reported in the failing dmesg line, or None
        detail      - the matching dmesg line, or None
    """
    try:
        out = subprocess.run(
            ["dmesg"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception as exc:
        print(f"[Camera Check] Could not read dmesg ({exc}) — skipping check.")
        return True, None, None

    fail_pattern = re.compile(
        r"(\w+) \d+-\d+: (failed to read chip id|Error reading reg)"
    )

    last_match = None
    for line in out.splitlines():
        m = fail_pattern.search(line)
        if m:
            last_match = m   # keep the most recent occurrence

    if last_match:
        return False, last_match.group(1), last_match.group(0)

    return True, None, None

# ═════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class Application:

    def __init__(self, args):
        self.model_path                = args.model_path
        self.frame_width               = FRAME_WIDTH
        self.frame_height              = FRAME_HEIGHT
        self.framerate                 = FRAMERATE
        self.conf_threshold            = CONFIDENCE_TH
        self.iou_threshold             = IOU_TH
        self.source                    = args.source
        self.video_path                = args.video_path
        self.record_fps                = RECORDING_FPS
        self.record_dir                = RECORDING_SAVE_DIR
        self.encrypt_video_without_avi = ENCRYPTION
        self.gui_available             = GUI_AVAILABLE

        if self.source == "Camera":
            if not has_gui_libs:
                raise RuntimeError(
                    "Camera source requires GStreamer, GTK, and cairo libraries "
                    "which are not installed on this platform. Please run with --source Video."
                )

            cam_ok, sensor_name, detail = _check_camera_sensor()
            if not cam_ok:
                raise RuntimeError(
                    f"[ALERT] CSI camera sensor '{sensor_name}' not detected "
                    f"({detail!r}). Check that the camera module is connected "
                    f"and properly seated, then restart."
                )

            self.gst_widget = GstWidget(self)
            if GUI_AVAILABLE:
                self.main_window = MainWindow(self)
            else:
                self.main_window = None
                self.gst_widget.start()   # no window -> "realize" never fires, so start manually
        else:
            self.video_widget = VideoFileWidget(self)

    def run(self):
        if self.source == "Camera":
            if GUI_AVAILABLE:
                self.main_window.show_all()
                self.main_window.connect("delete-event", Gtk.main_quit)
                Gtk.main()
            else:
                print("[Headless] Running without display — Ctrl+C to quit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    self.gst_widget.stop()
        else:
            self.video_widget.run()


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)   # Ctrl-C works normally

    parser = argparse.ArgumentParser(
        # ── CHANGED: generic description ─────────────────────────────────────
        description="YOLOv8n / YOLOv9t INT8 — detection + recording"
    )
    parser.add_argument(
        "--source",
        type=str, default=DETECTION_ON,
        choices=["Camera", "Video"],
        help="'Camera' (CSI via libcamerasrc), 'Video' (file via OpenCV), "
             "or (HTTP stream via OpenCV).  "
             f"Default: {DETECTION_ON}",
    )
    parser.add_argument(
        "--video_path",
        type=str, default=VIDEO_PATH,
        help="Video file path — required when --source Video",
    )
    parser.add_argument(
        "--model_path",
        type=str, default=MODEL_PATH,
        help=".nb (NPU) or .tflite (CPU) model file.  "
             "Model variant (YOLOv8n / YOLOv9t) is auto-detected from filename.",
    )

    args = parser.parse_args()

    # ── CHANGED: generic banner ───────────────────────────────────────────────
    label      = _model_label(args.model_path)

    print("=" * 60)
    print(f"  {label} Detection + Recorder")
    source_str = f"{args.source}"
    if args.source == "Video":
        source_str += f"  →  {args.video_path}"
    print(f"  Source     : {source_str}")
    print(f"  Model      : {args.model_path}")
    print(f"  Resolution : {FRAME_WIDTH}×{FRAME_HEIGHT}")
    print(f"  Confidence : {CONFIDENCE_TH}   IoU : {IOU_TH}")
    print("=" * 60)

    # Pin whole process to core 0; inference thread overrides itself to core 1
    try:
        os.sched_setaffinity(0, {0})
        print("[CPU] Process → core 0")
    except Exception as exc:
        print(f"[CPU] Process pin failed: {exc}")

    try:
        app = Application(args)
        app.run()
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        import traceback; traceback.print_exc()

    print("[Pipeline] Exited cleanly.")
    os._exit(0)
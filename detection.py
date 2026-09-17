"""
detection.py
────────────
Unified YOLOv8n / YOLOv9t INT8 person-vehicle detector.

Auto-selects backend by file extension:
  .nb      → STM32MP2 Vivante NPU  (stai_mpu, hardware-accelerated)
  .tflite  → ARM CPU               (tflite_runtime, 4-thread)

For .nb models, quantisation params are resolved in order:
  1. Live query of the stai_mpu tensor API  (get_scale / get_zero_point)
  2. Built-in lookup table keyed on filename substring

Preprocess uses the zero-allocation XOR sign-bit trick when
  in_scale ≈ 1/255 and in_zp = -128  (true for both YOLOv8n and YOLOv9t).
Falls back to a float path automatically for any other model.

NMS is performed by cv2.dnn.NMSBoxes (C++ path, faster than Python greedy).

Public API
──────────
  detector = Detector(model_path, conf_threshold=0.35, iou_threshold=0.45)
  results  = detector.detect(bgr_frame)
  # → [[x1, y1, x2, y2, label_str, conf_float], ...]
  #   coords are in the INPUT frame's pixel space (any resolution)
"""

import cv2
import numpy as np
from pathlib import Path

# ── Target COCO classes ───────────────────────────────────────────────────────
TARGET_CLASSES = {0: "Person", 2: "Vehicle", 3: "Vehicle", 5: "Vehicle", 7: "Vehicle"}
_TARGET_IDS    = sorted(TARGET_CLASSES.keys())          # [0, 2, 3, 5, 7]

# ── Fallback quant params for .nb models (when tensor API is unavailable) ─────
# Key = lowercase substring expected in the model filename.
_NB_QUANT = {
    "yolov8n": dict(
        in_scale  = 0.003921568859368563,   in_zp  = -128,
        out_scale = 0.004194467328488827,   out_zp = -128,
    ),
    "yolov9t": dict(
        in_scale  = 0.003921568859368563,   in_zp  = -128,
        out_scale = 0.004248024430125952,   out_zp = -128,
    ),
}


class Detector:
    """
    Unified YOLOv8n / YOLOv9t INT8 person-vehicle detector.
    Thread-safe for detect() after __init__ completes.
    """

    def __init__(
        self,
        model_path:     str,
        conf_threshold: float = 0.35,
        iou_threshold:  float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        ext = Path(model_path).suffix.lower()
        if ext == ".nb":
            self._load_nb(model_path)
        elif ext == ".tflite":
            self._load_tflite(model_path)
        else:
            raise ValueError(f"Unsupported extension '{ext}' — use .nb or .tflite")

        _, self._ih, self._iw, _ = self._input_shape

        # Pre-allocated I/O buffers — zero heap allocation per frame
        self._input_buf = np.zeros(self._input_shape, dtype=np.int8)
        self._rgb_buf   = np.empty((self._ih, self._iw, 3), dtype=np.uint8)

        # Decide preprocess strategy once at load time (not per-frame)
        # XOR trick is valid when in_scale ≈ 1/255 and in_zp = -128
        self._use_xor = (
            abs(self._in_scale - 1.0 / 255.0) < 1e-6 and self._in_zp == -128
        )

        #self._print_info()

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_nb(self, model_path: str):
        """
        NPU backend via stai_mpu.
        Reads quant params from tensor API first; falls back to lookup table.
        """
        from stai_mpu import stai_mpu_network
        self._net = stai_mpu_network(model_path=model_path, use_hw_acceleration=True)

        try:
            inp_t = self._net.get_input(0)
            out_t = self._net.get_output(0)
            self._input_shape  = tuple(inp_t.get_shape())
            self._output_shape = tuple(out_t.get_shape())
            self._in_scale     = float(inp_t.get_scale())
            self._in_zp        = int(inp_t.get_zero_point())
            self._out_scale    = float(out_t.get_scale())
            self._out_zp       = int(out_t.get_zero_point())
            self._quant_src    = "stai_mpu tensor API"

        except Exception as exc:
            # Tensor API unavailable — use the built-in table
            print(f"[Detector] Tensor API failed ({exc}) → using lookup table")
            stem = Path(model_path).stem.lower()
            key  = next((k for k in _NB_QUANT if k in stem), None)
            if key is None:
                raise RuntimeError(
                    f"Unknown .nb model: '{model_path}'.\n"
                    f"Add its quant params to _NB_QUANT, or ensure filename "
                    f"contains one of: {list(_NB_QUANT.keys())}"
                ) from exc
            p = _NB_QUANT[key]
            self._input_shape  = (1, 320, 320, 3)
            self._output_shape = (1, 84, 2100)
            self._in_scale,  self._in_zp  = p["in_scale"],  p["in_zp"]
            self._out_scale, self._out_zp = p["out_scale"], p["out_zp"]
            self._quant_src = f"lookup table (key='{key}')"

        self.backend = "NPU (.nb)"

    def _load_tflite(self, model_path: str):
        """CPU backend via tflite_runtime (falls back to full TensorFlow)."""
        try:
            import tflite_runtime.interpreter as tflite
            Interp = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interp = tf.lite.Interpreter

        self._interp = Interp(model_path=model_path, num_threads=4)
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]

        self._in_idx       = inp["index"]
        self._out_idx      = out["index"]
        self._input_shape  = tuple(inp["shape"])
        self._output_shape = tuple(out["shape"])
        self._in_scale     = float(inp["quantization"][0])
        self._in_zp        = int(inp["quantization"][1])
        self._out_scale    = float(out["quantization"][0])
        self._out_zp       = int(out["quantization"][1])
        self._quant_src    = "tflite metadata"
        self.backend       = "CPU (.tflite)"

    # ── Logging ───────────────────────────────────────────────────────────────

    def _print_info(self):
        pre = "XOR fast-path (pixel−128)" if self._use_xor else "float quantise"
        print(f"[Detector] Backend    : {self.backend}")
        print(f"[Detector] In  shape  : {self._input_shape}  "
              f"scale={self._in_scale:.12f}  zp={self._in_zp}")
        print(f"[Detector] Out shape  : {self._output_shape}  "
              f"scale={self._out_scale:.12f}  zp={self._out_zp}")
        print(f"[Detector] Quant src  : {self._quant_src}")
        print(f"[Detector] Preprocess : {pre}")
        print(f"[Detector] Classes    : {', '.join(TARGET_CLASSES.values())}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on a BGR frame.

        Args:
            frame: uint8 BGR ndarray — any resolution.
                   Resized internally to model input size when needed.

        Returns:
            list of [x1, y1, x2, y2, label_str, conf_float]
            Pixel coordinates are in the INPUT frame's coordinate space.
        """
        h, w  = frame.shape[:2]
        blob  = self._preprocess(frame)
        pred  = self._run_model(blob)       # (anchors, 84)  float32
        return self._parse(pred, w, h)

    # ── Private ───────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR uint8 → int8 NHWC.  Zero heap allocation after __init__."""

        # Resize only when frame doesn't match model input size
        if frame.shape[0] != self._ih or frame.shape[1] != self._iw:
            frame = cv2.resize(frame, (self._iw, self._ih))

        # BGR → RGB into pre-allocated buffer (no new array created)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)

        if self._use_xor:
            # Fast path — valid when in_scale≈1/255, in_zp=-128:
            #   int8 = pixel − 128  ≡  flip the MSB (XOR 0x80)
            # Zero float arithmetic, zero heap allocations.
            np.bitwise_xor(
                self._rgb_buf.view(np.int8), np.int8(-128),
                out=self._input_buf[0],
            )
        else:
            # General path — works for any in_scale / in_zp
            #   int8 = clip(round(pixel / (255 × scale) + zp), −128, 127)
            tmp  = self._rgb_buf.astype(np.float32)
            tmp *= 1.0 / (255.0 * self._in_scale)
            tmp += self._in_zp
            np.clip(tmp, -128, 127, out=tmp)
            self._input_buf[0] = tmp.astype(np.int8)

        return self._input_buf

    def _run_model(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference and dequantise → (anchors, 84) float32."""

        if self.backend == "NPU (.nb)":
            self._net.set_input(0, input_tensor)
            self._net.run()
            raw = self._net.get_output(0)                       # (1, 84, 2100) int8
        else:
            self._interp.set_tensor(self._in_idx, input_tensor)
            self._interp.invoke()
            raw = self._interp.get_tensor(self._out_idx)        # (1, 84, N) int8

        # Dequantise:  float = (int8 − zp) × scale
        out = (raw[0].astype(np.float32) - self._out_zp) * self._out_scale

        # Normalise layout to (anchors, 84) — output may be (84, anchors)
        return out.T if out.shape[0] == 84 else out

    def _parse(self, pred: np.ndarray, w: int, h: int) -> list:
        """
        Vectorised confidence filter + NMS — no Python loop over anchors.

        pred columns: cx, cy, bw, bh, score_cls0 … score_cls79  (normalised 0–1)
        """
        # ── Score only the 5 target class columns ─────────────────────────────
        sub_scores = pred[:, 4:][:, _TARGET_IDS]                        # (N, 5)
        best_sub   = np.argmax(sub_scores, axis=1)                      # (N,)
        best_conf  = sub_scores[np.arange(len(sub_scores)), best_sub]   # (N,)

        mask = best_conf >= self.conf_threshold
        if not mask.any():
            return []

        pred_m    = pred[mask]
        conf_m    = best_conf[mask]
        class_ids = np.array(_TARGET_IDS)[best_sub[mask]]

        # ── Box corners in the INPUT frame's pixel space ───────────────────────
        cx, cy = pred_m[:, 0], pred_m[:, 1]
        bw, bh = pred_m[:, 2], pred_m[:, 3]

        x1 = np.clip((cx - bw / 2) * w, 0, w).astype(int)
        y1 = np.clip((cy - bh / 2) * h, 0, h).astype(int)
        x2 = np.clip((cx + bw / 2) * w, 0, w).astype(int)
        y2 = np.clip((cy + bh / 2) * h, 0, h).astype(int)

        # ── NMS via C++ cv2.dnn.NMSBoxes ──────────────────────────────────────
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(
            boxes_xywh, conf_m.tolist(),
            self.conf_threshold, self.iou_threshold,
        )

        result = []
        if len(keep):
            for i in keep.flatten():
                result.append([
                    int(x1[i]), int(y1[i]),
                    int(x2[i]), int(y2[i]),
                    TARGET_CLASSES[int(class_ids[i])],
                    float(conf_m[i]),
                ])
        return result

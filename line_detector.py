"""
line_detector.py
────────────────
Virtual-line proximity detector — centroid + overlap based.

Logic (new)
───────────
  The virtual line is defined by two configurable points p1 and p2.
  One side of the line is marked as the ALERT side via "alert_side" in
  the config (a third point that lies on the alert side).

  For each detected bounding box the rule is:

    1. Box overlaps the line segment (Cohen-Sutherland intersection check)
       → RED  (regardless of centroid position)

    2. Centroid is on the ALERT side of the line  (signed distance > 0)
       → RED

    3. Centroid is on the SAFE side  (signed distance < 0)
       → GREEN

    4. Centroid is exactly on the line  (signed distance == 0)
       → RED  (treated same as alert side)

  Distance is measured as the perpendicular signed distance from the
  centroid to the INFINITE LINE through p1–p2, then clamped to the
  segment for the overlap check.

  Terminal alert prints ONCE on the GREEN → RED transition per track.
  When a track goes RED → GREEN the alert clears so the next crossing
  will print again.

Config file  (line_config.json)
───────────────────────────────
  {
    "p1":              [x1, y1],
    "p2":              [x2, y2],
    "alert_side_pt":   [x3, y3],   // any point on the alert side of the line
    "threshold_pixels": 0          // optional dead-band around the line (default 0)
  }

  alert_side_pt is a single reference point that you place on the side
  you want to trigger alerts.  E.g. if your line is horizontal and
  "below" should be the alert side, put [frame_width/2, frame_height-1].

Public API
──────────
  detector = ProximityDetector(config_path, frame_width, frame_height)
  alert_flags = detector.update(detections)   # list of bools, one per detection
  detector.draw_line_opencv(frame)
  detector.draw_line_cairo(ctx)
"""

import os
import json
import math
import cv2

CONFIG_PATH  = "line_config.json"
MATCH_RADIUS = 60.0   # px — max centroid movement between frames to re-link a track


# ─────────────────────────────────────────────────────────────────────────────
class ProximityDetector:

    def __init__(self, config_path=CONFIG_PATH, frame_width=640, frame_height=480):
        self.config_path  = config_path
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # Line endpoints (set by load())
        self.p1 = (0, frame_height // 2)
        self.p2 = (frame_width - 1, frame_height // 2)

        # The sign of the signed distance for a point on the alert side.
        # +1 means positive signed-distance → alert side.
        # -1 means negative signed-distance → alert side.
        self._alert_sign = 1

        self.threshold = 0.0   # dead-band around line in pixels (usually 0)
        self.mtime     = 0.0

        # Track list.  Each entry:
        #   cx, cy       – centroid at last update
        #   label        – class string
        #   is_alert     – bool: currently on alert side / overlapping line
        self.tracks = []

        self.load()

    # ── Resolution update ─────────────────────────────────────────────────────
    def set_resolution(self, width, height):
        self.frame_width  = int(width)
        self.frame_height = int(height)
        self.p1 = self._clamp(self.p1)
        self.p2 = self._clamp(self.p2)

    # ── Config hot-reload ─────────────────────────────────────────────────────
    def _clamp(self, pt):
        x = max(0, min(int(pt[0]), self.frame_width  - 1))
        y = max(0, min(int(pt[1]), self.frame_height - 1))
        return (x, y)

    def load(self):
        """Reload config if the file has been modified."""
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:
            return
        if mtime == self.mtime:
            return

        with open(self.config_path) as f:
            cfg = json.load(f)

        self.p1        = self._clamp(cfg.get("p1", [0, self.frame_height // 2]))
        self.p2        = self._clamp(cfg.get("p2", [self.frame_width - 1, self.frame_height // 2]))
        self.threshold = float(cfg.get("threshold_pixels", 0.0))

        # Determine alert-side sign from the reference point
        ref_raw = cfg.get("alert_side_pt",
                            [self.frame_width // 2, self.frame_height - 1])
        ref = self._clamp(ref_raw)
        ref_dist = self._signed_distance(ref[0], ref[1])
        if ref_dist > 0:
            self._alert_sign = 1
        elif ref_dist < 0:
            self._alert_sign = -1
        else:
            # Reference point is exactly on the line — fall back to lower side
            self._alert_sign = 1

        self.mtime  = mtime

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _signed_distance(self, px, py):
        """
        Signed perpendicular distance from point (px, py) to the infinite
        line through p1–p2.

          positive → one side of the line
          negative → the other side
          zero     → on the line

        The sign convention is fixed: positive is to the LEFT of the
        direction vector p1→p2 (standard 2-D cross-product convention).
        """
        x1, y1 = self.p1
        x2, y2 = self.p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0
        # Cross product (p1→p2) × (p1→point), normalised
        return ((px - x1) * dy - (py - y1) * dx) / length

    @staticmethod
    def _seg_intersects_rect(ax1, ay1, ax2, ay2, rx1, ry1, rx2, ry2):
        """
        Cohen-Sutherland line-clipping to test if segment (ax1,ay1)→(ax2,ay2)
        intersects or touches rectangle [rx1,ry1]→[rx2,ry2].
        """
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

        def code(x, y):
            c = INSIDE
            if   x < rx1: c |= LEFT
            elif x > rx2: c |= RIGHT
            if   y < ry1: c |= TOP
            elif y > ry2: c |= BOTTOM
            return c

        x0, y0, x1, y1 = float(ax1), float(ay1), float(ax2), float(ay2)
        c0, c1 = code(x0, y0), code(x1, y1)

        while True:
            if not (c0 | c1):   return True    # both inside
            if  c0 & c1:        return False   # both outside same half-plane
            c_out = c0 if c0 else c1
            dx = x1 - x0;  dy = y1 - y0
            if   c_out & BOTTOM: x = x0 + dx * (ry2 - y0) / dy if dy else x0; y = ry2
            elif c_out & TOP:    x = x0 + dx * (ry1 - y0) / dy if dy else x0; y = ry1
            elif c_out & RIGHT:  y = y0 + dy * (rx2 - x0) / dx if dx else y0; x = rx2
            else:                y = y0 + dy * (rx1 - x0) / dx if dx else y0; x = rx1
            if c_out == c0: x0, y0, c0 = x, y, code(x, y)
            else:           x1, y1, c1 = x, y, code(x, y)

    def _box_overlaps_line(self, bx1, by1, bx2, by2):
        """True if the line segment p1→p2 intersects the bounding box."""
        return self._seg_intersects_rect(
            self.p1[0], self.p1[1], self.p2[0], self.p2[1],
            bx1, by1, bx2, by2,
        )

    def _centroid_is_alert(self, cx, cy):
        """
        Returns True if the centroid (cx, cy) is on the alert side of the line
        (or exactly on it).

        Uses signed distance:
          sign matches _alert_sign → alert side
          distance == 0            → on the line → treat as alert
        """
        sd = self._signed_distance(cx, cy)
        # Apply threshold dead-band: within ±threshold treat as "on the line" → alert
        if abs(sd) <= self.threshold:
            return True   # on or within dead-band → RED
        return (sd * self._alert_sign) > 0

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, detections):
        """
        Update tracks and return alert flags.

        Args:
            detections: list of [x1, y1, x2, y2, label, conf]

        Returns:
            list of bool — True = RED (alert side or overlapping line)
        """
        self.load()

        if not detections:
            self.tracks = []
            return []

        alert_flags = []
        new_tracks  = []

        for det in detections:
            x1, y1, x2, y2, label, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # ── Determine alert state ─────────────────────────────────────────
            # Rule 1: box overlaps the line → RED
            # Rule 2/3/4: centroid side decides
            overlaps = self._box_overlaps_line(x1, y1, x2, y2)
            if overlaps:
                is_alert = True
            else:
                is_alert = self._centroid_is_alert(cx, cy)

            # ── Match to existing track ───────────────────────────────────────
            best_d      = float("inf")
            matched_idx = -1
            for idx, track in enumerate(new_tracks):
                # new_tracks only has entries added this frame — skip
                pass
            for idx, track in enumerate(self.tracks):
                if track["label"] == label:
                    d = math.hypot(cx - track["cx"], cy - track["cy"])
                    if d < best_d and d < MATCH_RADIUS:
                        best_d      = d
                        matched_idx = idx

            # ── Terminal alert on GREEN → RED transition ──────────────────────
            if matched_idx != -1:
                prev      = self.tracks[matched_idx]
                was_alert = prev["is_alert"]
                sd        = self._signed_distance(cx, cy)

                if is_alert and not was_alert:
                    print(
                        f"[ALERT] {label} entered alert zone. "
                        # f"Centroid=({cx:.0f},{cy:.0f})  "
                        # f"signed_dist={sd:.1f}px  "
                        # f"overlap={'yes' if overlaps else 'no'}"
                    )
                elif not is_alert and was_alert:
                    print(
                        f"[CLEAR] {label} left alert zone. "
                        # f"Centroid=({cx:.0f},{cy:.0f})  "
                        # f"signed_dist={sd:.1f}px"
                    )
            else:
                # Brand-new track — print alert immediately if already in alert zone
                if is_alert:
                    sd = self._signed_distance(cx, cy)
                    print(
                        f"[ALERT] New {label} detected in alert zone. "
                        # f"Centroid=({cx:.0f},{cy:.0f})  "
                        # f"signed_dist={sd:.1f}px  "
                        # f"overlap={'yes' if overlaps else 'no'}"
                    )

            new_tracks.append({
                "cx": cx, "cy": cy,
                "label": label,
                "is_alert": is_alert,
            })
            alert_flags.append(is_alert)

        self.tracks = new_tracks
        return alert_flags

    # ── Draw helpers ──────────────────────────────────────────────────────────

    def draw_line_opencv(self, frame):
        """Draw the virtual line and its endpoints on a BGR frame."""
        cv2.line(frame,   self.p1, self.p2, (0, 127, 255), 2)
        # cv2.circle(frame, self.p1, 4, (0, 127, 255), -1)
        # cv2.circle(frame, self.p2, 4, (0, 127, 255), -1)

    def draw_line_cairo(self, ctx):
        """Draw the virtual line on a Cairo context."""
        ctx.save()
        ctx.set_source_rgb(1.0, 0.5, 0.0)
        ctx.set_line_width(2)
        ctx.move_to(self.p1[0], self.p1[1])
        ctx.line_to(self.p2[0], self.p2[1])
        ctx.stroke()
        ctx.arc(self.p1[0], self.p1[1], 4, 0, 2 * math.pi)
        ctx.fill()
        ctx.arc(self.p2[0], self.p2[1], 4, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()
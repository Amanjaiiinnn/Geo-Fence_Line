"""
line_config_ui.py
─────────────────
Standalone PC tool to visually configure the virtual line for line_config.json.

Usage:
    python line_config_ui.py                        # looks for line_config.json in cwd
    python line_config_ui.py --config /path/to/line_config.json

Controls:
    Drag P1 or P2        →  reposition line endpoints
    Drag ALERT handle    →  set which side of the line triggers the alert
    Type in X/Y boxes    →  edit coordinates directly (live, no Enter needed)
    Resolution box       →  resize canvas to match your video resolution
    Threshold            →  pixel penetration depth that triggers an alert
    Save                 →  write to line_config.json
    Reset                →  reload last saved values from JSON
"""

import tkinter as tk
from tkinter import messagebox
import json
import os
import argparse

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
POINT_RADIUS = 9
HIT_SLACK    = 6
GRID_COLS    = 10
GRID_ROWS    = 8

BG_DARK      = "#1e1e2e"
BG_PANEL     = "#2a2a3e"
BG_CANVAS    = "#12121a"
GRID_COLOR   = "#2a2a4a"
LINE_COLOR   = "#ff8c00"
P1_COLOR     = "#ff8c00"
P2_COLOR     = "#00bfff"
ALERT_COLOR  = "#ff3333"
ACCENT       = "#7c6af5"
TEXT_MAIN    = "#e0e0f0"
TEXT_DIM     = "#6a6a8a"
BTN_SAVE_BG  = "#3a8c4e"
BTN_SAVE_FG  = "#ffffff"
BTN_RESET_BG = "#4a3a6e"
BTN_RESET_FG = "#ffffff"
ENTRY_BG     = "#1a1a2e"
ENTRY_OK     = "#2a2a3e"
ENTRY_ERR    = "#3e1a1a"

FONT_MONO   = ("Courier New", 10)
FONT_LABEL  = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI", 11, "bold")
FONT_COORDS = ("Courier New", 11, "bold")
FONT_SMALL  = ("Segoe UI", 8)


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
class LineConfigUI:
    def __init__(self, root: tk.Tk, config_path: str):
        self.root        = root
        self.config_path = config_path

        # State
        self.vid_w     = 640
        self.vid_h     = 480
        self.p1        = [100, 240]
        self.p2        = [540, 240]
        self.alert_pt  = [320, 400]
        self.threshold = 50.0
        self.dragging  = None

        # Guard so typing into fields doesn't trigger a re-render loop
        self._updating_fields = False

        self._load_config()   # load JSON first so vid_w/h/points are set
        self._build_ui()      # then build UI with correct values
        self._draw(sync_fields=True)

    # ── Config I/O ────────────────────────────────────────────────────────────

    def _load_config(self):
        """Load all values from JSON including frame resolution."""
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)

            # Restore frame resolution if saved
            if "frame_width" in cfg and "frame_height" in cfg:
                self.vid_w = int(cfg["frame_width"])
                self.vid_h = int(cfg["frame_height"])

            self.p1        = [int(cfg["p1"][0]),  int(cfg["p1"][1])]
            self.p2        = [int(cfg["p2"][0]),  int(cfg["p2"][1])]
            self.threshold = float(cfg.get("threshold_pixels", 50.0))

            if "alert_side_pt" in cfg:
                self.alert_pt = [int(cfg["alert_side_pt"][0]),
                                 int(cfg["alert_side_pt"][1])]
            else:
                self.alert_pt = [self.vid_w // 2, int(self.vid_h * 0.8)]

        except Exception as e:
            messagebox.showwarning("Load warning", f"Could not load config:\n{e}")

    def _save_config(self):
        try:
            threshold = float(self.thr_var.get())
        except ValueError:
            messagebox.showerror("Error", "Threshold must be a number.")
            return

        cfg = {
            "p1":               self.p1,
            "p2":               self.p2,
            "alert_side_pt":    self.alert_pt,
            "threshold_pixels": threshold,
            "frame_width":      self.vid_w,    # save resolution so it restores on next launch
            "frame_height":     self.vid_h,
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._status(f"Saved → {os.path.abspath(self.config_path)}", ok=True)
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("Virtual Line Configurator")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=BG_PANEL, pady=8)
        title_bar.pack(fill=tk.X)
        tk.Label(
            title_bar, text="⬛  Virtual Line Configurator",
            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_TITLE,
        ).pack(side=tk.LEFT, padx=14)
        tk.Label(
            title_bar, text=os.path.basename(self.config_path),
            bg=BG_PANEL, fg=TEXT_DIM, font=FONT_LABEL,
        ).pack(side=tk.RIGHT, padx=14)

        # ── Top controls: resolution + threshold ───────────────────────────────
        top = tk.Frame(self.root, bg=BG_DARK, pady=6, padx=12)
        top.pack(fill=tk.X)

        tk.Label(top, text="Frame resolution:", bg=BG_DARK,
                 fg=TEXT_DIM, font=FONT_LABEL).pack(side=tk.LEFT)

        self.w_var = tk.StringVar(value=str(self.vid_w))
        self.h_var = tk.StringVar(value=str(self.vid_h))
        self._entry(top, self.w_var, width=6).pack(side=tk.LEFT, padx=(6, 2))
        tk.Label(top, text="×", bg=BG_DARK, fg=TEXT_MAIN,
                 font=FONT_LABEL).pack(side=tk.LEFT)
        self._entry(top, self.h_var, width=6).pack(side=tk.LEFT, padx=(2, 8))
        self._btn(top, "Apply", self._apply_resolution,
                  bg=BG_PANEL, fg=TEXT_MAIN).pack(side=tk.LEFT)

        tk.Label(top, text="Threshold (px):", bg=BG_DARK,
                 fg=TEXT_DIM, font=FONT_LABEL).pack(side=tk.LEFT, padx=(24, 6))
        self.thr_var = tk.StringVar(value=str(int(self.threshold)))
        self._entry(top, self.thr_var, width=6).pack(side=tk.LEFT)

        # ── Canvas ────────────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=BG_DARK, padx=12)
        canvas_frame.pack()

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.vid_w, height=self.vid_h,
            bg=BG_CANVAS, cursor="crosshair",
            highlightthickness=1, highlightbackground=ACCENT,
        )
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>",          self._on_hover)

        # ── Coordinate readout + type-in fields ───────────────────────────────
        coord_frame = tk.Frame(self.root, bg=BG_PANEL, pady=8, padx=14)
        coord_frame.pack(fill=tk.X)

        # Column headers
        for col, (label, color) in enumerate([("P1", P1_COLOR),
                                               ("P2", P2_COLOR),
                                               ("ALERT", ALERT_COLOR)]):
            tk.Label(coord_frame, text=label, bg=BG_PANEL,
                     fg=color, font=FONT_TITLE).grid(
                row=0, column=col * 3, padx=(0, 4), sticky="w")

        # X / Y entry boxes for each point
        # Each point gets two Entry widgets (X and Y) with live trace
        self.p1_x_var    = tk.StringVar(value=str(self.p1[0]))
        self.p1_y_var    = tk.StringVar(value=str(self.p1[1]))
        self.p2_x_var    = tk.StringVar(value=str(self.p2[0]))
        self.p2_y_var    = tk.StringVar(value=str(self.p2[1]))
        self.alert_x_var = tk.StringVar(value=str(self.alert_pt[0]))
        self.alert_y_var = tk.StringVar(value=str(self.alert_pt[1]))

        def make_coord_pair(parent, x_var, y_var, col_offset, point_key):
            """Build X/Y entry pair for one point identified by point_key."""
            # point_key is a string: 'p1', 'p2', or 'alert_pt'
            # We look up self.<point_key> at write-time so that _load_config()
            # replacing the list object never breaks the trace.

            # X entry
            tk.Label(parent, text="X", bg=BG_PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).grid(row=1, column=col_offset, sticky="e")
            ex = tk.Entry(parent, textvariable=x_var, width=6,
                          bg=ENTRY_BG, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                          relief=tk.FLAT, font=FONT_MONO)
            ex.grid(row=1, column=col_offset + 1, padx=(2, 4))

            # Y entry
            tk.Label(parent, text="Y", bg=BG_PANEL, fg=TEXT_DIM,
                     font=FONT_SMALL).grid(row=2, column=col_offset, sticky="e")
            ey = tk.Entry(parent, textvariable=y_var, width=6,
                          bg=ENTRY_BG, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
                          relief=tk.FLAT, font=FONT_MONO)
            ey.grid(row=2, column=col_offset + 1, padx=(2, 4))

            def make_trace(var, entry_widget, coord_idx, max_val_fn, key):
                def _trace(*_):
                    if self._updating_fields:
                        return
                    raw = var.get()
                    if raw == "" or raw == "-":
                        entry_widget.config(bg=ENTRY_BG)
                        return
                    try:
                        val = int(raw)
                        val = max(0, min(val, max_val_fn()))
                        # Always look up the live attribute so Reset never breaks it
                        getattr(self, key)[coord_idx] = val
                        entry_widget.config(bg=ENTRY_OK)
                        self._draw()
                    except ValueError:
                        entry_widget.config(bg=ENTRY_ERR)
                return _trace

            x_var.trace_add("write", make_trace(
                x_var, ex, 0, lambda: self.vid_w - 1, point_key))
            y_var.trace_add("write", make_trace(
                y_var, ey, 1, lambda: self.vid_h - 1, point_key))

        make_coord_pair(coord_frame, self.p1_x_var,    self.p1_y_var,    0, "p1")
        make_coord_pair(coord_frame, self.p2_x_var,    self.p2_y_var,    3, "p2")
        make_coord_pair(coord_frame, self.alert_x_var, self.alert_y_var, 6, "alert_pt")

        # Mouse cursor readout
        self.mouse_var = tk.StringVar(value="")
        tk.Label(coord_frame, textvariable=self.mouse_var,
                 bg=BG_PANEL, fg=TEXT_DIM, font=FONT_MONO).grid(
            row=1, column=9, rowspan=2, padx=(20, 0), sticky="w")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bot = tk.Frame(self.root, bg=BG_DARK, pady=8, padx=12)
        bot.pack(fill=tk.X)
        tk.Label(
            bot,
            text="Drag handles  or  type X/Y values directly",
            bg=BG_DARK, fg=TEXT_DIM, font=FONT_LABEL,
        ).pack(side=tk.LEFT)
        self._btn(bot, "↺  Reset", self._reset,
                  bg=BTN_RESET_BG, fg=BTN_RESET_FG).pack(side=tk.RIGHT, padx=(8, 0))
        self._btn(bot, "💾  Save", self._save_config,
                  bg=BTN_SAVE_BG, fg=BTN_SAVE_FG, padx=14).pack(side=tk.RIGHT)

        # ── Status bar ────────────────────────────────────────────────────────
        loaded_msg = f"Loaded → {self.config_path}" if os.path.exists(self.config_path) \
                     else f"No existing config — will create {self.config_path}"
        self.status_var = tk.StringVar(value=loaded_msg)
        status_bar = tk.Frame(self.root, bg="#111122", pady=3)
        status_bar.pack(fill=tk.X)
        tk.Label(
            status_bar, textvariable=self.status_var,
            bg="#111122", fg=TEXT_DIM, font=FONT_LABEL,
            anchor="w", padx=12,
        ).pack(fill=tk.X)

    def _entry(self, parent, var, width=8):
        return tk.Entry(
            parent, textvariable=var, width=width,
            bg=BG_PANEL, fg=TEXT_MAIN, insertbackground=TEXT_MAIN,
            relief=tk.FLAT, font=FONT_MONO,
        )

    def _btn(self, parent, text, cmd, bg=BG_PANEL, fg=TEXT_MAIN, padx=10):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, activebackground=ACCENT,
            activeforeground="#ffffff", relief=tk.FLAT,
            font=FONT_LABEL, padx=padx, pady=4, cursor="hand2",
        )

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, sync_fields=False):
        c = self.canvas
        c.delete("all")

        # Grid
        for i in range(1, GRID_COLS):
            x = int(self.vid_w * i / GRID_COLS)
            c.create_line(x, 0, x, self.vid_h, fill=GRID_COLOR, width=1)
        for i in range(1, GRID_ROWS):
            y = int(self.vid_h * i / GRID_ROWS)
            c.create_line(0, y, self.vid_w, y, fill=GRID_COLOR, width=1)

        c.create_rectangle(0, 0, self.vid_w - 1, self.vid_h - 1,
                           outline="#333355", width=1)

        # Line shadow + line
        c.create_line(self.p1[0]+2, self.p1[1]+2,
                      self.p2[0]+2, self.p2[1]+2,
                      fill="#0a0a14", width=3)
        c.create_line(self.p1[0], self.p1[1], self.p2[0], self.p2[1],
                      fill=LINE_COLOR, width=2, dash=(8, 4))

        # Handles
        self._draw_handle(self.alert_pt, ALERT_COLOR, "ALERT SIDE")
        self._draw_handle(self.p1,       P1_COLOR,    "P1")
        self._draw_handle(self.p2,       P2_COLOR,    "P2")

        # Only sync text fields when called from drag/reset, not when called
        # from a field trace (that would overwrite what the user is typing)
        if sync_fields:
            self._sync_fields()

    def _draw_handle(self, pt, color, label):
        c    = self.canvas
        x, y = pt
        r    = POINT_RADIUS
        c.create_oval(x-r-4, y-r-4, x+r+4, y+r+4,
                      outline=color, fill="", width=1, stipple="gray25")
        c.create_oval(x-r, y-r, x+r, y+r,
                      fill=color, outline="white", width=2)
        c.create_oval(x-2, y-2, x+2, y+2, fill="white", outline="")
        c.create_text(x, y-r-10, text=label,
                      fill=color, font=("Segoe UI", 9, "bold"))

    def _sync_fields(self):
        """Push current point values into the text entry vars without firing traces."""
        self._updating_fields = True
        self.p1_x_var.set(str(self.p1[0]))
        self.p1_y_var.set(str(self.p1[1]))
        self.p2_x_var.set(str(self.p2[0]))
        self.p2_y_var.set(str(self.p2[1]))
        self.alert_x_var.set(str(self.alert_pt[0]))
        self.alert_y_var.set(str(self.alert_pt[1]))
        self._updating_fields = False

    # ── Interactions ──────────────────────────────────────────────────────────

    def _hit_test(self, x, y):
        r = POINT_RADIUS + HIT_SLACK
        for name, pt in [("p1", self.p1), ("p2", self.p2), ("alert", self.alert_pt)]:
            if ((x - pt[0])**2 + (y - pt[1])**2)**0.5 <= r:
                return name
        return None

    def _on_press(self, event):
        self.dragging = self._hit_test(event.x, event.y)

    def _on_drag(self, event):
        if self.dragging is None:
            return
        x = max(0, min(event.x, self.vid_w - 1))
        y = max(0, min(event.y, self.vid_h - 1))
        if   self.dragging == "p1":    self.p1       = [x, y]
        elif self.dragging == "p2":    self.p2       = [x, y]
        elif self.dragging == "alert": self.alert_pt = [x, y]
        self._draw(sync_fields=True)

    def _on_release(self, event):
        self.dragging = None

    def _on_hover(self, event):
        x = max(0, min(event.x, self.vid_w - 1))
        y = max(0, min(event.y, self.vid_h - 1))
        self.mouse_var.set(f"cursor  ({x:>4}, {y:>4})")
        self.canvas.config(cursor="fleur" if self._hit_test(x, y) else "crosshair")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _apply_resolution(self):
        try:
            w = int(self.w_var.get())
            h = int(self.h_var.get())
            if w < 100 or h < 100:
                raise ValueError("Minimum 100×100.")
            if w > 4096 or h > 4096:
                raise ValueError("Maximum 4096×4096.")
        except ValueError as e:
            messagebox.showerror("Invalid resolution", str(e))
            return

        self.vid_w = w
        self.vid_h = h
        self.p1       = [min(self.p1[0],       w-1), min(self.p1[1],       h-1)]
        self.p2       = [min(self.p2[0],       w-1), min(self.p2[1],       h-1)]
        self.alert_pt = [min(self.alert_pt[0], w-1), min(self.alert_pt[1], h-1)]
        self.canvas.config(width=w, height=h)
        self.root.update_idletasks()
        self._draw(sync_fields=True)
        self._status(f"Canvas resized to {w}×{h}")

    def _reset(self):
        self._load_config()
        # Sync resolution fields too
        self.w_var.set(str(self.vid_w))
        self.h_var.set(str(self.vid_h))
        self.canvas.config(width=self.vid_w, height=self.vid_h)
        self.root.update_idletasks()
        self._draw(sync_fields=True)
        self._status("Reset to saved values.")

    def _status(self, msg: str, ok: bool = False):
        self.status_var.set(("✓  " if ok else "   ") + msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Virtual line configurator")
    parser.add_argument("--config", default="line_config.json",
                        help="Path to line_config.json (default: ./line_config.json)")
    args = parser.parse_args()

    root = tk.Tk()
    LineConfigUI(root, config_path=args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
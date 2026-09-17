#!/usr/bin/env python3
"""
STM32MP2 VPU Hardware Video Encoder / Decoder
==============================================
Uses hantro_vpu hardware (VDEC /dev/video0 + VENC /dev/video1) via GStreamer.

Root cause of common failures
──────────────────────────────
• decodebin + encodebin  →  caps-negotiation error: VDEC outputs tiled NV12
  (NV12_16L32S) which encodebin rejects.  Fix: use explicit hw elements and
  insert videoconvert between them.

Usage
─────
  python3 vpu_tool.py encode    -i input.mp4   --codec vp8
  python3 vpu_tool.py encode    -i raw.yuv     --codec h264 --width 1920 --height 1080
  python3 vpu_tool.py decode    -i output.mp4  --display
  python3 vpu_tool.py decode    -i output.mp4  -o raw.yuv
  python3 vpu_tool.py transcode -i h264.mp4    --codec vp8 --bitrate 2000000
  python3 vpu_tool.py info      -i myfile.mp4
"""

import subprocess
import argparse
import os
import sys
import time
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VDEC_DEVICE = "/dev/video0"
VENC_DEVICE = "/dev/video1"

# Known container → explicit hw-decode chain
# videoconvert is added AFTER these in all encode/transcode pipelines
# to convert tiled NV12 → linear NV12 that VENC accepts.
DECODE_CHAINS = {
    ".mp4":   "qtdemux ! queue ! h264parse ! v4l2slh264dec",
    ".h264":  "h264parse ! v4l2slh264dec",
    ".264":   "h264parse ! v4l2slh264dec",
    ".webm":  "matroskademux ! queue ! v4l2slvp8dec",
    ".mkv":   "matroskademux ! queue ! v4l2slvp8dec",
    ".vp8":   "v4l2slvp8dec",
    ".jpg":   "jpegparse ! v4l2jpegdec",
    ".jpeg":  "jpegparse ! v4l2jpegdec",
    ".mjpeg": "jpegparse ! v4l2jpegdec",
}

# Direct hardware encoder elements (avoids encodebin negotiation)
HW_ENCODER = {
    "h264": "v4l2slh264enc",
    "vp8":  "v4l2slvp8enc",
    "jpeg": "v4l2jpegenc",
}

# Post-encoder mux + sink wrapper
ENCODE_TAIL = {
    "h264": "h264parse ! qtmux ! filesink location=\"{out}\"",
    "vp8":  "matroskamux ! filesink location=\"{out}\"",
    "jpeg": "filesink location=\"{out}\"",
}

OUTPUT_EXT = {"h264": ".mp4", "vp8": ".webm", "jpeg": ".jpg"}

RAW_EXTENSIONS = {".yuv", ".nv12", ".raw", ".i420"}

C = {
    "ok":    "\033[92m",
    "warn":  "\033[93m",
    "err":   "\033[91m",
    "info":  "\033[96m",
    "reset": "\033[0m",
    "bold":  "\033[1m",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cp(msg, kind="info"):
    print(f"{C.get(kind,'')}{msg}{C['reset']}")

def banner(title):
    w = 62
    print()
    print(C["bold"] + "=" * w + C["reset"])
    print(C["bold"] + f"  {title}" + C["reset"])
    print(C["bold"] + "=" * w + C["reset"])

def check_env():
    ok = True
    for t in ["gst-launch-1.0"]:
        if not shutil.which(t):
            cp(f"[MISSING] {t}", "err"); ok = False
    for dev, lbl in [(VDEC_DEVICE, "VDEC"), (VENC_DEVICE, "VENC")]:
        if os.path.exists(dev):
            cp(f"[OK] {lbl} → {dev}", "ok")
        else:
            cp(f"[WARN] {lbl} device {dev} not found", "warn")
    if not ok:
        sys.exit(1)

def irq_count(kw):
    try:
        for ln in Path("/proc/interrupts").read_text().splitlines():
            if kw in ln:
                return sum(int(p) for p in ln.split() if p.isdigit())
    except Exception:
        pass
    return 0

def hw_verify(kw, before, label):
    delta = irq_count(kw) - before
    if delta > 0:
        cp(f"[HW ✓] {label}: {delta} frame(s) processed by hardware", "ok")
    else:
        cp(f"[HW ?] No {label} interrupts detected", "warn")

def fsize(p):
    if not os.path.exists(p): return "?"
    s = os.path.getsize(p)
    return f"{s/1_048_576:.2f} MB" if s > 1_048_576 else f"{s/1024:.1f} KB"

def detect(p):
    """Returns ('raw'|'encoded'|'unknown', codec_hint_or_None)"""
    ext = Path(p).suffix.lower()
    if ext in RAW_EXTENSIONS:
        return "raw", "raw"
    hint = None
    if ext in (".mp4", ".h264", ".264"): hint = "h264"
    elif ext in (".webm", ".mkv", ".vp8"): hint = "vp8"
    elif ext in (".jpg", ".jpeg", ".mjpeg"): hint = "jpeg"
    kind = "encoded" if (ext in DECODE_CHAINS or hint) else "unknown"
    return kind, hint

def auto_name(inp, tag, ext):
    return f"{Path(inp).stem}_{tag}{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run(pipeline: str, title: str) -> bool:
    banner(title)
    cp("[Pipeline]", "info")
    pretty = pipeline.replace(" ! ", "\n    ! ")
    print(f"  gst-launch-1.0 -e \\\n    {pretty}\n")

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            f"gst-launch-1.0 -e {pipeline}",
            shell=True, executable="/bin/bash",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line: continue
            if "ERROR" in line or "error" in line: cp(f"  {line}", "err")
            elif "WARNING" in line: cp(f"  {line}", "warn")
            elif any(k in line for k in ["EOS", "Reached", "Execution ended"]): cp(f"  {line}", "ok")
            else: print(f"  {line}")
        proc.wait()
        elapsed = time.time() - t0
        if proc.returncode == 0:
            cp(f"\n[DONE] Finished in {elapsed:.2f}s", "ok")
            return True
        cp(f"\n[FAIL] Pipeline exited with code {proc.returncode}", "err")
        return False
    except KeyboardInterrupt:
        proc.terminate(); proc.wait()
        cp("\n[STOPPED] Interrupted", "warn")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Build encoder element string  (direct hw element, NOT encodebin)
# ─────────────────────────────────────────────────────────────────────────────

def encoder_element(codec, bitrate, quality, keyframe_interval, rotation):
    """
    Returns e.g.  v4l2slh264enc rate-control=1 bitrate=4000000 keyframe-interval=30
    Using direct elements avoids the encodebin caps-negotiation bug.
    """
    base = HW_ENCODER[codec]
    props = []

    if codec == "h264":
        if bitrate:
            props += [f"rate-control=1", f"bitrate={bitrate}"]
        if quality is not None:
            # lower = better quality
            q = max(0, min(51, quality))
            props += [f"qp-min={q}", f"qp-max={q}", f"quantizer={q}"]
        if keyframe_interval is not None:
            props.append(f"keyframe-interval={keyframe_interval}")

    elif codec == "vp8":
        if bitrate:
            props.append(f"bitrate={bitrate}")
        if quality is not None:
            q = max(0, min(63, quality))
            props += [f"min-quality={q}", f"max-quality={q}"]
        if keyframe_interval is not None:
            props.append(f"keyframe-interval={keyframe_interval}")

    if rotation:
        props.append(f"rotation={rotation}")

    return (base + " " + " ".join(props)).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def cmd_encode(args):
    """
    Encode any input file to H.264 / VP8 / JPEG using VENC hardware.

    Pipeline topology
    ─────────────────
    Raw input:     filesrc → rawparse → [videoconvert →] hw-encoder → mux → filesink
    Encoded input: filesrc → hw-demux → hw-decoder → videoconvert → hw-encoder → mux → filesink
                             (VDEC)                  (NV12 tiled→linear)   (VENC)
    """
    inp   = args.input
    codec = args.codec.lower()
    out   = args.output or auto_name(inp, f"encoded_{codec}", OUTPUT_EXT[codec])

    ftype, fhint = detect(inp)
    cp(f"[Input]  {inp}  ({ftype}, {fhint})", "info")
    cp(f"[Output] {out}  ({codec.upper()})", "info")

    enc_el  = encoder_element(codec, args.bitrate, args.quality,
                               args.keyframe_interval, args.rotation)
    mux_out = ENCODE_TAIL[codec].format(out=out)
    venc_b  = irq_count("venc")
    vdec_b  = irq_count("vdec")

    # ── Source / decode stage ─────────────────────────────────────────────
    if ftype == "raw":
        if not (args.width and args.height):
            cp("[ERROR] --width and --height required for raw input", "err")
            return False
        fmt = args.raw_format or "NV12"
        fps = args.framerate or 30
        # Raw input feeds directly into encoder (no VDEC used)
        pipeline = (
            f'filesrc location="{inp}" ! '
            f'video/x-raw,format={fmt},width={args.width},'
            f'height={args.height},framerate={fps}/1 ! '
            f'{enc_el} ! {mux_out}'
        )
        using_vdec = False

    elif ftype in ("encoded", "unknown"):
        ext = Path(inp).suffix.lower()
        decode_chain = DECODE_CHAINS.get(ext)

        if decode_chain:
            # Explicit hw-decode chain + videoconvert to fix tiled-NV12 issue
            pipeline = (
                f'filesrc location="{inp}" ! '
                f'{decode_chain} ! '
                f'videoconvert ! video/x-raw,format=NV12 ! queue ! '
                f'{enc_el} ! {mux_out}'
            )
        else:
            # Unknown container: let decodebin handle it, then videoconvert
            cp(f"[INFO] Unknown extension '{ext}', using decodebin fallback", "warn")
            pipeline = (
                f'filesrc location="{inp}" ! decodebin ! '
                f'videoconvert ! video/x-raw,format=NV12 ! queue ! '
                f'{enc_el} ! {mux_out}'
            )
        using_vdec = True
    else:
        cp(f"[ERROR] Cannot read: {inp}", "err")
        return False

    ok = run(pipeline, f"ENCODE  {Path(inp).name}  →  {Path(out).name}")

    if ok:
        hw_verify("venc", venc_b, "VENC (encoder)")
        if using_vdec:
            hw_verify("vdec", vdec_b, "VDEC (input decode)")
        if os.path.exists(out):
            cp(f"[Output] {out}  ({fsize(out)})", "ok")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# DECODE
# ─────────────────────────────────────────────────────────────────────────────

def cmd_decode(args):
    """
    Decode an encoded video to raw NV12 file OR display on Wayland.

    Pipeline topology
    ─────────────────
    filesrc → hw-demux → hw-decoder → [videoconvert →] filesink / waylandsink
                         (VDEC)
    """
    inp     = args.input
    display = args.display
    ext     = Path(inp).suffix.lower()
    ftype, fhint = detect(inp)

    if ftype == "raw":
        cp("[ERROR] Input appears to be raw video — nothing to decode.", "err")
        return False

    decode_chain = DECODE_CHAINS.get(ext, "decodebin")
    vdec_b = irq_count("vdec")

    if display:
        # videoconvert ensures waylandsink gets a format it supports
        pipeline = (
            f'filesrc location="{inp}" ! '
            f'{decode_chain} ! '
            f'videoconvert ! waylandsink fullscreen=true sync=false'
        )
        label = "Wayland display"
        ok = run(pipeline, f"DECODE + DISPLAY  {Path(inp).name}")
    else:
        out = args.output or auto_name(inp, "decoded", ".yuv")
        pipeline = (
            f'filesrc location="{inp}" ! '
            f'{decode_chain} ! '
            f'videoconvert ! video/x-raw,format=NV12 ! '
            f'filesink location="{out}"'
        )
        ok = run(pipeline, f"DECODE  {Path(inp).name}  →  {out}")
        label = out

    if ok:
        hw_verify("vdec", vdec_b, "VDEC (decoder)")
        if not display and os.path.exists(out):
            cp(f"[Output] {out}  ({fsize(out)})  [raw NV12]", "ok")
            cp(f"  Re-play: python3 vpu_tool.py decode -i {out} ... (needs width/height for raw)", "info")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCODE  (VDEC → VENC in one pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def cmd_transcode(args):
    """
    Transcode: VDEC hardware-decodes, VENC hardware-encodes, simultaneously.

    Pipeline topology
    ─────────────────
    filesrc → hw-demux → hw-decoder → videoconvert → hw-encoder → mux → filesink
                         (VDEC)     (tiled→NV12)       (VENC)
    """
    inp   = args.input
    codec = args.codec.lower()
    out   = args.output or auto_name(inp, f"transcoded_{codec}", OUTPUT_EXT[codec])
    ext   = Path(inp).suffix.lower()

    ftype, src_codec = detect(inp)
    if ftype == "raw":
        cp("[ERROR] Input is raw — use 'encode' instead of 'transcode'.", "err")
        return False

    cp(f"[Input]  {inp}  (src: {src_codec})", "info")
    cp(f"[Output] {out}  (dst: {codec.upper()})", "info")
    cp("[Mode]   VDEC ──► videoconvert ──► VENC  (single pipeline)", "info")

    decode_chain = DECODE_CHAINS.get(ext, "decodebin")
    enc_el  = encoder_element(codec, args.bitrate, args.quality,
                               args.keyframe_interval, args.rotation)
    mux_out = ENCODE_TAIL[codec].format(out=out)

    pipeline = (
        f'filesrc location="{inp}" ! '
        f'{decode_chain} ! '
        f'videoconvert ! video/x-raw,format=NV12 ! queue ! '
        f'{enc_el} ! {mux_out}'
    )

    vdec_b = irq_count("vdec")
    venc_b = irq_count("venc")

    ok = run(pipeline, f"TRANSCODE  {Path(inp).name}  →  {Path(out).name}")

    if ok:
        hw_verify("vdec", vdec_b, "VDEC (decode stage)")
        hw_verify("venc", venc_b, "VENC (encode stage)")
        if os.path.exists(out):
            cp(f"[Output] {out}  ({fsize(out)})", "ok")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# INFO
# ─────────────────────────────────────────────────────────────────────────────

def cmd_info(args):
    if args.input:
        banner(f"File: {args.input}")
        if not os.path.exists(args.input):
            cp(f"[ERROR] Not found: {args.input}", "err")
        else:
            cp(f"  Size  : {fsize(args.input)}", "info")
            cp(f"  Detect: {detect(args.input)}", "info")
            if shutil.which("gst-discoverer-1.0"):
                print()
                subprocess.run(["gst-discoverer-1.0", args.input], check=False)

    banner("VPU Hardware Capabilities")
    for dev, lbl in [(VDEC_DEVICE, "VDEC – Decoder"), (VENC_DEVICE, "VENC – Encoder")]:
        if not os.path.exists(dev):
            cp(f"\n  [{lbl}]  NOT FOUND", "warn"); continue
        cp(f"\n  [{lbl}]  {dev}", "ok")
        for flag, desc in [("--list-formats-out-ext", "Compressed INPUT formats"),
                           ("--list-formats-ext",     "Compressed OUTPUT formats")]:
            r = subprocess.run(["v4l2-ctl", "-d", dev, flag], capture_output=True, text=True)
            if r.stdout.strip():
                cp(f"    {desc}:", "info")
                for ln in r.stdout.splitlines():
                    if ln.strip(): print(f"      {ln}")

    banner("HW Interrupt Counters (frames processed since boot)")
    for kw, lbl in [("vdec","VDEC"), ("venc","VENC")]:
        cp(f"  {lbl} : {irq_count(kw)} frame(s)", "info")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="vpu_tool.py",
        description="STM32MP2 VPU Hardware Encoder / Decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────────────────────────────────────────────────────────────
  # Encode MP4 → VP8 WebM  (VDEC decodes, VENC encodes)
  python3 vpu_tool.py encode -i street.mp4 --codec vp8

  # Encode MP4 → H.264 at 4 Mbps CBR
  python3 vpu_tool.py encode -i street.mp4 --codec h264 --bitrate 4000000

  # Encode raw NV12 → H.264
  python3 vpu_tool.py encode -i raw.yuv --codec h264 --width 1920 --height 1080

  # Encode with best quality (qp=0 for H.264, quality=0 for VP8)
  python3 vpu_tool.py encode -i street.mp4 --codec h264 --quality 0

  # Decode to raw NV12 file
  python3 vpu_tool.py decode -i street.mp4

  # Decode and display on Wayland
  python3 vpu_tool.py decode -i street.mp4 --display

  # Transcode H.264 → VP8 using both chips simultaneously
  python3 vpu_tool.py transcode -i street.mp4 --codec vp8 --bitrate 2000000

  # Inspect file and hardware
  python3 vpu_tool.py info -i street.mp4
        """
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    def add_io(sp, ohelp="Output file (auto-named if omitted)"):
        sp.add_argument("-i","--input",  required=True, metavar="FILE")
        sp.add_argument("-o","--output", metavar="FILE", help=ohelp)

    def add_enc_opts(sp):
        sp.add_argument("-c","--codec", choices=["h264","vp8","jpeg"], default="h264",
                        help="Target codec (default: h264)")
        sp.add_argument("--bitrate", type=int, metavar="BPS",
                        help="Bitrate bps → enables CBR  e.g. 4000000")
        sp.add_argument("--quality", type=int, metavar="Q",
                        help="Quality: H.264 qp 0–51 (0=best), VP8 0–63 (0=best)")
        sp.add_argument("--keyframe-interval", type=int, default=30, metavar="N",
                        help="IDR/keyframe every N frames (default: 30)")
        sp.add_argument("--rotation", type=int, choices=[0,90,180,270], default=0,
                        help="Rotate frame before encoding (default: 0)")

    # encode
    enc = sub.add_parser("encode", help="Encode a file using VENC hardware")
    add_io(enc); add_enc_opts(enc)
    enc.add_argument("--width",      type=int, help="Frame width  (raw input only)")
    enc.add_argument("--height",     type=int, help="Frame height (raw input only)")
    enc.add_argument("--framerate",  type=int, default=30, help="FPS for raw input")
    enc.add_argument("--raw-format", default="NV12",
                     choices=["NV12","I420","YUY2","UYVY"],
                     help="Pixel format for raw input (default: NV12)")

    # decode
    dec = sub.add_parser("decode", help="Decode a file using VDEC hardware")
    add_io(dec, "Output raw NV12 file (auto-named if omitted)")
    dec.add_argument("--display", action="store_true",
                     help="Display via waylandsink instead of saving to file")

    # transcode
    xc = sub.add_parser("transcode", help="Decode + re-encode with VDEC + VENC")
    add_io(xc); add_enc_opts(xc)

    # info
    inf = sub.add_parser("info", help="Show file metadata and hardware info")
    inf.add_argument("-i","--input", metavar="FILE", help="File to analyse (optional)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    banner("STM32MP2 VPU Tool")
    check_env()

    if hasattr(args, "input") and args.input and args.command != "info":
        if not os.path.exists(args.input):
            cp(f"[ERROR] Input not found: {args.input}", "err")
            sys.exit(1)
        cp(f"[Input]  {args.input}  ({fsize(args.input)})", "info")

    ok = {"encode": cmd_encode, "decode": cmd_decode,
          "transcode": cmd_transcode, "info": cmd_info}[args.command](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

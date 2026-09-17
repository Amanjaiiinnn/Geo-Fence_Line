# Geo-Fence_Line — Detection and Recording (WiFi Camera)

This app detects people and vehicles in a camera or video stream and colours each box red when it's on the alert side of a virtual line. It records the annotated video, optionally encrypted with AES-256-GCM.

It runs in two places:

- **STM32MP2 board**: a CSI camera through GStreamer (`libcamerasrc`), `.nb` models on the NPU through `stai_mpu`, and a borderless, maximised display window through GTK.
- **Windows PC**: a video file, webcam or HTTP stream through OpenCV, with `.tflite` models on the CPU. There's no preview window on Windows (see [Known issues](#known-issues)); results go to the console and the recording.

## Contents

- [What it does](#what-it-does)
- [Quick start on Windows](#quick-start-on-windows)
- [Running on the STM32MP2 board](#running-on-the-stm32mp2-board)
- [Command-line options](#command-line-options)
- [Settings](#settings)
- [The virtual line](#the-virtual-line)
- [Recording and encryption](#recording-and-encryption)
- [Models](#models)
- [Other scripts and folders](#other-scripts-and-folders)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

---

## What it does

- **Detects** people and vehicles with a YOLOv8n or YOLOv9t INT8 model (320×320 input), running on every third frame. The latest results are drawn on every frame.
  - **Person**: COCO class 0
  - **Vehicle**: COCO classes 2 (car), 3 (motorcycle), 5 (bus) and 7 (truck)
  - Confidence threshold 0.35, NMS IoU 0.45
- **Checks each box against the virtual line** in `line_config.json`. A box is red when it overlaps the line, when its centre is on the alert side, or when its centre is within `threshold_pixels` of the line. Other boxes are green (Person) or light blue (Vehicle).
- **Prints to the console** when an object changes state:
  ```
  [ALERT] Person entered alert zone.
  [CLEAR] Person left alert zone.
  [ALERT] New Person detected in alert zone.
  ```
  An object is matched to the previous frame when the same class moved less than 60 px.
- **Records** from the moment it starts until it stops: annotated frames (boxes, line, a REC marker) at 15 fps to `recordings\rec_<YYYYMMDD_HHMMSS>.avi`, or to an encrypted `.enc` file when encryption is on.

---

## Quick start on Windows

### Requirements

- Python 3.12
- `numpy`, `opencv-python`, `tensorflow`, `cryptography` (needed even with encryption off)
- A video file, webcam or HTTP video stream

Known-good setup (the models load and detection runs on the CPU): Windows 11, Python 3.12.4, TensorFlow 2.19.0, OpenCV 4.13.0, NumPy 2.1.3.

```powershell
pip install numpy opencv-python tensorflow cryptography
```

### Run

```powershell
cd path\to\wifi-camera
python main.py --source Video --video_path path\to\video.mp4 --model_path models\yolov8n_full_integer_quant.tflite
```

- **Always pass `--video_path` and `--model_path`.** The defaults in `config.py` point to files on the original PC.
- Start from the project folder, because `line_config.json` and `recordings\` are relative paths.
- Other sources:
  - Webcam: `--video_path 0`
  - HTTP stream, such as DroidCam: `--video_path http://<phone-ip>:4747/video`
- A video file plays at its own frame rate, and the app stops at the end of it. A webcam or stream runs until you press **Ctrl+C**.
- **No preview window opens on Windows.** Watch the console, then open the recording in `recordings\` afterwards.
- `--source Camera` doesn't work on Windows; it needs GStreamer, GTK and cairo.

### Place the line

```powershell
python line_config_ui.py
```

- Drag **P1** and **P2** to place the line, and **ALERT** to the side that should trigger alerts. You can also type exact coordinates.
- Set the frame resolution to match your video, set the **Threshold** in pixels, then click **Save** to write `line_config.json`. **Reset** reloads the last saved file.
- The detector re-reads the file when it changes, even while the app is running.
- The editor shows an empty grid, not your video, so work out the coordinates from a frame of the video.
- `--config path\to\line_config.json` edits a different file.

---

## Running on the STM32MP2 board

### Board requirements

- An STM32MP2 board running OpenSTLinux with ST's X-LINUX-AI (the `stai_mpu` Python API)
- A CSI camera on the board's camera interface, with libcamera
- GStreamer plugins: `libcamerasrc`, `cairooverlay`, `gtkwaylandsink`, `fpsdisplaysink`, `videorate`, `videoconvert`, `queue`, `appsink`
- Python packages: PyGObject (`gi`, with GStreamer 1.0 and GTK 3), `pycairo`, `numpy`, OpenCV (`cv2`), `cryptography`

### Run

```bash
cd /path/to/wifi-camera

# CSI camera, YOLOv8n on the NPU
python3 main.py --source Camera --model_path models/yolov8n_full_integer_quant.nb

# CSI camera, YOLOv9t on the NPU
python3 main.py --source Camera --model_path models/yolov9t_full_integer_quant.nb
```

- The camera runs at 640×480 and 30 fps, in a borderless, maximised window. **Esc** quits and closes the recording properly.
- Without a display, it runs headless; stop it with **Ctrl+C**.
- Before starting, it searches `dmesg` for a camera sensor probe failure (`failed to read chip id` or `Error reading reg`) and refuses to start if one is found. On this board a missing sensor doesn't raise a GStreamer error, so without this check the app would record blank frames.
- The main process is pinned to CPU core 0 and the inference thread to core 1.
- `--source Video` also works on the board, with a `.nb` or `.tflite` model.

---

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--source` | `Video` | `Camera`: the board's CSI camera. `Video`: a file, webcam index or `http(s)://` stream through OpenCV |
| `--video_path` | `…\Desktop\wifi-camera\videos\entry3.mp4` | File, webcam index (`0`) or stream URL, for `--source Video` |
| `--model_path` | `…\Downloads\best_full_integer_quant (1).tflite` | `.nb` (NPU) or `.tflite` (CPU) model |

Both default paths are on the original PC and don't exist in this folder.

---

## Settings

Everything else is set in `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `DETECTION_ON`, `VIDEO_PATH`, `MODEL_PATH` | see above | Defaults for the command-line options |
| `FRAME_WIDTH`, `FRAME_HEIGHT`, `FRAMERATE` | 640, 480, 30 | Board camera format. Video files keep their own size |
| `CONFIDENCE_TH`, `IOU_TH` | 0.35, 0.45 | Detection and NMS thresholds |
| `INFER_EVERY_N_FRAMES` | 3 | Run the model on every Nth frame |
| `RECORDING` | `True` | Record every run |
| `RECORDING_SAVE_DIR` | `recordings` | Where recordings go |
| `RECORDING_FPS` | 15 | Frame rate written to the recording |
| `RECORDING_QUALITY` | 75 | MJPEG quality of the `.avi` |
| `ENCRYPTION` | `False` | `True`: write an encrypted `.enc` instead of an `.avi` |
| `ENCRYPTION_KEY_PATH` | `master.key` | Key file, relative to the folder you start in |
| `CLASS_COLORS`, `CLASS_COLORS_BGR` | — | Box colours for the board display and the OpenCV overlay |

---

## The virtual line

`line_config.json`:

| Key | Meaning |
|---|---|
| `p1`, `p2` | Line end points, in pixels of the processed frame |
| `alert_side_pt` | Any point on the side that should trigger alerts |
| `threshold_pixels` | A box whose centre is this close to the line also counts as on the alert side |
| `frame_width`, `frame_height` | Saved by the line editor; not used by the detector |

- Points outside the frame are clamped to its edges.
- If the reference point lies exactly on the line, the side with a positive distance is used.

---

## Recording and encryption

### Plain recordings

- One file per run: `recordings\rec_<YYYYMMDD_HHMMSS>.avi`, MJPEG at 15 fps.
- Frames are written on a separate thread. If the writer falls behind, the oldest waiting frames are dropped.

### Encrypted recordings

Set `ENCRYPTION = True` in `config.py`. Each run then writes only `recordings\rec_<YYYYMMDD_HHMMSS>.enc`:

- Every frame is JPEG-encoded (quality 75) and encrypted separately with AES-256-GCM.
- File format `ENCV2`: a 6-byte header (`ENCV2` and a version byte), then for each frame its length, a 12-byte nonce, an 8-byte frame number and the ciphertext with its tag. The frame number is authenticated too.
- The key is 32 random bytes in `master.key`, in the folder you start the app from. It's created automatically the first time if it doesn't exist.
- **Keep `master.key` safe and private.** Without it, recordings can't be decrypted.
- A `VideoWriter failed` warning appears in this mode; it's harmless, and the `.enc` file is still written.

### Decrypt a recording

```powershell
python decryption.py --enc_file recordings\rec_<date>_<time>.enc --key_path master.key
```

- The output is `<name>_decrypted.avi` next to the input, or the path given with `--output`.
- It's written at 15 fps.

### Other helpers in `encryption.py`

- Encrypt an existing `.avi`:
  ```powershell
  python -c "from encryption import encrypt_file; encrypt_file(r'recordings\rec.avi', r'recordings\rec.enc')"
  ```
- `StreamDecryptor(path).play_with_ffplay()` plays an `.enc` file without writing a decrypted copy. It needs `ffplay` (FFmpeg) on the PATH.

---

## Models

| File | Classes | Input → output | Works with `detection.py` |
|---|---|---|---|
| `models/yolov8n_full_integer_quant.tflite` / `.nb` | 80 COCO classes | 320×320×3 INT8 → 1×84×2100 INT8 | Yes |
| `models/yolov9t_full_integer_quant.tflite` / `.nb` | 80 COCO classes | 320×320×3 INT8 → 1×84×2100 INT8 | Yes |
| `models/optim_yolov8n_full_integer_quant.tflite` | 80 COCO classes | 320×320×3 INT8 → 1×84×2100 INT8 | Yes |
| `models/yolov8n_float16.tflite`, `yolov8n_float32.tflite` | 80 COCO classes | 320×320×3 float → 1×84×2100 float | No: fails with `ZeroDivisionError` (used by `trialonv8nfloat32.py` and `for-test\`) |
| `models/best_full_integer_quant19062026.tflite`, `best_full_integer_quant (1).tflite` | 2 classes | 320×320×3 INT8 → 1×6×2100 INT8 | No: runs, but the output is read as if it had 84 channels, so the boxes are meaningless |

- The `.tflite` shapes above were read from the files, and each file was run through `detection.py` on Windows.
- The `.nb` files are the NPU builds. For them, quantisation parameters come from `stai_mpu`; if that fails, a built-in table is used for file names containing `yolov8n` or `yolov9t`.
- `best_full_integer_quant (1).tflite` is identical to `wificamera.tflite`, the Person/Train detector in the metro thread project.

---

## Other scripts and folders

| Path | Runs on | What it is |
|---|---|---|
| `line_config_ui.py` | Windows | Tkinter editor for `line_config.json` |
| `test.py` | — | An identical copy of `line_config_ui.py` |
| `decryption.py` | Windows, board | Turns an `.enc` recording back into an `.avi` |
| `encryption.py` | — | Encryptor and decryptor module used by the recorder |
| `get_model_detail.py` | Board | `python3 get_model_detail.py <model>` prints a model's tensor details; `.tflite` needs `tflite_runtime` |
| `vpu_tool.py` | Board | STM32MP2 hardware video encode, decode and transcode through GStreamer (`encode`, `decode`, `transcode`, `info`) |
| `usb-yolov8n-detection.py` | Board | Older standalone detector for a USB camera (`v4l2src`) with a GTK display; options are in its docstring |
| `trialonv8nfloat32.py` | Windows | Float32 YOLOv8n test on a video; paths are hard-coded at the top |
| `for-test\` | Windows | Older detector test using the float16 model |
| `recordings\` | — | Saved recordings (about 1.4 GB in this copy) |
| `keys\master.key` | — | An AES key file. The app doesn't use it by default (see Known issues) |
| `build\`, `dist\` | — | PyInstaller output for `ftp_file_manager+.py`, whose source isn't in this folder |

`main.py`, `config.py` and `video_widget.py` each start with a commented-out copy of an older version, followed by the code that actually runs.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Video file not found`, or a model file error at start | Pass `--video_path` and `--model_path`; the defaults don't exist on this PC. |
| `Camera source requires GStreamer, GTK, and cairo libraries…` | On Windows, use `--source Video`. |
| `No module named 'cryptography'` | `pip install cryptography`. |
| `ZeroDivisionError` at start | You picked a float model; use an INT8 model. |
| Boxes all over the frame, or nonsense detections | You picked a 2-class model; use an 80-class COCO model. |
| No window on Windows | Expected; check the console and the recording. |
| Slow on Windows | The whole app runs on one CPU core; see Known issues. |
| `[CPU] Pin to core 1 failed` on Windows | A side effect of the same issue; harmless. |
| `CSI camera sensor '<name>' not detected` on the board | Reseat the camera and **reboot**. The old error stays in `dmesg` until a reboot. |
| `libcamerasrc plugin not found` | Install libcamera's GStreamer plugin on the board. |
| A recording won't open after Ctrl+C | The file wasn't closed properly; try a player that can handle incomplete AVI files. |
| The line is in the wrong place | Line points are pixels of the processed frame; set the resolution in the line editor to your video's size. |
| An `.enc` file won't decrypt | It was encrypted with a different key. Use the `master.key` from the folder the app was started in. |

## Known issues

1. **No preview window on Windows.** Display detection depends on GTK being installed, so the video mode runs headless on Windows.
2. **The whole app runs on one CPU core on Windows.** The process is pinned to core 0, so the inference thread can't move to core 1, and TFLite's four threads share that one core.
3. **The default `--model_path` and `--video_path` point to files on the original PC.** The default model is also a 2-class model that `detection.py` can't read correctly.
4. **Float models crash** with `ZeroDivisionError`.
5. **2-class models produce meaningless boxes**, because `detection.py` expects 80-class COCO outputs.
6. **The app uses `master.key` in the start folder, not `keys\master.key`**, and silently creates a new key if the file is missing. Recordings made with different keys can't be decrypted with each other's key.
7. **Ctrl+C doesn't close the recording.** It ends the process immediately; with no window on Windows, it's the only way to stop a webcam or stream.
8. **Recording is always on by default**, so every run writes a large file.
9. **`frame_width` and `frame_height` in `line_config.json` are ignored.** If your video's resolution differs from the one you drew the line for, the line lands in the wrong place.
10. **Encrypted mode prints a harmless `VideoWriter failed` warning.**
11. **`decryption.py` always writes at 15 fps**, whatever frame rate the recording used.
12. **The executable in `dist\` is probably unusable.** It's 281 KB, far smaller than the 679 MB package in `build\`, and its source file isn't included.
13. **`get_model_detail.py` can't inspect `.tflite` models on Windows**, because it needs `tflite_runtime`.

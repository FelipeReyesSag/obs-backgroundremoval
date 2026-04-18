# Plate Blur for OBS

An [OBS Studio](https://obsproject.com/) filter that automatically detects and
blurs license plates in your video feed in real time. Designed for IRL / driving
streamers who want an additional layer of privacy on top of manually hiding
plates.

> **Not a guarantee.** This plugin is an additional layer of protection, not a
> sole safeguard. Computer vision models miss things — especially plates that
> are small, heavily angled, motion-blurred, or partially occluded. Always test
> your setup, keep using your existing privacy habits, and don't rely on this
> plugin to save you from a mistake.

## How it works

- Add the **License Plate Blur** filter to any video source (webcam, game capture,
  screen capture, ...).
- The filter introduces a configurable delay (default **3 s**, adjustable 1–5 s).
  This is intentional: it gives the AI plenty of time to run on **every single
  frame** at full 1080p with no skipping. 3 s is imperceptible to streaming
  viewers and normal for livestreaming.
- A YOLOv9-t model from [fast-alpr](https://github.com/ankandrew/fast-alpr)
  runs on every buffered frame and outputs plate bounding boxes.
- A GPU shader Gaussian-blurs each detected region (with a safety padding)
  before the delayed frame is emitted.

The plugin **never reads plate text**. It only locates plates and blurs them.

## Install

> Binary installers will be published on the GitHub releases page once v0.1.0
> CI has produced them. Until then, build from source (below).

Pre-built packages, when available:

- **Windows** — `.exe` installer
- **macOS** — `.pkg` (universal)
- **Linux (Ubuntu)** — `.deb`

Pre-built installers bundle the YOLO model, so there is no separate model
download step after install.

## Build from source

Requires CMake 3.28+, a C++20 compiler, and the OBS plugin deps (pulled
automatically by the preset). Two extra things are needed that the preset
does **not** fetch for you:

1. A prebuilt [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases)
   matching the pin in `.github/workflows/plugin.yml` (currently `v1.23.2`).
   Download the platform-appropriate archive, extract it, and point
   `-DONNXRUNTIME_ROOT=...` (or `$ENV:ONNXRUNTIME_ROOT`) at the extracted
   directory. When unset, CMake looks at `.deps_vendor/onnxruntime` by default.
2. The YOLOv9-t ONNX weights, which are fetched by:

   ```bash
   # macOS / Linux
   scripts/fetch_model.sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy Bypass -File scripts/fetch_model.ps1
   ```

   Run this once from the repo root before `cmake --preset ...`; the weights
   are picked up at configure time and copied into the plugin bundle.

Then:

```bash
# macOS (universal)
cmake --preset macos
cmake --build --preset macos

# Windows (Developer PowerShell)
cmake --preset windows-x64
cmake --build --preset windows-x64 --config RelWithDebInfo

# Linux (libobs-dev / libopencv-dev / libcurl-dev from apt)
cmake --preset ubuntu-x86_64
cmake --build --preset ubuntu-x86_64
```

## Settings

| Setting | Range | Default | Notes |
|---|---|---|---|
| **Pipeline delay (ms)** | 1000–5000 | 3000 | Larger delay = more time for inference. Audio on the same source is *not* delayed — add a matching audio "Render Delay" filter if lip-sync matters. |
| **Detection confidence** | 0.1–0.9 | 0.25 | Lower = more blurs (including false positives on non-plates). Higher = fewer blurs (risk of missing real plates). Leave low for privacy. |
| **Blur strength (pixels)** | 1–30 | 12 | Gaussian blur radius applied inside each detected region. |
| **Box padding** | 0.0–0.5 | 0.10 | Fraction of plate size to expand each blur region. Errs on the side of covering slightly more than detected. |
| **Inference device** | CPU / CUDA / DirectML / CoreML / ... | platform-specific | Only providers compiled into the bundled ONNX Runtime are listed. Falls back to CPU if requested GPU provider is unavailable at runtime. |
| **Show detection overlay** | toggle | off | Tints detection regions red (without blur) — useful for tuning confidence and padding. |

## Hardware recommendations

- **Discrete GPU**: ideal. Inference on a 384×384 input with YOLOv9-t is <5 ms
  on most modern GPUs, so 60 fps source with 3 s buffer uses ~1.3 GB VRAM for
  the frame ring.
- **Integrated GPU**: works, but consider shortening the delay to 1–2 s and
  using CPU inference with 2–4 threads.
- **CPU-only**: possible at 30 fps; expect 15–40 ms per frame depending on
  silicon.

## Known limitations (v0.1)

- Uses the generic fast-alpr model. Accuracy on US/CA plates is good but not
  perfect; international plates are hit-or-miss.
- Audio is not auto-delayed (add a matching OBS audio delay filter manually).
- No per-frame tracking/smoothing — each frame's detections are independent.
  This means occasional one-frame detection gaps may briefly unblur a plate.
  (Lower the confidence threshold to compensate.)
- Memory usage scales with delay × resolution × fps. A 5 s delay at 1080p60
  holds ~2.5 GB of frame data in RAM.

## Attribution

This plugin is a fork of
[royshil/obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval)
with its portrait-segmentation logic replaced by license plate detection. The
original plugin's cross-platform CMake, CI, and ONNX Runtime wiring are reused
directly. Thanks to Roy Shilkrot and Kaito Udagawa for that scaffolding.

The plate detector is [fast-alpr](https://github.com/ankandrew/fast-alpr)'s
pretrained YOLOv9-t end2end ONNX model.

---

> SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>  
> SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>  
> SPDX-FileCopyrightText: 2026 Felipe Reyes  
>
> SPDX-License-Identifier: GPL-3.0-or-later  

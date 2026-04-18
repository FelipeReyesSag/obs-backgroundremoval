/*
 * SPDX-FileCopyrightText: 2026 Felipe Reyes
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PLATE_BLUR_CONSTS_H
#define PLATE_BLUR_CONSTS_H

// The fast-alpr end2end YOLOv9-t plate detector. 384x384 input, NMS baked in.
const char *const MODEL_PLATE_YOLO = "models/yolo-v9-t-384-license-plate-end2end.onnx";

const char *const USEGPU_CPU = "cpu";
const char *const USEGPU_CUDA = "cuda";
const char *const USEGPU_ROCM = "rocm";
const char *const USEGPU_MIGRAPHX = "migraphx";
const char *const USEGPU_TENSORRT = "tensorrt";
const char *const USEGPU_COREML = "coreml";
const char *const USEGPU_DIRECTML = "directml";

const char *const EFFECT_PLATE_BLUR_PATH = "effects/plate_blur.effect";

// Ring buffer capacity. 300 covers 5 s at 60 fps, which is the largest
// configurable delay. Anything larger is refused by the UI.
#define PLATE_BLUR_RING_CAPACITY 300

// Default inference throttling: run inference up to this many times per second.
// The render thread produces frames at the display rate (usually 60 Hz), but
// most cameras only produce ~30 new frames a second, so running inference any
// faster than 60 Hz just wastes GPU cycles.
#define PLATE_BLUR_MAX_INFERENCE_HZ 120

const char *const PLUGIN_INFO_TEMPLATE =
	"<b>Plate Blur for OBS</b> (%1)<br>"
	"Adds a configurable delay and automatically blurs detected license plates.<br>"
	"<i>This is a privacy aid, not a guarantee. Always test your stream.</i>";

#endif /* PLATE_BLUR_CONSTS_H */

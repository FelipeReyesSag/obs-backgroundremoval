// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef OBS_UTILS_H
#define OBS_UTILS_H

#include <obs-module.h>
#include <opencv2/core/mat.hpp>

// Render the filter's target source into `texrender`, stage it to `stagesurface`,
// and copy the staged pixels into `outBGRA` (cloned; safe to use after unmap).
//
// On success sets width/height to the captured frame size and returns true.
// On failure the surfaces may be resized or recreated as a side effect.
//
// `source` is the filter source.
// `texrenderInOut` may be NULL on entry to let the helper create one; it is
//   created/reused and written back via the pointer.
// `stagesurfaceInOut` follows the same create-on-demand pattern and is resized
//   if the target source's resolution changes between calls.
bool captureFilterTargetBGRA(obs_source_t *source, gs_texrender_t **texrenderInOut,
			     gs_stagesurf_t **stagesurfaceInOut, cv::Mat &outBGRA, uint32_t &width, uint32_t &height);

#endif /* OBS_UTILS_H */

// SPDX-FileCopyrightText: 2021-2026 Roy Shilkrot <roy.shil@gmail.com>
// SPDX-FileCopyrightText: 2023-2026 Kaito Udagawa <umireon@kaito.tokyo>
// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "obs-utils.hpp"

#include <obs-module.h>

bool captureFilterTargetBGRA(obs_source_t *source, gs_texrender_t **texrenderInOut, gs_stagesurf_t **stagesurfaceInOut,
			     cv::Mat &outBGRA, uint32_t &width, uint32_t &height)
{
	if (!source || !obs_source_enabled(source)) {
		return false;
	}
	obs_source_t *target = obs_filter_get_target(source);
	if (!target) {
		return false;
	}

	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);
	if (width == 0 || height == 0) {
		return false;
	}

	if (!*texrenderInOut) {
		*texrenderInOut = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	}
	gs_texrender_t *texrender = *texrenderInOut;

	gs_texrender_reset(texrender);
	if (!gs_texrender_begin(texrender, width, height)) {
		return false;
	}

	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(texrender);

	// Resize the staging surface if the target source dimensions changed.
	if (*stagesurfaceInOut) {
		const uint32_t sw = gs_stagesurface_get_width(*stagesurfaceInOut);
		const uint32_t sh = gs_stagesurface_get_height(*stagesurfaceInOut);
		if (sw != width || sh != height) {
			gs_stagesurface_destroy(*stagesurfaceInOut);
			*stagesurfaceInOut = nullptr;
		}
	}
	if (!*stagesurfaceInOut) {
		*stagesurfaceInOut = gs_stagesurface_create(width, height, GS_BGRA);
	}

	gs_stage_texture(*stagesurfaceInOut, gs_texrender_get_texture(texrender));

	uint8_t *video_data = nullptr;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(*stagesurfaceInOut, &video_data, &linesize)) {
		return false;
	}

	// Wrap the staged memory in a temporary cv::Mat and clone so the result
	// outlives the unmap call below.
	cv::Mat staged(static_cast<int>(height), static_cast<int>(width), CV_8UC4, video_data,
		       static_cast<size_t>(linesize));
	outBGRA = staged.clone();

	gs_stagesurface_unmap(*stagesurfaceInOut);
	return true;
}

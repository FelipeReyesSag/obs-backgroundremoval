/*
 * SPDX-FileCopyrightText: 2026 Felipe Reyes
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "plate-blur-filter.h"

struct obs_source_info plate_blur_filter_info = {
	.id = "plate_blur_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = plate_blur_filter_getname,
	.create = plate_blur_filter_create,
	.destroy = plate_blur_filter_destroy,
	.get_defaults = plate_blur_filter_defaults,
	.get_properties = plate_blur_filter_properties,
	.update = plate_blur_filter_update,
	.activate = plate_blur_filter_activate,
	.deactivate = plate_blur_filter_deactivate,
	.video_tick = plate_blur_filter_video_tick,
	.video_render = plate_blur_filter_video_render,
};

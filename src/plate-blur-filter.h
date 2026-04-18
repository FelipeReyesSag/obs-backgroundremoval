/*
 * SPDX-FileCopyrightText: 2026 Felipe Reyes
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PLATE_BLUR_FILTER_H
#define PLATE_BLUR_FILTER_H

#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *plate_blur_filter_getname(void *unused);
void *plate_blur_filter_create(obs_data_t *settings, obs_source_t *source);
void plate_blur_filter_destroy(void *data);
void plate_blur_filter_defaults(obs_data_t *settings);
obs_properties_t *plate_blur_filter_properties(void *data);
void plate_blur_filter_update(void *data, obs_data_t *settings);
void plate_blur_filter_activate(void *data);
void plate_blur_filter_deactivate(void *data);
void plate_blur_filter_video_tick(void *data, float seconds);
void plate_blur_filter_video_render(void *data, gs_effect_t *_effect);

#ifdef __cplusplus
}
#endif

#endif /* PLATE_BLUR_FILTER_H */

/*
 * SPDX-FileCopyrightText: 2026 Felipe Reyes
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <obs-module.h>

#include "plugin-support.h"
#include "update-checker/update-checker.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

extern struct obs_source_info plate_blur_filter_info;

bool obs_module_load(void)
{
	obs_register_source(&plate_blur_filter_info);
	obs_log(LOG_INFO, "Plate Blur for OBS loaded (version %s)", PLUGIN_VERSION);

	check_update();

	return true;
}

void obs_module_unload(void)
{
	obs_log(LOG_INFO, "Plate Blur for OBS unloaded");
}

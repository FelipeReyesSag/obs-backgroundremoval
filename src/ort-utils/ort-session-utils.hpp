// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ORT_SESSION_UTILS_H
#define ORT_SESSION_UTILS_H

#include <string>

#include "ORTModelData.hpp"

#define OBS_PLATE_BLUR_ORT_SESSION_ERROR_FILE_NOT_FOUND 1
#define OBS_PLATE_BLUR_ORT_SESSION_ERROR_INVALID_MODEL 2
#define OBS_PLATE_BLUR_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT 3
#define OBS_PLATE_BLUR_ORT_SESSION_ERROR_STARTUP 5
#define OBS_PLATE_BLUR_ORT_SESSION_SUCCESS 0

// Build an ORT session for the given model path with the requested provider.
// The requested provider is tried first and the session falls back to CPU if
// the accelerated provider can't be appended (logged as a warning).
//
// `useGPU` takes one of the USEGPU_* constants from plate-blur-consts.h.
int createOrtSession(ORTModelData &model, const std::string &modelPath, const std::string &useGPU,
		     uint32_t numThreads, std::string &actualProviderOut);

#endif /* ORT_SESSION_UTILS_H */

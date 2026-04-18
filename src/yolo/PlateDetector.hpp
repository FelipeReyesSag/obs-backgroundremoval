// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef PLATE_DETECTOR_HPP
#define PLATE_DETECTOR_HPP

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "../ort-utils/ORTModelData.hpp"

struct PlateBox {
	float x1; // source-image coords (not letterboxed)
	float y1;
	float x2;
	float y2;
	float confidence;
};

// Run the fast-alpr YOLOv9-t end2end plate detector on `imageBGRA` and return
// detections in source-image coordinate space. `confidenceThreshold` is applied
// after the model's built-in NMS; set it to 0 to see every raw detection.
//
// Assumes the model is BCHW float32 input (typical YOLOv9 export) and that
// inputDims[0] has shape [1,3,H,W]. The end2end output has 7 fields per
// detection in [batch_idx, x1, y1, x2, y2, score, class_id] form. Both
// channel-first ([1, 7, N]) and row-first ([1, N, 7]) layouts are accepted;
// 6-field exports without batch_idx also still work.
//
// Returns detections; on any runtime error logs and returns empty vector.
std::vector<PlateBox> detectPlates(ORTModelData &model, const cv::Mat &imageBGRA, float confidenceThreshold);

#endif /* PLATE_DETECTOR_HPP */

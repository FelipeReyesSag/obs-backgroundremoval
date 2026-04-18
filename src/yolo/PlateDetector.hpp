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
// inputDims[0] has shape [1,3,H,W]. The end2end output is a [1,N,6] tensor of
// [x1,y1,x2,y2,conf,cls] rows.
//
// Returns detections; on any runtime error logs and returns empty vector.
std::vector<PlateBox> detectPlates(ORTModelData &model, const cv::Mat &imageBGRA, float confidenceThreshold);

#endif /* PLATE_DETECTOR_HPP */

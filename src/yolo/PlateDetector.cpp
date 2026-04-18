// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "PlateDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>

#include <obs-module.h>
#include <opencv2/imgproc.hpp>

#include <plugin-support.h>

namespace {

// Letterbox an image into `dstHW x dstHW` keeping aspect ratio, padding the
// unused rows/cols with `padColor`. Returns the scale factor and the padding
// offsets (in letterboxed coords) so callers can map detections back.
void letterboxRGB(const cv::Mat &src, int dstHW, cv::Mat &dst, float &scale, int &padX, int &padY,
		  const cv::Scalar &padColor = cv::Scalar(114, 114, 114))
{
	const int srcW = src.cols;
	const int srcH = src.rows;
	const float r = std::min(static_cast<float>(dstHW) / static_cast<float>(srcW),
				 static_cast<float>(dstHW) / static_cast<float>(srcH));
	const int newW = static_cast<int>(std::round(srcW * r));
	const int newH = static_cast<int>(std::round(srcH * r));

	cv::Mat resized;
	if (newW != srcW || newH != srcH) {
		cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
	} else {
		resized = src;
	}

	padX = (dstHW - newW) / 2;
	padY = (dstHW - newH) / 2;
	dst.create(dstHW, dstHW, src.type());
	dst.setTo(padColor);
	resized.copyTo(dst(cv::Rect(padX, padY, newW, newH)));
	scale = r;
}

// Convert a BCHW image in `srcRGBfloat01` (H*W*3, values 0..1, HWC) to CHW
// contiguous float buffer inside `dst`.
void hwcToChw(const cv::Mat &srcRGBfloat01, std::vector<float> &dst)
{
	const int H = srcRGBfloat01.rows;
	const int W = srcRGBfloat01.cols;
	dst.resize(static_cast<size_t>(3) * H * W);
	std::vector<cv::Mat> planes(3);
	for (int c = 0; c < 3; ++c) {
		planes[c] = cv::Mat(H, W, CV_32FC1, dst.data() + static_cast<size_t>(c) * H * W);
	}
	cv::split(srcRGBfloat01, planes);
}

} // namespace

std::vector<PlateBox> detectPlates(ORTModelData &model, const cv::Mat &imageBGRA, float confidenceThreshold)
{
	std::vector<PlateBox> boxes;

	if (!model.session || imageBGRA.empty() || model.inputDims.empty() || model.outputDims.empty()) {
		return boxes;
	}

	// Figure out the model input H/W. The fast-alpr end2end model is BCHW with
	// shape [1, 3, 384, 384], so we read dims 2 and 3.
	const auto &inDims = model.inputDims[0];
	if (inDims.size() < 4) {
		obs_log(LOG_ERROR, "PlateDetector: unexpected input rank %zu", inDims.size());
		return boxes;
	}
	const int inH = static_cast<int>(inDims[2] > 0 ? inDims[2] : 384);
	const int inW = static_cast<int>(inDims[3] > 0 ? inDims[3] : 384);
	if (inH != inW) {
		obs_log(LOG_WARNING, "PlateDetector: non-square input %dx%d; using %d", inW, inH, inH);
	}
	const int dstHW = inH;

	cv::Mat imageRGB;
	cv::cvtColor(imageBGRA, imageRGB, cv::COLOR_BGRA2RGB);

	cv::Mat letterboxed;
	float scale = 1.0f;
	int padX = 0, padY = 0;
	letterboxRGB(imageRGB, dstHW, letterboxed, scale, padX, padY);

	cv::Mat letterboxedF;
	letterboxed.convertTo(letterboxedF, CV_32FC3, 1.0 / 255.0);

	std::vector<float> inputTensorValues;
	hwcToChw(letterboxedF, inputTensorValues);

	const std::array<int64_t, 4> shape{1, 3, dstHW, dstHW};
	Ort::MemoryInfo memInfo =
		Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memInfo, inputTensorValues.data(),
								 inputTensorValues.size(), shape.data(), shape.size());

	std::vector<const char *> inputNames;
	inputNames.reserve(model.inputNames.size());
	for (const auto &n : model.inputNames)
		inputNames.push_back(n.get());
	std::vector<const char *> outputNames;
	outputNames.reserve(model.outputNames.size());
	for (const auto &n : model.outputNames)
		outputNames.push_back(n.get());

	std::vector<Ort::Value> outputTensors;
	try {
		outputTensors = model.session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
						   outputNames.data(), outputNames.size());
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "PlateDetector inference failed: %s", e.what());
		return boxes;
	}

	if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
		obs_log(LOG_ERROR, "PlateDetector: empty or non-tensor output");
		return boxes;
	}

	// End2end YOLOv9 plate output: typically [1, N, 6] where each row is
	// [x1, y1, x2, y2, conf, cls] in input (letterboxed) coordinate space.
	// Some exports use [N, 6]. Handle both.
	const Ort::TensorTypeAndShapeInfo info = outputTensors[0].GetTensorTypeAndShapeInfo();
	const std::vector<int64_t> outShape = info.GetShape();
	const float *data = outputTensors[0].GetTensorData<float>();

	int64_t numRows = 0;
	int64_t cols = 0;
	if (outShape.size() == 3) {
		numRows = outShape[1];
		cols = outShape[2];
	} else if (outShape.size() == 2) {
		numRows = outShape[0];
		cols = outShape[1];
	} else {
		obs_log(LOG_ERROR, "PlateDetector: unexpected output rank %zu", outShape.size());
		return boxes;
	}

	if (cols < 5) {
		obs_log(LOG_ERROR, "PlateDetector: output has %lld cols (need >= 5)", static_cast<long long>(cols));
		return boxes;
	}

	const float invScale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
	const int srcW = imageBGRA.cols;
	const int srcH = imageBGRA.rows;

	boxes.reserve(static_cast<size_t>(numRows));
	for (int64_t i = 0; i < numRows; ++i) {
		const float *row = data + i * cols;
		const float conf = row[4];
		if (conf < confidenceThreshold)
			continue;
		// Un-letterbox the coordinates: subtract pad offsets, divide by scale.
		float x1 = (row[0] - padX) * invScale;
		float y1 = (row[1] - padY) * invScale;
		float x2 = (row[2] - padX) * invScale;
		float y2 = (row[3] - padY) * invScale;

		x1 = std::clamp(x1, 0.0f, static_cast<float>(srcW - 1));
		y1 = std::clamp(y1, 0.0f, static_cast<float>(srcH - 1));
		x2 = std::clamp(x2, 0.0f, static_cast<float>(srcW - 1));
		y2 = std::clamp(y2, 0.0f, static_cast<float>(srcH - 1));

		if (x2 <= x1 || y2 <= y1)
			continue;

		boxes.push_back(PlateBox{x1, y1, x2, y2, conf});
	}

	return boxes;
}

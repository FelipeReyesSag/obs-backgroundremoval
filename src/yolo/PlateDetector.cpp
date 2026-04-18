// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "PlateDetector.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
		// Throttle: this fires per-frame when a buggy EP (CoreML on dynamic
		// shapes) refuses every call. Log once per ~5s so the log viewer
		// stays usable while still surfacing the failure.
		using clock = std::chrono::steady_clock;
		static std::atomic<int64_t> last_log_ms{0};
		const auto now_ms =
			std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch()).count();
		const int64_t prev = last_log_ms.load();
		if (now_ms - prev > 5000) {
			last_log_ms.store(now_ms);
			obs_log(LOG_ERROR, "PlateDetector inference failed: %s", e.what());
		}
		return boxes;
	}

	if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
		obs_log(LOG_ERROR, "PlateDetector: empty or non-tensor output");
		return boxes;
	}

	// End2end YOLOv9 plate output. The export from ankandrew/open-image-models
	// emits a single output tensor with 7 fields per detection:
	//   [batch_idx, x1, y1, x2, y2, score, class_id]
	// in input (letterboxed) coordinate space.
	//
	// Two layouts have been observed in the wild:
	//   * channel-first: [1, 7, N]  (what runtime reports for our bundled model)
	//   * row-first:     [1, N, 7]  (some custom exports)
	// Handle both by sniffing which dim equals the number of fields.
	const Ort::TensorTypeAndShapeInfo info = outputTensors[0].GetTensorTypeAndShapeInfo();
	const std::vector<int64_t> outShape = info.GetShape();
	const float *data = outputTensors[0].GetTensorData<float>();

	int64_t numDets = 0;
	int64_t numFields = 0;
	bool channelFirst = false; // true → field f, det i is at data[f*N + i]
	if (outShape.size() == 3) {
		// [1, A, B]
		const int64_t A = outShape[1];
		const int64_t B = outShape[2];
		// Both 6 and 7 are valid field counts (with/without batch_idx). Treat
		// the *smaller* of the two trailing dims as the field axis when it's
		// in [5, 16]; otherwise fall back to row-first.
		if (A >= 5 && A <= 16 && (B > A || B < 5)) {
			numFields = A;
			numDets = B;
			channelFirst = true;
		} else {
			numDets = A;
			numFields = B;
			channelFirst = false;
		}
	} else if (outShape.size() == 2) {
		numDets = outShape[0];
		numFields = outShape[1];
		channelFirst = false;
	} else {
		obs_log(LOG_ERROR, "PlateDetector: unexpected output rank %zu", outShape.size());
		return boxes;
	}

	if (numFields < 5) {
		obs_log(LOG_ERROR, "PlateDetector: output has %lld fields (need >= 5)",
			static_cast<long long>(numFields));
		return boxes;
	}
	if (numDets <= 0) {
		return boxes;
	}

	// Field index map. 6-field exports drop batch_idx; 7-field exports include
	// it as field 0. Anything else: assume the standard 7-field end2end layout
	// and ignore extra fields.
	const bool hasBatchIdx = (numFields >= 7);
	const int fX1 = hasBatchIdx ? 1 : 0;
	const int fY1 = hasBatchIdx ? 2 : 1;
	const int fX2 = hasBatchIdx ? 3 : 2;
	const int fY2 = hasBatchIdx ? 4 : 3;
	const int fScore = hasBatchIdx ? 5 : 4;

	auto fetch = [&](int64_t det, int field) -> float {
		return channelFirst ? data[static_cast<int64_t>(field) * numDets + det]
				    : data[det * numFields + static_cast<int64_t>(field)];
	};

	const float invScale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
	const int srcW = imageBGRA.cols;
	const int srcH = imageBGRA.rows;

	boxes.reserve(static_cast<size_t>(numDets));
	for (int64_t i = 0; i < numDets; ++i) {
		const float conf = fetch(i, fScore);
		if (conf < confidenceThreshold)
			continue;
		// Un-letterbox the coordinates: subtract pad offsets, divide by scale.
		float x1 = (fetch(i, fX1) - padX) * invScale;
		float y1 = (fetch(i, fY1) - padY) * invScale;
		float x2 = (fetch(i, fX2) - padX) * invScale;
		float y2 = (fetch(i, fY2) - padY) * invScale;

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

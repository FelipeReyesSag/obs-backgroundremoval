// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif

#include <obs-module.h>

#include "ort-session-utils.hpp"
#include "../plate-blur-consts.h"
#include "plugin-support.h"

namespace {

#ifdef _WIN32
std::wstring toWide(const std::string &utf8)
{
	int outLength = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
	std::wstring wide(outLength, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), outLength);
	return wide;
}
#endif

// Parameters are only referenced when at least one execution provider header
// shipped with the linked ONNX Runtime. The Linux prebuilt only ships the
// CPU EP, in which case every #ifdef branch compiles out and the parameters
// would otherwise trip -Werror=unused-parameter.
bool tryAppendProvider([[maybe_unused]] Ort::SessionOptions &options, [[maybe_unused]] const std::string &useGPU,
		       [[maybe_unused]] std::string &actualProviderOut)
{
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
	if (useGPU == USEGPU_CUDA) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
		actualProviderOut = USEGPU_CUDA;
		return true;
	}
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
	if (useGPU == USEGPU_ROCM) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(options, 0));
		actualProviderOut = USEGPU_ROCM;
		return true;
	}
#endif
#ifdef HAVE_ONNXRUNTIME_MIGRAPHX_EP
	if (useGPU == USEGPU_MIGRAPHX) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(options, 0));
		actualProviderOut = USEGPU_MIGRAPHX;
		return true;
	}
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
	if (useGPU == USEGPU_TENSORRT) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, 0));
		actualProviderOut = USEGPU_TENSORRT;
		return true;
	}
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
	if (useGPU == USEGPU_DIRECTML) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, 0));
		actualProviderOut = USEGPU_DIRECTML;
		return true;
	}
#endif
#if defined(__APPLE__)
	if (useGPU == USEGPU_COREML) {
		// CoreML is flaky on dynamic shapes. Keep a safety net available at
		// runtime by allowing CPU-only fallback inside CoreML if needed.
		uint32_t coreml_flags = COREML_FLAG_ENABLE_ON_SUBGRAPH;
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags));
		actualProviderOut = USEGPU_COREML;
		return true;
	}
#endif
	return false;
}

} // namespace

int createOrtSession(ORTModelData &model, const std::string &modelPath, const std::string &useGPU, uint32_t numThreads,
		     std::string &actualProviderOut)
{
	if (modelPath.empty()) {
		obs_log(LOG_ERROR, "Empty model path");
		return OBS_PLATE_BLUR_ORT_SESSION_ERROR_FILE_NOT_FOUND;
	}

	Ort::SessionOptions sessionOptions;
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (useGPU != USEGPU_CPU) {
		sessionOptions.DisableMemPattern();
		sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	} else {
		sessionOptions.SetInterOpNumThreads(static_cast<int>(numThreads));
		sessionOptions.SetIntraOpNumThreads(static_cast<int>(numThreads));
	}

	actualProviderOut = USEGPU_CPU;
	try {
		if (useGPU != USEGPU_CPU) {
			if (!tryAppendProvider(sessionOptions, useGPU, actualProviderOut)) {
				obs_log(LOG_WARNING, "Requested provider '%s' not compiled in; falling back to CPU",
					useGPU.c_str());
			}
		}
	} catch (const std::exception &e) {
		obs_log(LOG_WARNING, "Failed to append provider '%s' (%s); falling back to CPU", useGPU.c_str(),
			e.what());
		actualProviderOut = USEGPU_CPU;
	}

	try {
#ifdef _WIN32
		const std::wstring wpath = toWide(modelPath);
		model.session.reset(new Ort::Session(*model.env, wpath.c_str(), sessionOptions));
#else
		model.session.reset(new Ort::Session(*model.env, modelPath.c_str(), sessionOptions));
#endif
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Failed to create ONNX session: %s", e.what());
		return OBS_PLATE_BLUR_ORT_SESSION_ERROR_STARTUP;
	}

	Ort::AllocatorWithDefaultOptions allocator;
	model.inputNames.clear();
	model.outputNames.clear();
	model.inputDims.clear();
	model.outputDims.clear();

	const size_t inputCount = model.session->GetInputCount();
	const size_t outputCount = model.session->GetOutputCount();

	for (size_t i = 0; i < inputCount; ++i) {
		model.inputNames.push_back(model.session->GetInputNameAllocated(i, allocator));
		auto shape = model.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		// Any dynamic dim becomes 1 here; callers fill in real sizes at inference time.
		for (auto &d : shape) {
			if (d < 0)
				d = 1;
		}
		model.inputDims.push_back(std::move(shape));
	}
	for (size_t i = 0; i < outputCount; ++i) {
		model.outputNames.push_back(model.session->GetOutputNameAllocated(i, allocator));
		auto shape = model.session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		for (auto &d : shape) {
			if (d < 0)
				d = 1;
		}
		model.outputDims.push_back(std::move(shape));
	}

	if (inputCount == 0 || outputCount == 0) {
		obs_log(LOG_ERROR, "Model has zero inputs or outputs");
		return OBS_PLATE_BLUR_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT;
	}

	obs_log(LOG_INFO, "ONNX Runtime initialized with provider: %s", actualProviderOut.c_str());
	for (size_t i = 0; i < inputCount; ++i) {
		const auto &d = model.inputDims[i];
		obs_log(LOG_INFO, "  input[%zu] name=%s dims=[%lld,%lld,%lld,%lld]", i, model.inputNames[i].get(),
			d.size() > 0 ? (long long)d[0] : -1, d.size() > 1 ? (long long)d[1] : -1,
			d.size() > 2 ? (long long)d[2] : -1, d.size() > 3 ? (long long)d[3] : -1);
	}
	for (size_t i = 0; i < outputCount; ++i) {
		const auto &d = model.outputDims[i];
		obs_log(LOG_INFO, "  output[%zu] name=%s dims=[%lld,%lld,%lld]", i, model.outputNames[i].get(),
			d.size() > 0 ? (long long)d[0] : -1, d.size() > 1 ? (long long)d[1] : -1,
			d.size() > 2 ? (long long)d[2] : -1);
	}

	return OBS_PLATE_BLUR_ORT_SESSION_SUCCESS;
}

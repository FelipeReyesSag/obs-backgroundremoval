// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef ORTMODELDATA_H
#define ORTMODELDATA_H

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <vector>

// Minimal ONNX Runtime per-session state. We hold the env, session, and the
// input/output metadata we introspected at load time. Tensor buffers are
// allocated per-inference inside the worker since YOLO's end2end output has
// a dynamic row count.
struct ORTModelData {
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Env> env;
	std::vector<Ort::AllocatedStringPtr> inputNames;
	std::vector<Ort::AllocatedStringPtr> outputNames;
	std::vector<std::vector<int64_t>> inputDims;
	std::vector<std::vector<int64_t>> outputDims;
};

#endif /* ORTMODELDATA_H */

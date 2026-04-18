// SPDX-FileCopyrightText: 2026 Felipe Reyes
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "plate-blur-filter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <plugin-support.h>

#include "plate-blur-consts.h"
#include "ort-utils/ORTModelData.hpp"
#include "ort-utils/ort-session-utils.hpp"
#include "obs-utils/obs-utils.hpp"
#include "yolo/PlateDetector.hpp"
#include "update-checker/update-checker.h"

namespace {

enum slot_state : uint8_t { SLOT_EMPTY = 0, SLOT_QUEUED = 1, SLOT_PROCESSED = 2, SLOT_LOST = 3 };

struct frame_slot {
	cv::Mat pixels_bgra; // owned clone; empty when SLOT_EMPTY
	std::vector<PlateBox> detections;
	uint64_t seq = 0;
	slot_state state = SLOT_EMPTY;
};

} // namespace

struct plate_blur_filter : public std::enable_shared_from_this<plate_blur_filter> {
	// --- OBS bindings ---
	obs_source_t *source = nullptr;

	// Capture staging (used only from render thread)
	gs_texrender_t *capture_texrender = nullptr;
	gs_stagesurf_t *capture_stagesurface = nullptr;

	// Offscreen intermediate for horizontal-blur pass
	gs_texrender_t *blur_intermediate = nullptr;

	// Textures uploaded each render from CPU data
	gs_texture_t *delayed_tex = nullptr;
	uint32_t delayed_tex_w = 0;
	uint32_t delayed_tex_h = 0;
	gs_texture_t *mask_tex = nullptr;
	uint32_t mask_tex_w = 0;
	uint32_t mask_tex_h = 0;

	// The blur shader
	gs_effect_t *blur_effect = nullptr;

	// --- ONNX state ---
	ORTModelData model;
	std::mutex model_mu;
	std::string requestedProvider;
	std::string activeProvider;
	uint32_t numThreads = 1;
	std::string modelPath;

	// --- Ring buffer (worker + render thread share this) ---
	std::vector<frame_slot> ring;
	size_t write_idx = 0;
	uint64_t write_seq = 0;
	std::mutex ring_mu;
	std::condition_variable ring_cv;

	// --- Settings ---
	int delay_ms = 3000;
	float conf_threshold = 0.25f;
	int blur_strength = 12;
	float box_padding = 0.10f;
	bool show_debug_overlay = false;

	// --- Worker ---
	std::thread worker;
	std::atomic<bool> worker_running{false};
	std::atomic<bool> shutdown_requested{false};

	std::atomic<bool> isDisabled{true};

	// Diagnostics
	std::atomic<uint64_t> frames_captured{0};
	std::atomic<uint64_t> frames_inferred{0};
	std::atomic<uint64_t> frames_dropped{0};

	~plate_blur_filter() { obs_log(LOG_INFO, "Plate blur filter destructor"); }
};

// -------------------------------------------------------------------------
//                             Helpers
// -------------------------------------------------------------------------
namespace {

// Compute delay_frames from the current OBS output FPS and the desired delay_ms.
size_t computeDelayFrames(int delay_ms)
{
	struct obs_video_info ovi;
	double fps = 60.0;
	if (obs_get_video_info(&ovi) && ovi.fps_num > 0 && ovi.fps_den > 0) {
		fps = static_cast<double>(ovi.fps_num) / static_cast<double>(ovi.fps_den);
	}
	size_t frames = static_cast<size_t>((fps * static_cast<double>(delay_ms)) / 1000.0 + 0.5);
	if (frames < 1)
		frames = 1;
	if (frames >= PLATE_BLUR_RING_CAPACITY)
		frames = PLATE_BLUR_RING_CAPACITY - 1;
	return frames;
}

// Upload BGRA pixel data into a GPU texture, recreating on size change.
// Must be called under obs_enter_graphics() by the caller.
void uploadBGRATexture(gs_texture_t **tex, uint32_t &cur_w, uint32_t &cur_h, const cv::Mat &bgra)
{
	if (bgra.empty())
		return;
	const uint32_t w = static_cast<uint32_t>(bgra.cols);
	const uint32_t h = static_cast<uint32_t>(bgra.rows);
	if (*tex && (cur_w != w || cur_h != h)) {
		gs_texture_destroy(*tex);
		*tex = nullptr;
	}
	if (!*tex) {
		const uint8_t *row = bgra.data;
		*tex = gs_texture_create(w, h, GS_BGRA, 1, &row, GS_DYNAMIC);
		cur_w = w;
		cur_h = h;
		return;
	}
	// Reuse existing texture; update its contents
	gs_texture_set_image(*tex, bgra.data, static_cast<uint32_t>(bgra.step), false);
}

// Same, for a single-channel R8 mask.
void uploadR8Texture(gs_texture_t **tex, uint32_t &cur_w, uint32_t &cur_h, const cv::Mat &r8)
{
	if (r8.empty())
		return;
	const uint32_t w = static_cast<uint32_t>(r8.cols);
	const uint32_t h = static_cast<uint32_t>(r8.rows);
	if (*tex && (cur_w != w || cur_h != h)) {
		gs_texture_destroy(*tex);
		*tex = nullptr;
	}
	if (!*tex) {
		const uint8_t *row = r8.data;
		*tex = gs_texture_create(w, h, GS_R8, 1, &row, GS_DYNAMIC);
		cur_w = w;
		cur_h = h;
		return;
	}
	gs_texture_set_image(*tex, r8.data, static_cast<uint32_t>(r8.step), false);
}

// Paint a 1-channel blur mask matching source dimensions. Each (padded) box
// contributes a soft-edged filled rectangle with value 255. Padding is
// `box_padding` as a fraction of the box's smaller dimension.
cv::Mat buildBlurMask(uint32_t w, uint32_t h, const std::vector<PlateBox> &boxes, float box_padding)
{
	cv::Mat mask(static_cast<int>(h), static_cast<int>(w), CV_8UC1, cv::Scalar(0));
	if (boxes.empty())
		return mask;

	for (const auto &b : boxes) {
		const float bw = b.x2 - b.x1;
		const float bh = b.y2 - b.y1;
		const float pad = box_padding * std::min(bw, bh);
		int x1 = std::max(0, static_cast<int>(std::floor(b.x1 - pad)));
		int y1 = std::max(0, static_cast<int>(std::floor(b.y1 - pad)));
		int x2 = std::min(static_cast<int>(w), static_cast<int>(std::ceil(b.x2 + pad)));
		int y2 = std::min(static_cast<int>(h), static_cast<int>(std::ceil(b.y2 + pad)));
		if (x2 <= x1 || y2 <= y1)
			continue;
		cv::rectangle(mask, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(255), cv::FILLED);
	}

	// Soft-feather the edges so the blur region fades smoothly.
	// Kernel is small relative to typical plate size (~40px) and scales with resolution.
	const int fk = std::max(3, static_cast<int>(std::min(w, h) / 200) | 1); // odd
	cv::GaussianBlur(mask, mask, cv::Size(fk, fk), 0);
	return mask;
}

} // namespace

// -------------------------------------------------------------------------
//                             Worker thread
// -------------------------------------------------------------------------
namespace {

void plate_blur_worker_loop(std::shared_ptr<plate_blur_filter> tf)
{
	obs_log(LOG_INFO, "Plate blur worker thread started");
	while (true) {
		cv::Mat frame;
		uint64_t seq = 0;
		size_t slot_idx = 0;
		{
			std::unique_lock<std::mutex> lk(tf->ring_mu);
			tf->ring_cv.wait(lk, [&]() {
				if (tf->shutdown_requested.load())
					return true;
				for (size_t i = 0; i < tf->ring.size(); ++i) {
					if (tf->ring[i].state == SLOT_QUEUED)
						return true;
				}
				return false;
			});
			if (tf->shutdown_requested.load())
				break;

			// Pick the OLDEST queued slot (FIFO) so every captured frame gets
			// inference before the read pointer reaches it.
			uint64_t oldest_seq = UINT64_MAX;
			bool found = false;
			for (size_t i = 0; i < tf->ring.size(); ++i) {
				if (tf->ring[i].state == SLOT_QUEUED && tf->ring[i].seq < oldest_seq) {
					oldest_seq = tf->ring[i].seq;
					slot_idx = i;
					found = true;
				}
			}
			if (!found)
				continue;
			frame = tf->ring[slot_idx].pixels_bgra.clone();
			seq = tf->ring[slot_idx].seq;
		}

		// Inference outside the lock.
		std::vector<PlateBox> boxes;
		{
			std::lock_guard<std::mutex> lk(tf->model_mu);
			if (tf->model.session) {
				try {
					boxes = detectPlates(tf->model, frame, tf->conf_threshold);
				} catch (const std::exception &e) {
					obs_log(LOG_ERROR, "Plate detection threw: %s", e.what());
				}
			}
		}

		// Write back results if the slot still holds the same frame.
		{
			std::lock_guard<std::mutex> lk(tf->ring_mu);
			if (slot_idx < tf->ring.size() && tf->ring[slot_idx].seq == seq &&
			    tf->ring[slot_idx].state == SLOT_QUEUED) {
				tf->ring[slot_idx].detections = std::move(boxes);
				tf->ring[slot_idx].state = SLOT_PROCESSED;
				tf->frames_inferred.fetch_add(1);
			}
		}
	}
	obs_log(LOG_INFO, "Plate blur worker thread exiting");
}

void start_worker(std::shared_ptr<plate_blur_filter> tf)
{
	if (tf->worker_running.load())
		return;
	tf->shutdown_requested.store(false);
	tf->worker_running.store(true);
	tf->worker = std::thread([tf]() {
		plate_blur_worker_loop(tf);
		tf->worker_running.store(false);
	});
}

void stop_worker(std::shared_ptr<plate_blur_filter> tf)
{
	{
		std::lock_guard<std::mutex> lk(tf->ring_mu);
		tf->shutdown_requested.store(true);
	}
	tf->ring_cv.notify_all();
	if (tf->worker.joinable())
		tf->worker.join();
	tf->worker_running.store(false);
}

} // namespace

// -------------------------------------------------------------------------
//                             Properties / defaults
// -------------------------------------------------------------------------
const char *plate_blur_filter_getname(void *)
{
	return obs_module_text("PlateBlurFilter");
}

void plate_blur_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "delay_ms", 3000);
	obs_data_set_default_double(settings, "conf_threshold", 0.25);
	obs_data_set_default_int(settings, "blur_strength", 12);
	obs_data_set_default_double(settings, "box_padding", 0.10);
	obs_data_set_default_bool(settings, "show_debug_overlay", false);
	obs_data_set_default_int(settings, "num_threads", 2);
#if defined(__APPLE__)
	obs_data_set_default_string(settings, "provider", USEGPU_COREML);
#elif defined(_WIN32)
	obs_data_set_default_string(settings, "provider", USEGPU_DIRECTML);
#else
	obs_data_set_default_string(settings, "provider", USEGPU_CPU);
#endif
}

obs_properties_t *plate_blur_filter_properties(void *)
{
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_int_slider(props, "delay_ms", obs_module_text("DelayMs"), 1000, 5000, 100);

	obs_properties_add_float_slider(props, "conf_threshold", obs_module_text("ConfidenceThreshold"), 0.1, 0.9,
					0.05);

	obs_properties_add_int_slider(props, "blur_strength", obs_module_text("BlurStrength"), 1, 30, 1);

	obs_properties_add_float_slider(props, "box_padding", obs_module_text("BoxPadding"), 0.0, 0.5, 0.05);

	obs_property_t *p_provider = obs_properties_add_list(props, "provider", obs_module_text("GPUProvider"),
							     OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p_provider, obs_module_text("CPU"), USEGPU_CPU);
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
	obs_property_list_add_string(p_provider, obs_module_text("GPUCUDA"), USEGPU_CUDA);
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
	obs_property_list_add_string(p_provider, obs_module_text("DirectML"), USEGPU_DIRECTML);
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
	obs_property_list_add_string(p_provider, obs_module_text("TENSORRT"), USEGPU_TENSORRT);
#endif
#ifdef HAVE_ONNXRUNTIME_MIGRAPHX_EP
	obs_property_list_add_string(p_provider, obs_module_text("GPUMIGRAPHX"), USEGPU_MIGRAPHX);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_provider, obs_module_text("CoreML"), USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "num_threads", obs_module_text("NumThreads"), 0, 8, 1);

	obs_properties_add_bool(props, "show_debug_overlay", obs_module_text("ShowDebugOverlay"));

	// Info blurb
	std::string info = std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	obs_properties_add_text(props, "info", info.c_str(), OBS_TEXT_INFO);

	return props;
}

// -------------------------------------------------------------------------
//                             Lifecycle
// -------------------------------------------------------------------------
namespace {

void load_blur_effect(plate_blur_filter *tf)
{
	obs_enter_graphics();
	char *p = obs_module_file(EFFECT_PLATE_BLUR_PATH);
	gs_effect_destroy(tf->blur_effect);
	tf->blur_effect = nullptr;
	if (p) {
		tf->blur_effect = gs_effect_create_from_file(p, nullptr);
		bfree(p);
	}
	if (!tf->blur_effect) {
		obs_log(LOG_ERROR, "Failed to load plate blur effect from '%s'", EFFECT_PLATE_BLUR_PATH);
	}
	obs_leave_graphics();
}

// Try to load the ONNX model file, preferring a user-data-dir override over
// the bundled `data/models/` path. Returns the full path or empty string.
std::string resolve_model_path()
{
	char *p = obs_module_file(MODEL_PLATE_YOLO);
	if (!p)
		return {};
	std::string s{p};
	bfree(p);
	return s;
}

} // namespace

void *plate_blur_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Plate blur filter created");
	try {
		auto instance = std::make_shared<plate_blur_filter>();
		instance->source = source;
		instance->ring.resize(PLATE_BLUR_RING_CAPACITY);

		instance->model.env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "obs-plate-blur"));

		obs_enter_graphics();
		instance->capture_texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
		instance->blur_intermediate = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
		obs_leave_graphics();

		load_blur_effect(instance.get());

		auto *ptr = new std::shared_ptr<plate_blur_filter>(instance);
		plate_blur_filter_update(ptr, settings);
		return ptr;
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "Plate blur filter create failed: %s", e.what());
		return nullptr;
	}
}

void plate_blur_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Plate blur filter destroyed");
	auto *ptr = static_cast<std::shared_ptr<plate_blur_filter> *>(data);
	if (!ptr)
		return;
	if (*ptr) {
		(*ptr)->isDisabled = true;
		stop_worker(*ptr);

		obs_enter_graphics();
		if ((*ptr)->capture_texrender)
			gs_texrender_destroy((*ptr)->capture_texrender);
		if ((*ptr)->blur_intermediate)
			gs_texrender_destroy((*ptr)->blur_intermediate);
		if ((*ptr)->capture_stagesurface)
			gs_stagesurface_destroy((*ptr)->capture_stagesurface);
		if ((*ptr)->delayed_tex)
			gs_texture_destroy((*ptr)->delayed_tex);
		if ((*ptr)->mask_tex)
			gs_texture_destroy((*ptr)->mask_tex);
		if ((*ptr)->blur_effect)
			gs_effect_destroy((*ptr)->blur_effect);
		obs_leave_graphics();
	}
	delete ptr;
}

void plate_blur_filter_update(void *data, obs_data_t *settings)
{
	auto *ptr = static_cast<std::shared_ptr<plate_blur_filter> *>(data);
	if (!ptr || !*ptr)
		return;
	auto tf = *ptr;

	tf->isDisabled = true;

	tf->delay_ms = static_cast<int>(obs_data_get_int(settings, "delay_ms"));
	tf->conf_threshold = static_cast<float>(obs_data_get_double(settings, "conf_threshold"));
	tf->blur_strength = static_cast<int>(obs_data_get_int(settings, "blur_strength"));
	tf->box_padding = static_cast<float>(obs_data_get_double(settings, "box_padding"));
	tf->show_debug_overlay = obs_data_get_bool(settings, "show_debug_overlay");

	const std::string newProvider = obs_data_get_string(settings, "provider");
	const uint32_t newNumThreads = static_cast<uint32_t>(obs_data_get_int(settings, "num_threads"));
	const std::string modelPath = resolve_model_path();

	if (modelPath.empty()) {
		obs_log(LOG_ERROR, "Plate detection model file not found. Expected at %s within plugin data. "
				   "Run scripts/fetch_model.sh (or .ps1) before loading the filter.",
			MODEL_PLATE_YOLO);
	}

	bool model_needs_reload = (tf->modelPath != modelPath) || (tf->requestedProvider != newProvider) ||
				  (tf->numThreads != newNumThreads) || !tf->model.session;

	if (model_needs_reload && !modelPath.empty()) {
		stop_worker(tf);
		{
			std::lock_guard<std::mutex> lk(tf->model_mu);
			tf->model.session.reset();
			tf->requestedProvider = newProvider;
			tf->numThreads = newNumThreads;
			tf->modelPath = modelPath;

			int rc = createOrtSession(tf->model, modelPath, newProvider, newNumThreads, tf->activeProvider);
			if (rc != OBS_PLATE_BLUR_ORT_SESSION_SUCCESS) {
				obs_log(LOG_ERROR, "createOrtSession failed with code %d; plate blur disabled", rc);
				tf->model.session.reset();
				return;
			}
		}
		start_worker(tf);
	}

	obs_log(LOG_INFO, "Plate Blur Filter settings updated:");
	obs_log(LOG_INFO, "  delay_ms=%d conf=%.2f blur_strength=%d padding=%.2f overlay=%d", tf->delay_ms,
		tf->conf_threshold, tf->blur_strength, tf->box_padding, (int)tf->show_debug_overlay);
	obs_log(LOG_INFO, "  provider=%s active=%s threads=%u", tf->requestedProvider.c_str(),
		tf->activeProvider.c_str(), (unsigned)tf->numThreads);

	tf->isDisabled = (tf->blur_effect == nullptr) || (!tf->model.session);
}

void plate_blur_filter_activate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<plate_blur_filter> *>(data);
	if (!ptr || !*ptr)
		return;
	(*ptr)->isDisabled = !((*ptr)->blur_effect) || !((*ptr)->model.session);
}

void plate_blur_filter_deactivate(void *)
{
	// We keep the worker running across deactivation so re-enabling is instant.
}

void plate_blur_filter_video_tick(void *, float)
{
	// No per-tick work; everything happens on the render thread + worker.
}

// -------------------------------------------------------------------------
//                             Render
// -------------------------------------------------------------------------
namespace {

// Push a freshly-captured frame into the ring. Returns the slot's seq number.
uint64_t enqueue_captured_frame(plate_blur_filter &tf, cv::Mat &&bgra)
{
	std::unique_lock<std::mutex> lk(tf.ring_mu);
	frame_slot &slot = tf.ring[tf.write_idx];
	if (slot.state == SLOT_QUEUED) {
		// Worker hasn't caught up; this frame never got inference. Count it.
		tf.frames_dropped.fetch_add(1);
	}
	slot.pixels_bgra = std::move(bgra);
	slot.detections.clear();
	slot.seq = ++tf.write_seq;
	slot.state = SLOT_QUEUED;
	tf.write_idx = (tf.write_idx + 1) % tf.ring.size();
	lk.unlock();
	tf.ring_cv.notify_one();
	tf.frames_captured.fetch_add(1);
	return tf.write_seq;
}

// Pull the frame to emit: the one that was written `delay_frames` ago.
// Returns false if nothing to emit yet (startup warming).
bool fetch_delayed_frame(plate_blur_filter &tf, size_t delay_frames, cv::Mat &outPixels,
			 std::vector<PlateBox> &outBoxes, bool &outWasProcessed)
{
	std::lock_guard<std::mutex> lk(tf.ring_mu);
	if (tf.write_seq < delay_frames + 1)
		return false;
	// Target seq = write_seq - delay_frames
	const uint64_t target_seq = tf.write_seq - delay_frames;
	// Index of that frame: (write_idx - delay_frames - 1) mod capacity
	const size_t cap = tf.ring.size();
	const size_t idx = (tf.write_idx + cap - delay_frames - 1) % cap;
	const frame_slot &slot = tf.ring[idx];
	if (slot.seq != target_seq || slot.state == SLOT_EMPTY)
		return false;
	outPixels = slot.pixels_bgra.clone(); // clone so we can unlock quickly
	outBoxes = slot.detections;
	outWasProcessed = (slot.state == SLOT_PROCESSED);
	return true;
}

// Render a full-screen quad through a given technique, with current
// gs_effect parameters already set. Assumes the caller has set up the render
// target (texrender or OBS filter output).
void draw_effect_pass(gs_effect_t *effect, const char *tech, uint32_t w, uint32_t h)
{
	while (gs_effect_loop(effect, tech)) {
		gs_draw_sprite(nullptr, 0, w, h);
	}
}

} // namespace

void plate_blur_filter_video_render(void *data, gs_effect_t *)
{
	auto *ptr = static_cast<std::shared_ptr<plate_blur_filter> *>(data);
	if (!ptr || !*ptr) {
		return;
	}
	auto tf = *ptr;

	if (tf->isDisabled || !tf->source) {
		if (tf->source)
			obs_source_skip_video_filter(tf->source);
		return;
	}

	// 1. Capture the current source frame into a CPU buffer (clone).
	cv::Mat captured;
	uint32_t src_w = 0, src_h = 0;
	if (!captureFilterTargetBGRA(tf->source, &tf->capture_texrender, &tf->capture_stagesurface, captured, src_w,
				     src_h)) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 2. Enqueue for the worker.
	enqueue_captured_frame(*tf, std::move(captured));

	// 3. Pull the delayed frame (N frames older).
	const size_t delay_frames = computeDelayFrames(tf->delay_ms);
	cv::Mat emit_pixels;
	std::vector<PlateBox> emit_boxes;
	bool processed = false;
	const bool have_frame = fetch_delayed_frame(*tf, delay_frames, emit_pixels, emit_boxes, processed);

	if (!have_frame) {
		// Startup warming: pass through the raw live frame.
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 4. Upload delayed frame + blur mask to GPU textures.
	cv::Mat mask = buildBlurMask(src_w, src_h, emit_boxes, tf->box_padding);
	uploadBGRATexture(&tf->delayed_tex, tf->delayed_tex_w, tf->delayed_tex_h, emit_pixels);
	uploadR8Texture(&tf->mask_tex, tf->mask_tex_w, tf->mask_tex_h, mask);

	if (!tf->delayed_tex || !tf->mask_tex || !tf->blur_effect) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 5. Pass 1: horizontal blur of the delayed frame into an intermediate texrender.
	gs_effect_t *eff = tf->blur_effect;
	gs_eparam_t *p_delayed = gs_effect_get_param_by_name(eff, "delayedFrame");
	gs_eparam_t *p_blurredH = gs_effect_get_param_by_name(eff, "blurredH");
	gs_eparam_t *p_mask = gs_effect_get_param_by_name(eff, "blurMask");
	gs_eparam_t *p_texel = gs_effect_get_param_by_name(eff, "texel_size");
	gs_eparam_t *p_radius = gs_effect_get_param_by_name(eff, "blur_radius_px");

	struct vec2 texel;
	texel.x = 1.0f / static_cast<float>(src_w);
	texel.y = 1.0f / static_cast<float>(src_h);

	if (p_delayed)
		gs_effect_set_texture(p_delayed, tf->delayed_tex);
	if (p_mask)
		gs_effect_set_texture(p_mask, tf->mask_tex);
	if (p_texel)
		gs_effect_set_vec2(p_texel, &texel);
	if (p_radius)
		gs_effect_set_float(p_radius, static_cast<float>(tf->blur_strength));

	gs_texrender_reset(tf->blur_intermediate);
	if (!gs_texrender_begin(tf->blur_intermediate, src_w, src_h)) {
		obs_source_skip_video_filter(tf->source);
		return;
	}
	{
		struct vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(src_w), 0.0f, static_cast<float>(src_h), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		draw_effect_pass(eff, "BlurH", src_w, src_h);
		gs_blend_state_pop();
	}
	gs_texrender_end(tf->blur_intermediate);

	gs_texture_t *intermediateTex = gs_texrender_get_texture(tf->blur_intermediate);
	if (!intermediateTex) {
		obs_source_skip_video_filter(tf->source);
		return;
	}

	// 6. Pass 2: composite into the filter output.
	if (p_blurredH)
		gs_effect_set_texture(p_blurredH, intermediateTex);
	// Re-bind every frame since the OBS filter path can rebind `image` after us.
	if (p_delayed)
		gs_effect_set_texture(p_delayed, tf->delayed_tex);
	if (p_mask)
		gs_effect_set_texture(p_mask, tf->mask_tex);

	const char *tech_name = tf->show_debug_overlay ? "DebugOverlay" : "BlurVComposite";

	if (!obs_source_process_filter_begin(tf->source, GS_BGRA, OBS_ALLOW_DIRECT_RENDERING)) {
		obs_source_skip_video_filter(tf->source);
		return;
	}
	obs_source_process_filter_tech_end(tf->source, eff, src_w, src_h, tech_name);
}

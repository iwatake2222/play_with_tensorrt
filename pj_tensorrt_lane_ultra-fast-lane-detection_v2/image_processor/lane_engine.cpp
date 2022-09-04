/* Copyright 2022 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "inference_helper_tensorrt.h"      // to call SetDlaCore
#include "lane_engine.h"

/*** Macro ***/
#define TAG "LaneEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#define USE_CULANE
//#define USE_TUSIMPLE
//#define USE_CULANECURVELANES

#if defined(USE_CULANE)
#define MODEL_NAME  "ufldv2_culane_res18_320x1600.onnx"
#define INPUT_DIMS  { 1, 3, 320, 1600 }
#elif defined(USE_TUSIMPLE)
#define MODEL_NAME  "ufldv2_tusimple_res18_320x800.onnx"
#define INPUT_DIMS  { 1, 3, 320, 800 }
#elif defined(USE_CULANECURVELANES)
#define MODEL_NAME  "ufldv2_curvelanes_res18_800x1600.onnx"
#define INPUT_DIMS  { 1, 3, 800, 1600 }
#endif

#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "loc_row"
#define OUTPUT_NAME_1 "loc_col"
#define OUTPUT_NAME_2 "exist_row"
#define OUTPUT_NAME_3 "exist_col"

#if defined(USE_CULANE)
static constexpr int32_t kNumRow = 72;
static constexpr int32_t kNumCol = 81;
static constexpr float kCropRatio = 0.6;
#elif defined(USE_TUSIMPLE)
static constexpr int32_t kNumRow = 56;
static constexpr int32_t kNumCol = 41;
static constexpr float kCropRatio = 0.8;
#elif defined(USE_CULANECURVELANES)
static constexpr int32_t kNumRow = 72;
static constexpr int32_t kNumCol = 81;
static constexpr float kCropRatio = 0.8;
#endif

void LaneEngine::GenerateAnchor()
{
    for (int32_t i = 0; i < kNumRow; i++) {
#if defined(USE_CULANE)
        row_anchor_.push_back(0.42 + i * (1.0 - 0.42) / (kNumRow - 1));
#elif defined(USE_TUSIMPLE)
        row_anchor_.push_back((160 + i * (710 - 160) / (kNumRow - 1)) / 720.0);
#elif defined(USE_CULANECURVELANES)
        row_anchor_.push_back(0.4 + i * (1.0 - 0.4) / (kNumRow - 1));
#endif
    }

    for (int32_t i = 0; i < kNumCol; i++) {
        col_anchor_.push_back(0.0 + i * (1.0 - 0.0) / (kNumCol - 1));
    }
}

/*** Function ***/
int32_t LaneEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* normalize for imagenet */
    input_tensor_info.normalize.mean[0] = 0.485f;
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_2, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_3, TENSORTYPE));
    
#if defined(USE_CULANECURVELANES)
    output_tensor_info_list_.push_back(OutputTensorInfo("282", TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo("288", TENSORTYPE));
#endif

    /* Create and Initialize Inference Helper */
    //inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntime));
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));

    if (!inference_helper_) {
        return kRetErr;
    }
    InferenceHelperTensorRt* p = dynamic_cast<InferenceHelperTensorRt*>(inference_helper_.get());
    if (p) p->SetDlaCore(-1);  /* Use GPU */
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    GenerateAnchor();

    return kRetOk;
}

int32_t LaneEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    return kRetOk;
}

static std::vector<int32_t> argmax_1(const std::vector<float>& v, const std::vector<int32_t>& dims)
{
    std::vector<int32_t> ret;
    ret.resize(dims[0] * dims[2] * dims[3]);

    for (int32_t i = 0; i < dims[2]; i++) {
        for (int32_t j = 0; j < dims[3]; j++) {
            int32_t offset = dims[3] * i + j;
            float max_val = 0;
            int32_t max_index = 0;
            for (int32_t k = 0; k < dims[1]; k++) {
                size_t index = k * dims[2] * dims[3] + offset;
                if (v[index] > max_val) {
                    max_val = v[index];
                    max_index = k;
                }
            }
            ret[offset] = max_index;
        }
    }
    return ret;
}

static int32_t sum_valid(const std::vector<int32_t>& v, int32_t num, int32_t interval, int32_t offset)
{
    int32_t sum = 0;
    for (int32_t i = 0; i < num; i++) {
        sum += v[i * interval + offset];
    }
    return sum;
}


std::vector<LaneEngine::Line<float>> LaneEngine::Pred2Coords(const std::vector<float>& loc_row, const std::vector<int32_t>& loc_row_dims, const std::vector<float>& exist_row, const std::vector<int32_t>& exist_row_dims,
                             const std::vector<float>& loc_col, const std::vector<int32_t>& loc_col_dims, const std::vector<float>& exist_col, const std::vector<int32_t>& exist_col_dims)
{
    std::vector<Line<float>> line_list(4);

    int32_t num_grid_row = loc_row_dims[1]; /* 200 */
    int32_t num_cls_row = loc_row_dims[2];  /* 72 */
    int32_t num_lane_row = loc_row_dims[3]; /* 4 */
    int32_t num_grid_col = loc_col_dims[1];
    int32_t num_cls_col = loc_col_dims[2];
    int32_t num_lane_col = loc_col_dims[3];

    auto max_indices_row = argmax_1(loc_row, loc_row_dims);  /* 1x200x72x4 -> 1x72x4 */
    auto valid_row = argmax_1(exist_row, exist_row_dims);
    auto max_indices_col = argmax_1(loc_col, loc_col_dims);
    auto valid_col = argmax_1(exist_col, exist_col_dims);

    for (int32_t i : { 1, 2 }) {
        if (sum_valid(valid_row, num_cls_row, num_lane_row, i) > num_cls_row / 2) {
            for (int32_t k = 0; k < num_cls_row; k++) {
                int32_t index = k * num_lane_row + i;
                if (valid_row[index] != 0) {
                    /* all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1))) */
                    std::vector<float> pred_all_list;
                    std::vector<int32_t> all_ind_list;
                    for (int32_t all_ind = std::max(0, max_indices_row[index] - 1); all_ind <= std::min(num_grid_row - 1, max_indices_row[index] + 1); all_ind++) {
                        pred_all_list.push_back(loc_row[all_ind * num_cls_row * num_lane_row + index]);
                        all_ind_list.push_back(all_ind);
                    }
                    std::vector<float> pred_all_list_softmax(pred_all_list.size());
                    CommonHelper::SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
                    float out_temp = 0;
                    for (int32_t l = 0; l < pred_all_list.size(); l++) {
                        out_temp += pred_all_list_softmax[l] * all_ind_list[l];
                    }
                    float x = (out_temp + 0.5) / (num_grid_row - 1.0);
                    float y = row_anchor_[k];
                    line_list[i].push_back(std::pair<float, float>(x, y));
                }
            }
        }
    }

    for (int32_t i : {0, 3}) {
        if (sum_valid(valid_col, num_cls_col, num_lane_col, i) > num_cls_col / 8) {
            for (int32_t k = 0; k < num_cls_col; k++) {
                int32_t index = k * num_lane_col + i;
                if (valid_col[index] != 0) {
                    std::vector<float> pred_all_list;
                    std::vector<int32_t> all_ind_list;
                    for (int32_t all_ind = std::max(0, max_indices_col[index] - 1); all_ind <= std::min(num_grid_col - 1, max_indices_col[index] + 1); all_ind++) {
                        pred_all_list.push_back(loc_col[all_ind * num_cls_col * num_lane_col + index]);
                        all_ind_list.push_back(all_ind);
                    }
                    std::vector<float> pred_all_list_softmax(pred_all_list.size());
                    CommonHelper::SoftMaxFast(pred_all_list.data(), pred_all_list_softmax.data(), pred_all_list.size());
                    float out_temp = 0;
                    for (int32_t l = 0; l < pred_all_list.size(); l++) {
                        out_temp += pred_all_list_softmax[l] * all_ind_list[l];
                    }
                    float y = (out_temp + 0.5) / (num_grid_col - 1.0);
                    float x = col_anchor_[k];
                    line_list[i].push_back(std::pair<float, float>(x, y));
                }
            }
        }
    }

    return line_list;
}
    


int32_t LaneEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do crop, resize and color conversion here because some inference engine doesn't support these operations */
#if defined(USE_CULANE)
    int32_t crop_x = 0;
    int32_t crop_y = original_mat.rows * 0.4;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows * 0.5;
#else
    int32_t crop_x = 0;
    int32_t crop_y = original_mat.rows * 0.0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows * 1.0;
#endif
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;
    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    std::vector<float> loc_row(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    std::vector<float> loc_col(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());
    std::vector<float> exist_row(output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + output_tensor_info_list_[2].GetElementNum());
    std::vector<float> exist_col(output_tensor_info_list_[3].GetDataAsFloat(), output_tensor_info_list_[3].GetDataAsFloat() + output_tensor_info_list_[3].GetElementNum());
    std::vector<int32_t> loc_row_dims = output_tensor_info_list_[0].tensor_dims;
    std::vector<int32_t> loc_col_dims = output_tensor_info_list_[1].tensor_dims;
    std::vector<int32_t> exist_row_dims = output_tensor_info_list_[2].tensor_dims;
    std::vector<int32_t> exist_col_dims = output_tensor_info_list_[3].tensor_dims;

    auto line_list = Pred2Coords(loc_row, loc_row_dims, exist_row, exist_row_dims, loc_col, loc_col_dims, exist_col, exist_col_dims);

    /* todo: I'm not sure the following code correct */
    /* Adjust height scale : https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/c80276bc2fd67d02579b6eeb57a76cb5a905aa3d/demo.py#L88 */
    /* It looks the demo code run inference with height = model_input_height / 0.6 . but our code cannot do this. so after running inference with height = model_input_height, adjust y position */
    const float kInferenceHeight = input_tensor_info.GetHeight() / kCropRatio;
    for (auto& line : line_list) {
        for (auto& p : line) {
            p.second = ((p.second * kInferenceHeight) - (kInferenceHeight - input_tensor_info.GetHeight())) / input_tensor_info.GetHeight();
        }
    }

    std::vector<Line <int32_t>> line_ret_list;
    for (auto& line : line_list) {
        Line <int32_t> line_ret;
        for (auto& p : line) {
            std::pair<int32_t, int32_t> p_ret;
            p_ret.first = p.first * crop_w + crop_x;
            p_ret.second = p.second * crop_h + crop_y;
            line_ret.push_back(p_ret);
        }
        line_ret_list.push_back(line_ret);
    }
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.line_list = line_ret_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

#ifndef DEPTH_STEREO_ENGINE_H_
#define DEPTH_STEREO_ENGINE_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "inference_helper.h"


class DepthStereoEngine {
public:
    enum {
        kRetOk = 0,
        kRetErr = -1,
    };

    typedef struct Result_ {
        cv::Mat           image;                // [height, width, 1]. CV_32FC1
        struct crop_ {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;
            crop_() : x(0), y(0), w(0), h(0) {}
        } crop;
        double            time_pre_process;		// [msec]
        double            time_inference;		// [msec]
        double            time_post_process;	// [msec]
        Result_() : time_pre_process(0), time_inference(0), time_post_process(0)
        {}
    } Result;

public:
    DepthStereoEngine() {}
    ~DepthStereoEngine() {}
    int32_t Initialize(const std::string& work_dir, const int32_t num_threads);
    int32_t Finalize(void);
    int32_t Process(const cv::Mat& image_l, const cv::Mat& image_r, Result& result);
    float GetMaxDisparity(void);


private:
    std::unique_ptr<InferenceHelper> inference_helper_;
    std::vector<InputTensorInfo> input_tensor_info_list_;
    std::vector<OutputTensorInfo> output_tensor_info_list_;
};

#endif

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

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "image_processor.h"

/*** Macro ***/
#define IMAGE_NAME   RESOURCE_DIR"/parrot.jpg"
#define WORK_DIR     RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 10

/* https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.cpp */
/* modified by iwatake2222 */
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True";
}

int32_t main()
{
    /*** Initialize ***/
    /* Initialize image processor library */
    ImageProcessor::InputParam input_param = { WORK_DIR, 4 };
    ImageProcessor::Initialize(&input_param);

#ifdef SPEED_TEST_ONLY
    /* Read an input image */
    cv::Mat original_image = cv::imread(IMAGE_NAME);

    /* Call image processor library */
    ImageProcessor::OutputParam output_param;
    ImageProcessor::Process(&original_image, &output_param);

    cv::imshow("original_image", original_image);
    cv::waitKey(1);

    /*** (Optional) Measure inference time ***/
    double time_pre_process = 0;
    double time_inference = 0;
    double time_post_process = 0;
    const auto& t0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
        ImageProcessor::Process(&original_image, &output_param);
        time_pre_process += output_param.time_pre_process;
        time_inference += output_param.time_inference;
        time_post_process += output_param.time_post_process;
    }
    const auto& t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeSpan = t1 - t0;
    printf("PreProcessing time  = %.3lf [msec]\n", time_pre_process / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("Inference time  = %.3lf [msec]\n", time_inference / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("PostProcessing time  = %.3lf [msec]\n", time_post_process / LOOP_NUM_FOR_TIME_MEASUREMENT);
    printf("Total Image processing time  = %.3lf [msec]\n", timeSpan.count() * 1000.0 / LOOP_NUM_FOR_TIME_MEASUREMENT);
    cv::waitKey(-1);

#else
    /* Initialize camera */
    static cv::VideoCapture cap;
#if 0
    cap = cv::VideoCapture(1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
#else
    cap = cv::VideoCapture(gstreamer_pipeline(640, 480, 640, 480, 30, 2), cv::CAP_GSTREAMER);
#endif
    // cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('B', 'G', 'R', '3'));
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    while (1) {
        const auto& time_all_0 = std::chrono::steady_clock::now();
        /*** Read image ***/
        const auto& time_cap_0 = std::chrono::steady_clock::now();
        cv::Mat original_image;
        cap.read(original_image);
        const auto& time_cap_1 = std::chrono::steady_clock::now();

        /* Call image processor library */
        const auto& time_process_0 = std::chrono::steady_clock::now();
        ImageProcessor::OutputParam output_param;
        ImageProcessor::Process(&original_image, &output_param);
        const auto& time_process_1 = std::chrono::steady_clock::now();

        cv::imshow("test", original_image);
        if (cv::waitKey(1) == 'q') break;

        const auto& time_all_1 = std::chrono::steady_clock::now();
        printf("Total time = %.3lf [msec]\n", (time_all_1 - time_all_0).count() / 1000000.0);
        printf("Capture time = %.3lf [msec]\n", (time_cap_1 - time_cap_0).count() / 1000000.0);
        printf("Image processing time = %.3lf [msec]\n", (time_process_1 - time_process_0).count() / 1000000.0);
        printf("========\n");
    }

#endif

    /* Fianlize image processor library */
    ImageProcessor::Finalize();

    return 0;
}

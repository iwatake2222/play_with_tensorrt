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
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "Anime2SketchEngine.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/*** Global variable ***/
std::unique_ptr<Anime2SketchEngine> s_anime2SketchEngine;

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}


int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam)
{
	if (s_anime2SketchEngine) {
		PRINT_E("Already initialized\n");
		return -1;
	}

	s_anime2SketchEngine.reset(new Anime2SketchEngine());
	if (s_anime2SketchEngine->initialize(inputParam->workDir, inputParam->numThreads) != Anime2SketchEngine::RET_OK) {
		s_anime2SketchEngine->finalize();
		s_anime2SketchEngine.reset();
		return -1;
	}
	return 0;
}

int32_t ImageProcessor_finalize(void)
{
	if (!s_anime2SketchEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	if (s_anime2SketchEngine->finalize() != Anime2SketchEngine::RET_OK) {
		return -1;
	}

	return 0;
}


int32_t ImageProcessor_command(int32_t cmd)
{
	if (!s_anime2SketchEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	switch (cmd) {
	case 0:
	default:
		PRINT_E("command(%d) is not supported\n", cmd);
		return -1;
	}
}


int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam)
{
	if (!s_anime2SketchEngine) {
		PRINT_E("Not initialized\n");
		return -1;
	}

	cv::Mat& originalMat = *mat;
	Anime2SketchEngine::RESULT styleTransferResult;
	s_anime2SketchEngine->invoke(originalMat, styleTransferResult);

	/* Return the results */
	originalMat = styleTransferResult.image;
	outputParam->timePreProcess = styleTransferResult.timePreProcess;
	outputParam->timeInference = styleTransferResult.timeInference;
	outputParam->timePostProcess = styleTransferResult.timePostProcess;

	return 0;
}


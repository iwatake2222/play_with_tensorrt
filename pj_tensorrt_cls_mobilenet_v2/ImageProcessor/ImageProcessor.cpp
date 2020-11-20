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
#include "CommonHelper.h"
#include "InferenceHelper.h"
#include "ImageProcessor.h"

/*** Macro ***/
#define TAG "ImageProcessor"
#define PRINT   COMMON_HELPER_PRINT
#define PRINT_E COMMON_HELPER_PRINT_E

#define CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT_E("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/* Model parameters */
#define MODEL_NAME   "mobilenetv2-7.onnx"
// #define MODEL_NAME   "mobilenetv2-7.trt"
#define LABEL_NAME   "synset.txt"


/*** Global variable ***/
static std::vector<std::string> s_labels;
static InferenceHelper *s_inferenceHelper;
static std::vector<TensorInfo> s_inputTensorList;
static std::vector<TensorInfo> s_outputTensorList;

/*** Function ***/
static cv::Scalar createCvColor(int32_t b, int32_t g, int32_t r) {
#ifdef CV_COLOR_IS_RGB
	return cv::Scalar(r, g, b);
#else
	return cv::Scalar(b, g, r);
#endif
}

static void readLabel(const char* filename, std::vector<std::string> & labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		PRINT(TAG, "failed to read %s\n", filename);
		return;
	}
	std::string str;
	while (getline(ifs, str)) {
		labels.push_back(str);
	}
}


int32_t ImageProcessor_initialize(const INPUT_PARAM *inputParam)
{
	s_inferenceHelper = InferenceHelper::create(InferenceHelper::TENSOR_RT);

	std::string modelFilename = std::string(inputParam->workDir) + "/model/" + MODEL_NAME;
	std::string labelFilename = std::string(inputParam->workDir) + "/model/" + LABEL_NAME;
	
	TensorInfo inputTensorInfo;
	TensorInfo outputTensorInfo;

	inputTensorInfo.name = "data";
	inputTensorInfo.type = TensorInfo::TENSOR_TYPE_FP32;
	inputTensorInfo.dims.batch = 1;
	inputTensorInfo.dims.width = 224;
	inputTensorInfo.dims.height = 224;
	inputTensorInfo.dims.channel = 3;
	inputTensorInfo.normalize.mean[0] = 0.485f;   	/* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
	inputTensorInfo.normalize.mean[1] = 0.456f;
	inputTensorInfo.normalize.mean[2] = 0.406f;
	inputTensorInfo.normalize.norm[0] = 0.229f;
	inputTensorInfo.normalize.norm[1] = 0.224f;
	inputTensorInfo.normalize.norm[2] = 0.225f;
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif

	s_inputTensorList.push_back(inputTensorInfo);

	outputTensorInfo.name = "mobilenetv20_output_flatten0_reshape0";
	outputTensorInfo.type = TensorInfo::TENSOR_TYPE_FP32;
	s_outputTensorList.push_back(outputTensorInfo);

	s_inferenceHelper->setNumThread(4);
	s_inferenceHelper->initialize(modelFilename.c_str(), s_inputTensorList, s_outputTensorList);

	/* read label */
	readLabel(labelFilename.c_str(), s_labels);

	return 0;
}

int32_t ImageProcessor_command(int32_t cmd)
{
	switch (cmd) {
	case 0:
	default:
		PRINT(TAG, "command(%d) is not supported\n", cmd);
		return -1;
	}
}


int32_t ImageProcessor_process(cv::Mat *mat, OUTPUT_PARAM *outputParam)
{
	TensorInfo& inputTensor = s_inputTensorList[0];

	/*** PreProcess ***/
	const auto& tPreProcess0 = std::chrono::steady_clock::now();
	inputTensor.preProcess(mat->data, mat->cols, mat->rows);
	const auto& tPreProcess1 = std::chrono::steady_clock::now();

	/*** Inference ***/
	const auto& tInference0 = std::chrono::steady_clock::now();
	s_inferenceHelper->invoke(s_inputTensorList, s_outputTensorList);
	const auto& tInference1 = std::chrono::steady_clock::now();

	/*** PostProcess ***/
	const auto& tPostProcess0 = std::chrono::steady_clock::now();
	/* Retrieve the result */
	std::vector<float_t> outputScoreList;
	outputScoreList.resize(s_outputTensorList[0].dims.width * s_outputTensorList[0].dims.height * s_outputTensorList[0].dims.channel);
	const float_t* valFloat = s_outputTensorList[0].getDataAsFloat();
	for (int32_t i = 0; i < (int32_t)outputScoreList.size(); i++) {
		outputScoreList[i] = valFloat[i];
	}

	/* Find the max score */
	int32_t maxIndex = (int32_t)(std::max_element(outputScoreList.begin(), outputScoreList.end()) - outputScoreList.begin());
	auto maxScore = *std::max_element(outputScoreList.begin(), outputScoreList.end());
	PRINT(TAG, "Result = %s (%d) (%.3f)\n", s_labels[maxIndex].c_str(), maxIndex, maxScore);
	const auto& tPostProcess1 = std::chrono::steady_clock::now();

	/* Draw the result */
	std::string resultStr;
	resultStr = "Result:" + s_labels[maxIndex] + " (score = " + std::to_string(maxScore) + ")";
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 0, 0), 3);
	cv::putText(*mat, resultStr, cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 1, createCvColor(0, 255, 0), 1);

	/* Return the results */
	outputParam->classId = maxIndex;
	snprintf(outputParam->label, sizeof(outputParam->label), "%s", s_labels[maxIndex].c_str());
	outputParam->score = maxScore;
	outputParam->timePreProcess = static_cast<std::chrono::duration<double_t>>(tPreProcess1 - tPreProcess0).count() * 1000.0;
	outputParam->timeInference = static_cast<std::chrono::duration<double_t>>(tInference1 - tInference0).count() * 1000.0;
	outputParam->timePostProcess = static_cast<std::chrono::duration<double_t>>(tPostProcess1 - tPostProcess0).count() * 1000.0;

	return 0;
}


int32_t ImageProcessor_finalize(void)
{
	s_inferenceHelper->finalize();
	delete s_inferenceHelper;

	return 0;
}

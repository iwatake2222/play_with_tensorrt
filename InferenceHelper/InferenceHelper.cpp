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

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelper.h"

#ifdef INFERENCE_HELPER_ENABLE_OPENCV
#include "InferenceHelperOpenCV.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
#include "InferenceHelperTensorRt.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
#include "InferenceHelperTensorflowLite.h"
#endif

/*** Macro ***/
#define TAG "InferenceHelper"
#define PRINT   COMMON_HELPER_PRINT
#define PRINT_E COMMON_HELPER_PRINT_E


InferenceHelper* InferenceHelper::create(const InferenceHelper::HELPER_TYPE type)
{
	InferenceHelper* p = NULL;
	switch (type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
	case TENSOR_RT:
		PRINT(TAG, "Use OpenCV \n");
		p = new InferenceHelperOpenCV();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
	case TENSOR_RT:
		PRINT(TAG, "Use TensorRT \n");
		p = new InferenceHelperTensorRt();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
	case TENSORFLOW_LITE:
		PRINT(TAG, "Use TensorflowLite\n");
		p = new InferenceHelperTensorflowLite();
		break;
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	case TENSORFLOW_LITE_EDGETPU:
		PRINT(TAG, "Use TensorflowLite EdgeTPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
	case TENSORFLOW_LITE_GPU:
		PRINT(TAG, "Use TensorflowLite GPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
	case TENSORFLOW_LITE_XNNPACK:
		PRINT(TAG, "Use TensorflowLite XNNPACK Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#endif
	default:
		PRINT_E(TAG, "not supported\n");
		exit(1);
		break;
	}
	p->m_helperType = type;
	return p;
}


TensorInfo::TensorInfo()
{
	name = "";
	type = TENSOR_TYPE_NONE;
	data = nullptr;;
	dims.batch = 1;
	dims.width = 1;
	dims.height = 1;
	dims.channel = 1;
	quant.scale = 0;
	quant.zeroPoint = 0;
	for (int32_t i = 0; i < 3; i++) {
		normalize.mean[i] = 0.0f;
		normalize.norm[i] = 1.0f;
	}
	m_dataFp32 = nullptr;
}


TensorInfo::~TensorInfo()
{
	if (m_dataFp32 != nullptr) {
		delete[] m_dataFp32;
	}
}

float_t* TensorInfo::getDataAsFloat()
{
	if (type == TENSOR_TYPE_UINT8) {
		int32_t dataNum = 1;
		dataNum = dims.batch * dims.channel * dims.height * dims.width;
		if (m_dataFp32 == nullptr) {
			m_dataFp32 = new float_t[dataNum];
		}
		for (int32_t i = 0; i < dataNum; i++) {
			const uint8_t* valUint8 = (uint8_t*)data;
			float_t valFloat = (valUint8[i] - quant.zeroPoint) * quant.scale;
			m_dataFp32[i] = valFloat;
		}
		return m_dataFp32;
	} else if (type == TENSOR_TYPE_FP32) {
		return (float_t*)data;
	} else {
		PRINT_E(TAG, "invalid call");
		return nullptr;
	}

	return nullptr;
}

#include <opencv2/opencv.hpp>
int32_t TensorInfo::preProcess(void *srcPixel, int32_t srcWidth, int32_t srcHeight)
{
	cv::Mat imgSrc = cv::Mat(cv::Size(srcWidth, srcHeight), CV_8UC3, srcPixel);	/* original input image data */
	cv::Mat imgInput;		/* memory is reserved in InferenceHelper */
	cv::Size sizeInput = cv::Size(dims.width, dims.height);
	if (type == TensorInfo::TENSOR_TYPE_FP32) {
		/* Resize the input image */
		cv::Mat imgResized;
		cv::resize(imgSrc, imgResized, sizeInput);
		/* Normalize the input image */
		if (dims.channel == 3) {
			imgInput = cv::Mat(cv::Size(dims.width, dims.height), CV_32FC3, data);
#if 1
			imgResized.convertTo(imgInput, CV_32FC3);
			cv::multiply(imgInput, cv::Scalar(cv::Vec<float_t, 3>(normalize.norm)), imgInput);
			cv::subtract(imgInput, cv::Scalar(cv::Vec<float_t, 3>(normalize.mean)), imgInput);
#else
			imgResized.convertTo(imgInput, CV_32FC3, 1.0 / 255);
			cv::subtract(imgInput, cv::Scalar(cv::Vec<float_t, 3>(normalize.mean)), imgInput);
			cv::divide(imgInput, cv::Scalar(cv::Vec<float_t, 3>(normalize.norm)), imgInput);
#endif
		} else {
			cv::cvtColor(imgResized, imgResized, cv::COLOR_BGR2GRAY);
			imgInput = cv::Mat(cv::Size(dims.width, dims.height), CV_32FC1, data);
#if 1
			imgResized.convertTo(imgInput, CV_32FC1);
			cv::multiply(imgInput, cv::Scalar(cv::Vec<float_t, 1>(normalize.norm)), imgInput);
			cv::subtract(imgInput, cv::Scalar(cv::Vec<float_t, 1>(normalize.mean)), imgInput);
#else
			imgResized.convertTo(imgInput, CV_32FC1, 1.0 / 255);
			cv::subtract(imgInput, cv::Scalar(cv::Vec<float_t, 1>(normalize.mean)), imgInput);
			cv::divide(imgInput, cv::Scalar(cv::Vec<float_t, 1>(normalize.norm)), imgInput);
#endif
		}
	} else if (type == TensorInfo::TENSOR_TYPE_UINT8) {
		if (dims.channel == 3) {
			imgInput = cv::Mat(cv::Size(dims.width, dims.height), CV_8UC3, data);
			cv::resize(imgSrc, imgInput, sizeInput);
		} else {
			imgInput = cv::Mat(cv::Size(dims.width, dims.height), CV_8UC1, data);
			cv::Mat imgResized;
			cv::resize(imgSrc, imgResized, sizeInput);
			cv::cvtColor(imgResized, imgInput, cv::COLOR_BGR2GRAY);
		}
	}
	
	if (data != imgInput.data) {
		PRINT_E(TAG, "Error at %s:%d\n", __FILE__, __LINE__);
		return RET_ERR;
	}
	return RET_OK;
}

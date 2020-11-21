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
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


InferenceHelper* InferenceHelper::create(const InferenceHelper::HELPER_TYPE type)
{
	InferenceHelper* p = NULL;
	switch (type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
	case OPEN_CV:
	case OPEN_CV_GPU:
		PRINT("Use OpenCV \n");
		p = new InferenceHelperOpenCV();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
	case TENSOR_RT:
		PRINT("Use TensorRT \n");
		p = new InferenceHelperTensorRt();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
	case TENSORFLOW_LITE:
		PRINT("Use TensorflowLite\n");
		p = new InferenceHelperTensorflowLite();
		break;
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
	case TENSORFLOW_LITE_EDGETPU:
		PRINT("Use TensorflowLite EdgeTPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
	case TENSORFLOW_LITE_GPU:
		PRINT("Use TensorflowLite GPU Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
	case TENSORFLOW_LITE_XNNPACK:
		PRINT("Use TensorflowLite XNNPACK Delegate\n");
		p = new InferenceHelperTensorflowLite();
		break;
#endif
#endif
	default:
		PRINT_E("not supported\n");
		exit(1);
		break;
	}
	p->m_helperType = type;
	return p;
}

#if 1
#include <opencv2/opencv.hpp>
void InferenceHelper::preProcessByOpenCV(const InputTensorInfo& inputTensor, bool isNCHW, cv::Mat& imgBlob)
{
	/* Generate mat from original data */
	cv::Mat imgSrc = cv::Mat(cv::Size(inputTensor.imageInfo.width, inputTensor.imageInfo.height), (inputTensor.imageInfo.channel == 3) ? CV_8UC3 : CV_8UC1, inputTensor.data);

	/* Crop image */
	if (inputTensor.imageInfo.width == inputTensor.imageInfo.cropWidth && inputTensor.imageInfo.height == inputTensor.imageInfo.cropHeight) {
		/* do nothing */
	} else {
		imgSrc = imgSrc(cv::Rect(inputTensor.imageInfo.cropX, inputTensor.imageInfo.cropY, inputTensor.imageInfo.cropWidth, inputTensor.imageInfo.cropHeight));
	}

	/* Resize image */
	if (inputTensor.imageInfo.cropWidth == inputTensor.tensorDims.width && inputTensor.imageInfo.cropHeight == inputTensor.tensorDims.height) {
		/* do nothing */
	} else {
		cv::resize(imgSrc, imgSrc, cv::Size(inputTensor.tensorDims.width, inputTensor.tensorDims.height));
	}

	/* Convert color type */
	if (inputTensor.imageInfo.channel == inputTensor.tensorDims.channel) {
		/* do nothing */
	} else if (inputTensor.imageInfo.channel == 3 && inputTensor.tensorDims.channel == 1) {
		cv::cvtColor(imgSrc, imgSrc, cv::COLOR_BGR2GRAY);
	} else if (inputTensor.imageInfo.channel == 1 && inputTensor.tensorDims.channel == 3) {
		cv::cvtColor(imgSrc, imgSrc, cv::COLOR_GRAY2BGR);
	}

	if (inputTensor.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
		/* Normalize image */
		if (inputTensor.tensorDims.channel == 3) {
#if 1
			imgSrc.convertTo(imgSrc, CV_32FC3);
			cv::multiply(imgSrc, cv::Scalar(cv::Vec<float_t, 3>(inputTensor.normalize.norm)), imgSrc);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float_t, 3>(inputTensor.normalize.mean)), imgSrc);
#else
			imgSrc.convertTo(imgSrc, CV_32FC3, 1.0 / 255);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float_t, 3>(inputTensor.normalize.mean)), imgSrc);
			cv::divide(imgSrc, cv::Scalar(cv::Vec<float_t, 3>(inputTensor.normalize.norm)), imgSrc);
#endif
		} else {
#if 1
			imgSrc.convertTo(imgSrc, CV_32FC1);
			cv::multiply(imgSrc, cv::Scalar(cv::Vec<float_t, 1>(inputTensor.normalize.norm)), imgSrc);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float_t, 1>(inputTensor.normalize.mean)), imgSrc);
#else
			imgSrc.convertTo(imgSrc, CV_32FC1, 1.0 / 255);
			cv::subtract(imgSrc, cv::Scalar(cv::Vec<float_t, 1>(inputTensor.normalize.mean)), imgSrc);
			cv::divide(imgSrc, cv::Scalar(cv::Vec<float_t, 1>(inputTensor.normalize.norm)), imgSrc);
#endif
		}
	} else {
		/* do nothing */
	}

	if (isNCHW) {
		/* Convert to 4-dimensional Mat in NCHW */
		imgSrc = cv::dnn::blobFromImage(imgSrc);
	}

	imgBlob = imgSrc;
	//memcpy(blobData, imgSrc.data, imgSrc.cols * imgSrc.rows * imgSrc.channels());

}

#else 
/* For the environment where OpenCV is not supported */
void InferenceHelper::preProcessByOpenCV(const InputTensorInfo& inputTensor, bool isNCHW, cv::Mat& imgBlob)
{
}
#endif

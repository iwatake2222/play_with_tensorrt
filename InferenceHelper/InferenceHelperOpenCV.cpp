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
#include <opencv2/dnn.hpp>

/* for My modules */
#include "CommonHelper.h"
#include "InferenceHelperOpenCV.h"

/*** Macro ***/
#define TAG "InferenceHelperOpenCV"
#define PRINT   COMMON_HELPER_PRINT
#define PRINT_E COMMON_HELPER_PRINT_E


/*** Function ***/
InferenceHelperOpenCV::InferenceHelperOpenCV()
{
}

InferenceHelperOpenCV::~InferenceHelperOpenCV()
{
}

int32_t InferenceHelperOpenCV::setNumThread(const int32_t numThread)
{
	cv::setNumThreads(numThread);
	return RET_OK;
}

int32_t InferenceHelperOpenCV::initialize(const std::string& modelFilename, std::vector<TensorInfo>& inputTensorInfoList, std::vector<TensorInfo>& outputTensorInfoList)
{
	/*** Create network ***/
	m_net = cv::dnn::readNetFromONNX(modelFilename);
	if (m_net.empty() == true) {
		PRINT_E(TAG, "Failed to create inference engine (%s)\n", modelFilename.c_str());
		return RET_ERR;
	}
	
	m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	// m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
	// m_net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);


	/*** Check tensor information  ***/
	/* Only one input tensor is supported. The input tensor name changes for some reasons. */
	if (inputTensorInfoList.size() != 1) {
		PRINT_E(TAG, "Invalid input tensor num (%zu)\n", inputTensorInfoList.size());
	}

	/* Reserve input memory */
	TensorInfo& inputTensor = inputTensorInfoList[0];
	//inputTensor.name = m_net.getLayer(0)->name;
	if (inputTensor.type == TensorInfo::TENSOR_TYPE_FP32) {
		m_imgInput = cv::Mat(cv::Size(inputTensor.dims.width, inputTensor.dims.height), (inputTensor.dims.channel == 3) ? CV_32FC3 : CV_32FC1);
	} else if (inputTensor.type == TensorInfo::TENSOR_TYPE_UINT8) {
		m_imgInput = cv::Mat(cv::Size(inputTensor.dims.width, inputTensor.dims.height), (inputTensor.dims.channel == 3) ? CV_8UC3 : CV_8UC1);
	} else {
		PRINT_E(TAG, "Unsupported tensor type (%d)\n", inputTensor.type);
		return RET_ERR;
	}	
	inputTensor.data = m_imgInput.data;		/* The user sets input image into this memory */

	/* Check output tensor name */
	for (const auto& outputTensorInfo : outputTensorInfoList) {
		bool isFound = false;
		for (const auto& layerNames : m_net.getLayerNames()) {
			if (outputTensorInfo.name == layerNames) {
				isFound = true;
				break;
			}
		}
		if (isFound == false) {
			PRINT_E(TAG, "Output name (%s) not found\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		}
	}

	return RET_OK;
};


int32_t InferenceHelperOpenCV::finalize(void)
{
	return RET_ERR;
}

int32_t InferenceHelperOpenCV::invoke(const std::vector<TensorInfo>& inputTensorInfoList, std::vector<TensorInfo>& outputTensorInfoList)
{
	/*** Convert to 4-dimensional Mat in NCHW ***/
	cv::Mat imgModelInput = cv::dnn::blobFromImage(m_imgInput);
	m_net.setInput(imgModelInput);

	/*** Run inference ***/
	std::vector<cv::String> outNameList;
	for (const auto& outputTensorInfo : outputTensorInfoList) {
		outNameList.push_back(outputTensorInfo.name);
	}
	m_outMatList.clear();
	m_net.forward(m_outMatList, outNameList);

	/*** Retrieve the results ***/
	if (m_outMatList.size() != outputTensorInfoList.size()) {
		PRINT_E(TAG, "Unexpected output tensor num (%zu)\n", m_outMatList.size());
		return RET_ERR;
	}
	for (int32_t i = 0; i < m_outMatList.size(); i++) {
		outputTensorInfoList[i].data = m_outMatList[i].data;
		outputTensorInfoList[i].dims.batch = 1;
		outputTensorInfoList[i].dims.width = m_outMatList[i].cols;
		outputTensorInfoList[i].dims.height = m_outMatList[i].rows;
		outputTensorInfoList[i].dims.channel = m_outMatList[i].channels();
	}

	return RET_OK;
}

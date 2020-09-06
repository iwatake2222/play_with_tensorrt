/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>

/* for TensorRT */
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "TensorRT/common.h"

#include "InferenceHelperTensorRt.h"

/*** Settings ***/
// #define USE_FP16
#define USE_INT8

#define OPT_MAX_WORK_SPACE_SIZE (1 << 30)
#define OPT_AVG_TIMING_ITERATIONS 8
#define OPT_MIN_TIMING_ITERATIONS 4

#ifdef USE_INT8
/* ★ Modify the following (use the same parameter as the model. Also, ppm must be the same size but not normalized.) */
#define CAL_DIR        "/home/iwatake/play_with_tensorrt/InferenceHelper/TensorRT/calibration/sample_ppm"
#define CAL_LIST_FILE  "list.txt"
#define CAL_INPUT_NAME "data"
#define CAL_BATCH_SIZE 10
#define CAL_NB_BATCHES 2
#define CAL_IMAGE_C    3
#define CAL_IMAGE_H    224
#define CAL_IMAGE_W    224
#define CAL_SCALE      (1.0 / 255.0)
#define CAL_BIAS       (0.0)

/* include BatchStream.h after defining parameters */
#include "TensorRT/BatchStream.h"
#include "TensorRT/EntropyCalibrator.h"
#endif

/*** Macro ***/
#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define _PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define _PRINT(...) printf(__VA_ARGS__)
#endif
#define PRINT(...) _PRINT("[InferenceHelperTensorRt] " __VA_ARGS__)

#define LOCAL_CHECK(x)                              \
  if (!(x)) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

/*** Function ***/
InferenceHelperTensorRt::InferenceHelperTensorRt()
{
}

int InferenceHelperTensorRt::initialize(const char *modelFilename, int numThreads, std::vector<std::pair<const char*, const void*>> customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return -1;
}

int InferenceHelperTensorRt::initialize(const char *modelFilename, int numThreads)
{
	/* check model format type */
	bool isTrtModel = false;
	bool isOnnxModel = false;
	bool isUffModel = false;
	std::string trtModelFilename = std::string(modelFilename);
	transform (trtModelFilename.begin(), trtModelFilename.end(), trtModelFilename.begin(), tolower);
	if (trtModelFilename.find(".onnx") != std::string::npos) {
		isOnnxModel = true;
		trtModelFilename = trtModelFilename.replace(trtModelFilename.find(".onnx"), std::string(".onnx").length(), ".trt\0");
	} else if (trtModelFilename.find(".trt") != std::string::npos) {
		isTrtModel = true;
	} else {
		PRINT("[ERROR] unsupoprted file format: %s\n", modelFilename);
	}

	/* create runtime and engine from model file */
	if (isTrtModel) {
		std::string buffer;
		std::ifstream stream(modelFilename, std::ios::binary);
		if (stream) {
			stream >> std::noskipws;
			copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
		}
		m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size(), NULL), samplesCommon::InferDeleter());
		stream.close();
		LOCAL_CHECK(m_engine != NULL);
		m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), samplesCommon::InferDeleter());
		LOCAL_CHECK(m_context != NULL);
	} else {
		/* create a TensorRT model from another format */
		auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
#if 0
		/* For older version of JetPack */
		auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetwork(), samplesCommon::InferDeleter());
#else
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch), samplesCommon::InferDeleter());
#endif
		auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), samplesCommon::InferDeleter());

		auto parserOnnx = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
		if (isOnnxModel) {
			if (!parserOnnx->parseFromFile(modelFilename, (int)nvinfer1::ILogger::Severity::kWARNING)) {
				PRINT("[ERROR] failed to parse onnx file");
				return -1;
			}
		}

		builder->setMaxBatchSize(1);
		config->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);
		config->setAvgTimingIterations(OPT_AVG_TIMING_ITERATIONS);
		config->setMinTimingIterations(OPT_MIN_TIMING_ITERATIONS) ;

#if defined(USE_FP16)
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		std::vector<std::string> dataDirs;
		dataDirs.push_back(CAL_DIR);
		nvinfer1::DimsNCHW imageDims{CAL_BATCH_SIZE, CAL_IMAGE_C, CAL_IMAGE_H, CAL_IMAGE_W};
		BatchStream calibrationStream(CAL_BATCH_SIZE, CAL_NB_BATCHES, imageDims, CAL_LIST_FILE, dataDirs);
		auto calibrator = std::unique_ptr<nvinfer1::IInt8Calibrator>(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "my_model", CAL_INPUT_NAME));
		config->setInt8Calibrator(calibrator.get());
#endif 

		m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
		LOCAL_CHECK(m_engine != NULL);
		m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext(), samplesCommon::InferDeleter());
		LOCAL_CHECK(m_context != NULL);
#if 1
		/* save serialized model for next time */
		nvinfer1::IHostMemory* trtModelStream = m_engine->serialize();
		std::ofstream ofs(std::string(trtModelFilename), std::ios::out | std::ios::binary);
		ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
		ofs.close();
		trtModelStream->destroy();
#endif
	}

	/* Allocate host/device buffers beforehand */
	allocateBuffers();

	return 0;
}

int InferenceHelperTensorRt::finalize(void)
{
	int numOfInOut = m_engine->getNbBindings();
	for (int i = 0; i < numOfInOut; i++) {
		const auto dataType = m_engine->getBindingDataType(i);
		switch (dataType) {
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kHALF:
		case nvinfer1::DataType::kINT32:
			delete[] (float*)(m_bufferListCPUReserved[i].first);
			break;
		case nvinfer1::DataType::kINT8:
			delete[] (int*)(m_bufferListCPUReserved[i].first);
			break;
		default:
			LOCAL_CHECK(false);
		}
	}

	for (auto p : m_bufferListGPU) {
		cudaFree(p);
	}

	return 0;
}

int InferenceHelperTensorRt::invoke(void)
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (int i = 0; i < (int)m_bufferListCPU.size(); i++) {
		if (m_engine->bindingIsInput(i)) {
			cudaMemcpyAsync(m_bufferListGPU[i], m_bufferListCPU[i].first, m_bufferListCPU[i].second, cudaMemcpyHostToDevice, stream);
		}
	}
	m_context->enqueue(1, &m_bufferListGPU[0], stream, NULL);
	for (int i = 0; i < (int)m_bufferListCPU.size(); i++) {
		if (!m_engine->bindingIsInput(i)) {
			cudaMemcpyAsync(m_bufferListCPU[i].first, m_bufferListGPU[i], m_bufferListCPU[i].second, cudaMemcpyDeviceToHost, stream);
		}
	}
	cudaStreamSynchronize(stream);

	cudaStreamDestroy(stream);

	return 0;
}


int InferenceHelperTensorRt::getTensorByName(const char *name, TensorInfo *tensorInfo)
{
	int index = m_engine->getBindingIndex(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return getTensorByIndex(index, tensorInfo);
}

int InferenceHelperTensorRt::getTensorByIndex(const int index, TensorInfo *tensorInfo)
{
	tensorInfo->index = index;

	const auto dims = m_engine->getBindingDimensions(index);
	for (int i = 0; i < dims.nbDims; i++) {
		tensorInfo->dims.push_back(dims.d[i]);
	}

	const auto dataType = m_engine->getBindingDataType(index);
	switch (dataType) {
	case nvinfer1::DataType::kFLOAT:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_FP32;
		break;
	case nvinfer1::DataType::kHALF:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_FP32;
		break;
	case nvinfer1::DataType::kINT8:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_UINT8;
		tensorInfo->quant.scale = 1.0;			// todo
		tensorInfo->quant.zeroPoint = 0.0;
		break;
	case nvinfer1::DataType::kINT32:
		tensorInfo->type = TensorInfo::TENSOR_TYPE_INT32;
		break;
	default:
		LOCAL_CHECK(false);
	}
	tensorInfo->data = m_bufferListCPU[index].first;
	return 0;
}

int InferenceHelperTensorRt::setBufferToTensorByName(const char *name, void *data, const int dataSize)
{
	PRINT("[WARNING] This method is not tested\n");
	int index = m_engine->getBindingIndex(name);
	if (index == -1) {
		PRINT("invalid name: %s\n", name);
		return -1;
	}

	return setBufferToTensorByIndex(index, data, dataSize);
}

int InferenceHelperTensorRt::setBufferToTensorByIndex(const int index, void *data, const int dataSize)
{
	PRINT("[WARNING] This method is not tested\n");
	LOCAL_CHECK(m_bufferListCPU[index].second == dataSize);
	m_bufferListCPU[index].first = data;
	return 0;
}


void InferenceHelperTensorRt::allocateBuffers()
{
	int numOfInOut = m_engine->getNbBindings();
	PRINT("numOfInOut = %d\n", numOfInOut);

	for (int i = 0; i < numOfInOut; i++) {
		PRINT("tensor[%d]->name: %s\n", i, m_engine->getBindingName(i));
		PRINT("  is input = %d\n", m_engine->bindingIsInput(i));
		int dataSize = 1;
		const auto dims = m_engine->getBindingDimensions(i);
		for (int i = 0; i < dims.nbDims; i++) {
			PRINT("  dims.d[%d] = %d\n", i, dims.d[i]);
			dataSize *= dims.d[i];
		}
		const auto dataType = m_engine->getBindingDataType(i);
		PRINT("  dataType = %d\n", (int)dataType);

		void *bufferCPU = NULL;
		void* bufferGPU = NULL;
		switch (dataType) {
		case nvinfer1::DataType::kFLOAT:
		case nvinfer1::DataType::kHALF:
		case nvinfer1::DataType::kINT32:
			bufferCPU = new float[dataSize];
			LOCAL_CHECK(bufferCPU);
			m_bufferListCPUReserved.push_back(std::pair<void*,int>(bufferCPU, dataSize * 4));
			cudaMalloc(&bufferGPU, dataSize * 4);
			LOCAL_CHECK(bufferGPU);
			m_bufferListGPU.push_back(bufferGPU);
			break;
		case nvinfer1::DataType::kINT8:
			bufferCPU = new int[dataSize];
			LOCAL_CHECK(bufferCPU);
			m_bufferListCPUReserved.push_back(std::pair<void*,int>(bufferCPU, dataSize * 1));
			cudaMalloc(&bufferGPU, dataSize * 1);
			LOCAL_CHECK(bufferGPU);
			m_bufferListGPU.push_back(bufferGPU);
			break;
		default:
			LOCAL_CHECK(false);
		}
	}
	m_bufferListCPU = m_bufferListCPUReserved;
}

#ifndef INFERENCE_HELPER_
#define INFERENCE_HELPER_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

class TensorInfo {
public:
	enum {
		TENSOR_TYPE_NONE,
		TENSOR_TYPE_UINT8,
		TENSOR_TYPE_FP32,
		TENSOR_TYPE_INT32,
		TENSOR_TYPE_INT64,
	};

public:
	TensorInfo() {
		name = "";
		id = -1;
		tensorType = TENSOR_TYPE_NONE;
		tensorDims.batch = 1;
		tensorDims.width = 1;
		tensorDims.height = 1;
		tensorDims.channel = 1;
	}
	~TensorInfo() {}

public:
	std::string name;
	int32_t     id;
	int32_t     tensorType;
	struct {
		int32_t batch;   // 0
		int32_t width;   // 1
		int32_t height;  // 2
		int32_t channel; // 3
	} tensorDims;
};

class InputTensorInfo : public TensorInfo {
public:
	enum {
		DATA_TYPE_IMAGE_BGR,
		DATA_TYPE_IMAGE_RGB,
		DATA_TYPE_BLOB_NHWC,
		DATA_TYPE_BLOB_NCHW,
	};

public:
	InputTensorInfo() {
		data = nullptr;
		dataType = DATA_TYPE_IMAGE_BGR;
		swapColor = false;
		imageInfo.width = 1;
		imageInfo.height = 1;
		imageInfo.channel = 1;
		imageInfo.cropX = 1;
		imageInfo.cropY = 1;
		imageInfo.cropWidth = 1;
		imageInfo.cropHeight = 1;
		for (int32_t i = 0; i < 3; i++) {
			normalize.mean[i] = 0.0f;
			normalize.norm[i] = 1.0f;
		}
	}
	~InputTensorInfo() {}

public:
	void* data;
	int32_t dataType;
	bool swapColor;

	struct {
		int32_t width;
		int32_t height;
		int32_t channel;
		int32_t cropX;
		int32_t cropY;
		int32_t cropWidth;
		int32_t cropHeight;
	} imageInfo;              // used when dataType == DATA_TYPE_IMAGE

	struct {
		float_t mean[3];
		float_t norm[3];
	} normalize;              // used when dataType == DATA_TYPE_IMAGE
};


class OutputTensorInfo : public TensorInfo {
public:
	OutputTensorInfo() {
		data = nullptr;
		quant.scale = 0;
		quant.zeroPoint = 0;
		m_dataFp32 = nullptr;
	}
	~OutputTensorInfo() {
		if (m_dataFp32 != nullptr) {
			delete[] m_dataFp32;
		}
	}
	const float_t* getDataAsFloat() {
		if (tensorType == TENSOR_TYPE_UINT8) {
			int32_t dataNum = 1;
			dataNum = tensorDims.batch * tensorDims.channel * tensorDims.height * tensorDims.width;
			if (m_dataFp32 == nullptr) {
				m_dataFp32 = new float_t[dataNum];
			}
			for (int32_t i = 0; i < dataNum; i++) {
				const uint8_t* valUint8 = (uint8_t*)data;
				float_t valFloat = (valUint8[i] - quant.zeroPoint) * quant.scale;
				m_dataFp32[i] = valFloat;
			}
			return m_dataFp32;
		} else if (tensorType == TENSOR_TYPE_FP32) {
			return (float_t*)data;
		} else {
			return nullptr;
		}
	}

public:
	void* data;
	struct {
		float_t scale;
		uint8_t zeroPoint;
	} quant;

private:
	float_t* m_dataFp32;
};


namespace cv {
	class Mat;
};

class InferenceHelper {
public:
	enum {
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef enum {
		TENSOR_RT,
		TENSORFLOW_LITE,
		TENSORFLOW_LITE_EDGETPU,
		TENSORFLOW_LITE_GPU,
		TENSORFLOW_LITE_XNNPACK,
		NCNN,
		NCNN_VULKAN,
		MNN,
		OPEN_CV,
		OPEN_CV_GPU,
	} HELPER_TYPE;

public:
	static InferenceHelper* create(const HELPER_TYPE typeFw);
	static void preProcessByOpenCV(const InputTensorInfo& inputTensor, bool isNCHW, cv::Mat& imgBlob);	// use this if the selected inference engine doesn't support pre-process

public:
	virtual ~InferenceHelper() {}
	virtual int32_t setNumThread(const int32_t numThread) = 0;
	virtual int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) = 0;
	virtual int32_t finalize(void) = 0;
	virtual int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) = 0;
	virtual int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) = 0;

protected:
	HELPER_TYPE m_helperType;
};

#endif

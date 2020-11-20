
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
		RET_OK = 0,
		RET_ERR = -1,
	};

	typedef enum {
		TENSOR_TYPE_NONE,
		TENSOR_TYPE_UINT8,
		TENSOR_TYPE_FP32,
		TENSOR_TYPE_INT32,
		TENSOR_TYPE_INT64,
	} TENSOR_TYPE;

public:
	TensorInfo();
	~TensorInfo();
	float_t* getDataAsFloat();
	int32_t preProcess(void *srcPixel, int32_t srcWidth, int32_t srcHeight);

public:
	std::string name;    // [In] tensor name. Need to set before initialize.
	TENSOR_TYPE type;    // [In] tensor type. Need to set before initialize. If the value is different from model structure, return error. 
	void*       data;    // InputTensor: [Out] pointer to raw input tensor data (fp32* at FP32 mode, UINT8 at UINT8 mode). Available after calling initialize
	                     //                    Input tensor data need to be written to this pointer before calling invoke 
	                     //                    Format is NHWC. Pre-process (Color conversion, resize and normalize need to be done) need to be done
	                     // OutputTensor: [Out] pointer to raw output tensor data (fp32* at FP32 mode, UINT8 at UINT8 mode). Available after calling invoke
	struct{
		int32_t batch;   // 0
		int32_t width;   // 1
		int32_t height;  // 2
		int32_t channel; // 3
	} dims;              // InputTensor: [In] If the value is different from model structure, return error. Need to set before initialize
	                     // OutputTensor: [Out] Automatically set at inference. Available after calling invoke

	struct {
		float_t scale;
		uint8_t zeroPoint;
	} quant;             // [Out] Automatically set by model structure if UINT8. Available after calling initialize

	struct {
		float_t mean[3];
		float_t norm[3];
	} normalize;        // InputTensor: [In] Need to set before calling preProcess. Not in use for OutputTensor

private:
	float_t  *m_dataFp32;
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
		OPEN_CV_OPENCL,
	} HELPER_TYPE;


public:
	virtual ~InferenceHelper() {}
	virtual int32_t setNumThread(const int32_t numThread) = 0;
	virtual int32_t initialize(const std::string& modelFilename, std::vector<TensorInfo>& inputTensorInfoList, std::vector<TensorInfo>& outputTensorInfoList) = 0;
	virtual int32_t finalize(void) = 0;
	virtual int32_t invoke(const std::vector<TensorInfo>& inputTensorInfoList, std::vector<TensorInfo>& outputTensorInfoList) = 0;
	
	static InferenceHelper* create(const HELPER_TYPE typeFw);

protected:
	HELPER_TYPE m_helperType;
};

#endif

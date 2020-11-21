# Play with TensorRT
Sample projects to use TensorRT

## Target Environment
- Platform
	- Linux (aarch64)
		- Tested in Jetson Nano (JetPack 4.3) and Jetson NX (JetPack 4.4)

- Projects
	- pj_tensorrt_cls_mobilenet_v2
		- Classification using MobileNet v2


## How to build application
### Common 
- Get source code
	```sh
	git clone https://github.com/iwatake2222/play_with_tensorrt.git
	cd play_with_tensorrt
	```

- Download resource
	- Download resource files (resource.zip) from https://github.com/iwatake2222/play_with_tensorrt/releases/ 
	- Extract it to `resource`

### Linux
```sh
cd pj_tensorrt_cls_mobilenet_v2   # for example
mkdir build && cd build
cmake ..
make
./main
```

### Option (Camera input)
```sh
cmake .. -DSPEED_TEST_ONLY=off
```

## Configuration for TensorRT
### Model format
- The model file name is specified in a class calling `InferenceHelperTensorRt.cpp` . Please find `MODEL_NAME` definition.
- `InferenceHelperTensorRt.cpp` automatically converts model according to the model format (extension).
	- `.onnx` : convert the model from onnx to trt, and save the converted trt model
	- `.uff` : convert the model from uff to trt, and save the converted trt model (WIP)
	- `.trt` : use pre-converted trt model
- I recommend you change the model filename once you conver the original onnx/uff model so that you can save your time
- In the source code on GitHub, onnx/uff model is used because trt model is not compatible with another environment

### Model conversion settings
- The parameters for model conversion is defiend in `InferenceHelperTensorRt.cpp`
- USE_FP16
	- define this for FP16 inference
- USE_INT8
	- define this for INT8 inference (you also need int8 calibration)
- OPT_MAX_WORK_SPACE_SIZE
	- `1 << 30`
- OPT_AVG_TIMING_ITERATIONS
 	- `8`
- OPT_MIN_TIMING_ITERATIONS
 	- `4`
- Parameters for Quantization Calibration
	- CAL_DIR
		- directory containing calibration images (ppm in the same size as model input size)
	- CAL_LIST_FILE
	 	- text file listing calibration images (filename only. no extension)
	- CAL_INPUT_NAME
	 	- input tensor name of the model
	- CAL_BATCH_SIZE
	 	- batch size for calibration
	- CAL_NB_BATCHES
	 	- the number of batches
	- CAL_IMAGE_C
	 	- the channel of calibration image. must be the same as model
	- CAL_IMAGE_H
	 	- the height of calibration image. must be the same as model
	- CAL_IMAGE_W
	 	- the width of calibration image. must be the same as model
	- CAL_SCALE
	 	- normalize parameter for calibration (probably, should use the same value as trainig)
	- CAL_BIAS
	 	- normalize parameter for calibration (probably, should use the same value as trainig)

### Quantization Calibration
- If you want to use int8 mode, you need calibration step
1. Create ppm images whose size is the same as model input size from training images
	- you can use `InferenceHelper/TensorRT/calibration/batchPrepare.py`
	- `python .\batchPrepare.py --inDir sample_org --outDir sample_ppm `
2. Copy the generated ppm files and list.txt to the target environment such as Jetson
3. Use `.onnx` model
4. Modify parameters for calibration such as `CAL_DIR` and define `USE_INT8`
5. Compile the project and run it
6. If it succeeds, trt model file is generated. You can use it after that

## Acknowledgements
- Some code are retrieved from the following projects:
	- https://github.com/nvidia/TensorRT (Apache-2.0 License)
- This project includes models from the following projects:
	- mobilenetv2-1.0
		- https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
		- https://github.com/onnx/models/blob/master/vision/classification/synset.txt


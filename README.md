# Play with TensorRT
Sample projects to use TensorRT

## Target Environment
- Platform
    - Linux (aarch64)
        - Tested in Jetson Xavier NX (JetPack 4.6)

## Usage
```
./main [input]

 - input = blank
    - use the default image file set in source code (main.cpp)
    - e.g. ./main
 - input = *.mp4, *.avi, *.webm
    - use video file
    - e.g. ./main test.mp4
 - input = *.jpg, *.png, *.bmp
    - use image file
    - e.g. ./main test.jpg
 - input = number (e.g. 0, 1, 2, ...)
    - use camera
    - e.g. ./main 0
- input = jetson
    - use camera via gstreamer on Jetson
    - e.g. ./main jetson
```

## How to build application
- Get source code
    ```sh
    git clone https://github.com/iwatake2222/play_with_tensorrt.git
    cd play_with_tensorrt
    git submodule update --init
    ```
- Download models
    - Download models (resource.zip) from https://github.com/iwatake2222/play_with_tensorrt/releases/ 
    - Extract it to `resource/`

- Build and run
    ```sh
    cd pj_tensorrt_cls_mobilenet_v2   # for example
    mkdir build && cd build
    cmake ..
    make
    ./main
    ```

## Configuration for TensorRT
### Model format
- The model file name is specified in `xxx_engine.cpp` . Please find `MODEL_NAME` definition
- `inference_helper_tensorrt.cpp` automatically converts model according to the model format (extension)
    - `.onnx` : convert the model from onnx to trt, and save the converted trt model
    - `.uff` : convert the model from uff to trt, and save the converted trt model (WIP)
    - `.trt` : use pre-converted trt model
- If *.trt file exists, InferenceHelper will use it to avoid re-conversion to save time
    - If you want to re-convert (for example, when you try another conversion settings), please delete `resource/model/*.trt`
    - Also, if you want to re-convert with INT8 calibration, please delete `CalibrationTable_cal.txt`

### DLA Cores (NVDLA)
- GPU is used by default
- Call `SetDlaCore(0)` or `SetDlaCore(1)` to use DLA

### Model conversion settings
- The parameters for model conversion is defiend in `inference_helper_tensorrt.cpp`
- USE_FP16
    - define this for FP16 inference
- USE_INT8_WITHOUT_CALIBRATION
    - define this for INT8 inference without calibration (I can't get good result with this)
- USE_INT8_WITH_CALIBRATION
    - define this for INT8 inference (you also need int8 calibration)
- OPT_MAX_WORK_SPACE_SIZE
    - `1 << 30`
- OPT_AVG_TIMING_ITERATIONS
     - not in use
- OPT_MIN_TIMING_ITERATIONS
     - not in use
- Parameters for Quantization Calibration
    - CAL_DIR
        - directory containing calibration images (ppm in the same size as model input size)
    - CAL_LIST_FILE
         - text file listing calibration images (filename only. no extension)
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
    - you can use `inference_helper/tensorrt/calibration/batchPrepare.py`
    - `python .\batchPrepare.py --inDir sample_org --outDir sample_ppm `
2. Copy the generated ppm files and list.txt to the target environment such as Jetson
3. Use `.onnx` model
4. Modify parameters for calibration such as `CAL_DIR` and define `USE_INT8`
5. Compile the project and run it
6. If it succeeds, trt model file is generated. You can use it after that


# License
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)

# Acknowledgements
- This project utilizes OSS (Open Source Software)
    - [NOTICE.md](NOTICE.md)
- Some images are retrieved from the followings:
    - dashcam_00.jpg, dashcam_01.jpg (Copyright Dashcam Roadshow 2020. https://www.youtube.com/watch?v=tTuUjnISt9s )

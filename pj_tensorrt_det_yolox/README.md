# YOLOX with TensorRT in C++
Sample project to run YOLOX + SORT

Click the image to open in YouTube

[![00_doc/yolox_sort_tensorrt.jpg](00_doc/yolox_sort_tensorrt.jpg)](https://youtu.be/dspn0dpZJHA)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tensorrt/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
        - copy `saved_model_yolox_nano_480x640/yolox_nano_480x640.onnx` to `resource/model/yolox_nano_480x640.onnx`
    - Build  `pj_tensorrt_det_yolox` project (this directory)

## Play more ?
- The project here uses very basic model and settings
- You can try another model such as bigger input size, quantized model, etc.:
    - Please modify `Model parameters` part in `detection_engine.cpp`
- You can try another inference framework like OpenCV, TensorFlow Lite, etc.
    - Please modify `Create and Initialize Inference Helper` part in `detection_engine.cpp` and cmake option
- You can tune model conversion paramters for TensorRT
    - Please modify `inference_helper_tensorrt.cpp`

## Acknowledgements
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/PINTO0309/PINTO_model_zoo

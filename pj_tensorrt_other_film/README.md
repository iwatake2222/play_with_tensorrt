# FILM (Frame Interpolation for Large Scene Motion) with TensorRT in C++
Sample project for FILM (Frame Interpolation for Large Scene Motion)

![00_doc/demo.jpg](00_doc/demo.jpg)

## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tensorrt/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/255_FILM/download_VGG.sh
        - copy `film_net_VGG_480x640/model_float32.tflite` to `resource/model/film_net_VGG_480x640.tflite`
        - copy `film_net_VGG_480x640/model_float32.onnx` to `resource/model/film_net_VGG_480x640.onnx`
    - Build  `pj_tensorrt_other_film` project (this directory)
    - Disable FP16 because model conversion may fails
        - Comment out `#define USE_FP16` in `InferenceHelper\inference_helper\inference_helper_tensorrt.cpp`

## Acknowledgements
- https://github.com/google-research/frame-interpolation
- https://github.com/PINTO0309/PINTO_model_zoo


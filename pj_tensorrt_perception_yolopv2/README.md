# YOLOPv2 with TensorRT in C++
Sample project to run YOLOPv2 + SORT + Bird's Eye View Transformation

![00_doc/demo.jpg](00_doc/demo.jpg)


## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tensorrt/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/326_YOLOPv2
        - copy `yolopv2_384x640` to `resource/model/yolopv2_384x640`
    - Build `pj_tensorrt_perception_yolopv2` project (this directory)

- Note:
    - Execution at the first time may take time due to model conversion
    - If you want to try quickly, use ONNX Runtime (enable `INFERENCE_HELPER_ENABLE_ONNX_RUNTIME` when run cmake, and use `kOnnxRuntime` )

## Acknowledgements
- https://github.com/CAIC-AD/YOLOPv2
- https://github.com/PINTO0309/PINTO_model_zoo

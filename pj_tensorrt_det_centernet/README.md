# CenterNet Object Detection with TensorRT in C++
Sample project to run CenterNet Object Detection

![00_doc/centernet.jpg](00_doc/centernet.jpg)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tensorrt/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/042_centernet/01_float32/download.sh
        - copy `ctdet_coco_dlav0_384.onnx` to `resource/model/ctdet_coco_dlav0_384.onnx`
    - Build  `pj_tensorrt_det_centernet` project (this directory)


## Acknowledgements
- https://github.com/xingyizhou/CenterNet.git
- https://github.com/PINTO0309/PINTO_model_zoo

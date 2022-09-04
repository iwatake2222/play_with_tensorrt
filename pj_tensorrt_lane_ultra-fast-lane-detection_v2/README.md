# Ultra-Fast-Lane-Detection-V2 with TensorRT in C++

![00_doc/demo.jpg](00_doc/demo.jpg)

https://user-images.githubusercontent.com/11009876/188304191-7361385b-ec9b-4aa8-a9a2-17723b2a53d7.mp4

*ufldv2_culane_res18_320x1600.onnx with TensorRT on GeForce RTX 3060 Ti*

[Link to full video on YouTube](https://www.youtube.com/watch?v=fbo-Z5v20D8)

## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tensorrt/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/324_Ultra-Fast-Lane-Detection-v2/download.sh
        - copy `ufldv2_culane_res18_320x1600.onnx` to `resource/model/ufldv2_culane_res18_320x1600.onnx`
    - Build  `pj_tensorrt_lane_ultra-fast-lane-detection_v2` project (this directory)

## Acknowledgements
- https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
- https://github.com/PINTO0309/PINTO_model_zoo
- Video
    - Drive Video by Dashcam Roadshow
    - 4K Tokyo Drive thru Ikebukuro, Shinjuku, Shibuya.mp4
    - https://www.youtube.com/watch?v=tTuUjnISt9s

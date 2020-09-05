# Play with Tensor RT
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


### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
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

## Acknowledgements
- This project includes models from the following projects:
	- mobilenetv2-1.0
		- https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
		- https://github.com/onnx/models/blob/master/vision/classification/synset.txt

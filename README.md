# Neural Speed

Neural Speed is an innovation library designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization and sparsity powered by [Intel Neural Compressor](https://github.com/intel/neural-compressor) and [llama.cpp](https://github.com/ggerganov/llama.cpp). We provide the experimental features as below:

- Modular design to support new models
- [Highly optimized low precision kernels](neural_speed/core/README.md)
- Utilize AMX, VNNI, AVX512F and AVX2 instruction set
- Support CPU (x86 platforms only) and Intel GPU (WIP)
- Support 4bits and 8bits quantization

> Neural Speed is under active development so APIs are subject to change.

You can refer [this blog](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176) to get the performance information.

## Quick Start
There are two methods for utilizing the Neural Speed:
1. We provide [Transformer-based API](https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md#llm-runtime-example-code) in [ITREX(intel extension for transformers)](https://github.com/intel/intel-extension-for-transformers), but you need to install Intel Extension for Transformers.

2. Neural Speed straight forward:

#### One-click Python scripts
Run LLM with one-click python script including conversion, quantization and inference.
```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

#### Quantize and Inference Step By Step
Besides the one-click script, Neural Speed also offers the detailed script: 1) convert and quantize, and 2) inference.

##### 1. Convert and Quantize LLM
Neural Speed assumes the compatible model format as [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). You can find GGUF models from HuggingFace or [llama.cpp](https://github.com/ggerganov/llama.cpp). An alternative approach involves converting the model by following the below steps:

```bash
# convert the model directly use model id in Hugging Face. (recommended)
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b
```

##### 2. Inference

We provide LLM inference script to run the quantized model. Please reach [us](mailto:itrex.maintainers@intel.com) if you want to run using C++ API directly.
```bash
numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"
```

> For details please refer to [Advanced Usage](./docs/advanced_usage.md).

## Supported Hardware
| Hardware | Optimization |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |
|Intel Arc GPU Series | WIP |
|Intel Data Center GPU Max Series | WIP |
|Intel Gaudi2 | Not yet |

## Supported Models


Neural Speed supports the following models: [list](./docs/supported_models.md)

## Install

### Build Python package
```shell
pip install .
```

### Build executable only

```shell
# Linux and WSL
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G Ninja
ninja
```

```powershell
# Windows
# Install VisualStudio 2022 and open 'Developer PowerShell for VS 2022'
mkdir build
cd build
cmake ..
cmake --build . -j --config Release
```


## Advanced Usage

### 1. Quantization and inferenece
You can do quantization and inference without Transformer-API: [Advanced Usage](./docs/advanced_usage.md).

### 2. Tensor Parallelism cross nodes/sockets

We support tensor parallelism strategy for distributed inference/training on multi-node and multi-socket. You can refer to [tensor_parallelism.md](./docs/tensor_parallelism.md) to enable this feature.


### 3. Contribution

You can consider adding your own models, please follow the document: [graph developer document](./developer_document.md). 

### 4. Custom Stopping Criteria

You can customize the stopping criteria according to your own needs by processing the input_ids to determine if text generation needs to be stopped.
Here is the document of Custom Stopping Criteria: [simple example with minimum generation length of 80 tokens](./docs/customized_stop.md)

### 5. Verbose Mode

Enable verbose mode and control tracing information using the `NEURAL_SPEED_VERBOSE` environment variable.

Available modes:
- 0: Print all tracing information. Comprehensive output, including: evaluation time and operator profiling.
- 1: Print evaluation time. Time taken for each evaluation.
- 2: Profile individual operator. Identify performance bottleneck within the model.

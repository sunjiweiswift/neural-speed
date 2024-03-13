# Neural Speed

Neural Speed is an innovation library designed to provide the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization and sparsity powered by [Intel Neural Compressor](https://github.com/intel/neural-compressor) and [llama.cpp](https://github.com/ggerganov/llama.cpp). Highlights of this project:

- Support LLAMA, LLAMA2, NeuralChat series, GPT-J, GPT-NEOX, Dolly-v2, MPT, Falcon, BLOOM, OPT, ChatGLM, ChatGLM2, Baichuan, Baichuan2, Qwen, Mistral, Whisper, CodeLlama, MagicCoder and StarCoder
- [Highly optimized low precision kernels](neural_speed/core/README.md), utilize AMX, VNNI, AVX512F, AVX_VNNI and AVX2 instruction set
- Up to 40x compared with [llama.cpp](https://github.com/ggerganov/llama.cpp), performance details: [blog](https://medium.com/@NeuralCompressor/llm-performance-of-intel-extension-for-transformers-f7d061556176) 
- NeurIPS' 2023: [Efficient LLM Inference on CPUs](https://arxiv.org/abs/2311.00502)
- Support 4bits and 8bits quantization
- Tensor Parallelism across sockets/nodes: [tensor_parallelism.md](./docs/tensor_parallelism.md)

> Neural Speed is under active development so APIs are subject to change.

## Supported Hardware
| Hardware | Optimization |
|-------------|:-------------:|
|Intel Xeon Scalable Processors | ✔ |
|Intel Xeon CPU Max Series | ✔ |
|Intel Core Processors | ✔ |

## Supported Models
LLAMA, LLAMA2, NeuralChat series, GPT-J, GPT-NEOX, Dolly-v2, MPT, Falcon, BLOOM, OPT, ChatGLM, ChatGLM2, Baichuan, Baichuan2, Qwen, Mistral, Whisper, CodeLlama, MagicCoder and StarCoder.

Neural Speed also supports GGUF models generated by [llama.cpp](https://github.com/ggerganov/llama.cpp), you need to download the model and use llama.cpp to create it. Validated models: [llama2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [falcon-7b](https://huggingface.co/tiiuae/falcon-7b), [falcon-40b](https://huggingface.co/tiiuae/falcon-40b), [mpt-7b](https://huggingface.co/mosaicml/mpt-7b), [mpt-40b](https://huggingface.co/mosaicml/mpt-40b) and [bloom-7b1](https://huggingface.co/bigscience/bloomz-7b1).

Please check more validated GGUF models from HuggingFace in [list](./docs/supported_models.md).

## Installation

### Build from Source
```shell
pip install -r requirements.txt
pip install .
```

>**Note**: GCC requires version 10+


## Quick Start (Transformer-like usage)

Install [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers) to use Transformer-like APIs.


### PyTorch Model from Hugging Face

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
prompt = "Once upon a time, there existed a little girl,"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### GGUF Model from Hugging Face

```python
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

# Specify the GGUF repo on the Hugginface
model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
# Download the the specific gguf model file from the above repo
model_file = "llama-2-7b-chat.Q4_0.gguf"
# make sure you are granted to access this model on the Huggingface.
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

prompt = "Once upon a time"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

### PyTorch Model from Modelscope
```python
import sys
from modelscope import AutoTokenizer
from transformers import TextStreamer
from neural_speed import Model

model_name = "qwen/Qwen1.5-7B-Chat"     # modelscope model_id or local model
prompt = "Once upon a time, there existed a little girl,"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = Model()
model.init(model_name, weight_dtype="int4", compute_dtype="int8", model_hub="modelscope")
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```

## Quick Start (llama.cpp-like usage)

### Single (One-click) Step

```
python scripts/run.py model-path --weight_dtype int4 -p "She opened the door and see"
```

### Multiple Steps

#### Convert and Quantize

```bash
# skip the step if GGUF model is from Hugging Face or generated by llama.cpp
python scripts/convert.py --outtype f32 --outfile ne-f32.bin EleutherAI/gpt-j-6b
```

#### Inference

```bash
# Linux and WSL
OMP_NUM_THREADS=<physic_cores> numactl -m 0 -C 0-<physic_cores-1> python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores> --color -p "She opened the door and see"
```

```bash
# Windows
python scripts/inference.py --model_name llama -m ne-q4_j.bin -c 512 -b 1024 -n 256 -t <physic_cores|P-cores> --color -p "She opened the door and see"
```

> For details please refer to [Advanced Usage](./docs/advanced_usage.md).

## Advanced Topics

### New model enabling
You can consider adding your own models, please follow the document: [graph developer document](./developer_document.md).

### Performance profiling
Enable `NEURAL_SPEED_VERBOSE` environment variable for performance profiling.

Available modes:
- 0: Print full information: evaluation time and operator profiling. Need to set `NS_PROFILING` to ON and recompile.
- 1: Print evaluation time. Time taken for each evaluation.
- 2: Profile individual operator. Identify performance bottleneck within the model. Need to set `NS_PROFILING` to ON and recompile.

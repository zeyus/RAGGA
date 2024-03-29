# RAGGA: Retrieval Augmented Generation General Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/ragga.svg)](https://pypi.org/project/ragga)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ragga.svg)](https://pypi.org/project/ragga)

-----

**Table of Contents**

- [RAGGA: Retrieval Augmented Generation General Assistant](#ragga-retrieval-augmented-generation-general-assistant)
  - [RAGGA](#ragga)
  - [Example Usage](#example-usage)
  - [Prerequisites](#prerequisites)
    - [CPU (Windows, Linux, and macOS)](#cpu-windows-linux-and-macos)
    - [GPU (Windows and Linux only)](#gpu-windows-and-linux-only)
  - [Installation](#installation)
    - [CPU (Windows, Linux, and macOS)](#cpu-windows-linux-and-macos-1)
    - [GPU (Windows and Linux only)](#gpu-windows-and-linux-only-1)
  - [Model Selection](#model-selection)
  - [License](#license)

## RAGGA

What is this? Load quantized LLMs and run them on your local devices. Interact with your own notes (currently markdown notes are supported by default) and ask questions about what you have written and get relevant answers from what you have written! Recommended model is TinyLlama-chat https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/README.md based on its speed of token generation and it's low system requirements. Try the 8-bit quantized model to start with and if that's still too beefy you can always try 5, 4, etc.

Basically, RAGGA is an local LLM based AI assistant that cares about your privacy, and does not require running things via untrusted third party APIs. The one exception is the web search feature, which can be completely disabled, so nothing ever leaves your system.

## Example Usage

Download one of the TinyLlama-chat quantized model (e.g. [8-bit quantized TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf)) and copy it to the models directory.

Install all the prerequistes and then install RAGGA.

Edit the [`config.yaml`](config.yaml) file to point to the model you downloaded in the `generator:` `llm_path:`, and update the `dataset:` `path:` key to point to your directory containing markdown files.

Run the [`tinyllama_example.py`](tinyllama_example.py) script to try it out.

## Prerequisites

Due to issues with [hatch not allowing pip options completely](https://github.com/pypa/hatch/issues/838):

- PyTorch needs to be installed manually
- llama-cpp-python needs to be installed manually
- meaning you need a c++ compiler (e.g. Visual Studio 2022 Community C++, build-essentials, etc)

When the issues are resolved with hatch, this will become significantly easier to install.

Both CPU and GPU will require `llama-cpp-python` to be installed. The `pip --pre` flag is required because the current stable release as of writing does not support phi-2.

### CPU (Windows, Linux, and macOS)

- Any python 3.11 installation, venv, conda, etc.
- Faiss (CPU) will be installed with the package
- Install PyTorch
  - Windows / MacOS
    - `pip install torch`
  - Linux
    - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Install llama-cpp-python
  - Windows / Linux / MacOS: Default (supports CPU without acceleration)
    - `pip install --pre llama-cpp-python`
- or Install llama-cpp-python with OpenBLAS Hardware Acceleration (optional)
  - Windows: OpenBLAS
    - `$env:CMAKE_ARGS = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"`
    - `pip install --pre llama-cpp-python`
  - Linux/MacOS: OpenBLAS
    - `CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python`

### GPU (Windows and Linux only)

Make sure you have CUDA 12.1 or higher installed.

- Install [miniforge](https://github.com/conda-forge/miniforge/releases) / [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Create a new environment with python 3.11 `conda create -n ragga python=3.11`
- Activate that environment and install faiss, pytorch and llama-cpp-python
  - `conda activate ragga`
  - `conda install faiss-gpu pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge`
  - Windows: llama-cpp-python with CUBLAS acceleration
    - `$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"`
    - `pip install --pre llama-cpp-python`
  - Linux: llama-cpp-python with CUBLAS acceleration
    - `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --pre llama-cpp-python`

**Note**: You do not need to use `conda`/`mamba` to install faiss-gpu, but as there are no wheels for it, you will need to compile it yourself, this is not covered here but see the [faiss build documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md#building-from-source)

## Installation

### CPU (Windows, Linux, and macOS)

```console
pip install 'ragga[cpu] @ https://github.com/zeyus/RAGGA/releases/download/v0.0.6/ragga-0.0.6-py3-none-any.whl'
```


### GPU (Windows and Linux only)

```console
pip install 'ragga @ https://github.com/zeyus/RAGGA/releases/download/v0.0.5/ragga-0.0.5-py3-none-any.whl'
```

## Model Selection

Subjective evaluation of 3 different models was performed by generating responses to general and dataset-specific questions for 4 publicly available Obsidian notes repositories. The results for response quality were ranked from 0-10 for each question, responses were shown anonymously so no indication of which model generated the responses was visible.

![reports/scores_by_model.png](reports/scores_by_model.png)

The phi-2 and phi-2 chat used the same base model but a different prompt template. For evaluation and rating details see [model_tests.py](model_tests.py) and [scripts/03_rank_output.py](scripts/03_rank_output.py).

## License

`ragga` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

# RAGGA: Retrieval Augmented Generation General Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/ragga.svg)](https://pypi.org/project/ragga)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ragga.svg)](https://pypi.org/project/ragga)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Prerequisites

Due to issues with [hatch not allowing pip options completely](https://github.com/pypa/hatch/issues/838):

- PyTorch needs to be installed manually
- llama-cpp-python needs to be installed manually

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
pip install 'ragga[cpu] @ https://github.com/zeyus/RAGGA/releases/download/v0.0.5/ragga-0.0.5-py3-none-any.whl'
```


### GPU (Windows and Linux only)

```console
pip install 'ragga @ https://github.com/zeyus/RAGGA/releases/download/v0.0.5/ragga-0.0.5-py3-none-any.whl'
```

## License

`ragga` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
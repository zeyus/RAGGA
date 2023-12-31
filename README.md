# RAGGA: Retrieval Augmented Generation General Assistant

[![PyPI - Version](https://img.shields.io/pypi/v/ragga.svg)](https://pypi.org/project/ragga)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ragga.svg)](https://pypi.org/project/ragga)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Prerequisites

Due to issues with [hatch not allowing pip options completely](https://github.com/pypa/hatch/issues/838), GPU support is a little bit tricky.

### GPU (Windows and Linux only)

Make sure you have CUDA 12.1 or higher installed.

- Install miniconda / mambaforge
- Create a new environment with python 3.11
- Activate that environment and use `conda install faiss-gpu pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge`


## Installation

Clone the repository and install it with pip:

```console
git clone https://github.com/zeyus/RAGGA.git
cd RAGGA
pip install .
```

## License

`ragga` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
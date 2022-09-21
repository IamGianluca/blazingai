## Introduction

A framework to accellerate the development cycle for ML projects.

This project leverage several other excellent libraries:

* [pytorch](https://github.com/pytorch/pytorch)
* [pytorch-lightning](https://github.com/Lightning-AI/lightning)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [timm](https://github.com/rwightman/pytorch-image-models)
* [torchmetrics](https://github.com/Lightning-AI/metrics)
* [transformers](https://github.com/huggingface/transformers)

## Installation

The package is not yet published in pypi. We are considering doing that on a later stage, when the package is more mature.

## Contribute

To contribute to the project, you can download the codebase and install the package in editable model with the following commands.

```
$ git clone git@github.com:IamGianluca/ml.git
$ cd ml
$ make install
```

It is also recommended to also install the development dependencies with `pip install -r requirements-dev.txt`. This will allow you to run additional helper commands like `make test` and `make format`.

When contributing to this repository, please use the following convention to label your commit messages.

* `BUG`: bug fixing
* `DEV`: development environment ― e.g., system dependencies
* `DOC`: documentation
* `ENH`: model architectures, feature engineering
* `MAINT`: maintenance ― e.g., refactoring
* `TST`: testing, continuous integration

## Introduction

A framework to accelerate the development cycle for AI/ML projects.

This project leverage several other excellent libraries:

* [pytorch](https://github.com/pytorch/pytorch)
* [lightning](https://github.com/Lightning-AI/lightning)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [timm](https://github.com/rwightman/pytorch-image-models)
* [torchmetrics](https://github.com/Lightning-AI/metrics)
* [transformers](https://github.com/huggingface/transformers)

## Installation

The package is not yet published in the Python Package Index (PyPI). We are considering doing that on a later stage, when the package is more mature.

## Contributing

To contribute to the project, you can download the codebase and install the library in editable mode with the following commands.

```
$ git clone git@github.com:IamGianluca/ml.git
$ cd ml
$ make install
```

After doing that, you will be able to run additional commands like `make test` and `make format`.

When contributing to this repository, please respect this naming convention to tag your commits:

* API relevant changes:
    * `feat`: Commits that add a new feature
    * `fix`: Commits that fixes a bug
* `refactor`: Commits that rewrite/restructure your code but does not change any behavior
    * `perf`: Special `refactor` commits that improve performance
* `style`: Commits that do not affect the meaning (white space, formatting, missing semi-colons, etc.)
* `test`: Commits that add missing tests or correct existing tests
* `docs`: Commits that affect documentation only
* `build`: Commits that affect build components like build tool, CI pipeline, dependencies, project version, etc...
* `ops`: Commits that affect operational components like infrastructure, deployment, backup, recovery, ...
* `chore`: Miscellaneous commits e.g., modifying `.gitignore`

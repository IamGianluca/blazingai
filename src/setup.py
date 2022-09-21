import ml
from setuptools import setup

setup(
    name="ml",
    version=ml.__version__,
    description="Personal ML library",
    author="Gianluca Rossi",
    author_email="gr.gianlucarossi@gmail.com",
    license="MIT",
    install_requires=[
        "kaggle",
        "loguru",
        "matplotlib",
        "numpy",
        "omegaconf",
        "pydicom",  # TODO: install as 'extra' requirement
        "pylibjpeg",  # required by pydicom
        "pylibjpeg-libjpeg",  # required by pydicom
        "pylibjpeg-openjpeg",  # required by pydicom
        "pytorch-lightning>=1.6.3",
        "scikit-learn",
        "timm",
        "torchmetrics>=0.8.2",
        "tqdm",
        "transformers",
    ],
    packages=["ml"],
    package_dir={"ml": "ml"},
)

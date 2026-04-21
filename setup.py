from setuptools import setup, find_packages

setup(
    name="torchquery",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
)
from setuptools import setup, find_packages
from glob import glob

so_files = glob("swish/python/swish_binding*.so")

setup(
    name="swish",
    version="0.1",
    description="Swish activation function",
    packages=find_packages(),
    package_data={
        "swish": ["python/*.so"],
    },
)
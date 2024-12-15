from setuptools import find_packages
from distutils.core import setup

if __name__ == "__main__":
    setup(
        name="mas",
        version="0.0.1",
        packages=find_packages(exclude=("experiments")),
        authors="Xiang Gao",
        description="Experiments into risk scoring system for post liver \
            transplant graft failure prediction.",
        python_requires=">=3.8",
        include_package_data=True,
    )
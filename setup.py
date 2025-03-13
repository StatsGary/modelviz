"""Setup file to determine imports"""
import re
from setuptools import setup, find_packages

def get_version():
    """Gets the current version of the package"""
    with open("modelviz/__init__.py", "r", encoding='utf-8') as f:
        version_line = next((line for line in f if line.startswith("__version__")), None)
        if version_line:
            return re.search(r"\"(.*?)\"", version_line).group(1)
        raise RuntimeError("Version not found in modelviz/__init__.py")

def read_requirements():
    """Read dependencies from requirements.txt"""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="modelviz",
    version=get_version(),
    author="Gary Hutson",
    author_email="hutsons-hacks@outlook.com",
    description="A package for EDA and Sci-Kit Learn visualisations and utilities",
    long_description=open("README.md", encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/StatsGary/modelviz",
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=7.0"]  
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
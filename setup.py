from setuptools import setup, find_packages

setup(
    name="modelviz",
    version="0.1.2",
    author="Gary Hutson",
    author_email="hutsons-hacks@engineer.com",
    description="A package for EDA and Sci-Kit Learn visualisations and utilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/your-repo",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",        # For DataFrame operations
        "numpy>=1.20",        # For numerical computations
        "scikit-learn>=0.24"  # For machine learning utilities
    ],
    extras_require={
        "dev": ["pytest>=7.0"]  # Development and testing dependencies
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)


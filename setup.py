import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adam_mini",
    version="1.1.1",
    author="NotImplemented",
    author_email="NotImplemented",
    description="Official implementation of the Adam_mini optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zyushun/Adam-mini",
    packages=setuptools.find_packages(exclude=("figures", "examples")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    extras_require={
        "torch": ["torch>=2.1.0"],
        "dev": [
            "twine",
            "setuptools",
            "wheel",
        ],
    },
)
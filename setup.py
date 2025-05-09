from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Package setup
setup(
    name="SeqLbToolkit",
    version="0.6.18",
    author="Yinghao Li",
    author_email="yinghaoli@gatech.edu",
    license="MIT",
    url="https://github.com/Yinghao-Li/seqlbtoolkit",
    description="Commonly-used functions for building sequence labeling models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "nlp",
        "sequence-labeling",
        "ml",
        "machine-learning",
        "natural-language-processing",
    ],
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
)

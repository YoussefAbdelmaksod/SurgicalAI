"""
Setup script for SurgicalAI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="surgicalai",
    version="0.1.0",
    author="SurgicalAI Team",
    author_email="contact@surgicalai.example.com",
    description="Advanced computer vision system for surgical tool detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/SurgicalAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "pycocotools>=2.0.6",
        "albumentations>=1.3.0",
        "pandas>=2.0.0",
        "optuna>=3.2.0",
        "scikit-learn>=1.3.0",
        "tensorboard>=2.13.0",
        "plotly>=5.15.0",
        "flask>=2.0.1",
        "timm>=0.5.4",
        "transformers>=4.8.2",
    ],
    entry_points={
        "console_scripts": [
            "surgicalai=app.main:main",
        ],
    },
)

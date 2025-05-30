#!/usr/bin/env python
"""
Setup script for the SurgicalAI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="SurgicalAI",
    version="1.0.0",
    author="SurgicalAI Team",
    author_email="your.email@example.com",
    description="A personalized guidance system for laparoscopic cholecystectomy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SurgicalAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "surgicalai-train=scripts.train_models:main",
            "surgicalai-inference=scripts.run_inference:main",
            "surgicalai-setup=scripts.setup_profiles:main",
            "surgicalai-init=scripts.initialize_system:main",
        ],
    },
)

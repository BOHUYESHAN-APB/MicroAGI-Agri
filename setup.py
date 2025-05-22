from setuptools import setup, find_packages

setup(
    name="microai_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "paddlepaddle>=3.0.0",
        "paddle-detection>=2.5.0",
        "albumentations>=1.3.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "visualdl>=2.5.0",
        "labelme>=5.1.1",
        "imgaug>=0.4.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)
from setuptools import setup, find_packages

setup(
    name="viola",
    version="1.0.0",
    description="Companion code for VIOLA (NeurIPS 2026)",
    author="Anonymous",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torch-geometric>=2.4",
        "scikit-learn>=1.3",
        "numpy>=1.24",
        "sentence-transformers>=2.2",
        "datasets>=2.14",
        "huggingface_hub>=0.20",
        "pandas>=2.0",
        "openai>=1.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
)

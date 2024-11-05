from setuptools import setup, find_packages

setup(
    name="sae-ts",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformer_lens",
        "sae_lens",
        "huggingface-hub",
        "plotly",
        "kaleido",
        "openai",
        "nest_asyncio",
    ],
    python_requires=">=3.8",
)

from setuptools import setup, find_packages

setup(
    name="llava_next_demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "Pillow",
        "requests"
    ]
) 
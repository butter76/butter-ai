from setuptools import setup, find_namespace_packages

setup(
    name="butter-ai",
    version="0.1.0",
    packages=find_namespace_packages(include=["projects.*"]),
    package_dir={"": "."},
    install_requires=[],
)
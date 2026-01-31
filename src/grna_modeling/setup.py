from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="grna-model",
    description="Model for gRNA generation and scoring",
    packages=find_packages(),
    install_requires=requirements,
)

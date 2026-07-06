from setuptools import setup, find_packages
setup(
    name="bayes-jnr",
    version="0.1.3",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0", "scipy>=1.4.0"],
)

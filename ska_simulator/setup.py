from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = "A small package to simulate SKA observations."
LONG_DESCRIPTION = "A small package to simulate SKA observations with cosmological signal, foregrounds and noise."

setup(
    name="convsn",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Rebecca Ceppas de Castro, Hannah Fronenberg, Adélie Gorce, Adrian Liu, Lisa McBride, Bobby Pascua",
    author_email="adelie.gorce@gmail.com",
    packages=find_packages(),
    dependencies='dynamic',
    install_requires=[
        "setuptools>=61.0",
        "astropy",
        "numpy",
        "cached_property",
        "scipy",
        "healpy",
        "uvtools"
        ],
    keywords='ska_simulator',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Cosmologists",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)

#!/usr/bin/env python
import imp
import io
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup  # noqa: F811


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_mpi', 'version.py'))
description = (
    "An MPI backend for the nengo python package. Supports running "
    "nengo simulations in parallel, using MPI as the communication protocol.")

long_description = read('README.md')

setup(
    name="nengo_mpi",
    version=version_module.version,
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    packages=find_packages(),
    scripts=[],
    data_files=[],
    url="https://github.com/e2crawfo/nengo_mpi",
    license="See LICENSE.rst",
    description=description,
    long_description=long_description,
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=[],
    install_requires=[],
    extras_require={},
    tests_require=[],
    zip_safe=False,
)

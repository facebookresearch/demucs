# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path
from importlib.machinery import SourceFileLoader


from setuptools import setup

NAME = 'demucs'
DESCRIPTION = 'Music source separation in the waveform domain.'

URL = 'https://github.com/facebookresearch/demucs'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez'
REQUIRES_PYTHON = '>=3.7.0'

HERE = Path(__file__).parent

# Trick taken from https://github.com/sigsep/sigsep-mus-db/blob/master/setup.py
VERSION = SourceFileLoader(
    'demucs', 'demucs/__init__.py'
).load_module().__version__

REQUIRED = [i.strip() for i in open(HERE / "requirements.txt")]

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['demucs'],
    install_requires=REQUIRED,
    extras_require={'dev': ['flake8']},
    include_package_data=True,
    license='MIT License',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

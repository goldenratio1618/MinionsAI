#!/usr/bin/env python
from distutils.core import setup

setup(name='MinionsAI',
      version='0.0',
      packages=['minionsai'],
      install_requires=['numpy', 'tqdm', 'trueskill', 'pytest', 'tabulate', 'pytest-timeout']
     )
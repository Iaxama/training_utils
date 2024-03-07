#!/usr/bin/env python

from setuptools import setup, find_packages, Require

setup(name='emg-training-utils', 
      version=1.0,
      install_requires=['numpy>=1.19.1,<=1.26.4',
                        'scipy==1.12.0'],
      packages=find_packages())
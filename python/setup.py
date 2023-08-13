#!/usr/bin/env python3

import re
import sys
import os
from setuptools import setup, find_packages

import subprocess

def get_version_from_tag():
    ret, val = subprocess.getstatusoutput('git describe --tags')
    return '0.0.0'
    # if ret != 0:
    m = re.match('v(\d+\.\d+\.\d+)(-.+)*', val)
    if not m:
        return '0.0.0'
    ver = m.group(1)
    revision = m.group(2)
    if revision:
        revision = revision[1:]
        return f'{ver}.{revision}'
    else:
        return ver

def iter_shared_objects():
    cur_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
    root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    for dirpath, dirnames, filenames in os.walk(cur_dir):
        for fn in filenames:
            if fn.endswith('.so'):
                yield os.path.join(dirpath, fn)
            if fn.endswith('.a'):
                yield os.path.join(dirpath, fn)


packages = ['tpu_plugin']
so_list = list(iter_shared_objects())
print(so_list)
setup(
    version=get_version_from_tag(),
    author='sophgo',
    description='A Pytorch Plugin for training with TPU',
    author_email='dev@sophgo.com',
    license='Apache',
    name='tpu_plugin',
    url='https://www.sophgo.com/',
    install_requires=[ "numpy","torch==1.13.1" ],
    dependency_links=['https://download.pytorch.org/whl/cpu/'], 
    packages=find_packages(),
    include_package_data=True,
    package_data={'tpu_plugin': so_list},
    platforms=['Linux_x86_64'])
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# - pip install 该cuda包的准备

# - 定位到所有的cpp文件和cuda文件
ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='mycuda',
    version='1.0',
    author='Sparsity',
    author_email='Zysun2002@163.com',
    description='first try of cuda',
    long_description='a tiny cuda program learnt from youtube',
    ext_modules=[
        CUDAExtension(
            name='mycuda',
            sources=sources,
            include_dirs=include_dirs,   #_ header文件位置
             #_ 优化的参数，optional
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
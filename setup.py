from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_requirements(fname):
    return read(fname).splitlines()


setup(
    name='img-classifier',
    version='0.1.0',
    packages=['main', 'main.utils'],
    url='https://github.com/jhole89/convnet-image-classifier',
    license='GPL',
    author='Joel Lutman',
    author_email='joellutman@gmail.com',
    description='A Convolutional Neural Network utility for image recognition',
    long_description=read('README.md'),
    install_requires=get_requirements('requirements.txt')
)

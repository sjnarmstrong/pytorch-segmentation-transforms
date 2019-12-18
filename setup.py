from setuptools import setup, find_packages
setup(
    name='SegTransforms',
    version='0.0.1',
    packages=find_packages(include=["segtf*"]),
    url='',
    license='MIT',
    author='sholto',
    author_email='sjnarmstrong@gmail.com',
    description='Project to implement transformations for segmentation datasets.',
    install_requires=[
        'pydantic',
	'torch',
	'torchvision'
    ],
)

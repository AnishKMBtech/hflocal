from setuptools import setup, find_packages

setup(
    name='hflocal',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.30.0',
        'torch>=2.0.0',
    ],
    author='Anish KM',
    author_email='your.email@example.com',
    description='A library to save, load, and use Hugging Face models locally.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AnishKMBtech/hflocal',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERP",
    version="1.1-rc1",
    license='LICENSE.txt',
    author="Charangan Vasantharajan",
    author_email="chaanuv@gmail.com",
    description="This is a pipeline designed for fine-tuning pre-trained transformers to perform Named Entity Recognition (NER) tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaarangan/NERP",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'transformers',
        'nltk',
        'pandas',
        'pyconll',
        'PyYAML',
        'sentencepiece',
        'scikit-learn',
        'TorchCRF',
        'loguru',
        'dictparse'
    ],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)
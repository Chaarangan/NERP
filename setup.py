import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERP",
    version="1.0.0",
    author="Charangan Vasantharajan",
    author_email="chaanuv@gmail.com",
    description="A Pipeline for Finetuning & Infering Transformers for Named-Entity Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaarangan/NERP",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'transformers',
        'sklearn',
        'nltk',
        'sentencepiece'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    dependency_links=['https://github.com/Chaarangan/NERDA/tree/save_and_load_tokenizer']
)

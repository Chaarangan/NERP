import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERP",
    version="0.97",
    author="Charangan Vasantharajan",
    author_email="chaanuv@gmail.com",
    description="A Pipeline for Finetuning & Infering Transformers for Named-Entity Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaarangan/NERP",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'nltk',
        'transformers',
        'scikit-learn',
        'sentencepiece',
        'progressbar',
        'pyconll',
        'pyyaml'
    ],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)

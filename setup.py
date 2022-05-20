import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NERP",
    version="0.98",
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
        'huggingface-hub==0.6.0',
        'joblib==1.1.0',
        'nltk==3.7',
        'pandas==1.4.2',
        'progressbar==2.5',
        'pyconll==3.1.0',
        'PyYAML==6.0',
        'scikit-learn==1.1.1',
        'scipy==1.8.1',
        'sentencepiece==0.1.96',
        'threadpoolctl==3.1.0',
        'tokenizers==0.12.1',
        'torch==1.11.0',
        'torchaudio==0.11.0',
        'torchvision==0.12.0',
        'tqdm==4.64.0',
        'transformers==4.19.2'
    ],
    tests_require=['pytest',
                   'pytest-cov'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True
)

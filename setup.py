from setuptools import setup, find_packages

setup(
    name='embed_software',
    version='0.0.2',
    url='https://github.com/nandanrao/embed-software',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'pandas',
        'scikit-learn',
        'tqdm',
        'dataset',
        'diskcache'
    ]
)

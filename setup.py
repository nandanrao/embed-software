from setuptools import setup

setup(
    name='embed_software',
    version='0.0.1',
    url='https://github.com/nandanrao/embed-software',
    py_modules=['lib'],
    zip_safe=False,
    install_requires=[
        'pandas',
        'scikit-learn',
        'tqdm',
        'dataset',
        'diskcache'
    ]
)

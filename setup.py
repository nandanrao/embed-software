from setuptools import setup

setup(
    name='embedsoftware',
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

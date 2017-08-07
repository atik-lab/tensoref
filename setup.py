from setuptools import setup
from tensoref import __version__

# setup.py
# find classifiers in https://pypi.python.org/pypi?%3Aaction=list_classifiers

setup(
    name='tensoref',
    version=__version__,
    packages=['tensoref'],
    install_requires=[
        'docopt',
        'tensorflow',
        'nltk',
        'xlrd',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    entry_points={
        'console_scripts': [
            'tensoref=tensoref.cli:main',
        ],
    },
)

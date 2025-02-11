"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
import re

# Get version
VERSIONFILE="discminer/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
#*************

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # $ pip install discminer
    # https://pypi.org/project/discminer/
    name='discminer',  # Required
    # https://packaging.python.org/guides/single-sourcing-package-version/
    version=verstr,  # Required
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Python package for parametric modelling of intensity channel maps from gas discs',  # Optional
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/andizq/discminer',  # Optional
    author='Andres F. Izquierdo',  # Optional
    author_email='andres.izquierdo.c@gmail.com',  # Optional
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='astronomy, discs, disks, planets, detection',  # Optional
    #package_dir={'discminer': 'discminer'},  # Optional
    packages=['discminer', 'discminer.tools', 'discminer.mining'],  # Required
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires='>=3.6, <4',
    # https://packaging.python.org/discussions/install-requires-vs-requirements/
    install_requires=
    [
        'numpy>=1.18',
        'scipy>=1.5',
        'matplotlib>=3.0',
        'astropy>=3',
        'emcee>=3',
        'corner',
        'cmasher',
        'radio-beam',
        'scikit-image',
        'scikit-learn',
        'spectral-cube>=0.6',
        'packaging>=20.9',
        'termtables',
        'pathos'
    ], 
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    package_data={  # Optional
        'discminer':
        [
            'icons/logo.txt',
            'icons/button*',
            'tools/discminer.mplstyle',
        ],
    },
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/andizq/discminer/issues',
        'Source': 'https://github.com/andizq/discminer/',
    },
    entry_points={
        'console_scripts': [
            'discminer = discminer.mining_control:main',
        ],
    }
)

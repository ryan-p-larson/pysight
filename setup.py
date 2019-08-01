# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()


setup(
    name='PySight',  # Required
    version='0.0.1',  # Required

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    python_requires='>=3.0',

    install_requires=[
        'dlib',
        'numpy',
        'scipy'
    ],  # Optional



    # If there are data files included in your packages that need to be
    # installed, specify them here.
    include_package_data=True,
    package_data={  # Optional
        'pysight': [
            'models/haarcascade_eye.xml',
            'models/haarcascade_frontalface_alt.xml',
            'models/shape_predictor_68_face_landmarks.dat'
        ]
    }
    # You may need to place data files outside of your packages
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],  # Optional

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    #entry_points={  # Optional
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)

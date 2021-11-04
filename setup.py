from setuptools import setup

setup(
    name='simbiofilm',
    version='0.0.1',
    description='Python biofilm simulation framework',
    # description=__doc__.split('\n')[0],
    # long_description=__doc__,
    # url='http://github.com/storborg/funniest',
    author='Matt Simmons',
    author_email='matt.simmons@compbiol.org',
    license='MIT',
    packages=['simbiofilm'],
    install_requires=[
        'numpy',
        'scipy',
        'numba'],
    tests_require=['pytest'],
)

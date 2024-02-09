from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here/'README.md').read_text(encoding='utf-8')

setup(

    name='elfdannagalac',

    version='0.1.0',

    description='ELectron Flux from Dm ANNihilation And propagation in GALAxy Clusters',

    long_description=long_description,

    long_description_content_type='text/markdown',

    author='Arlette Melo Galindo, Miguel Sánchez Conde, Rubén Alfaro, Sergio Hernández Cadena',

    author_email='skerzot@ciencias.unam.mx',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10'
        ],

    keywords=[
        'dark matter','indirect searches','science',
        'indirect dm searches','galaxy clusters'
        ],

    #   packages=find_packages(where='ctadmtool')
    packages=find_packages(exclude=('old',)),

    install_requires=['scipy','numpy','matplotlib'],

    include_package_data=True,

    package_data={'':[ 'data/cosmixs/*.dat','data/pppc4dmid/*.dat']}

)
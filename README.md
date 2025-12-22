# elf-danna-galac

Invoking a witch (or wizard?) to search for the black relics in galaxy clusters!

Science Package to compute the expected photon flux from the propagation and diffusion of electrons/positrons produced through dark matter (DM) annihilation/decay in galaxy clusters.
The name of this science project is the acronym for ELectron Flux induced by Dark matter ANNihilation And decay in GALAxy Clusters.

### Authors

1. Arlette Melo Galindo (IFT-UAM)
2. Miguel A. Sánchez Conde (IFT-UAM)
3. Sergio Hernández Cadena (TDLI-SJTU)


## Basic idea

DM represents at least 85% of the content of galaxy clusters. If this DM component is made of heavy particles (like WIMPs), products from their annihilation/decay should also be electron-positron pairs. We would expect this component to be "injected" (we refer to it as a new component) into the IntraCluster Medium (ICM) and propagate through diffusion due to the cluster's magnetic field. This additional contribution to the photon flux would be observable at wavelengths from Radio and X-ray up to TeV gamma-rays, according to the mass of the DM candidate.

## Analysis plan

We use `CRpropa` to estimate the photon flux obtained from the diffusion of electrons/positrons in the ICM after modelling the magnetic field. We also consider the Inverse Compton (IC) with other seed photon fields. We show the expected emission for an hypothetical observer just at the outskirsts of the cluster and for an observer at Earth. We use data from `cosmiXs` and `PPPC4-DMID` projects to estimate the final photon flux according to the differential energy spectrum of electrons and positrons obtained from DM annihilation.

## Installation

For convenience we provide the instructions to install the whole package including `CRpropa`. Installation was done in a remote server with Linux. Please note that, as for now, we use `"python 3.10.14"` and `"numpy == 1.26.4"` to be able to run `CRpropa`.

### Installing `CRpropa`

Please refer to the official [`CRpropa` documentation](https://crpropa.github.io/CRPropa3/pages/Installation.html) for more details. The following are the instructions with we were able to have a running virtual environment.

```bash
# The first part is to create the virtual environment and install all the necessary dependencies
# Most of them could be in a single line, but I don't like really long lines
conda create -n edmcanna -c conda-forge python=3.11 c-compiler cxx-compiler fortran-compiler cmake llvm openmpi gsl -y
conda activate edmcanna
conda install -c conda-forge ucx
conda install -c conda-forge swig -y
conda install -c conda-forge openmp -y
# conda install -c conda-forge conda-gcc-specs -y
conda install -c conda-forge numpy=1.26.4 scipy matplotlib jupyter notebook pandas astropy emcee -y
conda install -c conda-forge hdf5 cfitsio muparser -y
conda activate edmcanna
mkdir EDMClusters
cd EDMClusters

# This is to install fftw3
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar -zxf fftw-3.3.10.tar.gz
cd fftw-3.3.10/
./configure --prefix=${CONDA_PREFIX} --enable-openmp --enable-mpi --enable-sse --enable-single --enable-shared
make -j 4
make install

# As we want everything encapsulated, we change to the venv's directory
cd ${CONDA_PREFIX}

# Downoading CRpropa
wget https://github.com/CRPropa/CRPropa3/archive/refs/tags/3.2.1.tar.gz
tar -xzf 3.2.1.tar.gz
cd CRPropa3-3.2.1/

# This is to compile CRpropa using conda's gcc compilers
mkdir build && cd build
CMAKE_PREFIX_PATH=${CONDA_PREFIX} cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} .. ${CMAKE_ARGS} -DSIMD_EXTENSIONS:STRING=native -Wno-dev -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 8

# The test for MagneticLenses will fail
# After checking with one of the CRpropa developers,
# it turns out that the issue is known but no one knows
# how to solve it.
make test
make install
```
Other extra packages will be downloaded when installing this package, as loguru.

> [!NOTE]
> The numpy version fixed to 1.26.4 prevents to be able to install `CRpropa` in virtual environments with `python > 3.12`.
> The policy option when building `CRpropa` is needed for compatibility with recent versions of cmake

### Installation of elfdannagalac

As for now, the analysis scripts are under development, and the package is not available at PyPi for download. Then, a local installation is needed. (We wouldn't probably publish the package at PyPi).
Then once you clone the repository, you should change to that folder and use

```bash
python -m pip install .
```

Once the installation finished, you should have disponible in your command line interface (cli) the app `diffedm`

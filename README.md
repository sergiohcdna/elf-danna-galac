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
> The numpy version fixed to 1.26.4 prevents the installation of `CRpropa` in virtual environments with `python > 3.12`.
> The policy option when building `CRpropa` is needed for compatibility with recent versions of cmake

### Installation of elfdannagalac

As for now, the analysis scripts are under development, and the package is not available at PyPi for download. Then, a local installation is needed. (We wouldn't probably publish the package at PyPi).
Then once you clone the repository, you should change to that folder and use

```bash
python -m pip install .
```

Once the installation is finished, you should have disponible in your command line interface (cli) the apps:

* `diffedm`
* `countsmap`

## Running a simulation

The app `diffedm` is the basic script to run a simulation to get the expected photon emission from propagation of electron/positrons in the ICM of a galaxy cluster. `diffedm` is still under development.
The physics and other stuff are decribed in [Physics behind](#physics-behind).

To get the help message for `diffedm` you simply use:

```bash
diffedm --help
```

For your convenience, the output is:

```bash
usage: diffedm [-h] --srcz 0.0043 --dmprofile nfw --dmsource_type extended_smooth [--dmprofile_pars DMPROFILE_PARS [DMPROFILE_PARS ...]] [--process [anna,decay]] --emin 100
               --emax 1e5 --nparticles 10000 --brms 5 muG --lmin 0.02 Mpc --lmax 0.15 Mpc --cluster_center (16,0,0) Mpc (16,0,0) Mpc (16,0,0) Mpc --origin_box (14,-2,-2) Mpc
               (14,-2,-2) Mpc (14,-2,-2) Mpc --eta_index 0.33 --core_radius 0.25 Mpc [--diff_epsilon 0.1] [--diff_scale 0.1] [--diff_alpha 0.1] --ncells 512 --spacing 0.008
               Mpc [--xmin 14.0 Mpc] [--xmax 18.0 Mpc] [--ymin -2.0 Mpc] [--ymax 2.0 Mpc] [--zmin -2.0 Mpc] [--zmax 2.0 Mpc] [--maxtries 100000] [--ofname OFNAME] [--odir ./]

Galaxy clusters project: elf-danna-galac November/2025

options:
  -h, --help            show this help message and exit

Galaxy Cluster:
  Input params

  --srcz 0.0043         Redshift of the cluster
  --dmprofile nfw       Label for DM profile parametrization
  --dmsource_type extended_smooth
                        Point or Extended like DM injection sources
  --dmprofile_pars DMPROFILE_PARS [DMPROFILE_PARS ...]
                        Parameters to describe the DM density profile
  --process [anna,decay]
                        Annihilation or Decay?
  --emin 100            Minimum energy of electrons (GeV)
  --emax 1e5            Maximum energy of electrons (GeV)
  --nparticles 10000    Number of particles to be smulated
  --brms 5 muG          RMS value of the turbulence field [muG]
  --lmin 0.02 Mpc       Minimum scale of the turbulence field [Mpc]
  --lmax 0.15 Mpc       Maximum scale of the turbulence field [Mpc]
  --cluster_center (16,0,0) Mpc (16,0,0) Mpc (16,0,0) Mpc
                        Cartesian Position of the center of the cluster [Mpc]
  --origin_box (14,-2,-2) Mpc (14,-2,-2) Mpc (14,-2,-2) Mpc
                        Cartesian origin of the spatial grid [Mpc]
  --eta_index 0.33      Index of the Bfield dependance with radial distance
  --core_radius 0.25 Mpc
                        Value of the electron density core radius [Mpc]
  --diff_epsilon 0.1    Anisotrpy factor of the diffusion coefficient
  --diff_scale 0.1      Scaling factor to set the diffusion coefficient
  --diff_alpha 0.1      Spectral index for the Diff_coeff = E^{-alpha}
  --ncells 512          Number of points in the spatial grid
  --spacing 0.008 Mpc   Step used to construct the spatial grid [Mpc]
  --xmin 14.0 Mpc       Min X value for Source's position sampling [Mpc]
  --xmax 18.0 Mpc       Max X value for Source's position sampling [Mpc]
  --ymin -2.0 Mpc       Min Y value for Source's position sampling [Mpc]
  --ymax 2.0 Mpc        Max Y value for Source's position sampling [Mpc]
  --zmin -2.0 Mpc       Min Z value for Source's position sampling [Mpc]
  --zmax 2.0 Mpc        Max Z value for Source's position sampling [Mpc]
  --maxtries 100000     Maximum number of tries for source sampling
  --ofname OFNAME       Output file name [fits,fits.gz]
  --odir ./             Output directory to save files
```

A simulation using some default parameters should be like this:

```bash
diffedm --emin 100 --emax 1e4 --nparticles 100000 --ofname testSmoothHalo.fits.gz --cluster_center 16 0 0 --brms 5.0 --lmin 0.02 --lmax 0.15 --origin_box 14 -2 -2 --eta_index 0.33  --core_radius 0.25 --ncells 512 --spacing 0.008 --srcz 0.0043 --dmprofile nfw  --dmprofile_pars 0.3 157.556 0.008 5.605e3 --xmin 14.0 --xmax 18.0 --ymin -2.0 --ymax 2.0 --zmin -2.0 --zmax 2.0 --maxtries 1e6 --dmsource_type extended_smooth
```

An example of the expected output is:

```bash
2025-12-23 00:31:22.851 | INFO     | elfdannagalac.scripts.diff_edm:main:275 - Getting parameters
2025-12-23 00:31:22.852 | INFO     | elfdannagalac.tools.misc:checkDir:128 - . already exists
2025-12-23 00:31:22.852 | INFO     | elfdannagalac.scripts.diff_edm:main:302 - Calculation of Magnetic Field grid for cluster: 
2025-12-23 00:32:24.304 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:100 - Description of the field:
2025-12-23 00:32:24.305 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:101 - Correlation Length is: 39.182 kpc
2025-12-23 00:32:24.305 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:102 - RMS B field is: 5.0 microG
2025-12-23 00:32:24.305 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:103 - Mean B field is: 4.606491319465118 microG
2025-12-23 00:32:24.305 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:104 - B field at the center of the cluster: 5.218 microG
2025-12-23 00:48:58.419 | INFO     | elfdannagalac.scripts.diff_edm:main:346 - Preparing Diffusion module
2025-12-23 00:48:58.420 | INFO     | elfdannagalac.scripts.diff_edm:main:392 - Preparing dark matter source
2025-12-23 00:48:58.420 | INFO     | elfdannagalac.scripts.diff_edm:main:411 - You choose an extended source for this simulation
2025-12-23 00:48:58.420 | INFO     | elfdannagalac.scripts.diff_edm:main:412 - Considering the smooth contribution of the DM halo
2025-12-23 00:49:15.738 | INFO     | elfdannagalac.scripts.diff_edm:main:443 - Cosmic ray source
        SourceRedshift: Redshift z = 0.0043
    SourceIsotropicEmission: Random isotropic direction
    SourceParticleType: 11
    SourcePowerLawSpectrum: Random energy E = 1e-07 - 1e-05 EeV, dN/dE ~ E^-2

2025-12-23 00:49:15.738 | INFO     | elfdannagalac.scripts.diff_edm:main:444 - Particle 11, E = 5.27485e-07 EeV, x = 16.1587 0.0683999 -0.0395202 Mpc, p = -0.686251 0.160108 0.709524
2025-12-23 00:49:15.739 | INFO     | elfdannagalac.scripts.diff_edm:main:449 - Preparing CRpropa txtOutput to save data
2025-12-23 00:49:15.739 | INFO     | elfdannagalac.scripts.diff_edm:main:463 - Preparing Photon Observer
2025-12-23 00:49:15.778 | INFO     | elfdannagalac.scripts.diff_edm:main:496 - Adding Different modules to the simulation
crpropa::ModuleList: Number of Threads: 16
Run ModuleList
  Started Tue Dec 23 00:49:15 2025 : [ Finished ] 100%    Needed: 00:26:44  - Finished at Tue Dec 23 01:15:59 2025
2025-12-23 01:15:59.165 | INFO     | elfdannagalac.scripts.diff_edm:main:512 - Saving data to fits table
2025-12-23 01:24:31.929 | INFO     | elfdannagalac.tools.misc:elapsed_time:116 - Total Elapsed time: 00:53:09
```

Because, we need to estimate the grids to approximate the DM density and the magnetic field of the galaxy cluster, the total simulation for 1e5 particles takes approximately 55 min using 16 threads in a remote server with Alma 8.

## Getting a projected map of photons detected

From the output of the simulation, we can generate a plot of the number of photons reaching the detection surface (`ObserverSurface`) at the outskirts of the cluster. We only consider those photons going in the dierction of the observer at Earth within a field of view of the angular size of the cluster. All the calculations are packed in the app `countsmap`. An example of the help message for this app is:

```bash
countsmap --help

usage: countsmap [-h] --srcname Virgo Toy --srcz 0.0043 --srcpos (16,0,0) Mpc (16,0,0) Mpc (16,0,0) Mpc --srcdistance 16.0 Mpc --fov 8.0 deg
                 [--pixsize 0.1 deg] --rfile path/to/results.fits.gz [--frame icrs] [--odir ./]

Galaxy clusters project: elf-danna-galac December/2025

options:
  -h, --help            show this help message and exit

Galaxy Cluster:
  Input params

  --srcname Virgo Toy   Name of the cluster [to save files]
  --srcz 0.0043         Redshift of the cluster
  --srcpos (16,0,0) Mpc (16,0,0) Mpc (16,0,0) Mpc
                        Cartesian Position of the center of the cluster [Mpc]
  --srcdistance 16.0 Mpc
                        Distance to the cluster [Mpc]
  --fov 8.0 deg         FOV Angular size [deg]
  --pixsize 0.1 deg     Angular size of pixels used for binning [deg]
  --rfile path/to/results.fits.gz
                        Path to fits with results from CRpropa simulation
  --frame icrs          Name of frame used for SkyCoord representation
  --odir ./             Output directory to save files
```

And example of the output is:

```bash
countsmap --srcname "Virgo Toy" --srcz 0.0043 --srcpos -15.48616789 -2.09541755 3.43334085 --srcdistance 16.0 --fov 8.0 --pixsize 0.1 --rfile testVirgoSmoothHalo.fits.gz --frame icrs 
2025-12-25 04:39:04.321 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:112 - Getting parameters
2025-12-25 04:39:04.322 | INFO     | elfdannagalac.tools.misc:checkDir:128 - . already exists
2025-12-25 04:39:04.322 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:118 - Getting source position
2025-12-25 04:39:04.323 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:128 - Getting particle positions and directions
2025-12-25 04:39:27.400 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:134 - Getting FOV mask
2025-12-25 04:39:29.287 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:142 - Getting image data
2025-12-25 04:39:30.354 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:152 - Plotting
2025-12-25 04:39:32.171 | INFO     | elfdannagalac.tools.misc:elapsed_time:116 - Total Elapsed time: 00:00:27
```


## Physics behind

Here we simply describe the modeling behind some of the calculations, as for the magnetic field and DM density profiles.

### Magnetic field of the galaxy cluster

Observations indicate that the magnetic field in a galaxy cluster is turbulent with a strength $B$ decaying as a function of the distance to the center of the cluster, $r$. From radio, the spatial modulation (decay) of the magnetic field is proportional to the numeric density of electrons $n_e$ to some index $\eta$:

```math
B(r) = B_0\left(\frac{n_e}{n_0}\right)^{\eta}
```

$n_e$ can be determined from radio observations as well. Typically, a $\beta$ model is used to describe the observatoinal data:

```math
n_e = n_0\left(1+\frac{r}{r_\text{c}}\right)^{-\frac{3\beta}{2}}
```

To simulate the diffusion of electrons in a galaxy cluster, we model $B(r)$ in a cluster using `CRpropa`. We separate this in two steps. First, the turbulent magnetic field is computed over a grid covering the total spatial extension of the cluster. The rubulent field uses an instance of `SimpleGridTurbulence`, with appropiate $B_\text{RMS}$ and length scales $l_\text{min}$ and $l_\text{max}$. In a second step, the resulting turbulent field is scaled using a `Grid1f` object where the values are the ones obtained from $B(r)$ modulation function given previously.

### Dark Matter profiles and injection points 

We also need to tell `CRpropa` what will be the location of the sources to inject electrons in the DM halo of the galaxy cluster. From cosmological simulations, we expect two contributions to the DM halo of galaxy clusters, a smooth component from the main halo, and some (a lot of) subhalos embedded in the galaxy cluster. For the smooth component we use the same approach as in this [`CRpropa` tutorial](https://crpropa.github.io/CRPropa3/pages/example_notebooks/density/density_grid_sampling.html). We create a grid with values of the DM mass density profile (for now, we are only considering a Navarro-Frenk-White, NFW, profile) normalized to the maximum value of the density (either the saturation density $\rho_\text{sat}$ or the density at the radius given by the dimension of the grid, $`\rho_\text{DM}(r_\text{step})`$). Then, the particles to be simulated are injected with position sampled from the DM density profile of the main smooth halo. This give us the extended diffuse component.

For the subhalos, **work in progress**

## Coordinate system

All the vector positions and related calculations are done in the coordinate system of an observer at Earth. Then, for example, to compute the radial distance to the center of a galaxy cluster, we need to estimate the norm of the vector resulting from the difference between the vectors $\vec{r_\text{obs,cc}}$ and $\vec{r_\text{obs,particle}}$, $\left|\vec{r_\text{obs,particle}} - \vec{r_\text{obs,cc}}\right|$ (cc refers to cluster center).

## Observers

We place different observers to detect the photons and electrons at different distances from the center of the cluster. For now, all the observers are placed on spheres concentric to the cluster center. In the case of electrons, different distances from the center are used in order to get an idea of the 3D radial profile of injected electrons. The data collected from the electrons are also useful to get the 2D projected radial profile and compare directly with other observables from our simulation as the 2D projected radial profile for photons.

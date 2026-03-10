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
conda create -n edmcanna -c conda-forge python=3.13 c-compiler cxx-compiler fortran-compiler cmake llvm openmpi gsl -y
conda activate edmcanna
conda install -c conda-forge ucx
conda install -c conda-forge swig -y
conda install -c conda-forge openmp -y
# conda install -c conda-forge conda-gcc-specs -y
conda install -c conda-forge numpy scipy matplotlib jupyter notebook pandas astropy emcee -y
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
# wget https://github.com/CRPropa/CRPropa3/archive/refs/tags/3.2.1.tar.gz
# tar -xzf 3.2.1.tar.gz
git clone git@github.com:CRPropa/CRPropa3.git
cd CRPropa3/

# This is to compile CRpropa using conda's gcc compilers
mkdir build && cd build
CMAKE_PREFIX_PATH=${CONDA_PREFIX} cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} .. ${CMAKE_ARGS} -DSIMD_EXTENSIONS:STRING=native -Wno-dev -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 8

make test
make install
```
Other extra packages will be downloaded when installing this package, as loguru.

> [!NOTE]
> The policy option when building `CRpropa` is needed for compatibility with recent versions of cmake (but, needed now?)

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
$ diffedm --help
2026-03-05 21:59:54.800 | INFO     | elfdannagalac.scripts.diff_edm:main:256 - Getting parameters
usage: diffedm [-h] --srcname Virgo --srcz 0.0043 --dmprofile nfw --dmsource_type extended_smooth --m200 3.54e14 Msun --r200 1.406e3 kpc --ra 12 deg --dec 27 deg
               [--process [anna,decay]] --emin 100 --emax 1e5 --nparticles 10000 --brms 5 muG --lmin 0.02 Mpc --lmax 0.15 Mpc --eta_index 0.33 --core_radius 0.25 Mpc
               [--diff_epsilon 0.1] [--diff_scale 0.1] [--diff_alpha 0.1] --ncells 511 [--maxtries 100000] [--chunksize 32] [--ofname OFNAME] [--odir ./]

Galaxy clusters project: elf-danna-galac November/2025

options:
  -h, --help            show this help message and exit

Galaxy Cluster:
  Input params

  --srcname Virgo       Name of the cluster
  --srcz 0.0043         Redshift of the cluster
  --dmprofile nfw       Label for DM profile parametrization
  --dmsource_type extended_smooth
                        Point or Extended like DM injection sources
  --m200 3.54e14 Msun   Mass enclosed up to a radius where $ ho_{crit}=200$ [Msun]
  --r200 1.406e3 kpc    Radius where $ ho_{crit}=200$ [kpc]
  --ra 12 deg           Right Ascension [in deg]
  --dec 27 deg          Declination [deg]
  --process [anna,decay]
                        Annihilation or Decay?
  --emin 100            Minimum energy of electrons (GeV)
  --emax 1e5            Maximum energy of electrons (GeV)
  --nparticles 10000    Number of particles to be smulated
  --brms 5 muG          RMS value of the turbulence field [muG]
  --lmin 0.02 Mpc       Minimum scale of the turbulence field [kpc]
  --lmax 0.15 Mpc       Maximum scale of the turbulence field [kpc]
  --eta_index 0.33      Index of the Bfield dependance with radial distance
  --core_radius 0.25 Mpc
                        Value of the electron density core radius [kpc]
  --diff_epsilon 0.1    Anisotrpy factor of the diffusion coefficient
  --diff_scale 0.1      Scaling factor to set the diffusion coefficient
  --diff_alpha 0.1      Spectral index for the Diff_coeff = E^{-alpha}
  --ncells 511          Odd Number of points in the spatial grid
  --maxtries 100000     Maximum number of tries for source sampling
  --chunksize 32        Size of chunks to compute grids
  --ofname OFNAME       Output file name [fits,fits.gz]
  --odir ./             Output directory to save files
```

A simulation using some default parameters should be like this:

```bash
diffedm --srcname Abel --srcz 0.0308 --dmprofile nfw --dmsource_type extended_smooth --m200 3.545e14 --r200 1.5e3 --ra 12.0 --dec 27.0 --process decay --emin 100 --emax 1e5 --nparticles 10000 --brms 5.0 --lmin 20 --lmax 150 --eta_index 0.33  --core_radius 250 --ncells 511 --maxtries 1e8 --ofname testdecay.fits.gz
```

An example of the expected output is:

```bash
2026-03-05 22:06:20.312 | INFO     | elfdannagalac.scripts.diff_edm:main:256 - Getting parameters
2026-03-05 22:06:20.313 | INFO     | elfdannagalac.tools.misc:checkDir:128 - . already exists
2026-03-05 22:06:20.313 | INFO     | elfdannagalac.scripts.diff_edm:main:265 - Preparing simulation for DM decay in a cluster
2026-03-05 22:06:20.315 | WARNING  | elfdannagalac.dmsrc.halo:__init__:135 - Input R200 1500.000 kpc will be inconsistent with the rest of calculations.
 Setting R200 to the value obtained using the critical density: 1406.363 kpc
2026-03-05 22:12:34.292 | INFO     | elfdannagalac.dmsrc.halo:info:352 - 
Abel cluster configured with: 
        - Redshift: 0.031
        - D_lum: 129474.231 kpc
        - Coord: (12.000 deg,27.000 deg) [ICRS]
        - X: 112.841 Mpc
        - Y: 23.985 Mpc
        - Z: 58.780 Mpc
        - Cluster Radius [R200]: 1406.363 kpc
        - Total Mass [M200]: 3.545e+14 solMass
        - Scale radius: 86.108 kpc
        - Scale density: 2.313e+07 solMass / kpc3
        - Saturation Radius: 2.390e-07 kpc
        - Saturation Density: 8.333e+15 solMass / kpc3
        - Number of subhalos: 1225  (from [3.545e+09 solMass,3.545e+12 solMass])
        - Total mass in form of subhalos: 3.898e+13 solMass (11.0% of the cluster mass)
        - Annihilation emissivity [No sub]: 1.431e+21 solMass2 / kpc3 (6.058e+70 GeV2 / cm3)
        - Decay emissivity [No sub]: 3.545e+14 solMass (3.954e+71 GeV)
        - DM luminosity [Annihilation, No sub]: 3.49407e+42 erg / s
        - DM luminosity [Decay, No sub]: 6.33525e+41 erg / s

2026-03-05 22:12:34.292 | WARNING  | elfdannagalac.dmsrc.halo:info:363 - For the annihilation luminosity a candiate with a mass 100.00 GeV and thermal-average annihilation cross section 3.60e-24 cm3 / s were used. 
For decay, the luminosity was estimated assuming a lifetime of 1.00e27 s.
2026-03-05 22:12:34.292 | INFO     | elfdannagalac.scripts.diff_edm:main:300 - Processing Magnetic Field grid for cluster
2026-03-05 22:13:47.298 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:169 - Description of the field:
2026-03-05 22:13:47.298 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:170 - Correlation Length is: 39.18226 kpc
2026-03-05 22:13:47.298 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:171 - RMS B field is: 5.00000 microG
2026-03-05 22:13:47.298 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:172 - Mean B field is: 4.60698 microG
2026-03-05 22:13:47.298 | INFO     | elfdannagalac.magneticfield.bfields:get_cluster_field:173 - B field at the center of the cluster: 5.97097 microG
2026-03-05 22:14:56.666 | INFO     | elfdannagalac.scripts.diff_edm:main:343 - Preparing Diffusion module
2026-03-05 22:14:56.666 | INFO     | elfdannagalac.scripts.diff_edm:main:389 - Preparing dark matter source
2026-03-05 22:14:56.666 | INFO     | elfdannagalac.scripts.diff_edm:main:408 - You choose an extended source for this simulation
2026-03-05 22:14:56.666 | INFO     | elfdannagalac.scripts.diff_edm:main:409 - Considering the smooth contribution of the DM halo 
2026-03-05 22:14:56.667 | WARNING  | elfdannagalac.scripts.diff_edm:main:426 - To avoid numerical precision issues we do not use the saturation density during grid calculation. That implies larger computation times trying to sample regions where the probability is too low (1e-9). We will use the density evaluated at the size of the cells: 3.18965e+08 solMass / kpc3
2026-03-05 22:16:10.293 | INFO     | elfdannagalac.scripts.diff_edm:main:484 - Maximum of the PDF: 7.11199e-05 1 / kpc3
2026-03-05 22:16:10.294 | INFO     | elfdannagalac.scripts.diff_edm:main:485 - Cosmic ray source
        SourceRedshift: Redshift z = 0.0308
    SourceIsotropicEmission: Random isotropic direction
    SourceParticleType: 11
    SourcePowerLawSpectrum: Random energy E = 1e-07 - 0.0001 EeV, dN/dE ~ E^-2

2026-03-05 22:16:10.294 | INFO     | elfdannagalac.scripts.diff_edm:main:491 - Preparing CRpropa txtOutput to save data
2026-03-05 22:16:10.295 | INFO     | elfdannagalac.scripts.diff_edm:main:505 - Preparing Photon Observer
2026-03-05 22:16:10.295 | INFO     | elfdannagalac.scripts.diff_edm:main:526 - Observer centered at Vector(3.48192E+24, 7.40106E+23, 1.81376E+24) ([112841.43960637  23985.18841662  58780.07081334] kpc)
2026-03-05 22:16:10.335 | INFO     | elfdannagalac.scripts.diff_edm:main:541 - Adding Different modules to the simulation
crpropa::ModuleList: Number of Threads: 16
Run ModuleList
  Started Thu Mar  5 22:16:10 2026 : [ Finished ] 100%    Needed: 00:11:51  - Finished at Thu Mar  5 22:28:01 2026
2026-03-05 22:28:01.807 | INFO     | elfdannagalac.scripts.diff_edm:main:557 - Saving data to fits table
2026-03-05 22:28:56.147 | INFO     | elfdannagalac.tools.misc:elapsed_time:116 - Total Elapsed time: 00:22:35
```

~Because, we need to estimate the grids to approximate the DM density and the magnetic field of the galaxy cluster, the total simulation for 1e5 particles takes approximately 55 min using 16 threads in a remote server with Alma 8.~
The computation time has been reduced after we optimized the calculation of grids for the magnetic field and PDF for source sampling. Now, for a cubic grid of size $511^3$ cells, the software takes approx. 1 minute to compute all the values and load the `CRpropa` Grid instances. But, still, the computation times presented in example outputs were obtained from running the simulation on a server with Alma 8,, and accessing 16 threads. However, the computation of the grids is optimized even if no access to multithreading is available.

## Getting a projected map of photons detected

From the output of the simulation, we can generate a plot of the number of photons reaching the detection surface (`ObserverSurface`) at the outskirts of the cluster. We only consider those photons going in the dierction of the observer at Earth within a field of view of the angular size of the cluster. All the calculations are packed in the app `countsmap`. An example of the help message for this app is:

```bash
$ countsmap --help
2026-03-05 22:11:25.203 | INFO     | elfdannagalac.scripts.plotmaps:get_counts_map:112 - Getting parameters
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
countsmap --srcname Abel --srcz 0.0308 --srcpos 112.841 23.985 58.780 --srcdistance 129.474 --fov 1.6 --pixsize 0.05 --rfile testdecay.fits.gz
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

### Dark matter halo

A DM halo is described given its total mass and radial size. By default we use the value of the radius where the density of the DM halo is $200\rho_\text{crit}$, where $\rho_\text{crit}$ is the critical density of the uUniverse at redshift $z$. The input parameters to describe a halo are $M_{200}$ and $R_{200}$. For completeness, we verify that the value of $R_{200}$ is consistent with the provided total mass $M_{200}$. Other parameters are computed according to an specific density profile. Until now, we only consider the NFW profile. In this particular case, the NFW profile, we include a saturation radius where the density reach a constant value. The configuration of the DMHalo object includes the estimation of the scale radius $r_s$ and scale density $\rho_s$, the total luminosities for annihilation and decay of DM inside the DM halo without considering the effect of substructure; and the total number of subhalos in the range of masses $`[m_\text{min},m_\text{max}]`$ given a fraction $f_\text{sub}$ of the total mass $M_{200}$. 

The total luminosity for annihilation of DM inside the halo is:

```math
\mathfrak{L}_\text{anna} = 4\pi\frac{\langle\sigma v\rangle}{m_\text{DM}}\int_0^{R_{200}} {\rm d}r~r^2\rho^{2}(r)
```

while for decay, the luminosity is:

```math
\mathfrak{L}_\text{decay} = 4\pi\Gamma\int_0^{R_{200}} {\rm d}r~r^2\rho(r)
```

For example for a DM candidate with a mass of 100 GeV, and thermal annihilation cross-section $`\langle\sigma~v\rangle = 3.6\times10^{-24}~\text{cm}^3~\text{s}^{-1}`$ and decay lifetime $`\Gamma^{-1} = 10^{27}\text{s}`$, the luminosities are in the order of $`10^{42}\text{erg}~\text{s}^{-1}`$ and $`10^{41}\text{erg}~\text{s}^{-1}`$, respectively.

> [!IMPORTANT]
> The previous equations are the formal definitions of the luminosities. However, we provide only the result of the integral and the angular term in our code, as they are more meaningful for other calculations as the spatial PDFs used for source sampling.


The probability that a subhalo with mass $m_i$ and concentration $c_i$ is located at a position $r_i$ is given by:

```math
P_\text{sub}(r_i,m_i,c_i) = \frac{1}{\kappa_w} \frac{{\rm d}n}{{\rm d}V}\frac{{\rm d}n}{{\rm d}m}\frac{{\rm d}n}{{\rm d}c}
```

where $`\kappa_w`$ is obtained from the condition that $`P_\text{sub}`$ is normalized to 1:

```math
\int_0^{R_{200}}~\int_{m_\text{min}}^{m_\text{max}}~\int_{c_\text{min}=1}^{c_\text{max}(r_i,m_i)}~{\rm d}V~{\rm d}m~{\rm d}c P_\text{sub}(r_i,m_i,c_i) = 1
```

For the example given above, for a cluster with a mass of $`M_{200} = 3.4\times10^{14}~M_\odot`$, assuming that subhalos have masses in the range from $`[10^{-5}M_{200},10^{-1}M_{200}]`$, and that the total mass in form of subhalos represents $`\thicksim 11\%`$ of the total mass of the cluster (that is consistent with N-body simulations), gives a total number of subhalos of around 1200 subhalos that is also consistent, in average, with the number of observable galaxies for a galaxy cluster.

We consider the calculation of substructure because they are important when considering the annihilation of dark matter and the boost factor that results on the total gamma-ray emission.

All the previous calculations are included in the class `DMHalo`. An example of the configuration of an Abel-like galaxy cluster is shown in the ouput of the `diffedm` app above.

### Dark Matter profiles and injection points 

We also need to tell `CRpropa` what will be the location of the sources to inject electrons in the DM halo of the galaxy cluster. From cosmological simulations, we expect two contributions to the DM halo of galaxy clusters, a smooth component from the main halo, and some (a lot of) subhalos embedded in the galaxy cluster. For the smooth component we use the same approach as in this [`CRpropa` tutorial](https://crpropa.github.io/CRPropa3/pages/example_notebooks/density/density_grid_sampling.html). We create a grid with values of the DM mass density profile (for now, we are only considering a Navarro-Frenk-White, NFW, profile) normalized to the maximum value of the density (either the saturation density $\rho_\text{sat}$ or the density at the radius given by the dimension of the grid, $`\rho_\text{DM}(r_\text{step})`$). Then, the particles to be simulated are injected with position sampled from the DM density profile of the main smooth halo. This give us the extended diffuse component.

In particular, besides the spectrum of injected electrons, we expect a difference between the spatial signal obtained from annihilation and decay of DM particles with the former being more cuspy towards the center of the cluster and the last resulting in a more extended emission (yeah, exactly the same as in gamma-ray indirect DM searches). Additionally, we consider providing the spatial Probability Density Function (PDF) instead of the DM mass density profiles. For annihilation, because the expected injection rate of electrons is proportional to $`\rho^2`$, the spatial PDF is given by:

```math
\text{PDF}_\text{anna} = \frac{\rho^{2}(r)}{\mathfrak{L}_\text{anna}}
```

and for decay:

```math
\text{PDF}_\text{decay} = \frac{\rho(r)}{\mathfrak{L}_\text{decay}}
```

where $`\mathfrak{L}_\text{decay}`$ is just the enclosed mass up to radius $`R_{200}`$.

For the subhalos, **work in progress**

## Coordinate system

All the vector positions and related calculations are done in the coordinate system of an observer at Earth. Then, for example, to compute the radial distance to the center of a galaxy cluster, we need to estimate the norm of the vector resulting from the difference between the vectors $\vec{r_\text{obs,cc}}$ and $\vec{r_\text{obs,particle}}$, $\left|\vec{r_\text{obs,particle}} - \vec{r_\text{obs,cc}}\right|$ (cc refers to cluster center).

## Observers

We place different observers to detect the photons and electrons at different distances from the center of the cluster. For now, all the observers are placed on spheres concentric to the cluster center. In the case of electrons, different distances from the center are used in order to get an idea of the 3D radial profile of injected electrons. The data collected from the electrons are also useful to get the 2D projected radial profile and compare directly with other observables from our simulation as the 2D projected radial profile for photons.

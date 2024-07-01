# elf-danna-galac

Invoking a witch (or wizard?) to search for the black relics in clusters of galaxies!

Science Package to compute the expected photon flux from the propagation and diffusion of electrons/positrons produced via dark matter (DM) annihilation/decay in galaxy clusters.
The name of this science project is the acronym for ELectron Flux induced by Dark matter Annihilation And decay in GALaxy Clusters.

### Authors

1. Arlette Melo Galindo (UAM)
2. Miguel A. Sánchez Conde (IFT-UAM)
3. Sergio Hernández Cadena (TDLI-SJTU)


## Basic idea

DM represents at least 85% of the content of galaxy clusters. If this DM component is made of heavy particles (like WIMPs), products from their annihilation/decay should be electron-positron pairs. We should expect to this component to be "injected" (we refer to these pairs as a new component) into the IntraCluster Medium (ICM) and propagate through diffusion because of the presence of the cluster's magnetic field. We should expect to have an additional contribution to the photon flux with wavelengths from Radio and X-ray up to TeV gamma-rays, according to the mass of the DM candidate.

## Analysis plan

We use data from `cosmiXs` and `PPPC4-DMID` projects to compute the electron-positron flux for DM candidates with masses in the range from 100 GeV to 100 TeV. Then, we use `CRpropa` to estimate the photon flux obtained from the diffusion in the ICM after modelling the magnetic field. We also consider the Inverse Compton (IC) with other seed photon fields.

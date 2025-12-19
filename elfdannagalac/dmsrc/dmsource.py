###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing DM profiles used for source injection                         #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             December-2025                                                   #
###############################################################################

import numpy as np

from crpropa import GridProperties,Vector3d,Grid1f
from crpropa import DensityGrid
from crpropa import (
    Source,
    SourceMassDistribution,
    SourcePosition,
    SourceRedshift,
    SourceIsotropicEmission,
    SourceParticleType,
    SourcePowerLawSpectrum,
)

from crpropa import Mpc

from types import NoneType

# allowed_profiles = ["nfw","burkert","einasto"]
allowed_profiles      = ["nfw"]
allowed_spatial_types = ["pointlike","extended_smooth"]

def NFW_profile(
    r       : float,
    r_s     : float,
    rho_s   : float,
    r_sat   : float,
    rho_sat : float
) -> float:

    """
    Calculation of the mass density profile for a 
    NFW profile. Different to other cases, we need to include 
    a saturation radius to avoid divergence at the origin.
    No units are specifies, so in principle any self-consistent 
    election of values should work.
    ToDo: include units?
    
        :param r: Distance to the center of the DM halo
        :type r: float
        :param r_s: Scale radius of the DM halo
        :type r_s: float
        :param rho_s: Density at the scale radius
        :type rho_s: float
        :param r_sat: Radius where saturation density is reached
        :type r_sat: float
        :param rho_sat: Saturation density
        :type rho_sat: float
        :return: profile of mass density
        :rtype: float
    """

    x = r/r_s

    if r < r_sat:

        density = rho_sat

    else:

        density = rho_s/(x*(1+x)**2)

    return density

def dm_number_density(
    r       : float,
    dmpars  : list[float]|np.ndarray,
    dm_mass : float=5.,
    label   : str="nfw"
) -> float:

    """
    Get the number/particle density. Again, by default any self-consistent 
    set of units should work, but for now, we assume TeV/m**3 for mass density 
    and particles/m**3 for particle density.
    ToDo: Include units?
    
    :param r: Radius to the center of the DM halo
    :type r: float
    :param dmpars: List or array of parameters to get mass density
    :type dmpars: list[float] | np.ndarray
    :param dm_mass: Mass of the WIMP DM candidate. Default is 5
    :type dm_mass: float
    :param label: Name of the profile used. Default is NFW
    :type label: str
    """

    if label.lower() == "nfw":

        r_s,rho_s,r_sat,rho_sat = dmpars

        density = NFW_profile(r,r_s,rho_s,r_sat,rho_sat)

    number_rho = density/dm_mass

    return number_rho

def dm_mass_density(
    r       : float,
    dmpars  : list[float]|np.ndarray,
    label   : str="nfw"
) -> float:

    """
    Get the number/particle density. Again, by default any self-consistent 
    set of units should work, but for now, we assume TeV/m**3 for mass density 
    and particles/m**3 for particle density.
    ToDo: Include units?
    
    :param r: Radius to the center of the DM halo
    :type r: float
    :param dmpars: List or array of parameters to get mass density
    :type dmpars: list[float] | np.ndarray
    :param label: Name of the profile used. Default is NFW
    :type label: str
    """

    if label.lower() == "nfw":

        r_s,rho_s,r_sat,rho_sat = dmpars

        density = NFW_profile(r,r_s,rho_s,r_sat,rho_sat)

    return density

def SmoothDMHaloMassDensityGrid(
    dmhalo_center  : Vector3d,
    origin_lower   : Vector3d,
    ncells         : int,
    spacing        : float,
    dmpars         : list[float]|np.ndarray,
    dmprofile      : str="nfw"
    # save           : bool=False
) -> DensityGrid:
    """
    Get a Grid1f with the mass density of DM. 
    The density is computed in units of TeV/m**3.

    All vectors are given in the Observer's coordinate system (OCS). 
    All spatial quantities are assumed to be in Mpc.

    For the Grid1f we assume the same number of cells in each direction.
    As for now, we need to specify what is the type of gas density used 
    to compute the DensityGrid. This is specified by three booleans during 
    the declaration of the DensityGRid object. The booleans refer to the 
    cases where the DensityGrid represents H, HI or HII gas densities. 
    By default, we indicate that our DensityGrid is for H, but we don't 
    actually care about this, as we are only using this as a source 
    sampling function.

    This gonna take a while.

    Then, I will normalize by the maximum density to directly get a 
    mass distribution function normalized to one, and check if that 
    can accelerate the computation time. For the case of the NFW profile, 
    the maximum density should be the saturation density. Need to check 
    for the others DM halo profiles.
    
    :param cluster_center: Center of the cluster in the OCS. [Mpc]
    :type cluster_center: Vector3d
    :param origin_lower: Lower origin of the grid. [Mpc]
    :type origin_lower: Vector3d
    :param ncells: Number of cells used to get the grid
    :type ncells: int
    :param spacing: Separation between cells.[Mpc]
    :type spacing: float
    :param dmpars: Parameters to describe the DM mass density profile
    :type dmpars: list[float] | np.ndarray
    :param dmprofile: Name of DM profile used
    :type dmprofile: str
    :return: Grid of DM mass density [TeV/m**3]
    :rtype: DensityGrid
    """
    msg = "Unknown mass density profile"
    assert dmprofile in allowed_profiles,msg

    # rho_max = 5e+3

    # Checking the number of parameters
    if dmprofile.lower() == "nfw":

        msg = "Wrong number of parameters for the NFW profile"
        assert len(dmpars) == 4,msg

        rho_max = dmpars[-1]

    gridpos = GridProperties(origin_lower*Mpc,ncells,spacing*Mpc)
    gridpos.setClipVolume(True)

    dm_density = Grid1f(gridpos)

    for idx in range(ncells):

        for idy in range(ncells):

            for idz in range(ncells):

                dummyvec = Vector3d(idx,idy,idz)
                dummyvec = dummyvec*spacing + origin_lower
                dummyvec = dummyvec - dmhalo_center
                distance = np.sqrt(
                    dummyvec.x**2+dummyvec.y**2+dummyvec.z**2
                )

                rho = dm_mass_density(
                    distance,
                    dmpars,
                    label=dmprofile
                )

                dm_density.setValue(idx,idy,idz,rho/rho_max)

                if distance > 2.0:

                    dm_density.setValue(idx,idy,idz,0)

    dens = DensityGrid(dm_density,True,False,False)

    return dens

def SmoothDMHaloNumberDensityGrid(
    dmhalo_center  : Vector3d,
    origin_lower   : Vector3d,
    ncells         : int,
    spacing        : float,
    dmpars         : list[float]|np.ndarray,
    dmmass         : float,
    dmprofile      : str="nfw"
    # save           : bool=False
) -> DensityGrid:
    """
    Get a Grid1f with the number (particle) density of DM. 
    The density is computed in units of m**3.

    All vectors are given in the Observer's coordinate system (OCS). 
    All spatial quantities are assumed to be in Mpc.

    For the Grid1f we assume the same number of cells in each direction.
    As for now, we need to specify what is the type of gas density used 
    to compute the DensityGrid. This is specified by three booleans during 
    the declaration of the DensityGRid object. The booleans refer to the 
    cases where the DensityGrid represents H, HI or HII gas densities. 
    By default, we indicate that our DensityGrid is for H, but we don't 
    actually care about this, as we are only using this as a source 
    sampling function.

    This gonna take a while.

    Then, I will normalize by the maximum density to directly get a 
    mass distribution function normalized to one, and check if that 
    can accelerate the computation time. For the case of the NFW profile, 
    the maximum density should be the saturation density. Need to check 
    for the others DM halo profiles.

    :param dmhalo_center: Center of the cluster in the OCS. [Mpc]
    :type dmhalo_center: Vector3d
    :param origin_lower: Lower origin of the grid. [Mpc]
    :type origin_lower: Vector3d
    :param ncells: Number of cells used to get the grid
    :type ncells: int
    :param spacing: Separation between cells.[Mpc]
    :type spacing: float
    :param dmpars: Parameters to describe the DM mass density profile
    :type dmpars: list[float] | np.ndarray
    :param dmmass: Mass of the DM candidate. [TeV]
    :type dmmass: float
    :param dmprofile: Name of DM profile used
    :type dmprofile: str
    :return: Grid of DM mass density [TeV/m**3]
    :rtype: DensityGrid
    """

    msg = "Unknown mass density profile"
    assert dmprofile in allowed_profiles,msg

    # Checking the number of parameters
    if dmprofile.lower() == "nfw":

        msg = "Wrong number of parameters for the NFW profile"
        assert len(dmpars) == 4,msg

        rho_max = dmpars[-1]/dmmass

    gridpos = GridProperties(origin_lower*Mpc,ncells,spacing*Mpc)
    gridpos.setClipVolume(True)

    dm_density = Grid1f(gridpos)

    for idx in range(ncells):

        for idy in range(ncells):

            for idz in range(ncells):

                dummyvec = Vector3d(idx,idy,idz)
                dummyvec = dummyvec*spacing + origin_lower
                dummyvec = dummyvec - dmhalo_center
                distance = np.sqrt(
                    dummyvec.x**2+dummyvec.y**2+dummyvec.z**2
                )

                ndensity = dm_number_density(
                    distance,
                    dmpars,
                    dm_mass=dmmass,
                    label=dmprofile
                )

                dm_density.setValue(idx,idy,idz,ndensity/rho_max)

                if distance > 2.0:

                    dm_density.setValue(idx,idy,idz,0)

    dens = DensityGrid(dm_density,True,False,False)

    return dens

def DMSourceSmoothDistro(
    dmdensity   : DensityGrid,
    max_density : float,
    maxTries    : int,
    xmin        : float,
    xmax        : float,
    ymin        : float,
    ymax        : float,
    zmin        : float,
    zmax        : float
) -> SourceMassDistribution:

    """
    Get SourceFeature to add to the Source definition. 
    The DM densioty can be either mass or number density.
    
    :param dmdensity: [Mass or Number] DM density
    :type dmdensity: DensityGrid
    :param max_density: Max. value of the density used for normalization.
    :type max_density: float
    :param maxTries: Maximum number of trials to get a source.
    :type maxTries: int
    :param xmin: Lower Limit for source sampling in the x direction [Mpc]
    :type xmin: float
    :param xmax: Lower Limit for source sampling in the x direction [Mpc]
    :type xmax: float
    :param ymin: Lower Limit for source sampling in the x direction [Mpc]
    :type ymin: float
    :param ymax: Lower Limit for source sampling in the x direction [Mpc]
    :type ymax: float
    :param zmin: Lower Limit for source sampling in the x direction [Mpc]
    :type zmin: float
    :param zmax: Lower Limit for source sampling in the x direction [Mpc]
    :type zmax: float
    """

    dm_sources = SourceMassDistribution(dmdensity,max_density)
    dm_sources.setXrange(xmin*Mpc,xmax*Mpc)
    dm_sources.setYrange(ymin*Mpc,ymax*Mpc)
    dm_sources.setZrange(zmin*Mpc,zmax*Mpc)
    dm_sources.setMaximalTries(maxTries)

    return dm_sources

def preparePointLikeDMSource(
    dmsource_pos : Vector3d,
    redshift     : float,
    part_type    : int,
    emin         : float,
    emax         : float,
    pl_index     : float
) -> Source:

    """
    Prepare a pointlike DM source to inject particles of type part_type.

    All vectors are assumed to be in the Observer's coordinate system (OCS).

    Injection of particles follow a PowerLaw in energy spectrum between 
    emin and emax with index pl_index. The energies emin and emax are given 
    in GeV.
    
    :param dmsource_pos: Position  [Mpc]
    :type dmsource_pos: Vector3d
    :param redshift: Redshift
    :type redshift: float
    :param part_type: Particle type following CRpropa convention
    :type part_type: int
    :param emin: Minimum energy of particles to be injected
    :type emin: float
    :param emax: Maximum energy of particles to be injected
    :type emax: float
    :param pl_index: Spectral index of PowerLaw
    :type pl_index: float
    """

    dms = Source()

    dms.add(SourcePosition(dmsource_pos))
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms

def prepareSmoothExtendedDMSource(
    dmdensity   : DensityGrid,
    max_density : float,
    maxTries    : int,
    xmin        : float,
    xmax        : float,
    ymin        : float,
    ymax        : float,
    zmin        : float,
    zmax        : float,
    redshift    : float,
    part_type   : int,
    emin        : float,
    emax        : float,
    pl_index    : float
) -> Source:

    """
    Docstring for prepareSmoothExtendedDMSource
    
    :param dmdensity: [Mass or Number] DM density
    :type dmdensity: DensityGrid
    :param max_density: Max. value of the density used for normalization.
    :type max_density: float
    :param maxTries: Maximum number of trials to get a source.
    :type maxTries: int
    :param xmin: Lower Limit for source sampling in the x direction [Mpc]
    :type xmin: float
    :param xmax: Lower Limit for source sampling in the x direction [Mpc]
    :type xmax: float
    :param ymin: Lower Limit for source sampling in the x direction [Mpc]
    :type ymin: float
    :param ymax: Lower Limit for source sampling in the x direction [Mpc]
    :type ymax: float
    :param zmin: Lower Limit for source sampling in the x direction [Mpc]
    :type zmin: float
    :param zmax: Lower Limit for source sampling in the x direction [Mpc]
    :type zmax: float
    :param redshift: Redshift to the center of the DM halo
    :type redshift: float
    :param part_type: Particle type to be injected
    :type part_type: int
    :param emin: Minimum energy of particles to be injected
    :type emin: float
    :param emax: Maximum energy of particles to be injected
    :type emax: float
    :param pl_index: Spectral index of PowerLaw
    :type pl_index: float
    """

    dms = Source()

    dm_sources = SourceMassDistribution(dmdensity,max_density)
    dm_sources.setXrange(xmin*Mpc,xmax*Mpc)
    dm_sources.setYrange(ymin*Mpc,ymax*Mpc)
    dm_sources.setZrange(zmin*Mpc,zmax*Mpc)
    dm_sources.setMaximalTries(maxTries)

    dms.add(dm_sources)
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms


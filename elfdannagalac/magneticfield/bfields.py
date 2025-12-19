###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing magnetic field in the intra cluster medium                    #
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
from crpropa import SimpleTurbulenceSpectrum,SimpleGridTurbulence
from crpropa import ModulatedMagneticFieldGrid
from crpropa import muG,Mpc,kpc

from loguru import logger

def get_cluster_field(
    brms           : float,
    lmin           : float,
    lmax           : float,
    cluster_center : Vector3d,
    eta_index      : float,
    core_radius    : float,
    origin_lower   : Vector3d,
    ncells         : int,
    spacing        : float,
    seed           : int=42,
    # save           : bool=False
) -> ModulatedMagneticFieldGrid:
    
    """
    Vectors for the cluster center and the lower origin of the spatial grid 
    assume that are given in Mpc.
    The same is for the core radius and spacing.
    And also for the min and max values of the turbulence scales.
    Magnetic field strengths are given in microGauss (muG).
    All the vectors are in the observer's coordinates system (OCS) with 
    the x axis connecting the observer and cluster center (line of sight,los).
    We assume a cartesian system of coordinates.
    
        :param brms: RMS Value of the magnetic field strength [muG]
        :type brms: float
        :param lmin: Min. lenght scale of the turbulent field [Mpc]
        :type lmin: float
        :param lmax: Max. lenght scale of the turbulent field [Mpc]
        :type lmax: float
        :param cluster_center: Center of the cluster in the OCS [Mpc]
        :type cluster_center: Vector3d
        :param eta_index: Index of B field decrease with radius
        :type eta_index: float
        :param core_radius: Core radius of the e number density profile [Mpc]
        :type core_radius: float
        :param origin_lower: Location of the far lower end of the grid [Mpc]
        :type origin_lower: Vector3d
        :param ncells: Number of cells used to build the grid
        :type ncells: int
        :param spacing: Separation between cells
        :type spacing: float
        :param seed: Random number generation
        :type seed: int
        :Return a ModulatedMagneticFieldGrid for a galaxy cluster

    First, we create the spatial grid used to compute all the quantities 
    all points outside the box are set to zero by default.
    We use the same number of cells for each direction

    Then, we create the scalar grid to modulate the stength 
    of the cluster's magnetic field.
    The modulation is following results from 
    radio and X-ray observations, and from 
    MHD cosmological simulation where the 
    magnetific field scale with the numerical electron density.
    For the moment, we simplfy the expresion 
    of the spectral index to be $(3\eta)/2$
    
    This gonna take a while :c
    """

    # Create the grid used to save the data
    gridpos = GridProperties(origin_lower*Mpc,ncells,spacing*Mpc)
    gridpos.setClipVolume(True)

    # Then, we define the turbulence spectrum
    turbulence = SimpleTurbulenceSpectrum(brms*muG,lmin*Mpc,lmax*Mpc)

    # And this is the magnetic field
    Bfield = SimpleGridTurbulence(turbulence,gridpos,seed)

    # for completitud, we print some info
    l_corr = Bfield.getCorrelationLength()/kpc
    Brms   = Bfield.getBrms()/muG
    bmean  = Bfield.getMeanFieldStrength()/muG
    B0     = Bfield.getField(cluster_center*Mpc).getR()/muG

    logger.info("Description of the field:")
    logger.info(f"Correlation Length is: {l_corr:0.3f} kpc")
    logger.info(f"RMS B field is: {Brms} microG")
    logger.info(f"Mean B field is: {bmean} microG")
    logger.info(f"B field at the center of the cluster: {B0:0.3f} microG")

    scale = Grid1f(gridpos)

    for idx in range(ncells):

        for idy in range(ncells):

            for idz in range(ncells):

                dummyvec = Vector3d(idx,idy,idz)

                dummyvec = dummyvec*spacing + origin_lower
                dummyvec = dummyvec - cluster_center
                distance = np.sqrt(
                    dummyvec.x**2+dummyvec.y**2+dummyvec.z**2
                )

                val = (1+distance**2/core_radius**2)**(-3*eta_index/2)
                scale.setValue(idx,idy,idz,val)

                if distance > 2.0:

                    scale.setValue(idx,idx,idz,0)

    # Then, this is the modulated field
    my_bfield = ModulatedMagneticFieldGrid(Bfield.getGrid(),scale)

    return my_bfield

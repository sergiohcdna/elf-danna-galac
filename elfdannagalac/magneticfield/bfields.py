###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing magnetic field in the intra cluster medium                    #
#   - Reducing computation time for grid calculations                         #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             December-2025                                                   #
#             March-2026                                                      #
###############################################################################

import astropy.units as u
import numpy as np

from crpropa import GridProperties,Vector3d,Grid1f
from crpropa import SimpleTurbulenceSpectrum,SimpleGridTurbulence
from crpropa import ModulatedMagneticFieldGrid
from crpropa import muG

from loguru import logger

# I will avoid using the crpropa units as I need to hard-code
# most of them, and I want this to be a more general software
# I will update the function to use astropy units and Quantities

def get_cluster_field(
    cluster_center : u.Quantity,
    obsRadius      : u.Quantity,
    core_radius    : u.Quantity,
    brms           : u.Quantity,
    lmin           : u.Quantity,
    lmax           : u.Quantity,
    eta_index      : float,
    ncells         : int,
    chunksize      : int = 32,
    seed           : int = 42,
    # save           : bool=False
) -> ModulatedMagneticFieldGrid:
    
    r"""
    Calculation of a Turbulent magnetic field with a radial 
    damping term. This configuration is for galaxy clusters. 
    The calculation is done over a cubic grid centered on 
    cluster_center and number of cells ncells.

    The radial damping is computed using:

    $B(r) = \left(1+\frac{r^2}{r_\text{core}^2}\right)^{-3\eta /2}$

    The damping is following results from 
    radio and X-ray observations, and from 
    MHD cosmological simulation where the 
    magnetific field scale with the numerical electron density.
    For the moment, we simplfy the expresion 
    of the spectral index to be $(3\eta)/2$

    Because CRpropa uses units from the SI system, and we want ti 
    avoid hardcoding most of the distance units (like Mpc, kpc, etc),
    we use atropy units for management and convert all the distances 
    and positions to meters.

        :param cluster_center: Center of the cluster
        :type cluster_center: u.Quantity
        :param obsRadius: Radius of the observer/cluster
        :type obsRadius: u.Quantity
        :param core_radius: Core radius of the electron density
        :type core_radius: u.Quantity
        :param brms: RMS value of the magnetic strength [uG]
        :type brms: u.Quantity
        :param lmin: Minimum length scale of the magnetic field
        :type lmin: u.Quantity
        :param lmax: Maximum length scale of the magnetic field
        :type lmax: u.Quantity
        :param eta_index: Index of the electron density profile
        :type eta_index: float
        :param ncells: [Odd] number of cells in the grid
        :type ncells: int
        :param chunksize: Number of points for each chunk
        :type chunksize: int
        :param seed: Seed used for the random generator
        :type seed: int
        :return: Turbulent magnetic field
        :rtype: ModulatedMagneticFieldGrid
    """

    lunit   = core_radius.unit
    ccenter = cluster_center.to(lunit)
    obs_r   = obsRadius.to(lunit)
    lmin_   = lmin.to(lunit)
    lmax_   = lmax.to(lunit)
    box_or  = ccenter - obs_r
    box_f   = ccenter + obs_r
    step    = 2*obs_r/(ncells-1)
    brms_   = brms.to(u.uG)

    xs = np.linspace(box_or[0].value,box_f[0].value,num=ncells)
    ys = np.linspace(box_or[1].value,box_f[1].value,num=ncells)
    zs = np.linspace(box_or[2].value,box_f[2].value,num=ncells)

    box = Vector3d(
        box_or[0].to(u.m).value,
        box_or[1].to(u.m).value,
        box_or[2].to(u.m).value
    )

    center = Vector3d(
        ccenter[0].to(u.m).value,
        ccenter[1].to(u.m).value,
        ccenter[2].to(u.m).value
    )

    kappa = np.zeros((xs.size,ys.size,zs.size))

    for i in range(0,ncells,chunksize):

        istop  = np.min([i+chunksize,ncells])
        xsmall = xs[i:istop]

        for j in range(0,ncells,chunksize):

            jstop  = np.min([j+chunksize,ncells])
            ysmall = ys[j:jstop]

            for k in range(0,ncells,chunksize):

                kstop  = np.min([k+chunksize,ncells])
                zsmall = zs[k:kstop]

                Xchunk = xsmall[:,None,None]
                Ychunk = ysmall[None,:,None]
                Zchunk = zsmall[None,None,:]

                r = np.sqrt(
                    (Xchunk - ccenter[0].value)**2 + 
                    (Ychunk - ccenter[1].value)**2 + 
                    (Zchunk - ccenter[2].value)**2
                )*lunit

                val = (1+r**2/core_radius**2)**(-3*eta_index/2)

                val[r > obs_r] = 0

                kappa[i:istop,j:jstop,k:kstop] = val


    # Create the grid used to save the data
    gridpos = GridProperties(box,ncells,step.to(u.m).value)
    gridpos.setClipVolume(True)

    # Then, we define the turbulence spectrum
    turbulence = SimpleTurbulenceSpectrum(
        brms_.value*muG,
        lmin_.to(u.m).value,
        lmax_.to(u.m).value
    )

    # And this is the magnetic field
    Bfield = SimpleGridTurbulence(turbulence,gridpos,seed)

    # for completitud, we print some info
    l_corr = (Bfield.getCorrelationLength()*u.m).to(u.kpc)
    Brms   = Bfield.getBrms()/muG
    bmean  = Bfield.getMeanFieldStrength()/muG
    B0     = Bfield.getField(center).getR()/muG

    logger.info("Description of the field:")
    logger.info(f"Correlation Length is: {l_corr:0.5f}")
    logger.info(f"RMS B field is: {Brms:0.5f} microG")
    logger.info(f"Mean B field is: {bmean:0.5f} microG")
    logger.info(f"B field at the center of the cluster: {B0:0.5f} microG")

    scale = Grid1f(gridpos)

    for idx in range(gridpos.Nx):

        for idy in range(gridpos.Ny):

            for idz in range(gridpos.Nz):

                scale.setValue(idx,idy,idz,kappa[idx,idy,idz])

    # Then, this is the modulated field
    my_bfield = ModulatedMagneticFieldGrid(Bfield.getGrid(),scale)

    return my_bfield

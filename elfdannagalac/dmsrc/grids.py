###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing DM profiles used for source injection                         #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             April-2026                                                      #
###############################################################################

import astropy.units as u
import numpy as np

from crpropa import GridProperties,Vector3d,Grid1f
from crpropa import DensityGrid

from .profiles import dm_mass_density
from .smooth import get_smooth_dm_density

from ..tools.conversions import convert_density

from .halo import DMHalo

def SmoothDMHaloMassDensityGrid(
    dmhalo_center : u.Quantity,
    obs_radius    : u.Quantity,
    ncells        : int,
    halo          : DMHalo,
    r_inner       : u.Quantity,
    rho_inner     : u.Quantity,
    chunksize     : int = 32
) -> DensityGrid :

    lunit   = halo.rs.unit
    halo_c  = dmhalo_center.to(lunit)
    obs_r   = obs_radius.to(lunit)
    box_or  = halo_c - obs_r
    box_f   = halo_c + obs_r
    step    = 2*obs_r/(ncells-1)
    mass    = halo.m200

    xs = np.linspace(box_or[0].value,box_f[0].value,num=ncells)
    ys = np.linspace(box_or[1].value,box_f[1].value,num=ncells)
    zs = np.linspace(box_or[2].value,box_f[2].value,num=ncells)

    # Now, we precompute the values of the density with numpy

    rho_ = np.zeros((xs.size,ys.size,zs.size))

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
                    (Xchunk - halo_c[0].value)**2 + 
                    (Ychunk - halo_c[1].value)**2 + 
                    (Zchunk - halo_c[2].value)**2
                )*lunit

                rho_[i:istop,j:jstop,k:kstop] = halo.rho_tot(r).value

                if r_inner >= step:

                    rho_[np.where(r <= r_inner)] = rho_inner.value

    rho_    = rho_/mass.value

    # Because CRpropa use SI unit system
    # We convert length quantities to meters
    # and forget about multiplying by dimension factors
    # as we handle all the operations with astropy.units
    box = Vector3d(
        box_or[0].to(u.m).value,
        box_or[1].to(u.m).value,
        box_or[2].to(u.m).value
    )

    gridpos = GridProperties(box,ncells,step.to(u.m).value)
    gridpos.setClipVolume(True)

    # Then, we only need to assign the values to the CRpropa grid
    dm_density = Grid1f(gridpos)

    for idx in range(gridpos.Nx):

        for idy in range(gridpos.Ny):

            for idz in range(gridpos.Nz):

                dm_density.setValue(idx,idy,idz,rho_[idx,idy,idz])

    # dm_density.setInterpolationType(TRICUBIC)

    dens = DensityGrid(dm_density,True,False,False)

    return dens

def SmoothDMHaloMassSquaredDensityGrid(
    dmhalo_center : u.Quantity,
    obs_radius    : u.Quantity,
    ncells        : int,
    halo          : DMHalo,
    r_inner       : u.Quantity,
    rho_inner     : u.Quantity,
    chunksize     : int = 32,
    subhalos      : bool = False,
) -> DensityGrid:
    r"""
    Get a Grid1f with a PDF for sampling position following the 
    Squared DM density in a spherical halo. The PDF is normalized by 
    the annihilation luminosity

    $PDF = \frac{\rho(r)^2}{\mathfrak{L}_\text{Anna}}$

    All vectors are given in the Observer's coordinate system (OCS). 

    For the Grid1f we assume the same number of cells in each direction.
    As for now, we need to specify what is the type of gas density used 
    to compute the DensityGrid. This is specified by three booleans during 
    the declaration of the DensityGRid object. The booleans refer to the 
    cases where the DensityGrid represents H, HI or HII gas densities. 
    By default, we indicate that our DensityGrid is for H, but we don't 
    actually care about this, as we are only using this as a source 
    sampling function.

    Then, we normalize by the annihilation emissivity to directly get a 
    function normalized to one, and check if accelerate the computation time.

    We ask for an odd number of cells to make sure that the center of 
    the DM halo is an actual point (vertice) of the grid. This is to do 
    a correct source sampling.

        :param dmhalo_center: Center of the DM halo
        :type dmhalo_center: u.Quantity
        :param obs_radius: Radius of the observer/halo $R_{200}$
        :type obs_radius: u.Quantity
        :param ncells: [Odd] Number of cells
        :type ncells: int
        :param rs: Scale radius
        :type rs: u.Quantity
        :param rhos: Scale density
        :type rhos: u.Quantity
        :param rsat: Saturation radius
        :type rsat: u.Quantity
        :param rhosat: Saturation Density
        :type rhosat: u.Quantity
        :param rtrunc: Truncation radius
        :type rtrunc: u.Quantity
        :param total_dmlum: Total Annihilation luminosity
        :type total_dmlum: u.Quantity
        :param chunksize: Number of points to compute per chunk
        :type chunksize: int
        :param dmprofile: Label of the DM density profile
        :type dmprofile: str
        :return: PDF Grid for annihilation DM
        :rtype: DensityGrid
    """

    # We use the units of the scale radius as a natural choice
    # to compare and convert between the different units

    # As you can see, I am not checking the dimension of the arrays

    lunit   = halo.rs.unit
    halo_c  = dmhalo_center.to(lunit)
    obs_r   = obs_radius.to(lunit)
    box_or  = halo_c - obs_r
    box_f   = halo_c + obs_r
    step    = 2*obs_r/(ncells-1)

    # xs = np.arange(box_or[0].value,box_f[0].value,step=step.value)
    # ys = np.arange(box_or[1].value,box_f[1].value,step=step.value)
    # zs = np.arange(box_or[2].value,box_f[2].value,step=step.value)
    xs = np.linspace(box_or[0].value,box_f[0].value,num=ncells)
    ys = np.linspace(box_or[1].value,box_f[1].value,num=ncells)
    zs = np.linspace(box_or[2].value,box_f[2].value,num=ncells)

    # Now, we precompute the values of the density with numpy

    rho_ = np.zeros((xs.size,ys.size,zs.size))

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
                    (Xchunk - halo_c[0].value)**2 + 
                    (Ychunk - halo_c[1].value)**2 + 
                    (Zchunk - halo_c[2].value)**2
                )*lunit

                if subhalos:

                    dmd   = halo.rho_smooth(r).value
                    dmlum = halo.l_dm_anna_sub

                    if r_inner >= step:

                        dmd[np.where(r <= r_inner)] = rho_inner.value

                else:

                    dmd   = halo.rho_tot(r).value
                    dmlum = halo.l_dm_anna

                    if r_inner >= step:

                        dmd[np.where(r <= r_inner)] = rho_inner.value

                rho_[i:istop,j:jstop,k:kstop] = dmd**2

    rho_ = rho_/dmlum.value

    # Because CRpropa use SI unit system
    # We convert length quantities to meters
    # and forget about multiplying by dimension factors
    # as we handle all the operations with astropy.units
    box = Vector3d(
        box_or[0].to(u.m).value,
        box_or[1].to(u.m).value,
        box_or[2].to(u.m).value
    )

    gridpos = GridProperties(box,ncells,step.to(u.m).value)
    gridpos.setClipVolume(True)

    # Then, we only need to assign the values to the CRpropa grid
    dm_density = Grid1f(gridpos)

    for idx in range(gridpos.Nx):

        for idy in range(gridpos.Ny):

            for idz in range(gridpos.Nz):

                dm_density.setValue(idx,idy,idz,rho_[idx,idy,idz])

    # dm_density.setInterpolationType(TRICUBIC)

    dens = DensityGrid(dm_density,True,False,False)

    return dens

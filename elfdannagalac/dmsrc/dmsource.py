###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing DM profiles used for source injection                         #
#   - Including units                                                         #
#   - Replace list of parameters by DMHalo class                              #
#   - Removing some functions                                                 #
#   - Add a truncation radius to set densities to zero fi r > rtrunc          #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             December-2025                                                   #
#             February-2026                                                   #
###############################################################################

import astropy.units as u
import numpy as np

from scipy.integrate import quad

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
# from crpropa import TRICUBIC

from ..tools.conversions import convert_density,convert_mass

from ..tools.customerrors import DMProfileError

from loguru import logger

# allowed_profiles = ["nfw","burkert","einasto"]
allowed_profiles      = ["nfw"]
allowed_spatial_types = ["pointlike","extended_smooth"]

def get_rhosat(dmmass:u.Quantity,sigmav:u.Quantity)->u.Quantity:

    """
    Compute the saturation density. We use the formula given in:
    https://clumpy.gitlab.io/CLUMPY/v3.1.1/physics_profiles.html:

    $\rhosat = 3\times10^{18} \times (10^-26 / \langle\sigma v\rangle)~M_\odot~kpc^{-3}.$

    :param dmmass: Mass of the DM candidate
    :type dmmass: u.Quantity
    :param sigmav: Thermal-average annihilation cross section
    :type sigmav: u.Quantity
    :return: Saturation density
    :rtype: u.Quantity
    
    """

    dmm = convert_mass(dmmass,new_unit=u.GeV)
    sv  = sigmav.to(u.cm**3/u.s)

    rhosat = 3e18*(dmm.value/100.0)/(sv.value/1e-26)

    return rhosat*u.Msun/u.kpc**3

def NFW_profile(
    r           : u.Quantity,
    rs          : u.Quantity,
    rhos        : u.Quantity,
    rsat        : u.Quantity,
    rhosat      : u.Quantity,
    rtrunc      : u.Quantity,
    length_unit : u.Unit = u.Mpc
) -> u.Quantity:

    """
    Calculation of the mass density profile for a 
    NFW profile. Different to other cases, we need to include 
    a saturation radius to avoid divergence at the origin.  
    We include units.

    The function supports to work either with Quantities of 
    dimension 1 (float) or dimension greater than 2 (arrays). 
    The case for arrays is used during grid computation.
    
    :param r: Distance to the center of the DM halo
    :type r: u.Quantity
    :param rs: Scale radius of the DM halo
    :type rs: u.Quantity
    :param rhos: Density at the scale radius
    :type rhos: u.Quantity
    :param rsat: Radius where saturation density is reached
    :type rsat: u.Quantity
    :param rhosat: Saturation density
    :type rhosat: u.Quantity
    :param rtrunc: Truncation radius
    :type rtrunc: u.Quantity
    :param length_unit: Unit for distance comparison [default= u.Mpc]
    type length_unit: u.Unit
    :return: DM mass density
    :rtype: u.Quantity
    """

    if r.shape == ():

        # this is for the scalar part

        x = r.to(length_unit)/rs.to(length_unit)

        if r.to(length_unit) <= rsat.to(length_unit):

            density = rhosat

        elif r.to(length_unit) > rtrunc.to(length_unit):

            density = 0*(rhos.unit)

        else:

            density = rhos/(x*(1+x)**2)

    else:

        # this is for the array

        density = u.Quantity(np.zeros(r.shape),unit=rhos.unit)
        
        inner = r.to(length_unit) <= rsat.to(length_unit)
        inter = np.logical_and(
            r.to(length_unit) > rsat.to(length_unit), 
            r.to(length_unit) <= rtrunc.to(length_unit)
        )

        density[inner] = rhosat

        if np.any(inter):

            x = r[inter].to(length_unit) / rs.to(length_unit)

            density[inter] = rhos/(x*(1+x)**2)

    return density

def get_enclosed_mass_nfw(
    r           : u.Quantity,
    rs          : u.Quantity,
    rhos        : u.Quantity,
    rsat        : u.Quantity,
    rhosat      : u.Quantity,
    rtrunc      : u.Quantity,
    length_unit : u.Unit = u.Mpc
) -> u.Quantity:

    """
    Total DM mass enclosed up to a radius r for a spherical 
    halo with density described by the NFW profile.

    To apply the same operations through the different functions, 
    I will split the integral in the three different ranges as 
    for the calculation of the jfactor/dfactor  .
    This is only to include the term close to the saturation radius 
    where the density has a peak. 
    Again, this is only particular to the case of NFW profile. 
    Other DM profiles have not divergence at the center of the halo. 
    But, in this case, I need to consider different cases 
    according to the integration radii: 

    1. If r < r_sat (one integration)

    2. If rsat < r < rs (two integrations)

    3. If r > rs (three integrations)
        
    :param r: Distance from the center of the DM halo
    :type r: u.Quantity
    :param rs: Scale radius
    :type rs: u.Quantity
    :param rhos: Scale density (rho(rs) = rhos)
    :type rhos: u.Quantity
    :param rsat: Saturation radius
    :type rsat: u.Quantity
    :param rhosat: Saturation density
    :type rhosat: u.Quantity
    :param rtrunc: Truncation radius
    :type rtrunc: u.Quantity
    :param length_unit: Unit for distance comparison [default= u.Mpc]
    :type length_unit: u.Unit
    :return: Total enclosed DM mass
    :rtype: u.Quantity
    """

    r_      = r.to(length_unit)
    rs_     = rs.to(length_unit)
    rsat_   = rsat.to(length_unit)
    rtrunc_ = rtrunc.to(length_unit)
    rhos_   = convert_density(rhos,new_unit=u.Msun/length_unit**3)
    rhosat_ = convert_density(rhosat,new_unit=u.Msun/length_unit**3)

    int_args = (rs_,rhos_,rsat_,rhosat_,rtrunc_,length_unit)

    def integrand(r_,rs,rhos,rsat,rhosat,rtrunc,length_unit=length_unit):

        dm = (r_*length_unit)**2*NFW_profile(
            r_*length_unit,
            rs,
            rhos,
            rsat,
            rhosat,
            rtrunc,
            length_unit=length_unit
        )

        return dm.value

    if r_ < rsat_:

        total_mass = quad(
            integrand,
            0,
            r_.value,
            args=int_args
        )[0]

    if rsat_ < r_ < rs_:

        # Two integrals

        m1 = quad(
            integrand,
            0,
            rsat_.value,
            args=int_args
        )[0]

        m2 = quad(
            integrand,
            rsat_.value,
            r_.value,
            args=int_args
        )[0]

        total_mass = m1 + m2

    if r_ > rs_:

        # three integrals

        m1 = quad(
            integrand,
            0,
            rsat_.value,
            args=int_args
        )[0]

        m2 = quad(
            integrand,
            rsat_.value,
            rs_.value,
            args=int_args
        )[0]

        m3 = quad(
            integrand,
            rs_.value,
            r_.value,
            args=int_args
        )[0]

        total_mass = m1 + m2 + m3

    return 4*np.pi*total_mass*u.Msun

def dm_mass_density(
    r         : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    rtrunc    : u.Quantity,
    dmprofile : str = "nfw",
) -> u.Quantity:

    """
    Get the mass density. The density is computed in Units of 
    u.Msun/(rs.unit)**3. We set the truncation radius to 
    $R_{200}$.

    :param r: Distance from the center of the DM halo
    :type r: u.Quantity
    :param rs: Scale radius
    :type rs: u.Quantity
    :param rhos: Scale density
    :type rhos: u.Quantity
    :param rsat: Saturation radius
    :type rsat: u.Quantity
    :param rhosat: Saturation density
    :type rhosat: u.Quantity
    :param rtrunc: Truncation radius
    :type rtrunc: u.Quantity
    :param dmprofile: Label of the DM profile
    :type dmprofile: str
    :return: DM density
    :rtype: Quantity
    """

    if dmprofile.lower() == "nfw":

        density = NFW_profile(
            r.to(rs.unit),
            rs,
            rhos,
            rsat,
            rhosat,
            rtrunc,
            length_unit=rs.unit
        )

    else:

        logger.error(DMProfileError(f"Unknown {dmprofile} Profile"))

    return density

def SmoothDMHaloMassDensityGrid(
    dmhalo_center  : u.Quantity,
    obs_radius     : u.Quantity,
    ncells         : int,
    rs             : u.Quantity,
    rhos           : u.Quantity,
    rsat           : u.Quantity,
    rhosat         : u.Quantity,
    rtrunc         : u.Quantity,
    total_mass     : u.Quantity,
    chunksize      : int = 32,
    dmprofile      : str = "nfw"
    # save           : bool=False
) -> DensityGrid:
    
    r"""
    Get a Grid1f with a PDF for sampling position following the 
    DM density in a spherical halo. The PDF is normalized by 
    the decay luminosity (Total enclosed mass):

    $PDF = \frac{\rho(r)}{\mathfrak{L}_\text{Dec}}$

    All vectors are given in the Observer's coordinate system (OCS). 

    For the Grid1f we assume the same number of cells in each direction.
    As for now, we need to specify what is the type of gas density used 
    to compute the DensityGrid. This is specified by three booleans during 
    the declaration of the DensityGRid object. The booleans refer to the 
    cases where the DensityGrid represents H, HI or HII gas densities. 
    By default, we indicate that our DensityGrid is for H, but we don't 
    actually care about this, as we are only using this as a source 
    sampling function.

    Then, we normalize by the total mass to directly get a 
    mass distribution function normalized to one, and check if that 
    can accelerate the computation time.

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
        :param total_mass: Total mass of the DM halo
        :type total_mass: u.Quantity
        :param chunksize: Number of points to compute per chunk
        :type chunksize: int
        :param dmprofile: Label of the DM density profile
        :type dmprofile: str
        :return: PDF Grid for decaying DM
        :rtype: DensityGrid
    """

    # We use the units of the scale radius as a natural choice
    # to compare and convert between the different units

    # As you can see, I am not checking the dimension of the arrays

    lunit   = rs.unit
    halo_c  = dmhalo_center.to(lunit)
    obs_r   = obs_radius.to(lunit)
    box_or  = halo_c - obs_r
    box_f   = halo_c + obs_r
    step    = 2*obs_r/(ncells-1)
    rsat_   = rsat.to(lunit)
    rtrunc_ = rtrunc.to(lunit)
    rhos_   = convert_density(rhos,u.Msun/lunit**3)
    rhosat_ = convert_density(rhosat,u.Msun/lunit**3)
    mass    = total_mass.to(u.Msun,equivalencies=u.mass_energy())

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

                rho_[i:istop,j:jstop,k:kstop] = dm_mass_density(
                    r,
                    rs,
                    rhos_,
                    rsat_,
                    rhosat_,
                    rtrunc_,
                    dmprofile=dmprofile,
                ).value

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
    dmhalo_center  : u.Quantity,
    obs_radius     : u.Quantity,
    ncells         : int,
    rs             : u.Quantity,
    rhos           : u.Quantity,
    rsat           : u.Quantity,
    rhosat         : u.Quantity,
    rtrunc         : u.Quantity,
    total_dmlum    : u.Quantity,
    chunksize      : int = 32,
    dmprofile      : str = "nfw"
    # save           : bool=False
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

    lunit   = rs.unit
    halo_c  = dmhalo_center.to(lunit)
    obs_r   = obs_radius.to(lunit)
    box_or  = halo_c - obs_r
    box_f   = halo_c + obs_r
    step    = 2*obs_r/(ncells-1)
    rsat_   = rsat.to(lunit)
    rtrunc_ = rtrunc.to(lunit)
    rhos_   = convert_density(rhos,u.Msun/lunit**3)
    rhosat_ = convert_density(rhosat,u.Msun/lunit**3)
    dmlum   = total_dmlum.to(u.Msun**2/lunit**3,equivalencies=u.mass_energy())

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

                dmd = dm_mass_density(
                    r,
                    rs,
                    rhos_,
                    rsat_,
                    rhosat_,
                    rtrunc_,
                    dmprofile=dmprofile,
                ).value

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
    rmin        : float,
    rmax        : float,
    redshift    : float,
    part_type   : int,
    emin        : float,
    emax        : float,
    pl_index    : float
) -> Source:

    """
    Docstring for prepareSmoothExtendedDMSource
    
        :param dmdensity: [Mass] DM density
        :type dmdensity: DensityGrid
        :param max_density: Max. value of the density used for normalization.
        :type max_density: float
        :param maxTries: Maximum number of trials to get a source.
        :type maxTries: int
        :param rmin: Lower Limit for source sampling [3D vector,m]
        :type rmin: float
        :param rmax: Upper Limit for source sampling [3D vector,m]
        :type rmax: float
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
        :return: Source Mass Distribution
        :rtype: SourceMassDistribution
    """

    xmin,ymin,zmin = rmin
    xmax,ymax,zmax = rmax

    dms = Source()

    dm_sources = SourceMassDistribution(dmdensity,max_density)
    dm_sources.setXrange(xmin,xmax)
    dm_sources.setYrange(ymin,ymax)
    dm_sources.setZrange(zmin,zmax)
    dm_sources.setMaximalTries(maxTries)

    dms.add(dm_sources)
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms

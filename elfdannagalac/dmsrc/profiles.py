###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing DM profiles used for source injection                         #
#   - Including units                                                         #
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

from ..tools.conversions import convert_density,convert_mass

from ..tools.customerrors import DMProfileError

from loguru import logger

# allowed_profiles = ["nfw","burkert","einasto"]
allowed_profiles      = ["nfw"]
allowed_spatial_types = ["pointlike","extended_smooth"]

def get_rhosat(dmmass:u.Quantity,sigmav:u.Quantity)->u.Quantity:

    r"""
    Compute the saturation density. We use the formula given in:
    https://clumpy.gitlab.io/CLUMPY/v3.1.1/physics_profiles.html:

    $\rhosat = 3\times10^{18} \times (10^-26 / \langle\sigma v\rangle)~M_\odot~kpc^{-3}$

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

        logger.error(repr(DMProfileError(f"Unknown {dmprofile} Profile")))

    return density

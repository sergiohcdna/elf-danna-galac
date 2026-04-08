###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Luminosity for DM annihilation                                          #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             January-2026                                                    #
###############################################################################

import astropy.units as u
import numpy as np

from scipy.integrate import quad

from ..dmsrc.dmsource import NFW_profile
from ..dmsrc.smooth import get_smooth_dm_density
from ..substructure.population import SubHaloPopulation

from ..tools.conversions import convert_density

def luminosity_anna_nfw(
    rmax   : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity,
    rtrunc : u.Quantity,
) -> u.Quantity:

    r"""
    Annihilation luminosity for the NFW 
    DM profile. The integral is done without 
    considering the line-of-sigh projection. 
    We use the units of the scale radius to make 
    comparisons between the different radial distances. 
    The default units for mass and density are 
    $M_\odot$ and $M_odot~[\text{distance}]^{3}$. 
    The luminosity is:

    $J = 4\pi\int_0^{r_\text{max}} {\rm d}r r^2 \rho(r)^2$

    The integral is always done from the center of the host 
    halo to a maximum radius rmax. This should be enough for 
    all the purposes of this code. Please note, that we include 
    the angular factor $4\pi$ and the units of the emissivity are 
    $M_odot^2~[\text{distance}]^{3}$. The actual luminosity has 
    units of ergs/sec, then, this value needs to be multiplied 
    by a factor of $\langle\sigma v\rangle/m_\text{DM}$ to get 
    the correct units.
    
    :param rmax: Upper limit in radius for the integral
    :type rmax: u.Quantity
    :param rs: Scale radius
    :type rs: u.Quantity
    :param rhos: Scale density
    :type rhos: u.Quantity
    :param rsat: Saturation radius
    :type rsat: u.Quantity
    :param rhosat: Saturation density
    :type rhosat: u.Quantity
    :return: Dfactor [Msun**2/(rs.unit)**3]
    :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3
    units = dunit**2*lunit**3

    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)

    args = (rs,rhos_,rsat,rhosat_,rtrunc)

    def integrand(r,rs,rhos,rsat,rhosat,rtrunc):

        rhodm = NFW_profile(
            r*lunit,rs,rhos,rsat,rhosat,rtrunc,length_unit=lunit
        ).value

        return r**2*rhodm**2

    l_01 = quad(
        integrand,
        0,
        rsat.to(lunit).value,
        args=args
    )[0]

    l_02 = quad(
        integrand,
        rsat.to(lunit).value,
        rs.to(lunit).value,
        args=args
    )[0]

    l_03 = quad(
        integrand,
        rs.to(lunit).value,
        rmax.to(lunit).value,
        args=args
    )[0]

    return 4*np.pi*(l_01+l_02+l_03)*units

def luminosity_anna_nfw_tot(
    rmax      : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    rtrunc    : u.Quantity,
    m200      : u.Quantity,
    msub_min  : u.Quantity,
    msub_max  : u.Quantity,
    sigmac    : u.Quantity,
    indexpm   : float,
    h         : float,
    kw        : float,
    nsub      : int,
    sh_pop    : SubHaloPopulation,
    csublabel : str = "moline2017"
) -> u.Quantity:

    r"""
    Annihilation luminosity for the NFW DM profile 
    inclluding subhalos. The integral is done without 
    considering the line-of-sigh projection. 
    We use the units of the scale radius to make 
    comparisons between the different radial distances. 
    The default units for mass and density are 
    $M_\odot$ and $M_odot~[\text{distance}]^{3}$. 
    The luminosity is:

    $J = 4\pi\int_0^{r_\text{max}} {\rm d}r r^2 \rho_\text{smooth}(r)^2$

    The integral is always done from the center of the host 
    halo to a maximum radius rmax. This should be enough for 
    all the purposes of this code. Please note, that we include 
    the angular factor $4\pi$ and the units of the emissivity are 
    $M_odot^2~[\text{distance}]^{3}$. The actual luminosity has 
    units of ergs/sec, then, this value needs to be multiplied 
    by a factor of $\langle\sigma v\rangle/m_\text{DM}$ to get 
    the correct units.
    
    :param rmax: Upper limit in radius for the integral
    :type rmax: u.Quantity
    :param rs: Scale radius
    :type rs: u.Quantity
    :param rhos: Scale density
    :type rhos: u.Quantity
    :param rsat: Saturation radius
    :type rsat: u.Quantity
    :param rhosat: Saturation density
    :type rhosat: u.Quantity
    :return: Dfactor [Msun**2/(rs.unit)**3]
    :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3
    units = dunit**2*lunit**3

    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)

    args = (
        rs,rhos_,rsat,rhosat_,rtrunc,
        m200,msub_min,msub_max,
        sigmac,indexpm,h,kw,nsub,csublabel
    )

    def integrand(
        r,rs,rhos,rsat,rhosat,rtrunc,
        m200,msub_min,msub_max,sigmac,
        indexpm,h,kw,nsub,csublabel
    ):

        rho_smooth = get_smooth_dm_density(
            r*lunit,rs,rhos,rsat,rhosat,rtrunc,
            m200,msub_min,msub_max,sigmac,
            indexpm,h=h,clabel=csublabel
        )* kw * nsub


        return r**2*rho_smooth**2

    l_01 = quad(
        integrand,
        0,
        rsat.to(lunit).value,
        args=args
    )[0]

    l_02 = quad(
        integrand,
        rsat.to(lunit).value,
        rs.to(lunit).value,
        args=args
    )[0]

    l_03 = quad(
        integrand,
        rs.to(lunit).value,
        rmax.to(lunit).value,
        args=args
    )[0]

    lsubs  = np.sum(sh_pop.to_qtable()["lanna"]).to(units)
    lcross = np.sum(sh_pop.to_qtable()["cross"]).to(units)

    total_anna = 4*np.pi*(l_01+l_02+l_03)*units + lsubs + lcross

    return total_anna

###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Astrophysical factor for annihilation                                   #
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

from ..tools.conversions import convert_density

def jfactor_on_sphere_nfw(
    rmax   : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity
) -> u.Quantity:

    """
    Astrophysical J factor computed for the NFW 
    DM profile. The integral is done without 
    considering the line-of-sigh projection. 
    We use the units of the scale radius to make 
    comparisons between the different radial distances. 
    The default units for mass and density are 
    $M_\odot$ and $M_odot~[\text{distance}]^{3}$. 
    The J factor is:

    $J = 4\pi\int_0^{r_\text{max}} {\rm d}r \rho(r)^2$

    The integral is always done from the center of the host 
    halo to a maximum radius rmax. This should be enough for 
    all the purposes of this code. Please note, that we include 
    the angular factor $4\pi$. The units of the D factor are 
    $M_odot^2~[\text{distance}]^{5}$
    
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
    :return: Dfactor [Msun**2/(rs.unit)**5]
    :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msunh/lunit**3
    units = dunit**2*(lunit)

    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)


    def integrand(r,rs,rhos,rsat,rhosat):

        return NFW_profile(
            r*lunit,rs,rhos,rsat,rhosat,length_unit=lunit
        ).value**2

    jfactor_01 = quad(
        integrand,
        0,
        rsat.to(u.Mpc).value,
        args=(rs,rhos_,rsat,rhosat_)
    )[0]

    jfactor_02 = quad(
        integrand,
        rsat.to(u.Mpc).value,
        rs.to(u.Mpc).value,
        args=(rs,rhos,rsat,rhosat)
    )[0]

    jfactor_03 = quad(
        integrand,
        rs.to(u.Mpc).value,
        rmax.to(u.Mpc).value,
        args=(rs,rhos,rsat,rhosat)
    )[0]

    return 4*np.pi*(jfactor_01+jfactor_02+jfactor_03)*units

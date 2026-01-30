###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Astrophysical factor for decay                                          #
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


def dfactor_on_sphere_nfw(
    rmax   : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity
) -> u.Quantity:

    """
    Astrophysical D factor computed for the NFW 
    DM profile. The integral is done without 
    considering the line-of-sigh projection. 
    We use the units of the scale radius to make 
    comparisons between the different radial distances. 
    The default units for mass and density are 
    $M_\odot$ and $M_odot~[\text{distance}]^{3}$. 
    The D factor is:

    $D = 4\pi\int_0^{r_\text{max}} {\rm d}r \rho(r)$

    The integral is always done from the center of the host 
    halo to a maximum radius rmax. This should be enough for 
    all the purposes of this code. Please note, that we include 
    the angular factor $4\pi$. The units of the D factor are 
    $M_odot~[\text{distance}]^{2}$
    
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
    :return: Dfactor [Msun/(rs.unit)**2]
    :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3
    units = dunit*(lunit)

    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)


    def integrand(r,rs,rhos,rsat,rhosat):

        return NFW_profile(
            r*lunit,rs,rhos,rsat,rhosat,length_unit=lunit
        ).value

    dfactor_01 = quad(
        integrand,
        0,
        rsat.to(lunit).value,
        args=(rs,rhos_,rsat,rhosat_)
    )[0]

    dfactor_02 = quad(
        integrand,
        rsat.to(lunit).value,
        rs.to(lunit).value,
        args=(rs,rhos,rsat,rhosat)
    )[0]

    dfactor_03 = quad(
        integrand,
        rs.to(lunit).value,
        rmax.to(lunit).value,
        args=(rs,rhos,rsat,rhosat)
    )[0]

    return 4*np.pi*(dfactor_01+dfactor_02+dfactor_03)*units

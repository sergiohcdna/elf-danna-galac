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

from ..dmsrc.profiles import get_enclosed_mass_nfw


def luminosity_decay_nfw(
    rmax   : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity,
    rtrunc : u.Quantity,
) -> u.Quantity:

    """
    Decay luminosity computed for the NFW 
    DM profile. The integral is done without 
    considering the line-of-sigh projection. 
    We use the units of the scale radius to make 
    comparisons between the different radial distances. 
    The default units for mass and density are 
    $M_\odot$ and $M_odot~[\text{distance}]^{3}$. 
    The luminosity is:

    $D = 4\pi\int_0^{r_\text{max}} {\rm d}r r^2 \rho(r)$

    That is, the emissivity and the luminosity for decay 
    are proportional to the enclosed mass up to radius r_max.

    The integral is always done from the center of the host 
    halo to a maximum radius rmax. This should be enough for 
    all the purposes of this code. Please note, that we include 
    the angular factor $4\pi$ and the units of the emissivity are 
    $M_odot$. To get the actual luminosty, a factor of the 
    decay rate of DM needs to be included.
    
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

    mass = get_enclosed_mass_nfw(
        rmax,rs,rhos,rsat,rhosat,rtrunc,length_unit=rs.unit
    )

    return mass

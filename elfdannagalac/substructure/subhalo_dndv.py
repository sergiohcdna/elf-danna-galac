###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Probability function for subhalos given the distance                    #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             January-2026                                                    #
###############################################################################

import astropy.units as u

from ..dmsrc.dmsource import NFW_profile,get_enclosed_mass_nfw

def p_nsub_v(
    r      : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity,
    rtrunc : u.Quantity,
    m200   : u.Quantity,
) -> u.Quantity:
    
    """
    Compute the probability that a DM subhalo is located 
    at a distance r from the center of the host halo. 
    We assume that the subhalo spatial distribution follows 
    a NFW profile. This value has units of 1/V. 
    By default, we use the unit of the scale radius to 
    make comparison of the other radii used to describe 
    p(nsub|r).
    
    :param r: Distance to the center of the host halo
    :type r: u.Quantity
    :param rs: Scale radius of the host halo
    :type rs: u.Quantity
    :param rhos: Scale density of the host halo
    :type rhos: u.Quantity
    :param rsat: Saturation radius of the host halo
    :type rsat: u.Quantity
    :param rhosat: Saturation radius of the host halo
    :type rhosat: u.Quantity
    :param m200: M200 of the host halo
    :type m200: u.Quantity
    :return: p(n_sub|r)
    :rtype: Quantity
    """

    prob = NFW_profile(r,rs,rhos,rsat,rhosat,rtrunc,length_unit=rs.unit)

    return prob/m200

def p_nsub_v_int(
    r      : u.Quantity,
    rs     : u.Quantity,
    rhos   : u.Quantity,
    rsat   : u.Quantity,
    rhosat : u.Quantity,
    rtrunc : u.Quantity,
    m200   : u.Quantity,
) -> u.Quantity:

    """
    Compute the integral of the spatial probability 
    for DM subhalos from the origin to the radius r. 
    We assume that the subhalo spatial distribution follows 
    a NFW profile. This value has units of 1/V. 
    By default, we use the unit of the scale radius to 
    make comparison of the other radii used to describe 
    p(nsub|r).
    
    :param r: Distance to the center of the host halo
    :type r: u.Quantity
    :param rs: Scale radius of the host halo
    :type rs: u.Quantity
    :param rhos: Scale density of the host halo
    :type rhos: u.Quantity
    :param rsat: Saturation radius of the host halo
    :type rsat: u.Quantity
    :param rhosat: Saturation radius of the host halo
    :type rhosat: u.Quantity
    :param m200: M200 of the host halo
    :type m200: u.Quantity
    :return: p(n_sub|r)
    :rtype: Quantity
    """

    lunit = rs.unit

    prob_int = get_enclosed_mass_nfw(
        r,rs,rhos,rsat,rhosat,rtrunc,length_unit=lunit
    )/m200

    return prob_int

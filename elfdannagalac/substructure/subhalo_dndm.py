###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Probability function for subhalos given the mass                        #
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

from ..tools.conversions import convert_mass

def p_nsub_m(
    mass_sub : u.Quantity,
    index    : float=-1.9,
    norm     : float=1.0,
) -> u.Quantity:

    """
    Probability of a DM subhalo to have mass m. 
    The normalization can be used to normalize 
    the function. However, for our purposes we 
    normalize the whole probability in posterior 
    steps and not here. Keeping the norm equal to 1 
    helps to relatively speed the calculations of 
    other integrals.
    
    :param mass_sub: Mass of DM subhalo
    :type mass_sub: u.Quantity
    :param index: Index of the SHMF
    :type index: float
    :param norm: Normalization of the SHMF
    :type norm: float
    :return: p(n_sub|m)
    :rtype: Quantity
    """

    m_sh = convert_mass(mass_sub,new_unit=u.Msun)

    dndm = norm*(m_sh.value)**index

    return dndm/u.M_sun

def p_nsub_m_int(
    mmin_sub : u.Quantity,
    mmax_sub : u.Quantity,
    index    : float=-1.9,
    norm     : float=1.0,
) -> float|np.ndarray:

    """
    Integral of p(n_sub|m) in the range from 
    msub_min to msub_max. This function is not 
    used in other calculations, because the 
    subhalo probability functions are, in general, 
    not separable. This is only used for plot or 
    educational purposes. Because we are using a 
    power-law for the SHMF, then we use the 
    analytical solutions.
    
    :param mmin_sub: Min Mass of DM subhalo
    :type mass_sub: u.Quantity
    :param mmax_sub: Max Mass of DM subhalo
    :type mass_sub: u.Quantity
    :param index: Index of the SHMF
    :type index: float
    :param norm: Normalization of the SHMF
    :type norm: float
    :return: $\int_{mmin}^{mmax} p(n_{sub}|m) {\rm m}c$
    :rtype: Quantity
    """


    m_min = convert_mass(mmin_sub,new_unit=u.Msun)
    m_max = convert_mass(mmax_sub,new_unit=u.Msun)

    # Because we will use power laws, the integral has analytical solution
    # We just check for the value of the index accordingly
    # the integral is dimensionless

    if index == -1:

        pm = norm*np.log(m_max.value/m_min.value)

    else:

        pm = norm*((m_max.value)**(index+1)-(m_min.value)**(index+1))/index

    return pm

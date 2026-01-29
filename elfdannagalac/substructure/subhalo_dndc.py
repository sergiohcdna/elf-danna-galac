###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Probability function for subhalos given the concentration               #
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

from ..dmsrc.concentrations import get_c_sub
from scipy.integrate import quad

from loguru import logger

def p_nsub_c(
    c_sub     : float,
    sigma_c   : float,
    mass_halo : u.Quantity,
    r_halo    : u.Quantity|None = None,
    r_200     : u.Quantity|None = None,
    h         : float|None      = None,
    clabel    : str ="moline2017"
) -> float | np.ndarray :

    """
    Probability of having a subhalo with concentration c.
    Allowed scaling relations are Moliné +,2017 and 
    Sánchez-Conde +, 2014. We use the probability function 
    used in Moliné +, 2017. They used log_10 for the 
    calculations in the scaling relation (then the log(10)). 
    The probability function is a log-normal distribution 
    with width sigma_c and mean c=c(m;r).
    
    :param c_sub: Concentration value to test
    :type c_sub: float
    :param sigma_c: Width of the log-normal distribution
    :type sigma_c: float
    :param mass_halo: Mass of the DM subhalo
    :type mass_halo: u.Quantity
    :param r_halo: Location of the DM subhalo in the main halo
    :type r_halo: u.Quantity | None
    :param r_200: R200 of the main halo
    :type r_200: u.Quantity | None
    :param h: Reduced Hubble constant, H0/100 [dimensionless]
    :type h: float | None
    :param clabel: Label of the scaling relation
    :type clabel: str
    :return: Probability of having a subhalo with concentration c
    :rtype: float | ndarray
    """

    c_mean = get_c_sub(mass_halo,r_halo,r_200,h,clabel=clabel)
    diff   = np.log10(c_sub) - np.log10(c_mean)
    prob   = np.exp(-diff**2/(2*sigma_c**2))

    return prob/(np.sqrt(2*np.pi)*c_sub*sigma_c*np.log(10))

def p_nsub_c_int(
    c_sub_min : float,
    c_sub_max : float,
    sigma_c   : float,
    mass_halo : u.Quantity,
    r_halo    : u.Quantity|None = None,
    r_200     : u.Quantity|None = None,
    h         : float|None      = None,
    clabel    : str ="sanchez2014"
) -> float:

    """
    Integral of p(dmsub|c) in the range from 
    csub_min to csub_max
    
    :param c_sub_min: Min value of c
    :type c_sub_min: float
    :param c_sub_max: Max value of c
    :type c_sub_max: float
    :param sigma_c: Width of the log-normal distribution
    :type sigma_c: float
    :param mass_halo: Mass of the DM subhalo
    :type mass_halo: u.Quantity
    :param r_halo: Distance of the DMsubhalo to the center of the main halo
    :type r_halo: u.Quantity | None
    :param r_200: R200 of the main halo
    :type r_200: u.Quantity | None
    :param h: Reduced Hubble constant, H0/100 [dimensionless]
    :type h: float | None
    :param clabel: Label of the Scaling relation
    :type clabel: str
    :return: $\int_{cmin}^{cmax} p(n_{sub}|c) {\rm d}c$
    :rtype: float
    """

    c_args = (sigma_c,mass_halo,r_halo,r_200,h,clabel)
    intval = quad(p_nsub_c,c_sub_min,c_sub_max,args=c_args)

    # We discard the integration errors

    return intval[0]

###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Scaling relation for concentration parameters                           #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             January-2026                                                    #
###############################################################################


## Notes to copy to the help and the documentation
# in the project
# there is a term in c_moliné that depend in the 
# position of the subhalo x_sub
# and there is also a dependance in the cosmological
# parameter h. That in the original paper is h =0.73 or 0.71

import astropy.units as u
import numpy as np

from ..tools.conversions import convert_mass

from loguru import logger

from ..tools.customerrors import DMConcentrationError

c_sanchez2014 = np.array(
    [
        37.5153,-1.5093,1.636e-2,3.66e-4,-2.0e-5,5.32e-7,
    ]
)

c_moline2017 = np.array(
    [
        -0.195,0.089,0.089
    ]
)

allowed_concentrations = ["moline2017","sanchez2014"]

def get_c(
    mass_halo : u.Quantity,
    clabel    : str = "sanchez2014"
) -> float:

    """
    Compute concentration parameter for DM halos. 
    The only accepted scaling relation (for now) 
    is the relation by Sánchez-Conde +,2014 that 
    can be extrapolated to the lower end of the 
    mass spectrum of halos and subhalos.
    
    :param mass_halo: Mass of the DM halo
    :type mass_halo: u.Quantity
    :param clabel: Label of the scaling relation
    :type clabel: str
    :return: Concentration of the DM halo
    :rtype: float
    """

    m = convert_mass(mass_halo,u.Msun).value
    c = 0

    if clabel.lower() == "sanchez2014":

        for i,ci in enumerate(c_sanchez2014):

            c += ci*np.log(m)**i

    else:

        msg = "Unknown/Wrong c relation"
        logger.error(repr(DMConcentrationError(msg)))
        c = 0

    return c

def get_c_sub(
    mass_halo : u.Quantity,
    r_halo    : u.Quantity|None = None,
    r_200     : u.Quantity|None = None,
    h         : float|None      = None,
    clabel    : str             = "sanchez2014"
) -> float | np.ndarray:


    """
    Compute the concentration parameter for a 
    DM subhalo. Accepted scaling relations are from 
    Moliné +, 2017 and Sánchez-Conde +, 2014. Both 
    relations can be extrapolated to very low masses 
    of subhalos. Please note that the relation in 
    Moliné +, 2017 has a term that depends on the 
    position of the subhalo in the main halo, and the 
    reduced Hubble constant h (that in the paper is 
    setted to 0.71 and 0.73 for two different N-body 
    simulations).
    
    :param mass_halo: Mass of the DM subhalo
    :type mass_halo: u.Quantity
    :param r_halo: Distance to the center of the main halo
    :type r_halo: u.Quantity | None
    :param r_200: R200 of the main halo
    :type r_200: u.Quantity | None
    :param h: Reduced Hubble constant (H0/100) [dimensionless]
    :type h: float | None
    :param clabel: Label of the scaling relation
    :type clabel: str
    :return: Concentration of a subhalo with mass m and located at a
             distance r from the center of the main halo
    :rtype: float | ndarray
    """

    if clabel.lower() == "sanchez2014":

        c = get_c(mass_halo,clabel=clabel)

    elif clabel.lower() == "moline2017":

        assert r_halo is not None
        assert r_200 is not None
        assert h is not None

        m = convert_mass(mass_halo).value

        xsub = (r_halo.to(u.Mpc)/r_200.to(u.Mpc)).value

        x = 0

        for i,ci in enumerate(c_moline2017):

            x += (ci*np.log10(h*m/1e8))**(i+1)

        c = 19.9*(1+x)*(1-0.54*np.log10(xsub))

    else:

        msg = "Unknown/Wrong c relation"
        logger.error(repr(DMConcentrationError(msg)))
        c = 0

    return c


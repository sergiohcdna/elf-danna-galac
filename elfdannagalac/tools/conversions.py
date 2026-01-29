###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Functions used to convert density and mass variables                    #
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

from astropy.units import UnitConversionError

def convert_density(
    rho_old  : u.Quantity,
    new_unit : u.Unit = u.Msun/u.Mpc**3
) -> u.Quantity:

    """
    Unit conversion for density parameters. 
    It supports Particle Physics Units through 
    the equivalences keyword. By default, the 
    returned value is in units of solMass/Mpc**3.
    
    :param rho_old: Old density 
    :type rho_old: u.Quantity
    :param new_unit: New unit to do the conversion
    type new_unit: u.Unit
    :return: Density [default in M_sun/Mpc**3]
    :rtype: Quantity
    """

    try:

        rho_new = rho_old.to(new_unit)

    except UnitConversionError:

        rho_new = rho_old.to(new_unit,equivalencies=u.mass_energy())

    return rho_new

def convert_mass(
    mass_old : u.Quantity,
    new_unit : u.Unit = u.Msun
) -> u.Quantity:

    """
    Unit conversion for mass parameters. 
    It supports Particle Physics Units through 
    the equivalences keyword.  By default, the 
    returned value is in units of Msun.
    
    :param mass_old: Old mass quantity
    :type mass_old: u.Quantity
    :param new_unit: New unit to do the conversion
    type new_unit: u.Unit
    :return: Mass [default in M_sun]
    :rtype: Quantity
    """

    try:

        mass_new = mass_old.to(new_unit)

    except UnitConversionError:

        mass_new = mass_old.to(new_unit,equivalencies=u.mass_energy())

    return mass_new

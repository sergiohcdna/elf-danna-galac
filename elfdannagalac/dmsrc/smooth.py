###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Smooth component of a DM halo                                           #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             April-2026                                                      #
###############################################################################

import astropy.units as u

from .dmsource import NFW_profile
from ..substructure.subhalos import rhosub

from ..tools.customerrors import DMProfileError

from loguru import logger


def get_smooth_dm_density(
    r         : u.Quantity,
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
    dmprofile : str = "nfw",
    csublabel : str = "moline2017"
) -> u.Quantity:

    if dmprofile.lower() == "nfw":

        rho_tot = NFW_profile(
            r.to(rs.unit),
            rs,
            rhos,
            rsat,
            rhosat,
            rtrunc,
            length_unit=rs.unit
        )

        rho_sub = rhosub(
            r.to(rs.unit),
            msub_min,
            msub_max,
            rs,
            rhos,
            rsat,
            rhosat,
            rtrunc,
            m200,
            sigmac,
            indexpm,
            h      = h,
            clabel = csublabel
        )* kw * nsub

    else:

        logger.error(repr(DMProfileError(f"Unknown {dmprofile} Profile")))

    return rho_tot - rho_sub

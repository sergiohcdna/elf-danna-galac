###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Calculation of spectrum                                                 #
#   - Calculation of differential luminosity                                  #
#   - Calculation of SED                                                      #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             April-2026                                                      #
###############################################################################

import astropy.units as u
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from .dmspectra import ALLOWED_PROCESSES
from ..tools.customerrors import DMProcessError

from loguru import logger

def powerlawPDF(
    eng   : u.Quantity,
    emin  : u.Quantity,
    emax  : u.Quantity,
    index : float
) -> float|np.ndarray:

    e    = eng.to(u.GeV).value
    elow = emin.to(u.GeV).value
    eup  = emax.to(u.GeV).value

    if index == -1:

        pdf = np.log(elow/eup)*e**index

    else :

        alpha = index+1
        pdf   = alpha*e**index/(eup**alpha - elow**alpha)

    return pdf

def dmdecayNorm(
    interpolator : RegularGridInterpolator,
    dmmass       : u.Quantity,
    emin         : u.Quantity,
    emax         : u.Quantity
) -> float:

    from scipy.integrate import quad

    elow = emin.to(u.GeV).value
    eup  = emax.to(u.GeV).value
    dmm  = dmmass.to(u.GeV).value

    log10xmin = np.log10(elow/dmm)
    log10xmax = np.log10(eup/dmm)

    norm,_ = quad(
        lambda x,mass : interpolator((mass,x))/(mass*10**x*np.log(10)),
        log10xmin,
        log10xmax,
        args=(dmm)
    )

    return norm

def dmdecayPDF(
    interpolator : RegularGridInterpolator,
    eng          : u.Quantity,
    dmmass       : u.Quantity,
    norm         : float
):

    e      = eng.to(u.GeV).value
    dmm    = dmmass.to(u.GeV).value
    log10x = np.log10(e/dmm)

    val = interpolator((dmm,log10x))/(dmm*10**log10x*np.log(10))

    return val/norm

def Qe_dmdecay(
    dmmass   : u.Quantity,
    M200     : u.Quantity,
    lifetime : u.Quantity,
    norm     : float
) -> u.Quantity:
    
    m     = M200.to(u.GeV,equivalencies=u.mass_energy())
    dmm   = dmmass.to(u.GeV)
    dmtau = lifetime.to(u.s)

    rate = 4*np.pi*m*norm/dmm/dmtau

    return rate


def Qe_dmanna(
    dmmass    : u.Quantity,
    sigma_v   : u.Quantity,
    c200      : float,
    rho_scale : u.Quantity,
    r_scale   : u.Quantity,
    norm      : float
) -> u.Quantity:

    dmm  = dmmass.to(u.GeV).value
    sv   = sigma_v.to(u.cm**3/u.s).value
    rhos = rho_scale.to(u.GeV/u.cm**3,equivalencies=u.mass_energy())
    rs   = r_scale.to(u.cm)
    term = 1 - 1/((1+c200)**3)
    rate = 4*np.pi*rhos**2*rs**3*sv*term*norm/(3*dmm*2)

    return rate

def weigths_PL_sim(
    nsim_e        : int,
    ndot_e        : u.Quantity,
    dmmass        : u.Quantity,
    eng_e_sim     : u.Quantity,
    emin_sim      : u.Quantity,
    emax_sim      : u.Quantity,
    dndedm_interp : RegularGridInterpolator,
    dmNorm        : float,
    process       : str   = "decay",
    index         : float = -2.0
) -> u.Quantity :

    msg = "Unknown process. Options are: anna,decay"

    if process.lower() not in ALLOWED_PROCESSES:
        logger.error(repr(DMProcessError(msg)))

    # Convert to common set of units
    emin = emin_sim.to(u.GeV)
    emax = emax_sim.to(u.GeV)
    e_e0 = eng_e_sim.to(u.GeV)
    ndot = ndot_e.to(1/u.s)
    dmm  = dmmass.to(u.GeV)

    # Get PDFs for the energies of the injected electrons
    pdfPL = powerlawPDF(e_e0,emin,emax,index)

    # Get PDFs for the case of DM
    if process.lower() == "decay":

        pdfdm = dmdecayPDF(dndedm_interp,e_e0,dmm,dmNorm)

    elif process.lower() == "anna":

        # For now, I will only use the same function as in the case
        # for decay
        pdfdm = dmdecayPDF(dndedm_interp,e_e0,dmm,dmNorm)

    spectrum_weight = ndot * pdfdm / nsim_e / pdfPL

    return spectrum_weight
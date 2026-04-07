###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Probability function for subhalos given the c, mass and distance        #
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

from scipy.integrate import quad,dblquad,tplquad
from scipy.special import erf

from ..dmsrc.concentrations import get_c_sub
from .subhalo_dndc import p_nsub_c
from .subhalo_dndm import p_nsub_m
from .subhalo_dndv import p_nsub_v

from ..tools.conversions import convert_mass,convert_density

# GENERAL NOTE:
# The most difficult integral comes from the mass term 
# And, I am still trying to figure out how to optimize 
# the integration intervals to speed up the calculations. 
# The problem is the power-law used to describe dndm. 
# Then, for smaller and smaller values of mass 
# scipy routines take more and more time.
# For example, for subhalo masses in the range from 
# 1e-5*M_200 to 1e-3*M_200, the integral took 3 min.
# We will compute the total mass in form of subhalos 
# in three log-intervals between 1e-5*M_200 and 1e-2*M_200. 
# Following previous works, we can assume that the 
# total mass in the range from 1e-5*M_200 to 1e-2*M_200 
# is approximately 0.11*M_200.

def p_nsub_tot(
    c_sub     : float,
    mass_halo : u.Quantity,
    r         : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    r200      : u.Quantity,
    m200      : u.Quantity,
    sigma_c   : float      = 0.13,
    index     : float      = -1.9,
    norm      : float      = 1.0,
    h         : float|None = 0.71,
    clabel    : str        = "moline2017"
) -> u.Quantity :

    """
    This is to get the total probability 
    of a subhalo to have mass m, concentration c,
    and to be located at a position r from the 
    center of the cluster. 
    The normalization for the mass term need 
    to evaluate the concentration probability too.

    The function is defined over the mass range [msh_min,msh_max], 
    and the spherical volume from [0,r200]. 
    The concentration part is more dificult, because 
    the function behaves like a Dirac delta function. 
    We can reduce the range were the function is valid 
    to avoid integration issues. For example, following 
    http://dx.doi.org/10.1103/PhysRevD.95.063003, 
    we can define de range [1,exp(ln(c_mean)+8*sigma_c)] 
    Please note that the probabilities are not separable
    
        :param c_sub: Concentration of DM subhalo
        :type c_sub: float
        :param mass_halo: Mass of DM subhalo
        :type mass_halo: u.Quantity
        :param r: Distance to the center of the host halo
        :type r: u.Quantity
        :param rs: Scale radius of the host halo
        :type rs: u.Quantity
        :param rhos: Scale density of the host halo
        :type rhos: u.Quantity
        :param rsat: Saturation radius of the host halo
        :type rsat: u.Quantity
        :param rhosat: Saturation density of the host halo
        :type rhosat: u.Quantity
        :param r200: R200 of the host halo
        :type r200: u.Quantity
        :param m200: M200 of the host halo
        :type m200: u.Quantity
        :param sigma_c: Width of the dn/dc distribution [default is 0.13]
        :type sigma_c: float
        :param index: Index of the SHMF (dn/dm) [default is -1.9]
        :type index: float
        :param norm: Normalization of the SHMF (dn/dm) [default is 1]
        :type norm: float
        :param h: Reduced Hubble constant H0/100 [default is 0.71]
        :type h: float | None
        :param clabel: Label of c-M relation [default is moline2017]
        :type clabel: str
        :return: p(n_sub|r,m,c) [1/mass/distance**3]
        :rtype: Quantity
    """


    dndm = p_nsub_m(mass_halo,index,norm)
    dndv = p_nsub_v(r,rs,rhos,rsat,rhosat,r200,m200)
    dndc = p_nsub_c(c_sub,sigma_c,mass_halo,r,r200,h,clabel=clabel)

    tot_prob = dndv*dndm*dndc

    return tot_prob

def msub_tot(
    rmin      : u.Quantity,
    rmax      : u.Quantity,
    msub_min  : u.Quantity,
    msub_max  : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    r200      : u.Quantity,
    m200      : u.Quantity,
    sigma_c   : float  = 0.2,
    index     : float  = -1.9,
    norm      : float  = 1.0,
    h         : float  = 0.71,
    clabel    : str    = "moline2017"
) -> u.Quantity:

    """
    Computes the total mass in form of subhalos 
    located in the spherical shell between rmin and rmax.
    
        :param rmin: Minimum radius
        :type rmin: u.Quantity
        :param rmax: Maximum radius
        :type rmax: u.Quantity
        :param msub_min: Minimum mass of the dm subhalos
        :type msub_min: u.Quantity
        :param msub_max: Maximum mass of the dm subhalos
        :type msub_max: u.Quantity
        :param rs: Scale radius of the host halo
        :type rs: u.Quantity
        :param rhos: Scale density ot the host halo
        :type rhos: u.Quantity
        :param rsat: Saturation radius of the host halo
        :type rsat: u.Quantity
        :param rhosat: Saturation radius of the host halo
        :type rhosat: u.Quantity
        :param r200: R200 of the host halo
        :type r200: u.Quantity
        :param m200: M200 of the host halo
        :type m200: u.Quantity
        :param sigma_c: Width of the dn/dc distribution [default is 0.13]
        :type sigma_c: float
        :param index: Index of the SHMF (dn/dm) [default is -1.9]
        :type index: float
        :param norm: Normalization of the SHMF (dn/dm) [default is 1]
        :type norm: float
        :param h: Reduced Hubble constant H0/100 [default is 0.71]
        :type h: float | None
        :param clabel: Label of c-M relation [default is moline2017]
        :type clabel: str
        :return: Total mass in form of subhalos
        :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3

    rmin_   = rmin.to(lunit)
    rmax_   = rmax.to(lunit)
    rs_     = rs.to(lunit)
    rsat_   = rsat.to(lunit)
    r200_   = r200.to(lunit)
    mmin_   = convert_mass(msub_min,new_unit=u.Msun)
    mmax_   = convert_mass(msub_max,new_unit=u.Msun)
    m200_   = convert_mass(m200,new_unit=u.Msun)
    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)

    def mass_integrand(
        # c_sub     : float,
        mass_halo : float,
        r         : float,
        rs        : u.Quantity,
        rhos      : u.Quantity,
        rsat      : u.Quantity,
        rhosat    : u.Quantity,
        r200      : u.Quantity,
        m200      : u.Quantity,
        sigma_c   : float      = 0.2,
        index     : float      = -1.9,
        norm      : float      = 1.0,
        h         : float|None = 0.71,
        clabel    : str        = "moline2017"        
    ):

        dndm   = p_nsub_m(mass_halo*u.Msun,index,norm).value
        dndv   = p_nsub_v(r*lunit,rs,rhos,rsat,rhosat,r200,m200).value
        c_mean = get_c_sub(mass_halo*u.M_sun,r*lunit,r200_,h=h,clabel=clabel)
        c_max  = np.exp(np.log(c_mean) + 8*sigma_c)
        diff   = np.log(c_max) - np.log(c_mean)
        lnnorm = np.log(10)*np.sqrt(2)*sigma_c
        deltac = 0.5*(erf(diff/lnnorm) - erf(-np.log(c_mean)/lnnorm))


        return 4*np.pi*mass_halo*r**2*dndm*dndv*(deltac)

    args=(
        rs_,
        rhos_,
        rsat_,
        rhosat_,
        r200_,
        m200_,
        sigma_c,
        index,
        norm,
        h,
        clabel
    )

    msub = dblquad(
        mass_integrand,
        rmin_.value,
        rmax_.value,
        mmin_.value,
        mmax_.value,
        args=args,
    )


    return msub[0]*u.M_sun

def nsub_tot(
    rmin      : u.Quantity,
    rmax      : u.Quantity,
    msub_min  : u.Quantity,
    msub_max  : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    r200      : u.Quantity,
    m200      : u.Quantity,
    sigma_c   : float  = 0.13,
    index     : float  = -1.9,
    norm      : float  = 1.0,
    h         : float  = 0.71,
    clabel    : str    = "moline2017"
) -> float:

    """
    Computes the number of subhalos in the spherical shell 
    [rmin,rmax] and the mass range from [msub_min,msub_max].

    This function is just a base function used to estimate 
    normalization factors for the total subhalo PDFs. Then, 
    we are just interested in the value of the integral.
    
        :param rmin: Minimum radius
        :type rmin: u.Quantity
        :param rmax: Maximum radius
        :type rmax: u.Quantity
        :param msub_min: Minimum mass of the dm subhalos
        :type msub_min: u.Quantity
        :param msub_max: Maximum mass of the dm subhalos
        :type msub_max: u.Quantity
        :param rs: Scale radius of the host halo
        :type rs: u.Quantity
        :param rhos: Scale density ot the host halo
        :type rhos: u.Quantity
        :param rsat: Saturation radius of the host halo
        :type rsat: u.Quantity
        :param rhosat: Saturation radius of the host halo
        :type rhosat: u.Quantity
        :param r200: R200 of the host halo
        :type r200: u.Quantity
        :param m200: M200 of the host halo
        :type m200: u.Quantity
        :param sigma_c: Width of the dn/dc distribution [default is 0.13]
        :type sigma_c: float
        :param index: Index of the SHMF (dn/dm) [default is -1.9]
        :type index: float
        :param norm: Normalization of the SHMF (dn/dm) [default is 1]
        :type norm: float
        :param h: Reduced Hubble constant H0/100 [default is 0.71]
        :type h: float | None
        :param clabel: Label of c-M relation [default is moline2017]
        :type clabel: str
        :return: Total number of subhalos
        :rtype: float
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3

    rmin_   = rmin.to(lunit)
    rmax_   = rmax.to(lunit)
    rs_     = rs.to(lunit)
    rsat_   = rsat.to(lunit)
    r200_   = r200.to(lunit)
    mmin_   = convert_mass(msub_min,new_unit=u.Msun)
    mmax_   = convert_mass(msub_max,new_unit=u.Msun)
    m200_   = convert_mass(m200,new_unit=u.Msun)
    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)

    def n_integrand(
        # c_sub     : float,
        mass_halo : float,
        r         : float,
        rs        : u.Quantity,
        rhos      : u.Quantity,
        rsat      : u.Quantity,
        rhosat    : u.Quantity,
        r200      : u.Quantity,
        m200      : u.Quantity,
        sigma_c   : float      = 0.2,
        index     : float      = -1.9,
        norm      : float      = 1.0,
        h         : float|None = 0.71,
        clabel    : str        = "moline2017"        
    ):

        dndm   = p_nsub_m(mass_halo*u.Msun,index,norm).value
        dndv   = p_nsub_v(r*lunit,rs,rhos,rsat,rhosat,r200,m200).value
        c_mean = get_c_sub(mass_halo*u.M_sun,r*lunit,r200_,h=h,clabel=clabel)
        c_max  = np.exp(np.log(c_mean) + 8*sigma_c)
        diff   = np.log(c_max) - np.log(c_mean)
        lnnorm = np.log(10)*np.sqrt(2)*sigma_c
        deltac = 0.5*(erf(diff/lnnorm) - erf(-np.log(c_mean)/lnnorm))


        return 4*np.pi*r**2*dndm*dndv*(deltac)

    args=(
        rs_,
        rhos_,
        rsat_,
        rhosat_,
        r200_,
        m200_,
        sigma_c,
        index,
        norm,
        h,
        clabel
    )

    nsub = dblquad(
        n_integrand,
        rmin_.value,
        rmax_.value,
        mmin_.value,
        mmax_.value,
        args=args,
    )


    return nsub[0]

def nsub_r(
    r         : u.Quantity,
    msub_min  : u.Quantity,
    msub_max  : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    r200      : u.Quantity,
    m200      : u.Quantity,
    sigma_c   : float  = 0.2,
    index     : float  = -1.9,
    norm      : float  = 1.0,
    h         : float  = 0.71,
    clabel    : str    = "moline2017"
) -> float:

    """
    Computes the number of subhalos in the mass range 
    from [msub_min,msub_max] at a distance r from the 
    center of the host halo, i.e. the numeric density 
    of subhalos at a distance r. We do not integrate over 
    the angular part, so technically the units of this are 
    $[\text{distance}]^{-3}~\text{sr}^{-1}$, but the result 
    is just in $[\text{distance}]^{-3}$. 

    This function is just a base function used to estimate 
    normalization factors for the total subhalo PDFs. Then, 
    we are just interested in the value of the integral.

    For a proper implementation you can refer to the DMHalo class
    
        :param r: Radial distance to the center of the host halo
        :type r: u.Quantity
        :param msub_min: Minimum mass of the dm subhalos
        :type msub_min: u.Quantity
        :param msub_max: Maximum mass of the dm subhalos
        :type msub_max: u.Quantity
        :param rs: Scale radius of the host halo
        :type rs: u.Quantity
        :param rhos: Scale density ot the host halo
        :type rhos: u.Quantity
        :param rsat: Saturation radius of the host halo
        :type rsat: u.Quantity
        :param rhosat: Saturation radius of the host halo
        :type rhosat: u.Quantity
        :param r200: R200 of the host halo
        :type r200: u.Quantity
        :param m200: M200 of the host halo
        :type m200: u.Quantity
        :param sigma_c: Width of the dn/dc distribution [default is 0.13]
        :type sigma_c: float
        :param index: Index of the SHMF (dn/dm) [default is -1.9]
        :type index: float
        :param norm: Normalization of the SHMF (dn/dm) [default is 1]
        :type norm: float
        :param h: Reduced Hubble constant H0/100 [default is 0.71]
        :type h: float | None
        :param clabel: Label of c-M relation [default is moline2017]
        :type clabel: str
        :return: subhalo numeric density
        :rtype: float
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3
    units = 1/lunit**3

    r_      = r.to(lunit)
    rs_     = rs.to(lunit)
    rsat_   = rsat.to(lunit)
    r200_   = r200.to(lunit)
    mmin_   = convert_mass(msub_min,new_unit=u.Msun)
    mmax_   = convert_mass(msub_max,new_unit=u.Msun)
    m200_   = convert_mass(m200,new_unit=u.Msun)
    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)

    

    def n_integrand(
        c_sub     : float,
        mass_halo : float,
        r         : u.Quantity,
        rs        : u.Quantity,
        rhos      : u.Quantity,
        rsat      : u.Quantity,
        rhosat    : u.Quantity,
        r200      : u.Quantity,
        m200      : u.Quantity,
        sigma_c   : float      = 0.2,
        index     : float      = -1.9,
        norm      : float      = 1.0,
        h         : float|None = 0.71,
        clabel    : str        = "moline2017"        
    ):

        nprob = p_nsub_tot(
            c_sub,
            mass_halo*u.M_sun,
            r,
            rs,
            rhos,
            rsat,
            rhosat,
            r200,
            m200,
            sigma_c,
            index,
            norm,
            h,
            clabel=clabel
        )

        return nprob.value
    
    def compute_c(m):

        c_mean = get_c_sub(
            m*u.M_sun,
            r_,
            r200_,
            h=h,
            clabel=clabel
        )

        return np.exp(np.log(c_mean) + 8*sigma_c)

    args=(
        r_,
        rs_,
        rhos_,
        rsat_,
        rhosat_,
        r200_,
        m200_,
        sigma_c,
        index,
        norm,
        h,
        clabel
    )

    nsub = dblquad(
        n_integrand,
        mmin_.value,
        mmax_.value,
        1.0,
        compute_c,
        args=args,
    )

    return nsub[0]*units

def rhosub(
    r         : u.Quantity,
    msub_min  : u.Quantity,
    msub_max  : u.Quantity,
    rs        : u.Quantity,
    rhos      : u.Quantity,
    rsat      : u.Quantity,
    rhosat    : u.Quantity,
    r200      : u.Quantity,
    m200      : u.Quantity,
    sigma_c   : float  = 0.13,
    index     : float  = -1.9,
    norm      : float  = 1.0,
    h         : float  = 0.71,
    clabel    : str    = "moline2017"
) -> u.Quantity:

    """
    Computes the density associated to subhalos 
    at a distance r from the center of the host halo.
    
        :param msub_min: Minimum mass of the dm subhalos
        :type msub_min: u.Quantity
        :param msub_max: Maximum mass of the dm subhalos
        :type msub_max: u.Quantity
        :param rs: Scale radius of the host halo
        :type rs: u.Quantity
        :param rhos: Scale density ot the host halo
        :type rhos: u.Quantity
        :param rsat: Saturation radius of the host halo
        :type rsat: u.Quantity
        :param rhosat: Saturation radius of the host halo
        :type rhosat: u.Quantity
        :param r200: R200 of the host halo
        :type r200: u.Quantity
        :param m200: M200 of the host halo
        :type m200: u.Quantity
        :param sigma_c: Width of the dn/dc distribution [default is 0.13]
        :type sigma_c: float
        :param index: Index of the SHMF (dn/dm) [default is -1.9]
        :type index: float
        :param norm: Normalization of the SHMF (dn/dm) [default is 1]
        :type norm: float
        :param h: Reduced Hubble constant H0/100 [default is 0.71]
        :type h: float | None
        :param clabel: Label of c-M relation [default is moline2017]
        :type clabel: str
        :return: Total mass in form of subhalos
        :rtype: Quantity
    """

    lunit = rs.unit
    dunit = u.Msun/lunit**3

    r_      = r.to(lunit)
    rs_     = rs.to(lunit)
    rsat_   = rsat.to(lunit)
    r200_   = r200.to(lunit)
    mmin_   = convert_mass(msub_min,new_unit=u.Msun)
    mmax_   = convert_mass(msub_max,new_unit=u.Msun)
    m200_   = convert_mass(m200,new_unit=u.Msun)
    rhos_   = convert_density(rhos,new_unit=dunit)
    rhosat_ = convert_density(rhosat,new_unit=dunit)
    dndv    = p_nsub_v(r_,rs_,rhos_,rsat_,rhosat_,r200_,m200_)

    def mass_integrand(
        mass_halo : float,
        r         : float,
        r200      : u.Quantity,
        sigma_c   : float      = 0.13,
        index     : float      = -1.9,
        norm      : float      = 1.0,
        h         : float|None = 0.71,
        clabel    : str        = "moline2017"        
    ):

        dndm   = p_nsub_m(mass_halo*u.Msun,index,norm).value
        c_mean = get_c_sub(mass_halo*u.M_sun,r*lunit,r200,h=h,clabel=clabel)
        c_max  = np.exp(np.log(c_mean) + 8*sigma_c)
        diff   = np.log(c_max) - np.log(c_mean)
        lnnorm = np.log(10)*np.sqrt(2)*sigma_c
        deltac = 0.5*(erf(diff/lnnorm) - erf(-np.log(c_mean)/lnnorm))

        return mass_halo*dndm*deltac

    args=(
        r_.value,
        r200_,
        sigma_c,
        index,
        norm,
        h,
        clabel
    )

    m_av = quad(
        mass_integrand,
        mmin_.value,
        mmax_.value,
        args=args,
    )

    return m_av[0]*u.M_sun*dndv


###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - DM halo class                                                           #
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

from astropy.constants import G
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

from ..astrofactors.jfactor import luminosity_anna_nfw
from ..astrofactors.dfactor import luminosity_decay_nfw
from .concentrations import get_c,get_c_sub
from .dmsource import get_rhosat,get_enclosed_mass_nfw,NFW_profile
from ..substructure.subhalos import msub_tot,nsub_tot,p_nsub_tot,nsub_r
from ..tools.conversions import convert_mass,convert_density

from ..tools.customerrors import DMProfileError,SubHaloMassError

from loguru import logger

class DMHalo():

    def __init__(
        self,
        name      : str,
        ra        : u.Quantity,
        dec       : u.Quantity,
        z         : float,
        m200      : u.Quantity,
        r200      : u.Quantity,
        dmmass    : u.Quantity = 100*u.GeV,
        dmsigmav  : u.Quantity = 3.6e-26*u.cm**3/u.s,
        fsub      : float      = 0.2,
        msub_min  : u.Quantity = 1e8*u.M_sun,
        msub_max  : u.Quantity = 1e12*u.M_sun,
        index_pm  : float      = -1.9,
        sigma_c   : float      = 0.13,
        h         : float      = 0.73,
        mpoints   : int        = 10,
        clabel    : str        = "sanchez2014",
        dmprofile : str        = "nfw",
        csublabel : str        = "moline2017"
    ) -> None:

        r"""
        Create a DM halo given the radius ($R_{200}$) where the density is 
        200 times the critical density of the universe at redshift z, 
        and the mass enclosed up to $R_{200}$, $M_{200}$. The class computes 
        all the parameters related to the DM halo as the saturation density 
        and saturation radius and the concentration parameter. For a given 
        value of substructure (fraction of the total mass and the mass range) 
        it also computes the total number of subhalos and the normalization 
        of the subhalo PDF. Additionally, computes the astrophysical factors 
        (without projecting along the line of sight) and returns the convenient 
        functions $\rho(r)$ and $\rho(r)^2$.

        The equatorial coordinates (J200) and redshift of the source need to be 
        provided to estimate the luminosity distance and return the cartesian 
        coordinates of the center of the halo. 
        
            :param name: Name of the DM halo
            :type name: str
            :param ra: Right Ascension (J200) [deg]
            :type ra: u.Quantity
            :param dec: Declination (J200) [deg]
            :type dec: u.Quantity
            :param z: Redshift to the source
            :type z: float
            :param m200: Total mass of the halo
            :type m200: u.Quantity
            :param r200: Radius where the density is $200\rho_c$ 
            :type r200: u.Quantity
            :param dmmass: Mass of the DM particle candidate
            :type dmmass: u.Quantity
            :param dmsigmav: Thermal average annihilation cross section
            :type dmsigmav: u.Quantity
            :param fsub: Fraction of the total mass in form of subhalos
            :type fsub: float
            :param msub_min: Minimum mass of a subhalo
            :type msub_min: u.Quantity
            :param msub_max: Maximum mass of a subhalo
            :type msub_max: u.Quantity
            :param index_pm: Index of the SubHalo Mass Function
            :type index_pm: float
            :param sigma_c: Width of the c-log-normal distribution
            :type sigma_c: float
            :param h: Reduced hubble constant
            :type h: float
            :param mpoints: Number of points to do the integral of the SHMF
            :type mpoints: int
            :param clabel: Name of the c-mass relation for the host halo
            :type clabel: str
            :param dmprofile: Name of the DM profile
            :type dmprofile: str
            :param csublabel: Name of the c-mass relation for the subhalos
            :type csublabel: str
        """

        lunit = r200.unit
        dunit = u.Msun/lunit**3

        rhosat = get_rhosat(dmmass,dmsigmav)
        cosmo  = FlatLambdaCDM(100*h,Om0=0.3,Ob0=0.04)
        rhoc   = 3*cosmo.H0**2*(cosmo.Om0*(1+z)**3 + cosmo.Ode0)/(8*np.pi*G)
        rhoc   = convert_density(rhoc,new_unit=u.Msun/lunit**3)

        self._name      = name
        self._ra        = ra
        self._dec       = dec
        self._z         = z
        self._clabel    = clabel,
        self._m200      = convert_mass(m200,new_unit=u.Msun)

        # We need to check that R200 is consistent with the value of M200
        # We use M200 as the main halo parameter
        r200_ = np.cbrt(3*self._m200/(800*np.pi*rhoc))

        if r200_/r200 >= 0.01:
            msg = (
                f"Input R200 {r200:0.3f} will be inconsistent "
                "with the rest of calculations.\n "
                "Setting R200 to the value obtained using "
                f"the critical density: {r200_:0.3f}"
            )
            logger.warning(msg)

            self._r200 = r200_.to(lunit)

        else:

            self._r200 = r200.to(lunit)

        self._c200      = get_c(self._m200,clabel=clabel)
        self._rs        = self._r200/self._c200
        logterm         = np.log(1+self._c200)-self._c200/(1+self._c200)
        self._rhos      = 200*rhoc*self._c200**3/(3*logterm)
        self._dmprofile = dmprofile
        self._csublabel = csublabel
        self._dmmas     = dmmass
        self._dmsigmav  = dmsigmav
        self._rhosat    = convert_density(rhosat,new_unit=dunit)
        self._rsat      = self._rs*(self._rhos/self._rhosat)
        self._lanna     = self.get_lanna()
        self._ldecay    = self.get_ldecay()
        self._fsub      = fsub
        self._indexpm   = index_pm
        self._sigmac    = sigma_c
        self._h         = h
        self._mpoints   = mpoints
        self._msub_min  = convert_mass(msub_min,new_unit=u.Msun)

        msg = (
            "Max. value of Sub halo masses can not be larger "
            "than 1% of the total cluster mass."
        )
        if convert_mass(msub_max,new_unit=u.Msun) > 0.01*self._m200:
            logger.error(repr(SubHaloMassError(msg)))

        self._msub_max = convert_mass(msub_max,new_unit=u.Msun)

        self._lannapp = (self._lanna*(
            (1.0*u.M_sun).to(u.GeV,equivalencies=u.mass_energy())**2
        )/u.M_sun**2).to(u.GeV**2/u.cm**3)
        self._ldecaypp = (
            self._ldecay.to(u.GeV,equivalencies=u.mass_energy())
        )

        self._kw   = self.get_kw()
        self._nsub = self.get_nnorm()
        self._msub = self.get_msub(self._r200,self._msub_min,self._msub_max)

        self._dlum  = cosmo.luminosity_distance(self._z).to(lunit)
        self._coord = SkyCoord(ra=ra,dec=dec,frame="icrs",distance=self._dlum)
        self._cart  = self._coord.cartesian

        return

    @property
    def m200(self):

        return self._m200

    @property
    def c200(self):

        return self._c200

    @property
    def rs(self):

        return self._rs

    @property
    def rhos(self):

        return self._rhos

    @property
    def r200(self):

        return self._r200

    @property
    def r_sat(self):

        return self._rsat

    @property
    def rho_sat(self):

        return self._rhosat

    @property
    def msub_min(self):
        
        return self._msub_min

    @property
    def msub_max(self):
        
        return self._msub_max

    @property
    def fsub(self):
        
        return self._fsub

    @property
    def index_shmf(self):
        
        return self._indexpm

    @property
    def l_dm_anna(self):

        return self._lanna

    @property
    def l_dm_decay(self):

        return self._ldecay

    @property
    def lanna_pp(self):

        return self._lannapp

    @property
    def ldecay_pp(self):

        return self._ldecaypp

    @property
    def nsubs(self):

        return self._nsub
    
    @property
    def sigmac(self):

        return self._sigmac
    
    @property
    def h(self):

        return self._h

    @property
    def z(self):

        return self._z

    @property
    def msubs(self):

        return self._msub

    @property
    def dlum(self):

        return self._dlum

    @property
    def skycoord(self):

        return self._coord
    
    @property
    def cartcoord(self):

        return self._coord.cartesian

    @property
    def cM_host(self):

        return self._clabel

    @property
    def cM_sub(self):

        return self._csublabel

    @property
    def DMrhoLabel(self):

        return self._dmprofile

    @property
    def info(self):

        ldma = self._lannapp*self._dmsigmav/self._dmmas
        ldma = ldma.to(u.erg/u.s,equivalencies=u.mass_energy())
        ldmd = self._ldecaypp/(1e27*u.s)
        ldmd = ldmd.to(u.erg/u.s,equivalencies=u.mass_energy())

        msg = (
            f"\n{self._name} cluster configured with: \n"
            f"\t- Redshift: {self._z:0.3f}\n"
            f"\t- D_lum: {self._dlum:0.3f}\n"
            f"\t- Coord: ({self._coord.ra:0.3f},{self._coord.dec:0.3f}) [ICRS]\n"
            f"\t- X: {self._cart.x.to(u.Mpc):0.3f}\n"
            f"\t- Y: {self._cart.y.to(u.Mpc):0.3f}\n"
            f"\t- Z: {self._cart.z.to(u.Mpc):0.3f}\n"
            f"\t- Cluster Radius [R200]: {self._r200:0.3f}\n"
            f"\t- Total Mass [M200]: {self._m200:0.3e}\n"
            f"\t- Scale radius: {self._rs:0.3f}\n"
            f"\t- Scale density: {self._rhos:0.3e}\n"
            f"\t- Saturation Radius: {self._rsat:0.3e}\n"
            f"\t- Saturation Density: {self._rhosat:0.3e}\n"
            f"\t- Number of subhalos: {self._nsub} "
            f" (from [{self._msub_min:0.3e},{self._msub_max:0.3e}])\n"
            f"\t- Total mass in form of subhalos: {self._msub:0.3e} "
            f"({self._fsub*100.0}% of the cluster mass)\n"
            f"\t- Annihilation emissivity [No sub]: {self._lanna:0.3e} "
            f"({self._lannapp:0.3e})\n"
            f"\t- Decay emissivity [No sub]: {self._ldecay:0.3e} "
            f"({self._ldecaypp:0.3e})\n"
            f"\t- DM luminosity [Annihilation, No sub]: {ldma:0.5e}\n"
            f"\t- DM luminosity [Decay, No sub]: {ldmd:0.5e}\n"
        )

        logger.info(msg)

        msg = (
            "For the annihilation luminosity a candiate with "
            f"a mass {self._dmmas:0.2f} and thermal-average "
            f"annihilation cross section {self._dmsigmav:0.2e} "
            "were used. \n"
            "For decay, the luminosity was estimated assuming a "
            f"lifetime of 1.00e27 s."
        )

        logger.warning(msg)

        return


    def get_m200(self) -> u.Quantity:

        if self._dmprofile.lower() == "nfw":

            m200 = get_enclosed_mass_nfw(
                self._r200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                length_unit=self._rs.unit
            )

        else :

            logger.error(repr(DMProfileError("Unknown DM profile")))

            m200 = 0 *u.M_sun

        return m200

    def get_lanna(self) -> u.Quantity:

        if self._dmprofile.lower() == "nfw":

            e_tot = luminosity_anna_nfw(
                self._r200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
            )

        else:

            logger.error(repr(DMProfileError("Unknown DM profile")))
            e_tot = 0*u.M_sun**2/u.Mpc**3

        return e_tot

    def get_ldecay(self) -> u.Quantity:

        if self._dmprofile.lower() == "nfw":

            e_tot = luminosity_decay_nfw(
                self._r200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
            )

        else:

            logger.error(repr(DMProfileError("Unknown DM profile")))
            e_tot = 0*u.M_sun

        return e_tot

    def c_subhalo(
        self,
        m_sub : u.Quantity,
        r_sub : u.Quantity = None,
    ):

        csub = get_c_sub(m_sub,r_sub,self._r200,self._h,clabel=self._csublabel)

        return csub

    def get_kw(self) -> float:

        edges = np.logspace(
            np.log10(self._msub_min.value/self._m200.value),
            np.log10(self._msub_max.value/self._m200.value),
            self._mpoints
        )

        norm = 0

        for i,j in zip(edges[:-1],edges[1:]):

            norm += nsub_tot(
                0*u.Mpc,
                self._r200,
                i*self._m200,
                j*self._m200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                self._m200,
                sigma_c=self._sigmac,
                norm=1.0,
                h=self._h,
                clabel=self._csublabel
            )

        return 1/norm

    def get_nnorm(self) -> int:

        # This function is used to estimate the 
        # normalization in the number of subhalos 
        # function. We recquire that the faction 
        # of mass in the form of subhalos is fsub 
        # in the range of msub_min and msub_mass.

        pnorm = self._kw

        edges = np.logspace(
            np.log10(self._msub_min.value/self._m200.value),
            np.log10(self._msub_max.value/self._m200.value),
            self._mpoints
        )

        mtot = 0*u.M_sun

        for i,j in zip(edges[:-1],edges[1:]):

            mtot += msub_tot(
                0*u.Mpc,
                self._r200,
                i*self._m200,
                j*self._m200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                self._m200,
                sigma_c=self._sigmac,
                norm=1.0
            )*pnorm

        nnorm = self._fsub*self._m200/mtot

        return int(nnorm)

    def nsub_prob(
        self,
        csub : float,
        msub : u.Quantity,
        rsub : u.Quantity,
    ) -> float:

        msub_ = convert_mass(msub)
        rsub_ = rsub.to(u.Mpc)

        pnorm = self._kw

        dndw = p_nsub_tot(
                csub,
                msub_,
                rsub_,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                self._m200,
                sigma_c=self._sigmac,
                norm=1.0,
                h=self._h,
                clabel=self._csublabel
            )

        return pnorm*dndw

    def dnsubdr(self,rsub : u.Quantity) -> u.Quantity :

        rsub_ = rsub.to(u.Mpc)
        dndr  = 0/u.Mpc**3
        pnorm = self._kw
        nsubs = self._nsub

        edges = np.logspace(
            np.log10(self._msub_min.value/self._m200.value),
            np.log10(self._msub_max.value/self._m200.value),
            self._mpoints
        )

        for i,j in zip(edges[:-1],edges[1:]):

            dndr += nsub_r(
                rsub_,
                i*self._m200,
                j*self._m200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                self._m200,
                sigma_c = self._sigmac,
                index   = self._indexpm,
                norm    = 1.0,
                h       = self._h,
                clabel  = self._csublabel
            )

        return dndr*pnorm*nsubs

    def get_msub(
        self,
        rsubmax : u.Quantity,
        msubmin : u.Quantity,
        msubmax : u.Quantity
    ) -> u.Quantity :

        pnorm = self._kw
        nsubs = self._nsub
        rsmax = rsubmax.to(u.Mpc)
        msmin = convert_mass(msubmin)
        msmax = convert_mass(msubmax)

        edges = np.logspace(
            np.log10(msmin.value/self._m200.value),
            np.log10(msmax.value/self._m200.value),
            self._mpoints
        )

        m1 = 0*u.Msun

        for i,j in zip(edges[:-1],edges[1:]):

            m1 += msub_tot(
                0*u.Mpc,
                rsmax,
                i*self._m200,
                j*self._m200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                self._r200,
                self._m200,
                sigma_c=self._sigmac,
                norm=1.0
            )*pnorm*nsubs

        return m1

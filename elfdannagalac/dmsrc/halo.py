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

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

from ..astrofactors.jfactor import jfactor_on_sphere_nfw
from ..astrofactors.dfactor import dfactor_on_sphere_nfw
from .concentrations import get_c,get_c_sub
from .dmsource import get_rhosat,get_enclosed_mass_nfw
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
        rs        : u.Quantity,
        rhos      : u.Quantity,
        r200      : u.Quantity,
        dmmass    : u.Quantity = 1e5*u.GeV,
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

        lunit = rs.unit
        dunit = u.Msun/lunit**3

        rhosat = get_rhosat(dmmass,dmsigmav)

        self._name      = name
        self._ra        = ra
        self._dec       = dec
        self._z         = z
        self._rs        = rs.to(lunit)
        self._rhos      = convert_density(rhos,new_unit=dunit)
        self._r200      = r200.to(lunit)
        self._clabel    = clabel,
        self._dmprofile = dmprofile
        self._csublabel = csublabel
        self._dmmas     = dmmass
        self._dmsigmav  = dmsigmav
        self._rhosat    = convert_density(rhosat,new_unit=dunit)
        self._rsat      = self._rs*(self._rhos/self._rhosat)
        self._m200      = self.get_m200()
        self._c200      = get_c(self._m200,clabel=clabel)
        self._jfactor   = self.jfactor_sph()
        self._dfactor   = self.dfactor_sph()
        self._fsub      = fsub
        self._indexpm   = index_pm
        self._sigmac    = sigma_c
        self._h         = h
        self._mpoints   = mpoints
        self._msub_min  = convert_mass(msub_min,new_unit=u.Msun)

        msg = (
            "Max. value of Sub halo masses can not be larger "
            "than 1% of total cluster mass."
        )
        if convert_mass(msub_max,new_unit=u.Msun) > 0.01*self._m200:
            logger.error(repr(SubHaloMassError(msg)))

        self._msub_max = convert_mass(msub_max,new_unit=u.Msun)

        self._jfactorpp = (self._jfactor*(
            (1.0*u.M_sun).to(u.GeV,equivalencies=u.mass_energy())**2
        )/u.M_sun**2).to(u.GeV**2/u.cm**5)
        self._dfactorpp = (
            self._dfactor.to(u.GeV/u.cm**2,equivalencies=u.mass_energy())
        )

        self._kw   = self.get_kw()
        self._nsub = self.get_nnorm()
        self._msub = self.get_msub(self._r200,self._msub_min,self._msub_max)

        cosmo = FlatLambdaCDM(100*self._h,Om0=0.3,Ob0=0.04)

        self._dlum  = cosmo.luminosity_distance(self._z).to(u.Mpc)
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
    def jfactor(self):

        return self._jfactor

    @property
    def dfactor(self):

        return self._dfactor

    @property
    def jfactor_pp(self):

        return self._jfactorpp

    @property
    def dfactor_pp(self):

        return self._dfactorpp

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
    def info(self):

        msg = (
            f"\n{self._name} cluster configured with: \n"
            f"\t- Redshift: {self._z:0.3f}\n"
            f"\t- D_lum: {self._dlum:0.3f}\n"
            f"\t- Coord: ({self._coord.ra:0.3f},{self._coord.dec:0.3f}) [ICRS]\n"
            f"\t- X: {self._cart.x.to(u.Mpc):0.3f}\n"
            f"\t- Y: {self._cart.y.to(u.Mpc):0.3f}\n"
            f"\t- Z: {self._cart.z.to(u.Mpc):0.3f}\n"
            f"\t- Scale radius: {self._rs:0.3f}\n"
            f"\t- Scale density: {self._rhos:0.3e}\n"
            f"\t- Saturation Radius: {self._rsat:0.3e}\n"
            f"\t- Saturation Density: {self._rhosat:0.3e}\n"
            f"\t- Cluster Radius [R200]: {self._r200:0.3f}\n"
            f"\t- Total Mass [M200]: {self._m200:0.3e}\n"
            f"\t- Number of subhalos: {self._nsub} "
            f" (from [{self._msub_min:0.3e},{self._msub_max:0.3e}])\n"
            f"\t- Total mass in form of subhalos: {self._msub:0.3e} "
            f"({self._fsub*100.0}% of the cluster mass)\n"
            f"\t- Anna J factor [No sub]: {self._jfactor:0.3e} "
            f"({self._jfactorpp:0.3e})\n"
            f"\t- Decay D factor [No sub]: {self._dfactor:0.3e} "
            f"({self._dfactorpp:0.3e})\n"
        )

        logger.info(msg)

        return


    def get_m200(self) -> u.Quantity:

        if self._dmprofile.lower() == "nfw":

            m200 = get_enclosed_mass_nfw(
                self._r200,
                self._rs,
                self._rhos,
                self._rsat,
                self._rhosat,
                length_unit=self._rs.unit
            )

        else :

            logger.error(repr(DMProfileError("Unknown DM profile")))

            m200 = 0 *u.M_sun

        return m200

    def jfactor_sph(self):

        if self._dmprofile.lower() == "nfw":

            j_tot = jfactor_on_sphere_nfw(
                self._r200,self._rs,self._rhos,self._rsat,self._rhosat
            )

        else:

            logger.error(repr(DMProfileError("Unknown DM profile")))
            j_tot = 0*u.M_sun**2/u.Mpc**5

        return j_tot

    def dfactor_sph(self):

        if self._dmprofile.lower() == "nfw":

            d_tot = dfactor_on_sphere_nfw(
                self._r200,self._rs,self._rhos,self._rsat,self._rhosat
            )

        else:

            logger.error(repr(DMProfileError("Unknown DM profile")))
            d_tot = 0*u.M_sun/u.Mpc**2

        return d_tot

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
        # normalization in the number of subhalo
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

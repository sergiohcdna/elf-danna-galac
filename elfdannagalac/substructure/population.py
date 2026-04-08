###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Population container for subhalos                                       #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             April-2026.                                                     #
###############################################################################

import astropy.units as u
import numpy as np

from astropy.table import QTable

from loguru import logger
from typing import Union

class SubhaloView:
    """
    Read-only view of a single subhalo within a container
    """

    __slots__ = ('_container','_idx')
    
    def __init__(self,container,idx):

        self._container = container
        self._idx       = idx
    
    @property
    def mass(self):

        return self._container._msh[self._idx]
    
    @property
    def r(self):

        return self._container._rsh[self._idx]
    
    @property
    def x(self):

        return self._container._xsh[self._idx]
    
    @property
    def y(self):

        return self._container._ysh[self._idx]
    
    @property
    def z(self):

        return self._container._zsh[self._idx]
    
    @property
    def c(self):

        return self._container._csh[self._idx]
    
    @property
    def r200(self):

        return self._container._r200[self._idx]

    @property
    def rs(self):

        return self._container._rs[self._idx]

    @property
    def rhos(self):

        return self._container._rhos[self._idx]
    
    @property
    def lanna(self):

        return self._container._lanna[self._idx]

    @property
    def cross(self):

        return self._container._cross[self._idx]

    def __repr__(self):
        return f"<Subhalo {self._idx}: mass={self.mass}, r={self.r}>"

class SubHaloPopulation():

    def __init__(
        self,
        masses     : u.Quantity,
        rpositions : u.Quantity,
        xsh        : u.Quantity,
        ysh        : u.Quantity,
        zsh        : u.Quantity,
        csh_vals   : np.ndarray,
        r200_vals  : u.Quantity,
        rhos_vals  : u.Quantity,
        rs_vals    : u.Quantity,
        lanna_vals : u.Quantity,
        cross_vals : u.Quantity
    ) -> None:
        
        n = len(masses)

        pars = (
            rpositions,xsh,ysh,zsh,csh_vals,
            r200_vals,rhos_vals,rs_vals,
            lanna_vals,cross_vals
        )

        if not all(len(arr) == n for arr in pars):
            msg = "Initial population arrays have not the same size"
            logger.error(repr(ValueError(msg)))

        lunit = rpositions.unit

        self._msh   = masses.to(u.Msun)
        self._rsh   = rpositions
        self._xsh   = xsh.to(lunit)
        self._ysh   = ysh.to(lunit)
        self._zsh   = zsh.to(lunit)
        self._csh   = csh_vals
        self._r200  = r200_vals
        self._rhos  = rhos_vals
        self._rs    = rs_vals
        self._lanna = lanna_vals
        self._cross = cross_vals
        self._nsub  = n

    def __len__(self) -> int:

        """
        Number of subhalos
        """
        return self._nsub

    def __repr__(self) -> str:
        return (f"<SubhaloContainer: {len(self)} subhalos, "
                f"mass unit={self._msh.unit},length unit={self._rsh.unit}>")


    def __iter__(self):

        """
        Iterate over subhalos, yielding a SubhaloView for each
        """

        for i in range(len(self)):
            yield self[i]

    def __getitem__(self,index):

        """
        Return a SubhaloView (lightweight view of one subhalo) at given index.
        Supports slicing as well
        """

        if isinstance(index,slice):
            # Return a new container for slices (efficient, no copy unless needed)
            return SubHaloPopulation(
                mass       = self.mass[index],
                rpositions = self.r[index],
                xsh        = self.x[index],
                ysh        = self.y[index],
                zsh        = self.z[index],
                csh_vals   = self.c[index],
                r200_vals  = self.r200[index],
                rhos_vals  = self.rhos[index],
                rs_vals    = self.rs[index],
                lanna_vals = self.lanna[index],
                cross_vals = self.cross[index]
            )
        
        else:

            return SubhaloView(self,index)

    @classmethod
    def from_values(
        cls,
        mass_vals   : Union[list,np.ndarray],
        rpos_vals   : Union[list,np.ndarray],
        xvals       : Union[list,np.ndarray],
        yvals       : Union[list,np.ndarray],
        zvals       : Union[list,np.ndarray],
        cvals       : Union[list,np.ndarray],
        r200vals    : Union[list,np.ndarray],
        rhosvals    : Union[list,np.ndarray],
        rsvals      : Union[list,np.ndarray],
        lannavals   : Union[list,np.ndarray],
        crossvals   : Union[list,np.ndarray],
        mass_unit   : u.Unit = u.Msun,
        length_unit : u.Unit = u.kpc
    ) -> "SubHaloPopulation":
        
        """
        Create a SubHalo Population container from values and units

        :param mass_vals: Description
        :type mass_vals: Union[list, np.ndarray]
        :param rpos_vals: Description
        :type rpos_vals: Union[list, np.ndarray]
        :param xvals: Description
        :type xvals: Union[list, np.ndarray]
        :param yvals: Description
        :type yvals: Union[list, np.ndarray]
        :param zvals: Description
        :type zvals: Union[list, np.ndarray]
        :param cvals: Description
        :type cvals: Union[list, np.ndarray]
        :param mass_unit: Description
        :type mass_unit: u.unit
        :param length_unit: Description
        :type length_unit: u.Unit
        :return: Description
        :rtype: SubHaloPopulation
        """

        masses     = np.asarray(mass_vals)*mass_unit
        rpositions = np.asarray(rpos_vals)*length_unit
        x_sh       = np.asarray(xvals)*length_unit
        y_sh       = np.asarray(yvals)*length_unit
        z_sh       = np.asarray(zvals)*length_unit
        csh_vals   = np.asarray(cvals)
        r200_vals  = np.asarray(r200vals)*length_unit
        rhos_vals  = np.asarray(rhosvals)*mass_unit/length_unit**3
        rs_vals    = np.asarray(rsvals)*length_unit
        lanna_vals = np.asarray(lannavals)*mass_unit**2/length_unit**3
        cross_vals = np.asarray(crossvals)*mass_unit**2/length_unit**3

        return cls(
            masses,rpositions,x_sh,y_sh,z_sh,csh_vals,
            r200_vals,rhos_vals,rs_vals,lanna_vals,cross_vals
        )

    @classmethod
    def from_qtable(cls,table: QTable) -> "SubHaloPopulation":
        """
        Create container from an astropy QTable (must have columns: mass,
        rpositions,x,y,z,concentration).
        """
        return cls(
            masses     = table["msh"],
            rpositions = table["rsh"],
            xsh        = table["xsh"],
            ysh        = table["ysh"],
            zsh        = table["zsh"],
            csh_vals   = table["csh"],
            r200_vals  = table["r200"],
            rhos_vals  = table["rhos"],
            rs_vals    = table["rs"],
            lanna_vals = table["lanna"],
            cross_vals = table["cross"]
        )

    def to_qtable(self) -> QTable:

        """
        Convert container to an astropy QTable (preserves units).
        """

        return QTable(
            {
                "msh"   : self._msh,
                "rsh"   : self._rsh,
                "xsh"   : self._xsh,
                "ysh"   : self._ysh,
                "zsh"   : self._zsh,
                "csh"   : self._csh,
                "r200"  : self._r200,
                "rhos"  : self._rhos,
                "rs"    : self._rs,
                "lanna" : self._lanna,
                "cross" : self._cross
            }
        )

    def save_to_fits(
        self,
        filename  : str,
        overwrite : bool = True
    ) -> None:
        
        """
        Save container to a FITS file (units preserved automatically).
        """

        self.to_qtable().write(filename,format='fits',overwrite=overwrite)

    @classmethod
    def load_from_fits(cls,filename:str) -> "SubHaloPopulation":

        """
        Load container from a FITS file.
        """

        tbl = QTable.read(filename,format='fits')

        return cls.from_qtable(tbl)


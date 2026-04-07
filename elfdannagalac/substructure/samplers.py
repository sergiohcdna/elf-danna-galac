###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Samplers for position and masses of a set of subhalos                   #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             April-2026.                                                     #
###############################################################################

import numpy as np

from scipy.stats import rv_continuous


class dndm_PL(rv_continuous):

    def __init__(
        self,
        alpha : float,
        m_min : float,
        m_max : float,
    ):

        r"""
        Mass sampler for subhalos in a DM halo following a 
        power-law for the subhalo mass function (SHMF). The 
        SHMF is defined in the range between $m_\text{min}$ 
        and $m_\text{max}$. 

        We use analytical expressions for all the calculations.

        :param alpha: Power-Law index
        :type alpha: float
        :param m_min: Minimum mass of the subhalos
        :type m_min: float
        :param m_max: Maximum mass of the subhalos
        :type m_max: float
        """

        super().__init__(a=m_min, b=m_max)
        self.alpha = alpha
        self.m_min = m_min
        self.m_max = m_max

        if alpha == 1:

            self._norm = np.log(m_max/m_min)

        else:
            
            # self._norm = (m_max**(1-alpha) - m_min**(1-alpha)) / (1-alpha)
            self._norm = (m_max**(alpha+1) - m_min**(alpha+1)) / (alpha+1)

    def _pdf(self,x:float):

        return x**(self.alpha) / self._norm
    
    def _cdf(self,x:float):

        if self.alpha == 1:

            return np.log(x/self.m_min) / self._norm
        
        else:

            diffmin = x**(1+self.alpha) - self.m_min**(1+self.alpha)
            diffmax = self.m_max**(1+self.alpha) - x**(1+self.alpha)

            return diffmin/diffmax


class dndvCoredNFW(rv_continuous):
    def __init__(self,r_s:float,r_sat:float,r_max:float):

        r"""
        Radial distribution for subhalos in a NFW profile: 
        constant density for $r < r_\text{sa}t$, NFW for $r >= r_\text{sat}$.
        
        Parameters:
            :param r_s: scale radius
            :param r_sat: Saturation Radius (0 < r_sat < r_s)
            :param r_max: outer truncation radius
        """

        super().__init__(a=0, b=r_max)

        self.r_s   = r_s
        self.r_sat = r_sat
        self.r_max = r_max
        self.x_sat = r_sat / r_s
        self.x_max = r_max / r_s
        
        self.inner_const = 1.0 / (self.x_sat * (1 + self.x_sat)**2)
        self.I_in_scaled = self.inner_const * (r_sat**3 / 3.0)
        
        term_max = np.log(1 + self.x_max) - self.x_max/(1 + self.x_max)
        term_sat = np.log(1 + self.x_sat) - self.x_sat/(1 + self.x_sat)
        self.I_out_scaled = self.r_s**3 * (term_max - term_sat)
        
        self.Z_scaled = self.I_in_scaled + self.I_out_scaled
    
    def _pdf(self,r:float|np.ndarray):
        
        if np.isscalar(r):

            if r <= self.r_sat:

                return (r**2 * self.inner_const) / self.Z_scaled
            
            else:

                x   = r / self.r_s
                val = r**2 / (x * (1 + x)**2)

                return val / self.Z_scaled
        else:

            # Vectorized
            result   = np.empty_like(r)
            mask_in  = r <= self.r_sat
            mask_out = ~mask_in

            result[mask_in] = r[mask_in]**2 * self.inner_const

            x = r[mask_out] / self.r_s

            result[mask_out] = r[mask_out]**2 / (x * (1 + x)**2)

            return result / self.Z_scaled
    
    def _cdf(self,r:float|np.ndarray):

        if np.isscalar(r):

            if r <= 0:

                return 0.0
            
            if r >= self.r_max:

                return 1.0
            
            if r <= self.r_sat:

                return (self.inner_const * r**3 / 3.0) / self.Z_scaled
            
            else:

                x = r / self.r_s

                term_r     = np.log(1 + x) - x/(1 + x)
                term_sat   = np.log(1 + self.x_sat) - self.x_sat/(1 + self.x_sat)
                I_out_part = self.r_s**3 * (term_r - term_sat)

                return (self.I_in_scaled + I_out_part) / self.Z_scaled
        else:
            
            return np.array([self._cdf(ri) for ri in r])


#################################################
#################################################

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os
# import logging
from loguru import logger
from ..tools.misc import ValidString,ValidValue

# logger    = logging.getLogger(__name__)
# fmt       = ('%(asctime)s[%(levelname)s] @ %(filename)s.%(funcName)s ' +
#              '(%(lineno)d): %(message)s')
# dmhandler = logging.StreamHandler()
# dmformat  = logging.Formatter(fmt)
# dmhandler.setLevel(logging.WARNING)
# dmhandler.setFormatter(dmformat)



# Channels for PPPC4DMID are taken from:
# http://www.marcocirelli.net/PPPC4DMID.html
# We do not consider contribution of helicities
# or longitudinal/transverse momentum
ALLOWED_CHANNELS_PPPC4DMID = ('e','Mu','Tau',
                              'q','c','b','t',
                              'W','Z','g','Gamma','h',
                              'Nue','NuMu','NuTau')

# Channels for cosmiXs are taken from:
# https://github.com/ajueid/CosmiXs
# The same for helicities and momentum
# For convention, in the table, 'a' refers to photons
ALLOWED_CHANNELS_COSMIXS = ('e','mu','tau',
                            'u','d','s','c','b','t',
                            'W','Z','g','h','a','aZ','HZ',
                            'nue','numu','nutau')

# EBL models are taken from:
# https://github.com/me-manu/ebltable
# ALLOWED_EBLMODELS = ('franceschini','franceschini2017',
#                      'kneiske','finke','dominguez','dominguez-upper',
#                      'dominguez-lower','inuoe','inuoe-low-pop3',
#                      'inuoe-up-pop3','gilmore','gilmore-fixed')

ALLOWED_PROCESSES = ('anna','decay')

ALLOWED_PROJECTS = ('cosmixs','pppc4dmid')

# @ValidString("_eblmodel",empty_allowed=False,options=ALLOWED_EBLMODELS)
@ValidString("_process",empty_allowed=False,options=ALLOWED_PROCESSES)
@ValidString("_project",empty_allowed=False,options=ALLOWED_PROJECTS)
# @ValidValue("_z",min_val=0)
@ValidValue("_mass",min_val=10, max_val=1.e+5)
class dmspectrum():
    """
    Class to compute spectra for electron
    from annihilations or decay of dark matter particles.
    The calculation is based on tables from PPPC4DMID project:
        http://www.marcocirelli.net/PPPC4DMID.html
    and from the new cosmiXs project:
        https://github.com/ajueid/CosmiXs
    You can select between both projects.
    Theres is a local copy of files with PPPC4DMID and cosmiXs tables
    """

    def __init__(self,dm_mass,emin,emax,channel,process='anna',
                 project='cosmixs',epoints=100, nbins=10):
        """
        Initiate dark matter class

        Parameters
        ----------

        dm_mass : Mass of dark matter particle in GeV
                  Because cosmiXs and PPPC4DMID use GeV
        emin    : Minimum energy to compute spectra (GeV)
        emax    : Maximum energy to compute spectra (GeV)
        channel : Annihilation/Decay channel
        process : Annihilation (anna) or Decay (decay) of
                  dark matter particles
        project : DATA Project used to compute the spectrum
        epoints : Number of points in energy spectrum
        nbins   : Number of energy bins used to compute weights
        """

        # First, setting some properties of the class that don't
        # need more checks
        self._mass     = dm_mass
        self._process  = process
        self._project  = project
        self._epoints  = epoints
        self._nbins    = nbins
   
        # This is the same variable that both projects
        # use to give the spectrum.
        # Using log10x to be able to compare relatively 
        # "usual" values (small numbers) and avoid
        # possible precission problems
        xmin      = emin/dm_mass
        log10xmin = np.log10(xmin)

        # Setting extra properties of the class
        # based in what project we are using to
        # interpolate the data
        # The properties are: channel, emin, and emax
        if project == 'cosmixs':

            # Check for the channel
            if channel not in ALLOWED_CHANNELS_COSMIXS:

                msg = ('\nChannel is not valid\n' +
                       f'Options are {ALLOWED_CHANNELS_COSMIXS}')
                raise ValueError(msg)

            self._channel = channel

            # check for emin
            # cosmiXs report log10x starting at -8.9950
            if log10xmin < -8.995:

                val = np.power(10,-8.955)*dm_mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                self._emin = val

            else:

                self._emin = emin

        elif project == 'pppc4dmid':

            if channel not in ALLOWED_CHANNELS_PPPC4DMID:

                msg = ('\nChannel is not valid\n' +
                       f'Options are {ALLOWED_CHANNELS_PPPC4DMID}')
                raise ValueError(msg)

            self._channel = channel

            # check for emin
            # PPPC4DMID report log10x starting at -8.9
            if log10xmin < -8.9:

                val = np.power(10,-8.9)*dm_mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                self._emin = val

            else:

                self._emin = emin

        # Check for emax
        # emax cannot larger than dm_mass for annihilation
        # and dm_mass/2 for decay. So, we need to check for both cases
        # The same for e_min, right?
        # At this starting point, only annihilation is ok
        # But now, I am including decay too
        if process == 'anna':

            if emax > dm_mass:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                        'Setting Max energy to mass of the particle')
                logger.warning(msg)

                self._emax = dm_mass

            else:

                self._emax = emax

        elif process == 'decay':

            if emax > dm_mass/2:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                        'Setting Max energy to half mass of the particle')
                logger.warning(msg)

                self._emax = dm_mass/2

            else:

                self._emax = emax

        # Get array with values used to get the spectrum, array of energy bin edges and array with weights
        self._energy   = self._earray(emin,emax,epoints)
        self._ebins    = self._engbins(emin,emax,self.nbins)
        self._weights  = self._Weights(self._energy,self.spectrum(),self._ebins, self._nbins)

        #   Return
        return

    @staticmethod
    def _earray(emin,emax,epoints):
        """
        Return energy array to compute the spectra.
        The calculation is based in the number of points
        Return an np.array instance. The energies are
        computed assuming logarithmic distance.
        """
        logemin  = np.log10(emin)
        logemax  = np.log10(emax)
        energies = np.logspace(logemin,logemax,epoints)

        #   Return
        return energies
    
    @staticmethod
    def _engbins(emin,emax,nbins):
        """"
        Returns np.array with energy bins' edges 
        to compute spectrum weights according to 
        the given number of bins.Also computed assuming
        logarithmic distance.
        """
        min_E=np.log10(emin)
        max_E=np.log10(emax)
        E_i=np.logspace(min_E,max_E,nbins+1)

        return E_i
    
    @staticmethod
    def _Weights(engs,E_spectrum,ebins,nbins):
        """"
        Returns np.array with electron spectrum weights
        according to the given number of bins. 
        """
        Weights=[]
        index_list=np.zeros((nbins,), dtype=list)
        N_tot = sum(E_spectrum)
        E_i=ebins
        
        for i in range(len(E_i)-1):                                     
            index=[]

            if i != (len(E_i)-1)-1:
                for j, item in enumerate(engs):
                    if ((item >= E_i[i]) and (item < E_i[i+1])):
                        index.append(j)

                index_list[i]=(index)
                spec_i=[]

                for num in index_list[i]:
                    spec_i.append(E_spectrum[num])

                Weights.append(sum(spec_i)/N_tot)

            else:
                for j, item in enumerate(engs):
                    if ((item >= E_i[i]) and (item <= E_i[i+1])):
                        index.append(j)

                index_list[i]=(index)
                spec_i=[]

                for num in index_list[i]:
                    spec_i.append(E_spectrum[num])

                Weights.append(sum(spec_i)/N_tot)

        return Weights


    @property
    def mass(self):
        """
        Return mass of the candidate
        """

        # Return
        return self._mass
    
    @mass.setter
    def mass(self,dm_mass):
        """
        Set the mass of the candidate

        Parameters
        ------------------------
            dm_mass: Mass (in GeV) [10 GeV,100 TeV]
        """

        # Check that the mass is valid
        if not (10 <= dm_mass <= 1.e+5):
            raise ValueError(('\nMass of DM particle ' +
                              f'with value {dm_mass} ' +
                              'is out of range: [5,1.e+5] GeV'))
        
        # Set mass
        self._mass = dm_mass

        # Return
        return
    
    @property
    def emin(self):

        return self._emin

    @emin.setter
    def emin(self,e_min):
        
        xmin = np.log10(e_min/self._mass)

        if self._project == 'cosmixs':
            # check for emin
            # cosmiXs report log10x starting at -8.9950
            if xmin < -8.995:
                
                val = np.power(10,-8.955)*self._mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                self._emin = val

            else:

                self._emin = e_min

        elif self._project == 'pppc4dmid':

            # check for emin
            # PPPC4DMID report log10x starting at -8.9
            if xmin < -8.9:

                val = np.power(10,-8.9)*self._mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                self._emin = val

            else:

                self._emin = e_min

        # I need to update the energy values too!!
        # Get array with values used to get the spectrum
        self._energy = self._earray(e_min,self._emax,self._epoints)

        # Return
        return

    @property
    def emax(self):

        return self._emax
    
    @emax.setter
    def emax(self,e_max):
        # Check for emax
        # emax cannot larger than dm_mass for annihilation
        # and dm_mass/2 for decay. So, we nee to check for both cases
        # The same for e_min, right?
        # At this starting point, only annihilation is ok
        # But now, I am including decay too c:

        if self._process == 'anna':

            if e_max > self._mass:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                       'Setting Max energy to mass of the particle')
                logger.warning(msg)

                self._emax = self._mass

            else:

                self._emax = e_max

        elif self._process == 'decay':

            if e_max > self._mass/2:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                       'Setting Max energy to half mass of the particle')
                logger.warning(msg)

                self._emax = self._mass/2

            else:

                self._emax = e_max

        # I need to update the energy values too!!
        # Get array with values used to get the spectrum
        self._energy = self._earray(self._emin,e_max,self._epoints)
        self._ebins = self._engbins(self._emin,e_max,self._nbins)

        # Return
        return

    @property
    def engs(self):

        return self._energy

    @engs.setter
    def engs(self,e_min,e_max,e_points):

        if e_min > e_max:

            raise ValueError('Invalid Values for Emin and Emax')

        xmin = np.log10(e_min/self._mass)

        if self._project=='cosmixs':
            # check for emin
            # cosmiXs report log10x starting at -8.9950
            if xmin < -8.995:
                
                val = np.power(10,-8.955)*self._mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                e_min = val
                self._emin = e_min

            else:

                self._emin = e_min

        elif self._project == 'pppc4dmid':

            # check for emin
            # PPPC4DMID report log10x starting at -8.9
            if xmin < -8.9:

                val = np.power(10,-8.9)*self._mass

                msg = ('Min energy is below the allowed value\n'+ 
                       f'Setting to the min value {val:.3e}')
                logger.warning(msg)

                e_min      = val
                self._emin = val

            else:

                self._emin = e_min

        if self._process == 'anna':

            if e_max > self._mass:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                       'Setting Max energy to mass of the particle')
                logger.warning(msg)

                self._emax = self._mass

            else:

                self._emax = e_max

        elif self._process == 'decay':

            if e_max > self._mass/2:

                msg = ('Maximum energy cannot exceed the energy available.\n'+
                       'Setting Max energy to half mass of the particle')
                logger.warning(msg)

                self._emax = self._mass/2

            else:

                self._emax = e_max

        self._energy = self._array(e_min,e_max,e_points)
        self._ebins = self._engbins(e_min,e_max,self._nbins)

        return

    @property
    def channel(self):

        return self._channel
    
    @channel.setter
    def channel(self,ch):

        if self._project == 'cosmixs' and ch not in ALLOWED_CHANNELS_COSMIXS:

            msg = ('Invalid channel' +
                   f'Options are: {ALLOWED_CHANNELS_COSMIXS}')
            logger.error(msg)

        if self._project == 'pppc4dmid' and ch not in ALLOWED_CHANNELS_PPPC4DMID:

            msg = ('Invalid channel' +
                   f'Options are: {ALLOWED_CHANNELS_PPPC4DMID}')
            logger.error(msg)

        self._channel = ch

        return

    @property
    def process(self):

        return self._process

    @process.setter
    def process(self,dmprocess):

        if dmprocess not in ALLOWED_PROCESSES:

            msg = ('Invalid Process.\n'+
                   f'Options are {ALLOWED_PROCESSES}')
            raise ValueError(msg)
        
        self._process = dmprocess

        return
    
    @property
    def dataproject(self):

        return self._project


    @staticmethod
    def _interpolator(ch,dproject):

        BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        dataloc = os.path.join(BASEDIR,'data',dproject)
        fname   = 'AtProduction-Positrons.dat'
        fname   = os.path.join(dataloc,fname)

        data = np.genfromtxt(fname,names=True)

        masses = np.unique(data['mDM'])
        log10x = np.unique(data['Log10x'])

        dndlogx = np.zeros((masses.size,log10x.size))

        for index,mass in enumerate(masses):

            mass    = int(mass)
            indices = np.where(data['mDM'] == mass)
            phis    = data[ch][indices]

            for pindex,phi in enumerate(phis):

                dndlogx[index][pindex] = phi

        points   = (masses,log10x)
        dminterp = RegularGridInterpolator(points,dndlogx,method='linear',
                                           bounds_error=False,fill_value=0.0)

        return dminterp

    def spectrum(self):

        dm_interp = self._interpolator(self._channel,self._project)

        if self._process == 'anna':

            log10xval = np.log10(self._energy/self._mass)
            dndlogx   = dm_interp((self._mass,log10xval),method='linear')

        elif self._process == 'decay':

            log10xval = np.log10(self._energy/(0.5*self._mass))
            dndlogx   = dm_interp((0.5*self._mass,log10xval),method='linear')

        dndlogx = dndlogx.flatten()
        dnde    = dndlogx / (self._energy*np.log(10))

        return dnde
    
    @property
    def nbins(self):

        return self._nbins
    
    @nbins.setter
    def nbins(self,numbins):

        if (type(numbins) is int) and (numbins >= 0):

            self._nbins = numbins

            self._ebins = self._engbins(self._emin,self._emax,numbins)

            self._weights  = self._Weights(self._energy,self.spectrum(),self._ebins, numbins)
            
        else:

            raise ValueError(('\nValue of nbins must be a positive integer.'))
        

        return
        
    @property
    def ebins(self):

        return self._ebins
    
    @ebins.setter
    def ebins(self):

        #Get array with bin edges used to clculate weights
        self._ebins = self._engbins(self._emin,self._emax,self._nbins)

        # self._weights  = self._Weights(self._energy,self.spectrum(),self._ebins, self._nbins)

        #Return
        return

    @property
    def weights(self):

        return self._weights
    
    @weights.setter    
    def weights(self):

        #Get array with spectrum weights
        self._weights = self._Weights(self._energy,self.spectrum(),self._ebins,self._nbins)

        #Return
        return
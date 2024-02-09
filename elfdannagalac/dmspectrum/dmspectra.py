#################################################
#################################################

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os
import logging
from ..tools.misc import ValidString,ValidValue

dmslog    = logging.getLogger(__name__)
fmt       = ('%(asctime)s[%(levelname)s] @ %(filename)s.%(funcName)s ' +
             '(%(lineno)d): %(message)s')
dmhandler = logging.StreamHandler()
dmformat  = logging.Formatter(fmt)
dmhandler.setLevel(logging.WARNING)
dmhandler.setFormatter(dmformat)



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
@ValidValue("_mass",min_val=5, max_val=1.e+5)
class dmspectrum():
    """
    Class to compute spectra for electron
    from annihilations or decay of dark matter particles.
    The calculation is based tables from PPPC4DMID project:
        http://www.marcocirelli.net/PPPC4DMID.html
    and from the new cosmiXs project:
        https://github.com/ajueid/CosmiXs
    You can select between both projects.
    Theres is a local copy of files with PPPC4DMID and cosmiXs tables
    """

    def __init__(self,dm_mass,emin,emax,channel,process='anna',
                 project='cosmixs',epoints=100):
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
        """

        # First, setting some properties of the class that don't
        # need more checks
        self._mass     = dm_mass
        self._process  = process
        self._project  = project
        self._epoints  = epoints

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
                       'Options are {0}'.format(ALLOWED_CHANNELS_COSMIXS))
                raise ValueError(msg)

            self._channel = channel

            # check for emin
            # cosmiXs report log10x starting at -8.9950
            if log10xmin < -8.995:

                val = np.power(10,-8.955)*dm_mass

                msg = ('Min energy is below the allowed value\n'+ 
                       'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

                self._emin = val

            else:

                self._emin = emin

        elif project == 'pppc4dmid':

            if channel not in ALLOWED_CHANNELS_PPPC4DMID:

                msg = ('\nChannel is not valid\n' +
                       'Options are {0}'.format(ALLOWED_CHANNELS_COSMIXS))
                raise ValueError(msg)

            self._channel = channel

            # check for emin
            # PPPC4DMID report log10x starting at -8.9
            if log10xmin < -8.9:

                val = np.power(10,-8.9)*dm_mass

                msg = ('Min energy is below the allowed value\n'+ 
                       'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

                self._emin = val

            else:

                self._emin = emin

        # Check for emax
        # emax cannot larger than dm_mass for annihilation
        # and dm_mass/2 for decay. So, we nee to check for both cases
        # The same for e_min, right?
        # At this starting point, only annihilation is ok
        if emax > dm_mass:

            msg = ('Maximum energy cannot exceed the energy available.\n'+
                    'Setting Max energy to the mass of the particle')
            dmslog.warning(msg)

            self._emax = dm_mass

        else:

            self._emax = emax

        # Get array with values used to get the spectrum
        self._energy   = self._earray(emin,emax,epoints)

        #   Return
        return

    @staticmethod
    def _earray(emin,emax,epoints):
        """
        Return energy array to compute the spectra.
        The calculation is based in the number of points
        Return an np.array instance. The energies are
        computed assuming logarithmic distance
        """
        logemin  = np.log10(emin)
        logemax  = np.log10(emax)
        energies = np.logspace(logemin,logemax,epoints)

        #   Return
        return energies


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
            dm_mass: Mass (in GeV) [5 GeV,100 TeV]
        """

        # Check that the mass is valid
        if not (5 <= dm_mass <= 1.e+5):
            raise ValueError(('\nMass of DM particle ' +
                              'with value {0} '.format(dm_mass) +
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
                        'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

                self._emin = val

            else:

                self._emin = e_min

        elif self._project == 'pppc4dmid':

            # check for emin
            # PPPC4DMID report log10x starting at -8.9
            if xmin < -8.9:

                val = np.power(10,-8.9)*self._mass

                msg = ('Min energy is below the allowed value\n'+ 
                       'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

                self._emin = val

            else:

                self._emin = e_min

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

        if e_max > self._mass:

            msg = ('Maximum energy cannot exceed the energy available.\n'+
                    'Setting Max energy to the mass of the particle')
            dmslog.warning(msg)

            self._emax = self._mass

        else:

            self._emax = e_max

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
                        'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

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
                       'Setting to the min value {:.3e}'.format(val))
                dmslog.warning(msg)

                e_min      = val
                self._emin = val

            else:

                self._emin = e_min

        if e_max > self._mass:

            msg = ('Maximum energy cannot exceed the energy available.\n'+
                    'Setting Max energy to the mass of the particle')
            dmslog.warning(msg)

            self._emax = self._mass
            e_max      = self._mass

        else:

            self._emax = e_max

        energies = self._array(e_min,e_max,e_points)

        self._energy = energies

        return

    @property
    def channel(self):

        return self._channel
    
    @channel.setter
    def channel(self,ch):

        if self._project == 'cosmixs' and ch not in ALLOWED_CHANNELS_COSMIXS:

            msg = ('Invalid channel' +
                   'Options are: {0}'.format(ALLOWED_CHANNELS_COSMIXS))
            dmslog.error(msg)

        if self._project == 'pppc4dmid' and ch not in ALLOWED_CHANNELS_PPPC4DMID:

            msg = ('Invalid channel' +
                'Options are: {0}'.format(ALLOWED_CHANNELS_PPPC4DMID))
            dmslog.error(msg)

        self._channel = ch

        return

    @property
    def process(self):

        return self._process

    @process.setter
    def process(self,dmprocess):

        if dmprocess not in ALLOWED_PROCESSES:

            msg = ('Invalid Process.\n'+
                   'Options are {0}'.format(ALLOWED_PROCESSES))
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
        dminterp = RegularGridInterpolator(points,dndlogx,method='cubic',
                                           bounds_error=False,fill_value=None)

        return dminterp

    def spectrum(self):

        if self._process == 'anna':

            dm_interp = self._interpolator(self._channel,self._project)


            log10xval = np.log10(self._energy/self._mass)
            dndlogx   = dm_interp((self._mass,log10xval),method='linear')
            dndlogx   = dndlogx.flatten()
            dnde      = dndlogx / self._energy / np.log(10)

        return dnde
from scipy.stats import rv_discrete

from crpropa import SourceFeature
from crpropa import GeV

import time

class SourceDMSpectrum(SourceFeature):
    """
    Set the energy of particles from a DMSpectrum
    Hard coded to electrons, for now
    """

    def __init__(self,emin,emax,dminterpolator):
        # Energies should be in GeV

        SourceFeature.__init__(self)

        self._emin = emin*GeV
        self._emax = emax*GeV

        # update the interpolator to the new energies 
        dminterpolator.emin = emin
        dminterpolator.emax = emax

        self._dminterpolator = dminterpolator

        # Get other properties from the dminterpolator instance
        self._dmmass    = dminterpolator.mass
        self._dmchannel = dminterpolator.channel
        self._dmprocess = dminterpolator.process
        self._dmproject = dminterpolator.dataproject

        # Get the spectrum
        self._dmspectrum = dminterpolator.spectrum()

        # Get the energies
        self._energies = dminterpolator.engs

        # Get the normalized spectrum
        self._specnorm = self._dmspectrum / self._dmspectrum.sum()

        # Create the random generator
        values = (self._energies,self._specnorm)

        self._dmdistro = rv_discrete(name='DMEngdistribution',values=values)


    def setDescription(self):

        msg = ('Set the energy of electrons/positrons from the '+
               'energy spectrum induced by DM annihilation/decay')

        description = msg

        return description

    def prepareParticle(self,particleState):

        self._dmdistro.random_state = int(time.time())
        speng = self._dmdistro.rvs()

        particleState.setEnergy(speng*GeV)


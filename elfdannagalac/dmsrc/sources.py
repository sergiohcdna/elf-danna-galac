###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing DM sources used for particle injection                        #
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

from crpropa import Vector3d
from crpropa import DensityGrid
from crpropa import (
    Source,
    SourceMassDistribution,
    SourcePosition,
    SourceRedshift,
    SourceIsotropicEmission,
    SourceParticleType,
    SourcePowerLawSpectrum,
    SourceMultiplePositions,
    SourceList
)

from ..substructure.population import SubHaloPopulation

def preparePointLikeDMSource(
    dmsource_pos : Vector3d,
    redshift     : float,
    part_type    : int,
    emin         : float,
    emax         : float,
    pl_index     : float
) -> Source:

    """
    Prepare a pointlike DM source to inject particles of type part_type.

    All vectors are assumed to be in the Observer's coordinate system (OCS).

    Injection of particles follow a PowerLaw in energy spectrum between 
    emin and emax with index pl_index. The energies emin and emax are given 
    in GeV.
    
    :param dmsource_pos: Position  [Mpc]
    :type dmsource_pos: Vector3d
    :param redshift: Redshift
    :type redshift: float
    :param part_type: Particle type following CRpropa convention
    :type part_type: int
    :param emin: Minimum energy of particles to be injected
    :type emin: float
    :param emax: Maximum energy of particles to be injected
    :type emax: float
    :param pl_index: Spectral index of PowerLaw
    :type pl_index: float
    """

    dms = Source()

    dms.add(SourcePosition(dmsource_pos))
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms

def prepareSmoothExtendedDMSource(
    dmdensity   : DensityGrid,
    max_density : float,
    maxTries    : int,
    rmin        : float,
    rmax        : float,
    redshift    : float,
    part_type   : int,
    emin        : float,
    emax        : float,
    pl_index    : float
) -> Source:

    """
    Docstring for prepareSmoothExtendedDMSource
    
        :param dmdensity: [Mass] DM density
        :type dmdensity: DensityGrid
        :param max_density: Max. value of the density used for normalization.
        :type max_density: float
        :param maxTries: Maximum number of trials to get a source.
        :type maxTries: int
        :param rmin: Lower Limit for source sampling [3D vector,m]
        :type rmin: float
        :param rmax: Upper Limit for source sampling [3D vector,m]
        :type rmax: float
        :param redshift: Redshift to the center of the DM halo
        :type redshift: float
        :param part_type: Particle type to be injected
        :type part_type: int
        :param emin: Minimum energy of particles to be injected
        :type emin: float
        :param emax: Maximum energy of particles to be injected
        :type emax: float
        :param pl_index: Spectral index of PowerLaw
        :type pl_index: float
        :return: Source Mass Distribution
        :rtype: SourceMassDistribution
    """

    xmin,ymin,zmin = rmin
    xmax,ymax,zmax = rmax

    dms = Source()

    dm_sources = SourceMassDistribution(dmdensity,max_density)
    dm_sources.setXrange(xmin,xmax)
    dm_sources.setYrange(ymin,ymax)
    dm_sources.setZrange(zmin,zmax)
    dm_sources.setMaximalTries(maxTries)

    dms.add(dm_sources)
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms

def prepareSubHaloDMSource(
    shpop     : SubHaloPopulation,
    ldmanna   : float,
    redshift  : float,
    part_type : int,
    emin      : float,
    emax      : float,
    pl_index  : float
) -> Source:
    
    shsources = SourceMultiplePositions()

    for sh in shpop:

        shsources.add(
            Vector3d(
                sh.x.to(u.m).value,
                sh.y.to(u.m).value,
                sh.z.to(u.m).value
            ),
            (sh.lanna.value+sh.cross.value)/ldmanna
        )

    dms = Source()

    dms.add(shsources)
    dms.add(SourceRedshift(redshift))
    dms.add(SourceIsotropicEmission())
    dms.add(SourceParticleType(part_type))
    dms.add(SourcePowerLawSpectrum(emin,emax,pl_index))

    return dms

def prepareDMHaloSource(
    dmdensity   : DensityGrid,
    shpop       : SubHaloPopulation,
    max_density : float,
    ldmanna     : float,
    maxTries    : int,
    rmin        : float,
    rmax        : float,
    redshift    : float,
    part_type   : int,
    emin        : float,
    emax        : float,
    pl_index    : float,
    w_smooth    : float,
    w_subhalo   : float,
) -> Source:

    smooth = prepareSmoothExtendedDMSource(
        dmdensity,
        max_density,
        maxTries,
        rmin,
        rmax,
        redshift,part_type,
        emin,
        emax,
        pl_index
    )

    subhalo = prepareSubHaloDMSource(
        shpop,
        ldmanna,
        redshift,
        part_type,
        emin,
        emax,
        pl_index
    )

    slist = SourceList()
    slist.add(smooth,w_smooth)
    slist.add(subhalo,w_subhalo)

    return slist

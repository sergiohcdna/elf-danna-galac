###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Preparing observers for particle detection                              #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             December-2025                                                   #
###############################################################################

from crpropa import Observer,ObserverSurface
from crpropa import ObserverTimeEvolution
from crpropa import Sphere,Vector3d
from crpropa import Mpc
from crpropa import TextOutput

from crpropa import (
    ObserverElectronVeto,
    ObserverNucleusVeto,
    ObserverNeutrinoVeto,
    ObserverPhotonVeto
)


def preparePhotonObserver(
    obsCenter    : Vector3d,
    obsRadius    : float,
    step         : float,
    nsteps       : int,
    outtxt       : TextOutput,
    deactivate   : bool=True,
    electronveto : bool=True,
    protonveto   : bool=True,
    neutrinoveto : bool=True,
) -> Observer:

    """
    Default observer to detect photons in galaxy cluster simulations. 
    The observer is assumed to be on a surface. We consider temporal 
    evolution of the system. We also apply, by default, veto on all the 
    other particles, as we are only interested on photon detection.
    On detection, photons are deactivated, and parameters for every 
    particle are saved into a txt file.
    
    :param obsCenter: Center of the Sphere [Mpc]
    :type obsCenter: Vector3d
    :param obsRadius: Radius of the Sphere [Mpc]
    :type obsRadius: float
    :param step: Used for temporal evolution [Mpc]
    :type step: float
    :param nsteps: Number of steps used for temporal evolution
    :type nsteps: int
    :param outtxt: Instance of txt output to record particles
    :type outtxt: TextOutput
    :param deactivate: Deactivate particles on Detection. Default is True
    :type deactivate: bool
    :param electronveto: Apply Veto to not record electrons. Default is True
    :type electronveto: bool
    :param protonveto: Apply Veto to not record protons. Default is True
    :type protonveto: bool
    :param neutrinoveto: Apply Veto to not record neutrinos. Default is True
    :type neutrinoveto: bool
    :return: Observer on a Sphere
    :rtype: Observer
    """

    thisobserver = Observer()

    thisobserver.add(
        ObserverSurface(
            Sphere(
                obsCenter*Mpc,
                obsRadius*Mpc
            )
        )
    )

    thisobserver.add(ObserverTimeEvolution(step,step,nsteps))
    thisobserver.setDeactivateOnDetection(deactivate)
    thisobserver.onDetection(outtxt)

    if electronveto:

        thisobserver.add(ObserverElectronVeto())

    if protonveto:

        thisobserver.add(ObserverNucleusVeto())


    if neutrinoveto:

        thisobserver.add(ObserverNeutrinoVeto())

    return thisobserver


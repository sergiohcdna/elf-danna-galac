###############################################################################
# Scienca Case 1: x                                                           #
#   - DM point source                                                         #
#   - Injection of electrons from DM annihilation                             #
#   - Obsever on a sphere                                                     #
#   - Spherical boundary                                                      #
#   - Just output for photons                                                 #
#   - Constant magnetic vector                                                #
#   - Constant diffusion coefficient                                          #
#   - only IC with CMB photons                                                #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             July-2024                                                       #
###############################################################################


import os
import time

from crpropa import (
    Vector3d,muG,GeV,pc,kpc,Mpc,
    UniformMagneticField,
    DiffusionSDE,
    SphericalBoundary,Sphere,
    MaximumTrajectoryLength,
    Source,SourcePosition,SourceRedshift,
    SourceIsotropicEmission,SourceParticleType,
    TextOutput,Output,
    Observer,ObserverSurface,ObserverPhotonVeto,
    ObserverElectronVeto,ObserverTimeEvolution,
    CMB,EMInverseComptonScattering,
    ModuleList,
)

from elfdannagalac.crpropafeatures.srcDMSpectra import SourceDMSpectrum
from elfdannagalac.dmspectrum import dmspectra

import argparse as ap

from elfdannagalac.tools.misc import elapsed_time,checkDir

if __name__ == '__main__':

    this_start = time.time()

    msg     = ('Galaxy clusters project: elf-danna-galac \n'
               'June/2024')
    options = ap.ArgumentParser(description=msg)

    src = options.add_argument_group('Dark matter and galaxy cluster','Input params')
    src.add_argument('--dmmass',help='Mass of the dark matter candidate (GeV)',
                     type=float,required=True,metavar='1000.0')
    src.add_argument('--emin',help='Minimum energy of electrons (GeV)',
                     type=float,required=True,metavar='100')
    src.add_argument('--emax',help='Maximum energy of electrons (GeV)',
                     type=float,required=True,metavar='0.9*dmmas')
    src.add_argument('--nparticles',help='Number of particles to be smulated',
                     type=int,required=True,metavar='10000')
    src.add_argument('--bfield',help='Strength of magnetic field along z direction (uG)',
                     type=float,required=True,metavar='25')
    src.add_argument('--channel',help='annihilation/decay channel',type=str,
                     required=False,metavar='tau',default='tau')
    src.add_argument('--process',help='Annihilation (anna) or decay (decay)',
                     type=str,required=False,default='anna',metavar='anna')
    src.add_argument('--project',help='Data project to get the spectrum [cosmixs,pppc4dmid]',
                     type=str,required=False,default='cosmixs',metavar='cosmixs')
    src.add_argument('--ofname',help='Output file name (txt)',
                     type=str,required=False,default='output.txt')
    src.add_argument('--odir',help='Output directory to save files',
                     type=str,required=False,default='./',metavar='./')

    args = options.parse_args()

    checkDir(args.odir)

    # This is to get the spectrum using dmspctra class
    # and to define the source used to asign the energy
    # to electrons in the CRpropa simulation
    dme_spec = dmspectra.dmspectrum(
        args.dmmass,
        args.emin,
        args.emax,
        args.channel,
        project=args.project,
        process=args.process
    )

    srcdmspectra = SourceDMSpectrum(
        args.emin,
        args.emax,
        dme_spec
    )

    # Now, I will define some parameters associated with the CRpropa simulation
    # Later, we can think to put this in a special function c:

    # Magnetic field
    # The value of the magnetic field is only for testting purposes
    # We assumed a uniform magnetic field along the z direction
    ConstMagVec = Vector3d(0*muG,0*muG,args.bfield*muG)
    BField      = UniformMagneticField(ConstMagVec)

    # parameters used for field line tracking
    precision = 1e-4
    minStep   = 10*pc
    maxStep   = 1*kpc

    # The following lines are used to describe the diffusion coefficient
    # We still need to check more literature about this.
    # ratio between parallel and perpendicular diffusion coefficient
    epsilon = .1

    # Difffusion Module
    # D_xx=D_yy= 1e23 m^2 / s, D_zz=10*D_xx
    # The normalization is adjusted and the energy dependence 
    # is deactivated (setting power law index alpha=0)
    # SDE refers to the nnumerical method used to solve
    # the diffusion equations: Stochastic diferential equations
    Dif = DiffusionSDE(BField,precision,minStep,maxStep,epsilon)
    Dif.setScale(1./10)
    Dif.setAlpha(0.)

    # Now, we define the boundary. For this test, we consider
    # an spherical boundary centered in the galaxy cluster.
    # We select a boundary radius greater than the 
    # radius of the spherical observer
    boundaryCenter  = Vector3d(16,0,0)*Mpc
    boundaryRadius  = 2.1*Mpc
    clusterBoundary = SphericalBoundary(boundaryCenter,boundaryRadius)

    # This is to specify that the simulation ends after some time
    # in pur case, just at the outskirts of the cluster
    # Boundary
    # Simulation ends after t=maxTra/c
    maxTra = MaximumTrajectoryLength(3.0*Mpc)

    # Now, we define parameters associated with the source of DM
    # as position, and redshift
    srcpos = Vector3d(16,0,0)*Mpc
    srcz   = 0.0043

    # and this is for the particle content in the simulation
    # For now, only electrons
    # But, if we would like to include positrons
    # we need to call a method for multiple particle types,
    # and pass -11, to consider positrons
    part_type = 11

    # So, to describe the source we consider:
    #   - Position (for now a point source)
    #   - Redshift (for temporal evolution, until the outskirst of the cluster)
    #   - Emission type (isotropic in our case)
    #   - Particle type (Particles injected)
    #   - Particle energy (following the distribution from DM spectra class)
    s = Source()
    s.add(SourcePosition(srcpos))
    s.add(SourceRedshift(srcz))
    s.add(SourceIsotropicEmission())
    s.add(SourceParticleType(part_type))
    s.add(srcdmspectra)

    # The next part is to configure the output
    # I will try the option to use the text format
    fname = os.path.join(args.odir,args.ofname)
    Out   = TextOutput(fname)

    # First, I disable all the columns, as we don't need everythin
    # Also, I am not completely sure what to save, hahaha
    # For now, I will save the 
    #   - Total trajectory of the photons
    #   - Position, current, but I think on detection should be better (?)
    #   - Energy of the photons
    #   - Weight. This is because it is impossible to track all the photons
    #             from synchrotron
    # Finally, I will set units to Mpc and GeV
    Out.disableAll()
    # Out.enable(Output.TrajectoryLengthColumn)
    Out.enable(Output.CurrentPositionColumn)
    # Out.enable(Output.CurrentIdColumn)
    Out.enable(Output.CurrentEnergyColumn)
    Out.enable(Output.WeightColumn)
    Out.setLengthScale(Mpc)
    Out.setEnergyScale(GeV)

    # Now, we configure the observer
    # For this test, we will consider an observer on a sphere
    
    # N is the total number of particles in the simulation
    # n is the number of steps
    # step refers to the size. In this case, size 2 Mpc
    N    = args.nparticles
    n    = 1000
    step = 5*Mpc/n

    # Now, here comes the parameters to define the sphere
    obsCenter = Vector3d(16.,0,0)*Mpc
    obsRadius = 2.0*Mpc

    # And this is to create an instance of the observer class
    obs = Observer()

    # We disable the output for electrons, as we are only interested in photons
    # setDeactivateOnDetection is set to True by default, but just in case
    obs.add(ObserverSurface(Sphere(obsCenter,obsRadius)))
    obs.add(ObserverElectronVeto())
    obs.add(ObserverTimeEvolution(step,step,n))
    obs.setDeactivateOnDetection(True)
    obs.onDetection(Out)

    ############## TEST ####################
    # I will add a second observer to deactivate all the electrons
    # I suspect this is the reason why the simulation takes too long
    # when removing the condition of MaximumTrajectoryLength
    # First, setting the output for electrons
    # This should be useful to test the electron spectrum
    efname = f'Electrons{args.ofname}'
    efname = os.path.join(args.odir,efname)
    outElectrons = TextOutput(efname)
    outElectrons.disableAll()
    outElectrons.enable(Output.TrajectoryLengthColumn)
    # outElectrons.enable(Output.CurrentPositionColumn)
    # outElectrons.enable(Output.CurrentIdColumn)
    # outElectrons.enable(Output.CurrentEnergyColumn)
    # outElectrons.enable(Output.CreatedIdColumn)
    outElectrons.enable(Output.SourcePositionColumn)
    outElectrons.enable(Output.SourceEnergyColumn)
    outElectrons.enable(Output.CurrentEnergyColumn)
    outElectrons.enable(Output.SerialNumberColumn)
    # outElectrons.enable(Output.CandidateTagColumn)
    outElectrons.setLengthScale(Mpc)
    outElectrons.setEnergyScale(GeV)

    # For this observer, we will consider an sphere centered in the galaxy cluster
    # with a radius of 0.5Mpc, based on the timescale for synchrtron energy-losses
    # for electrons with energies of 100 GeV --> ~1.e+6 yr
    # Here, I will disable the ouput for photons, as we are not concerned
    # about photons at this point
    obsElectron   = Observer()
    obsElecRadius = 0.3*Mpc

    obsElectron.add(ObserverSurface(Sphere(obsCenter,obsElecRadius)))
    obsElectron.add(ObserverPhotonVeto())
    obsElectron.add(ObserverTimeEvolution(step,step,n))
    obsElectron.setDeactivateOnDetection(False)
    obsElectron.onDetection(outElectrons)

    # Finally, the inverse compton module
    # And here, we are only considering the CMB as seed photon field
    # 0.5 referes to the "screening" (I don't remember the correct term now)
    # But it is required to weight the output of photons
    # because the total number of photons is beyond the computational limit
    cmb   = CMB()
    iccmb = EMInverseComptonScattering(cmb,True,0.5)

    # module list
    # Add modules to the list and run the simulation
    sim = ModuleList()

    sim.add(Dif)
    sim.add(obsElectron)
    sim.add(obs)
    sim.add(clusterBoundary)
    sim.add(maxTra)
    sim.add(iccmb)

    sim.setShowProgress(True)
    sim.run(s,N,True)
    outElectrons.close()
    Out.close()

    # Plotting and post-processing is done in another script

    msg = 'Total Elapsed time: '
    elapsed_time(this_start,msg)

###############################################################################
# NEW Science Case 1: x                                                       #
#   - DM point source                                                         #
#   - Injection of electrons PowerLaw                                         #
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
#             November-2025                                                   #
###############################################################################

import astropy.units as u
import numpy as np

from crpropa import pc,kpc,Mpc,GeV
from crpropa import Vector3d
from crpropa import DiffusionSDE
from crpropa import SphericalBoundary,MaximumTrajectoryLength
from crpropa import CMB,EMInverseComptonScattering
from crpropa import ModuleList

from ..magneticfield.bfields import get_cluster_field
from ..observer.observers import preparePhotonObserver

from ..dmsrc.profiles import dm_mass_density

from ..dmsrc.grids import (
    SmoothDMHaloMassDensityGrid,
    SmoothDMHaloMassSquaredDensityGrid
)

from ..dmsrc.profiles import allowed_spatial_types,allowed_profiles
from ..dmsrc.halo import DMHalo

from ..dmsrc.sources import (
    preparePointLikeDMSource,
    prepareSmoothExtendedDMSource,
    prepareDMHaloSource,
)

from ..dmspectrum.dmspectra import ALLOWED_PROCESSES

from ..tools.utils import create_table,prepareOutput

import argparse as ap
import time

from loguru import logger
from pathlib import Path

from ..tools.misc import checkDir,elapsed_time

def main():

    this_start = time.time()

    msg     = ('Galaxy clusters project: elf-danna-galac \n'
               'November/2025')
    options = ap.ArgumentParser(description=msg)

    src = options.add_argument_group('Galaxy Cluster','Input params')

    src.add_argument(
        "--srcname",
        help="Name of the cluster",
        type=str,
        required=True,
        metavar="Virgo"
    )
    src.add_argument(
        "--srcz",
        help="Redshift of the cluster",
        type=float,
        required=True,
        metavar="0.0043"
    )
    src.add_argument(
        "--dmprofile",
        help="Label for DM profile parametrization",
        type=str,
        required=True,
        choices=allowed_profiles,
        metavar="nfw"
    )
    src.add_argument(
        "--dmsource_type",
        help="Point or Extended like DM injection sources",
        type=str,
        required=True,
        choices=allowed_spatial_types,
        metavar="extended_smooth"
    )
    src.add_argument(
        "--m200",
        help="Mass enclosed up to a radius where $\rho_{crit}=200$ [Msun]",
        type=float,
        required=True,
        metavar="3.54e14 Msun"
    )
    src.add_argument(
        "--r200",
        help="Radius where $\rho_{crit}=200$ [kpc]",
        type=float,
        required=True,
        metavar="1.406e3 kpc"
    )
    src.add_argument(
        "--ra",
        help="Right Ascension [in deg]",
        type=float,
        required=True,
        metavar="12 deg"
    )
    src.add_argument(
        "--dec",
        help="Declination [deg]",
        type=float,
        required=True,
        metavar="27 deg"
    )
    src.add_argument(
        "--process",
        help="Annihilation or Decay?",
        type=str,
        required=False,
        default="anna",
        choices=ALLOWED_PROCESSES,
        metavar="[anna,decay]"
    )
    src.add_argument(
        "--subhalos",
        help="Consider Subhalos?",
        action=ap.BooleanOptionalAction
    )
    src.add_argument(
        '--emin',
        help='Minimum energy of electrons (GeV)',
        type=float,
        required=True,
        metavar='100'
    )
    src.add_argument(
        '--emax',
        help='Maximum energy of electrons (GeV)',
        type=float,
        required=True,
        metavar='1e5'
    )
    src.add_argument(
        '--nparticles',
        help='Number of particles to be smulated',
        type=int,
        required=True,
        metavar='10000'
    )
    src.add_argument(
        "--brms",
        help="RMS value of the turbulence field [muG]",
        type=float,
        required=True,
        metavar="5 muG"
    )
    src.add_argument(
        "--lmin",
        help="Minimum scale of the turbulence field [kpc]",
        type=float,
        required=True,
        metavar="0.02 Mpc"
    )
    src.add_argument(
        "--lmax",
        help="Maximum scale of the turbulence field [kpc]",
        type=float,
        required=True,
        metavar="0.15 Mpc"
    )
    src.add_argument(
        "--eta_index",
        help="Index of the Bfield dependance with radial distance",
        type=float,
        required=True,
        metavar="0.33"
    )
    src.add_argument(
        "--core_radius",
        help="Value of the electron density core radius [kpc]",
        type=float,
        required=True,
        metavar="0.25 Mpc"
    )
    src.add_argument(
        "--diff_epsilon",
        help="Anisotrpy factor of the diffusion coefficient",
        type=float,
        required=False,
        default=0.1,
        metavar="0.1"
    )
    src.add_argument(
        "--diff_scale",
        help="Scaling factor to set the diffusion coefficient",
        type=float,
        required=False,
        default=0.1,
        metavar="0.1"
    )
    src.add_argument(
        "--diff_alpha",
        help="Spectral index for the Diff_coeff = E^{-alpha}",
        type=float,
        required=False,
        default=0.1,
        metavar="0.1"
    )
    src.add_argument(
        "--ncells",
        help="Odd Number of points in the spatial grid",
        type=int,
        required=True,
        metavar="511"
    )
    src.add_argument(
        "--maxtries",
        help="Maximum number of tries for source sampling",
        type=float,
        required=False,
        metavar=100000
    )
    src.add_argument(
        "--chunksize",
        help="Size of chunks to compute grids",
        type=int,
        required=False,
        default=32,
        metavar="32"
    )
    src.add_argument(
        '--ofname',
        help='Output file name [fits,fits.gz]',
        type=str,
        required=False,
        default='output.fits'
    )
    src.add_argument(
        '--odir',
        help='Output directory to save files',
        type=str,
        required=False,
        default='./',
        metavar='./'
    )

    # To avoid multiple initialization parameters, we will use 
    # the parameters of the magnetic field grid to get the grid 
    # for the DM mass.
    # Until discussed, we will use the DM mass density for 
    # source's position sampling. This is for the smooth component.

    logger.info("Getting parameters")
    args = options.parse_args()

    outpath = Path(args.odir)
    checkDir(outpath)

    msg = "Unknown file extension to save results"
    assert args.ofname.lower().endswith(("fits.gz","fits")),logger.error(msg)

    logger.info(f"Preparing simulation for DM {args.process} in a cluster")

    cluster = DMHalo(
        args.srcname,
        args.ra*u.deg,
        args.dec*u.deg,
        args.srcz,
        args.m200*u.Msun,
        args.r200*u.kpc,
        dmsigmav=3.6e-24*u.cm**3/u.s,
        fsub=0.11,
        msub_min=1e-5*args.m200*u.Msun,
        msub_max=1e-2*args.m200*u.Msun,
        index_pm=-1.9,
        mpoints=5
    )

    cluster.info

    cluster_center = cluster.cartcoord.xyz
    obsRadius      = cluster.r200
    spacing        = 2*obsRadius/(args.ncells - 1)

    cx,cy,cz  = cluster_center.to(u.m).value
    c_center  = Vector3d(cx,cy,cz)

    e_emin   = args.emin*GeV
    e_emax   = args.emax*GeV
    pl_index = -2.0

    # Next is just to configure the dedicated
    # magnetic field modules

    # Magnetic field

    logger.info("Processing Magnetic Field grid for cluster")
    # Now, we can just create our magnetic field c:
    # with turbulence and modulation c:
    Bfield = get_cluster_field(
        cluster_center,
        obsRadius,
        args.core_radius*u.kpc,
        args.brms*u.uG,
        args.lmin*u.kpc,
        args.lmax*u.kpc,
        args.eta_index,
        args.ncells,
    )

    # parameters used for field line tracking
    precision = 1e-4
    minStep   = 10*pc
    maxStep   = 1*kpc

    # The following lines are used to describe the diffusion coefficient
    # We still need to check more literature about this.
    # ratio between parallel and perpendicular diffusion coefficient

    # Difffusion Module
    # D_xx=D_yy= 1e23 m^2 / s, D_zz=10*D_xx
    # The normalization is adjusted and the energy dependence 
    # is deactivated (setting power law index alpha=0)
    # SDE refers to the nnumerical method used to solve
    # the diffusion equations: Stochastic diferential equations
    # Now, we can start increasing the complexity of the
    # diffusion modeling by consideting the spectral dependence
    # with a power law, and also scaling the Diffusion coefficient
    # to the corresponding energies.
    # There aren't too much references about the actusl value
    # of the diffusion coefficient.
    # We assume that the diffusion coefficient at energies of 10 PeV
    # is around ~1e26 m**2/s (https://arxiv.org/abs/1811.03062v1)
    # The user need to estimate the scale factor to get the correct
    # diffusion coefficient at the energies that will be simulated
    # Additionally, I think, for the initial tests, it is safe to assume
    # that the diffusion coefficient is uniform across the ICM
    # But I will let epsilon and alpha as input parameters

    logger.info("Preparing Diffusion module")
    Dif = DiffusionSDE(Bfield,precision,minStep,maxStep,args.diff_epsilon)
    Dif.setScale(args.diff_scale)
    Dif.setAlpha(args.diff_alpha)

    # Now, we define the boundary. For this test, we consider
    # an spherical boundary centered in the galaxy cluster.
    # We select a boundary radius greater than the 
    # radius of the spherical observer
    # By default, the boundary center, the cluster center 
    # and the observer center are the same
    boundaryCenter  = c_center
    boundaryRadius  = 1.1*obsRadius.to(u.m).value
    clusterBoundary = SphericalBoundary(boundaryCenter,boundaryRadius)

    # This is to specify that the simulation ends after some time
    # in our case, just at the outskirts of the cluster
    # Boundary
    # Simulation ends after t=maxTra/c
    maxTra = MaximumTrajectoryLength(5.0*Mpc)

    # and this is for the particle content in the simulation
    # For now, only electrons
    # But, if we would like to include positrons
    # we need to call a method for multiple particle types,
    # and pass -11, to consider positrons
    part_type = 11

    # Now, we are able to process if we want to simulate 
    # a pointlike or a extended source for injection of particles 
    # obtained from DM annihilation/decay
    # We can test for each case and prepare the DM source
    # used for injection of particles.
    # For now, we only inject one type of particle
    # ToDo: Add several types of particles
    # For the case of the pointlike source, we assume
    # that the center of the cluster, the center of the DM halo, 
    # and the position of the pointlike source are the same

    # So, to describe the source we consider:
    #   - Source distribution following the Smooth DM Halo Mass distro.
    #   - Redshift (for temporal evolution, afueritas of the cluster)
    #   - Emission type (isotropic in our case)
    #   - Particle type (Particles injected)
    #   - Particle energy (following a PL)

    logger.info("Preparing dark matter source")
    if args.dmsource_type.lower() == "pointlike":

        logger.info("You choose a point-like source for this simulation")
        # Now, we define parameters associated with the source of DM
        # as position, and redshift
        srcpos = c_center

        s = preparePointLikeDMSource(
            srcpos,
            cluster.z,
            part_type,
            e_emin,
            e_emax,
            pl_index,
        )

    if args.dmsource_type.lower() == "extended_smooth":

        logger.info("You choose an extended source for this simulation")
        logger.info("Considering the smooth contribution of the DM halo ")

        if args.subhalos and args.process.lower() == "anna":

            rho_trunc = cluster.rho_smooth(spacing)

        else:

            rho_trunc = cluster.rho_tot(spacing)

        msg = (
            "To avoid numerical precision issues we do not use the "
            "saturation density during grid calculation. That implies "
            "larger computation times trying to sample regions where "
            "the probability is too low (1e-9). We will use the density "
            f"evaluated at the size of the cells: {rho_trunc:0.5e}"
        )
        logger.warning(msg)
        # First we need to prepare the DMDensity Grid 
        # then, we can define the source to inject the particles.
        # By default, we are assuming that each sampled source inject 
        # particles isotropically.

        rmin = cluster_center - obsRadius
        rmax = cluster_center + obsRadius

        if args.process.lower() == "decay":

            dmgrid = SmoothDMHaloMassDensityGrid(
                cluster_center,
                obsRadius,
                args.ncells,
                cluster,
                spacing,
                rho_trunc,
                args.chunksize,
            )

            maxdens = rho_trunc/cluster.l_dm_decay

        if args.process.lower() == "anna":

            dmgrid = SmoothDMHaloMassSquaredDensityGrid(
                cluster_center,
                obsRadius,
                args.ncells,
                cluster,
                spacing,
                rho_trunc,
                args.chunksize,
                subhalos=args.subhalos
            )

            maxdens = rho_trunc**2/cluster.l_dm_anna

        if args.subhalos and args.process.lower() == "anna":

            l_sub   = np.sum(cluster.shpop.to_qtable()["lanna"])
            l_cross = np.sum(cluster.shpop.to_qtable()["cross"])
            l_tot   = cluster.l_dm_anna_sub

            w_smooth  = (l_tot - l_sub - l_cross) / l_tot
            w_subhalo = (l_sub + l_cross) / l_tot

            msg = (
                "\nUsing smooth and subhalo contributions with weights: \n"
                f"\t- Smooth: {w_smooth:0.5f} \n"
                f"\t- Subhalo: {w_subhalo:0.5f}\n"
            )

            logger.info(msg)

            s = prepareDMHaloSource(
                dmgrid,
                cluster.shpop,
                maxdens.value,
                cluster.l_dm_anna_sub.value,
                int(args.maxtries),
                rmin.to(u.m).value,
                rmax.to(u.m).value,
                cluster.z,
                part_type,
                e_emin,
                e_emax,
                pl_index,
                w_smooth.value,
                w_subhalo.value
            )

        else:

            s = prepareSmoothExtendedDMSource(
                dmgrid,
                maxdens.value,
                int(args.maxtries),
                rmin.to(u.m).value,
                rmax.to(u.m).value,
                cluster.z,
                part_type,
                e_emin,
                e_emax,
                pl_index,
            )

    logger.info(f"Maximum of the PDF: {maxdens:0.5e}")
    logger.info(s.getDescription())

    # Here we prepare the CRpropa file output 
    # with some default parameters.

    logger.info("Preparing CRpropa txtOutput to save data")
    if args.ofname.lower().endswith("fits"):

        fname = outpath/args.ofname.replace("fits","txt")

    if args.ofname.lower().endswith("fits.gz"):

        fname = outpath/args.ofname.replace("fits.gz","txt")

    Out = prepareOutput(fname)

    # Now, we configure the observer
    # For this test, we will consider an observer on a sphere
    
    logger.info("Preparing Photon Observer")

    # N is the total number of particles in the simulation
    # n is the number of steps
    # step refers to the size. In this case, size 5 kpc
    N    = args.nparticles
    n    = 1000
    step = 5*Mpc/n 
    # Radio de larmor dados el valor mínimo de energía de los electrones 
    # y la intensidad máxima del campo magnético

    # Now, here comes the parameters to define the sphere
    # By default, we define that the center for the observer,
    # the DM halo, and the galaxy cluster, are the same
    # obsCenter = c_center*Mpc

    # And this is to create an instance of the observer class
    obs = preparePhotonObserver(
        c_center,obsRadius.to(u.m).value,step,n,Out
    )

    logger.info(f"Observer centered at {c_center} ({cluster_center})")

    # Finally, the inverse compton module
    # And here, we are only considering the CMB as seed photon field
    # 0.5 referes to the "screening" (I don't remember the correct term now)
    # But it is required to weight the output of photons
    # because the total number of photons is beyond the computational limit
    # Se puede agregar un parámetro de limit en los módulos de interacción
    # para indicar que sí en un determinado paso la interacción no ocurrió
    # en el siguiente paso, la meanfreepath para esa interacción
    # se hace más pequeño
    cmb   = CMB()
    iccmb = EMInverseComptonScattering(cmb,True,0.5)
    # iccmb = EMInverseComptonScattering(cmb,True,0.5,limit)

    logger.info("Adding Different modules to the simulation")
    # module list
    # Add modules and run the simulation
    sim = ModuleList()

    sim.add(Dif)
    sim.add(obs)
    sim.add(clusterBoundary)
    sim.add(maxTra)
    sim.add(iccmb)

    sim.setShowProgress(True)
    sim.run(s,N,True)
    Out.close()


    logger.info("Saving data to fits table")
    create_table(fname,outpath/args.ofname)

    # Plotting and post-processing is done in another script

    msg = 'Total Elapsed time: '
    elapsed_time(this_start,msg)



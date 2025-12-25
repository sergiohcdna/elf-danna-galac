##########################################################
# Used to generate projection maps:                      #
#       - Counts                                         #
#                                                        #
#--------------------------------------------------------#
#           THE ELF-DANNA-GALAC Task force               #
#               - Arlette Melo Galindo                   #
#               - Miguel A. Sánchez Conde                #
#               - Sergio Hernández Cadena                #
#--------------------------------------------------------#
#             December-2025                              #
##########################################################

import astropy.units as u
import matplotlib.pyplot as plt

from astropy.coordinates import CartesianRepresentation,SkyCoord
from matplotlib.colors import LogNorm

from ..projection.roiprojection import (
    get_particle_coordinates,
    get_mask_fov,
    get_proj_image
)

import argparse as ap
import time

from loguru import logger
from pathlib import Path

from ..tools.misc import checkDir,elapsed_time

def get_counts_map():

    this_start = time.time()

    msg     = ('Galaxy clusters project: elf-danna-galac \n'
               'December/2025')
    options = ap.ArgumentParser(description=msg)

    src = options.add_argument_group('Galaxy Cluster','Input params')

    src.add_argument(
        "--srcname",
        help="Name of the cluster [to save files]",
        type=str,
        required=True,
        metavar="Virgo Toy"
    )
    src.add_argument(
        "--srcz",
        help="Redshift of the cluster",
        type=float,
        required=True,
        metavar="0.0043"
    )
    src.add_argument(
        "--srcpos",
        help="Cartesian Position of the center of the cluster [Mpc]",
        nargs=3,
        required=True,
        type=float,
        metavar="(16,0,0) Mpc"
    )
    src.add_argument(
        "--srcdistance",
        help="Distance to the cluster [Mpc]",
        type=float,
        required=True,
        metavar="16.0 Mpc"
    )
    src.add_argument(
        "--fov",
        help="FOV Angular size [deg]",
        type=float,
        required=True,
        metavar="8.0 deg"
    )
    src.add_argument(
        "--pixsize",
        help="Angular size of pixels used for binning [deg]",
        type=float,
        required=False,
        metavar="0.1 deg",
        default=0.1
    )
    src.add_argument(
        "--rfile",
        help="Path to fits with results from CRpropa simulation",
        type=str,
        required=True,
        metavar="path/to/results.fits.gz"
    )
    src.add_argument(
        "--frame",
        help="Name of frame used for SkyCoord representation",
        type=str,
        required=False,
        metavar="icrs",
        default="icrs"
    )
    src.add_argument(
        '--odir',
        help='Output directory to save files',
        type=str,
        required=False,
        default='./',
        metavar='./'
    )

    logger.info("Getting parameters")
    args = options.parse_args()

    outpath = Path(args.odir)
    checkDir(outpath)

    logger.info("Getting source position")
    cx,cy,cz = args.srcpos
    src_pos  = CartesianRepresentation(
        x=cx,
        y=cy,
        z=cz,
        unit=u.Mpc
    )
    src_coord = SkyCoord(src_pos,frame=args.frame)

    logger.info("Getting particle positions and directions")
    coords = get_particle_coordinates(args.rfile)

    ang_size = args.fov*u.deg
    pix_size = args.pixsize*u.deg

    logger.info("Getting FOV mask")
    fov_mask = get_mask_fov(
        coords["ppos_car"],
        coords["dir_car"],
        src_pos,
        ang_size
    )

    logger.info("Getting image data")
    imagen = get_proj_image(
        src_coord,
        coords["ppos_sky"],
        ang_size,
        pix_size,
        fov_mask,
        coords["weights"]
    )

    logger.info("Plotting")

    _,ax = plt.subplots(figsize=(7,7))

    im = ax.imshow(
        imagen["data"],
        origin='lower',
        extent=[
            imagen["ra_e"][0].deg,
            imagen["ra_e"][-1].deg,
            imagen["dec_e"][0].deg,
            imagen["dec_e"][-1].deg,
        ],
        cmap="magma",
        aspect="equal",
        norm=LogNorm(),
    )

    plt.colorbar(im,ax=ax,label="Counts")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")


    # # Add source direction marker
    ax.plot(
        src_coord.ra.wrap_at(180*u.deg).deg,
        src_coord.dec.deg,
        'k+',
        markersize=15,
        markeredgewidth=2,
        label=args.srcname
    )
    ax.legend()

    ofname = args.srcname.replace(" ","")
    ofname = outpath/f"{ofname}CountsMap.png"
    plt.savefig(ofname,dpi=400)

    msg = "Total Elapsed time: "
    elapsed_time(this_start,msg)


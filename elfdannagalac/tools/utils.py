###############################################################################
# Diffusion of electrons  in the intraclluster medium of galaxy clusters      #
#   - Misc. functions used to convert output from CRpropa to fits table       #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             December-2025                                                   #
###############################################################################

import astropy.units as u
import numpy as np

from astropy.table import QTable

from crpropa import Output,TextOutput
from crpropa import Mpc,GeV

from pathlib import Path

from ..io.parquetout import AsyncDynamicParquetOutput

# #	D	time	z	SN	ID	E	X	Y	Z	Px	Py	Pz	SN0	ID0	E0	X0	Y0	Z0	P0x	P0y	P0z	SN1	ID1	E1	X1	Y1	Z1	P1x	P1y	P1z	W	tag

def create_table(
    ifname    : Path,
    ofname    : Path,
    rm_ifname : bool=True
):
    """
    Convert output from txt file to a more convenient QTable

    :param ifname: Path to input File obtained from CRpropa simulation
    :type ifname: Path
    :param ofname: Path to save data in a fits/fits.gz file
    :type ofname: Path
    :param rm_ifname: Delate original input file. Default is True
    :type rm_ifname: bool

    By default we use the extended version of the output from CRpropa.
    (Yes, We are saving everything)
    """

    with open(ifname,"r") as ff:

        names = ff.readline().strip("\n")

    names = names.split()[1:]

    dtypes = [
        float,float,float,
        int,int,float,float,float,float,float,float,float,
        int,int,float,float,float,float,float,float,float,
        int,int,float,float,float,float,float,float,float,
        float,"U10",
    ]

    units = [
        u.Mpc,u.Myr,None,
        None,None,u.GeV,u.Mpc,u.Mpc,u.Mpc,None,None,None,
        None,None,u.GeV,u.Mpc,u.Mpc,u.Mpc,None,None,None,
        None,None,u.GeV,u.Mpc,u.Mpc,u.Mpc,None,None,None,
        None,None
    ]

    ph_evs = np.genfromtxt(
        ifname,
        delimiter="\t",
        # skip_header=26,
        comments="#",
        dtype=dtypes,
        names=names
    )

    out_table = QTable(
        data=ph_evs,
        names=names,
        dtype=dtypes,
        units=units,
    )

    out_table.write(
        ofname,
        # format="fits",
        overwrite=True
    )

    if rm_ifname:

        ifname.unlink()

    return

def prepareOutput(ofname:Path) -> TextOutput:

    """
    Prepare TextOutput for CRpropa simulation. 
    By default, we save everything. 
    Also, we enable the weightining by default. 
    For particle types as electrons, the weight should 
    be zero, and then can be discarded during the post-processing.
    Also, Distances and Energies are given in Mpc and GeV.

    I know, the function is too short, but Iin the case, 
    we want to check the spectrum of electrons/positrons at 
    different radii, then function comes in handy, c:
    
    :param ofname: Path to the output txt file
    :type ofname: Path
    :return: Instance of TextOutput
    :rtype: TextOutput
    """

    thisout = TextOutput(str(ofname),Output.Everything)

    thisout.enable(Output.WeightColumn)
    thisout.setLengthScale(Mpc)
    thisout.setEnergyScale(GeV)

    return thisout

def prepareParquetOutput(ofname:Path):

    Out = AsyncDynamicParquetOutput(
        ofname,
        # fields=Output.Everything,
        energy_unit   = GeV,
        batch_size    = 50000,
        queue_maxsize = 1000
    )
    Out.enable(Output.WeightColumn)
    Out.enable(Output.CandidateTagColumn)

    return Out

###############################################################################
# Scienca Case 1: x                                                           #
#   - Get electron weights from the DM spectrum as an input for CRpropa       #
#-----------------------------------------------------------------------------#
#                      THE ELF-DANNA-GALAC Task force                         #
#                      - Arlette Melo Galindo                                 #
#                      - Miguel A. Sánchez Conde                              #
#                      - Sergio Hernández Cadena                              #
#-----------------------------------------------------------------------------#
#             September-2024                                                  #
###############################################################################

from elfdannagalac.dmspectrum import dmspectra

import argparse as ap
from elfdannagalac.tools.misc import elapsed_time

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
    src.add_argument('--channel',help='annihilation/decay channel',type=str,
                     required=False,metavar='tau',default='tau')
    src.add_argument('--process',help='Annihilation (anna) or decay (decay)',
                     type=str,required=False,default='anna',metavar='anna')
    src.add_argument('--project',help='Data project to get the spectrum [cosmixs,pppc4dmid]',
                     type=str,required=False,default='cosmixs',metavar='cosmixs')
    src.add_argument('--nbins',help='Number of bins for electron spectrum, used for CRpropa',
                     type=int,required=False,default=10,metavar='10')

    args = options.parse_args()

    # This is to get the spectrum using dmspctra class
    dme_spec = dmspectra.dmspectrum(
        args.dmmass,
        args.emin,
        args.emax,
        args.channel,
        project=args.project,
        process=args.process
    )

    # From this, we use the nbins argument to compute the
    # weights, meaning the fraction of particles used
    # during the CRpropa simulation
    dme_spec.nbins = args.nbins

    ebin_edges = dme_spec.ebins
    eweights   = dme_spec.weights

    # For now, using print, until we add the part of logging
    print('CRpropa electron-weights')

    for idx in range(args.nbins):

        edgel  = ebin_edges[idx]
        edgeu  = ebin_edges[idx+1]
        weight = eweights[idx]
        print(f'\t{edgel:03e} -- {edgeu:03e} --> {weight:03e}')

    msg = 'Total Elapsed time: '
    elapsed_time(this_start,msg)

##########################################################
# sky_projection taken from threeML, HAL and astromodels #
# Adapted to our special case of Simulation Processing   #
# And a more compact version                             #
# Eliminate innecessary dependances                      #
# as we don't care about astromodels                     #
# We want the minimal version                            #
#                                                        #
# However, for now, we don't use the sky_projection      #
# as we are not converting.transforming to pixel coords  #
# and save the image into a healpix map (TODO)           #
#                                                        #
#--------------------------------------------------------#
#           THE ELF-DANNA-GALAC Task force               #
#               - Arlette Melo Galindo                   #
#               - Miguel A. Sánchez Conde                #
#               - Sergio Hernández Cadena                #
#--------------------------------------------------------#
#             November-2025                              #
##########################################################

from __future__ import absolute_import, division

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import CartesianRepresentation,SkyCoord,ICRS
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area,celestial_frame_to_wcs

from matplotlib.colors import LogNorm

from collections import OrderedDict
from loguru import logger
from pathlib import Path


_EQUATORIAL = "equatorial"
_GALACTIC = "galactic"

_RING = "RING"
_NESTED = "NESTED"

_fits_header = """
NAXIS   =                    2
NAXIS1  =                   %i
NAXIS2  =                   %i
CTYPE1  = 'RA---TAN'
CRPIX1  =                   %i
CRVAL1  =                   %s
CDELT1  =                  -%f
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--TAN'
CRPIX2  =                   %i
CRVAL2  =                   %s
CDELT2  =                   %f
CUNIT2  = 'deg     '
COORDSYS= '%s'
"""


def _get_header(
    ra         : float,
    dec        : float,
    pixel_size : float,
    coordsys   : str,
    h          : int,
    w          : int
) -> fits.Header:

    assert 0 <= ra <= 360

    header = fits.Header.fromstring(
        _fits_header % (
            h,w,h / 2,ra,pixel_size,w / 2,dec,pixel_size,coordsys
        ),
        sep="\n",
    )

    return header

def _get_all_ra_dec(
    input_wcs: WCS,
    h: int,
    w: int
) -> tuple[np.ndarray, ...]:
    # An array of all the possible permutation of (i,j) for i=0..999 and j=0..999

    xx = np.arange(0.5,h + 0.5,1,dtype=np.int16)
    yy = np.arange(0.5,w + 0.5,1,dtype=np.int16)

    _ij_grid = cartesian((xx,yy))

    # Convert pixel coordinates to world coordinates
    world = input_wcs.all_pix2world(_ij_grid,0,ra_dec_order=True)

    return world[:,0],world[:,1]

# def radec_to_vec(ra: np.ndarray | float, dec: np.ndarray | float) -> np.ndarray:
#     assert 0 <= ra <= 360
#     assert -90 <= dec <= 90

#     # Healpix uses the convention -180 - 180 for longitude, instead
#     # we get RA between 0 and 360, so we need to wrap
#     wrap_angle = 180.0

#     lon = np.mod(ra - wrap_angle, 360.0) - (360.0 - wrap_angle)

#     vec = hp.dir2vec(lon, dec, lonlat=True)

#     return vec


def cartesian(
    arrays : np.ndarray | tuple | list
) -> tuple[np.ndarray]:
    return np.dstack(
        np.meshgrid(*arrays,indexing="ij")).reshape(-1,len(arrays)
    )


def _get_radians(my_angle : float | u.Quantity) -> float:

    if isinstance(my_angle,u.Quantity):
        my_angle_radians = my_angle.to(u.rad).value

    else:
        my_angle_radians = np.deg2rad(my_angle)

    return my_angle_radians

def get_particle_coordinates(
    rfile : Path,
    frame : str="icrs",
) -> OrderedDict:

    """
    Read CRpropa fits file from simulation output 
    and get the positions and directions vectors 
    for all detected particles. Positions are given 
    in the frame "frame"
    
    :param rfile: Path to fits file with results
    :type rfile: Path
    :param frame: Name of the frame used for SkyCoord and 
                  Cartesian Representation
    :type frame: str

    :return: Dictionary with positions (Sky and Car), directions
             for all particles given in rfile
    :rtype: OrderedDict
    """

    coords = OrderedDict()

    with fits.open(rfile) as hdul:
        data = hdul[1].data
        # cols = hdul[1].columns

    # We assume that all positions are
    # in the same units
    positions = CartesianRepresentation(
        x=data.field("X"),
        y=data.field("Y"),
        z=data.field("Z"),
        unit=u.Unit(data.columns["X"].unit)
    )

    directions = CartesianRepresentation(
        x=data.field("Px"),
        y=data.field("Py"),
        z=data.field("Pz"),
        unit=u.dimensionless_unscaled
    )

    coords["ppos_car"] = positions
    coords["dir_car"]  = directions
    coords["ppos_sky"] = SkyCoord(positions,frame=frame)
    coords["weights"]  = data.field("W")

    return coords

def get_mask_fov(
    positions  : CartesianRepresentation,
    directions : CartesianRepresentation,
    src_pos    : CartesianRepresentation,
    fov_deg    : u.Quantity,
) -> np.ndarray[np.bool_]:

    """
    Obtain the mask used to indicate if 
    any particle is moving towards the  observer
    and inside its field of view (fov). 
    First, we use the line-of-sight (los) directions 
    to indicate which particles are moving to the 
    observer. Then, using the direction to the 
    source, we construct the image plane, and 
    estimate the angle from the los, then compare 
    with the observer's fov.
    
    :param positions: Cartesian Positions of detected particles [Mpc]
    :type positions: CartesianRepresentation
    :param directions: Cartesian Directions of detected particles [-]
    :type directions: CartesianRepresentation
    :param src_pos: Cartesian position of the source
    :type src_pos: CartesianRepresentation
    :param fov_deg: Angular size of field of view [deg]
    :type fov_deg: u.Quantity

    :return: Mask
    :rtype: ndarray[bool_, Any]
    """

    # The first part is to get the particles
    # traveling towards the observer, or
    # just the line of sight direction
    # Notice: all the vectors are given
    # in the observer coordinate system
    los_direction = positions/positions.norm()

    # then we compute cos(angle) to decide if a
    # particle is moving towards the observer
    cos_angle   = los_direction.dot(directions)
    toward_mask = cos_angle < 0

    apparent_directions               = los_direction.copy()
    apparent_directions[toward_mask]  = apparent_directions[toward_mask]
    apparent_directions[~toward_mask] = apparent_directions[~toward_mask]

    # Then, we compute the unitary direction vector (z-axis)
    # of the source and construct the projection plane
    z_axis  = src_pos / src_pos.norm()

    if np.abs(z_axis.x) < 0.9:
        x_axis = z_axis.cross(
            CartesianRepresentation(
                x=1.0,
                y=0.0,
                z=0.0,
                unit=u.dimensionless_unscaled
            )
        )
    else:
        x_axis = z_axis.cross(
            CartesianRepresentation(
                x=0.0,
                y=1.0,
                z=0.0,
                unit=u.dimensionless_unscaled
            )
        )

    x_axis = x_axis / x_axis.norm()

    # y-axis completes right-handed system
    y_axis = z_axis.cross(x_axis)
    y_axis = y_axis / y_axis.norm()
  
    # Now, we get the apparent componets of the
    # particle position vectors
    # apparent_x = apparent_directions.dot(x_axis)
    # apparent_y = apparent_directions.dot(y_axis)
    apparent_z = apparent_directions.dot(z_axis)

    # We can estimate the number of particles failing in 
    # the acceptance cone of the observer (fov)
    # This will give us the other part of the mask
    theta  = np.arccos(np.clip(apparent_z,-1.0,1.0)).to(u.deg)
    in_fov = theta <= fov_deg

    final_mask = toward_mask & in_fov

    return final_mask

def get_proj_image(
    src_coord       : SkyCoord,
    particle_coords : SkyCoord,
    fov_deg         : u.Quantity,
    pixel_size      : u.Quantity,
    fov_mask        : np.ndarray[np.bool_],
    weights         : np.ndarray
) -> OrderedDict:

    """
    Obtain projected image from a 2D histogram 
    in RA and DEC of all detected particles.
    
    :param src_coord: SkyCoord of the source
    :type src_coord: SkyCoord
    :param particle_coords: SkyCoord of detected particles
    :type particle_coords: SkyCoord
    :param fov_deg: Angular size of the observer's field of view [deg]
    :type fov_deg: u.Quantity
    :param pixel_size: Angular size of pixels used for binning [deg]
    :type pixel_size: u.Quantity
    :param fov_mask: Mask to indicate incoming particles
    :type fov_mask: np.ndarray[np.bool_]
    :param weights: Weights from CRpropa due to thining
    :type weights: np.ndarray

    :return: Dictionary with binned data and edge arrays
             for RA and DEC
    :rtype: OrderedDict
    """

    imagen = OrderedDict()

    pixel_scale = pixel_size.to(u.rad)
    n_pixels    = int(2 * fov_deg.to(u.rad) / pixel_scale)

    hist_range = [
        [
            src_coord.ra.wrap_at(180*u.deg) - fov_deg,
            src_coord.ra.wrap_at(180*u.deg) + fov_deg
        ],
        [
            src_coord.dec - fov_deg,
            src_coord.dec + fov_deg
        ]
    ]

    hist,ra_edges,dec_edges = np.histogram2d(
        particle_coords.ra.wrap_at(180*u.deg)[fov_mask],
        particle_coords.dec[fov_mask],
        bins=n_pixels,
        range=hist_range,
        weights=weights[fov_mask]
    )

    imagen["data"]  = hist.T
    imagen["ra_e"]  = ra_edges
    imagen["dec_e"] = dec_edges

    return imagen

class sky_projection:
    def __init__(
        self,
        ra_center: float,
        dec_center: float,
        pixel_size_deg: float,
        npix_height: int,
        npix_width: int,
    ) -> None:
        assert npix_height % 2 == 0, "Number of height pixels must be even"
        assert npix_width % 2 == 0, "Number of width pixels must be even"

        if isinstance(npix_height, float):
            assert npix_height.is_integer(), "This is a bug"

        if isinstance(npix_width, float):
            assert npix_width.is_integer(), "This is a bug"

        self._npix_height = int(npix_height)
        self._npix_width = int(npix_width)

        assert 0 <= ra_center <= 360.0, "RA must be between 0 and 360"
        assert -90.0 <= dec_center <= 90.0, "Dec must be between -90 and 90"

        self._ra_center = float(ra_center)
        self._dec_center = float(dec_center)

        self._pixel_size_deg = float(pixel_size_deg)

        # Build projection, i.e., a World Coordinate System object

        self._wcs = WCS(
            _get_header(
                ra_center,
                dec_center,
                pixel_size_deg,
                "icrs",
                npix_height,
                npix_width,
            )
        )

        # Pre-compute all R.A., Decs
        self._ras, self._decs = _get_all_ra_dec(
            self._wcs,
            npix_height,
            npix_width
        )

        # Make sure we have the right amount of coordinates
        assert self._ras.shape[0] == self._decs.shape[0]
        assert self._ras.shape[0] == npix_width * npix_height

        # Pre-compute pixel area
        self._pixel_area = proj_plane_pixel_area(self._wcs)

        # Cache for angular distances from a point
        # (see get_spherical_distances_from)
        self._distance_cache = {}

    @property
    def ras(self) -> np.ndarray:
        """
        :return: Right Ascension for all pixels
        """
        return self._ras

    @property
    def decs(self) -> np.ndarray:
        """
        :return: Declination for all pixels
        """
        return self._decs

    @property
    def ra_center(self) -> float:
        """
        :return: R.A. for the center of the projection
        """
        return self._ra_center

    @property
    def dec_center(self) -> float:
        """
        :return: Declination for the center of the projection
        """
        return self._dec_center

    @property
    def pixel_size(self) -> float:
        """
        :return: size (in deg) of the pixel
        """
        return self._pixel_size_deg

    @property
    def wcs(self) -> WCS:
        """
        :return: WCS instance describing the projection
        """
        return self._wcs

    @property
    def npix_height(self) -> int:
        """
        :return: height of the projection in pixels
        """
        return self._npix_height

    @property
    def npix_width(self) -> int:
        """
        :return: width of the projection in pixels
        """
        return self._npix_width

    @property
    def project_plane_pixel_area(self) -> float:
        """
        :return: area of the pixels
        """
        return self._pixel_area


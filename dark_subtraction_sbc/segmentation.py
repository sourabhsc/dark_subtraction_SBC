import os
import glob
from astropy.io import fits
import matplotlib.pylab as pylab
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import configparser
from configparser import ExtendedInterpolation
from matplotlib.patches import Circle
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
from subprocess import call


def smooth_gal(file_name, gal_key, smooth_scale, params):
    hdulist = fits.open(file_name)
    data = gaussian_filter(hdulist[1].data, float(smooth_scale))
    DQ = hdulist[3].data
    hot_pix_x = np.array(params['hot_pix_x'].split(','))
    hot_pix_y = np.array(params['hot_pix_y'].split(','))

    DQ[int(hot_pix_x[0]), int(hot_pix_y[0])] = float(params['dq'])
    DQ[int(hot_pix_x[1]), int(hot_pix_y[1])] = float(params['dq'])
    file_smooth = file_name.replace("flt.fits", "flt_smooth.fits")
    data[DQ != 0] = float(params['dq_bad'])
    hdu = fits.PrimaryHDU(data=data)

    header = hdu.header
    header.add_history("smooth galaxy image to be used to create sextrator segmentation maps for %s" % (file_name))
    header['ROOTRAW'] = (gal_key, 'keyword for raw images')
    header['SMTHSCL'] = (smooth_scale, 'smoothing scale for gausssian filter')
    header['CREATOR'] = ('SSC', 'FEB 24 2018')

    hdu.writeto(file_smooth, overwrite=True)

    return file_smooth


def sextractor_seg(file_smooth, sex_config, params):
    catalog = file_smooth.replace("smooth.fits", "catalog.cat")
    seg_map = file_smooth.replace("smooth.fits", "seg_map.fits")

    DETECT_THRESH = float(params['detect_thresh'])
    DETECT_MINAREA = float(params['detect_minarea'])
    ANALYSIS_THRESH = float(params['analysis_thresh'])

    cmd = "sextractor %s \
    -c %s \
    -CATALOG_NAME %s             \
    -CHECKIMAGE_TYPE SEGMENTATION  \
    -CHECKIMAGE_NAME %s\
    -DETECT_MINAREA   %s\
    -DETECT_THRESH    %s\
    -ANALYSIS_THRESH  %s"\
    % (file_smooth, sex_config, catalog, seg_map, DETECT_MINAREA, DETECT_THRESH, ANALYSIS_THRESH)
    print (cmd)
    call(cmd, shell=True)
    return seg_map


def galaxy_mask(file_smooth, gal_key, smooth_scale, seg_map, sex_config):
    hdu = fits.open(file_smooth)
    data = hdu[0].data
    seg = fits.open(seg_map)
    masks = np.where(seg[0].data == 0)
    data[masks] = 0.0
    ###### removing additional artifacts from corners etc.
    #file_mask = file_smooth.replace("smooth", "mask1")
    '''
    for j in range(1024):
        for k in range(1024):
            if k < 100 or j < 100:
                data[j][k] = 0.0
            if gal_num != 3 and gal_num != 4:
                if j > 650:
                    data[j][k] = 0.0

    #4) Also replace all pixels with j, k such that k<100 or j<100 set, data[j][k]==0.0 \
    #5) ### j, k are opposite to as it appears on ds9 window\
    !!!!!!!!!!! this functionality has been removed for now as it requires too much customization 
    '''

    file_mask = file_smooth.replace("smooth", "mask")
    header = hdu[0].header
    header['COMMENT'] = (" Image creation steps :- 1) smooth FLT image using gaussian filter of a smoothing scale %s,\
     2) use segmentation maps = %s, created by sextrator using config file = %s \
     3) Replace flux values in smooth FLT images at all the pixels with\
      zeros in segmentaion map with zero\
     4) output = %s" % (smooth_scale, seg_map, sex_config, file_mask))

    header['SEXCFG'] = (sex_config, 'sextractor config file')
    hdu.writeto(file_mask, overwrite=True)

    return file_mask

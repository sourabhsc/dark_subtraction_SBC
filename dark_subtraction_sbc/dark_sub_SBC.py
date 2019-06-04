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
from subprocess import call
from astropy.table import Table

import shutil
from segmentation import *
plt.rcdefaults()
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
          'axes.labelsize': 'xx-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'xx-large',
          'ytick.labelsize': 'xx-large'}
pylab.rcParams.update(params)


def directory_check(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#### for plotting circles ##########


def circle(x, y, rad, col_circ1, ax4):
    circle = Circle((x, y),
                    rad, clip_on=False, linewidth=0.5,
                    edgecolor=col_circ1, facecolor=(0, 0, 0, .0125))
    ax4.add_artist(circle)

############## creating masks with a given center pixel and width of annuli#######


def masks_circular(cent_x, cent_y, width, aper_lim, nx, ny):
    rad1 = np.arange(1., aper_lim, width)
    y, x = np.mgrid[0:ny, 0:nx]
    masks_annulus = [np.where(((x - cent_x)**2 + (y - cent_y)**2 >= rad1[k]**2)
                              & ((x - cent_x)**2 + (y - cent_y)**2 <= rad1[k + 1]**2))
                     for k in range(len(rad1) - 1)]

    masks = [np.where((x - cent_x)**2 + (y - cent_y)**2 < rad1[k]**2) for k in range(len(rad1))]
    rad_annulus = ([(a + b) / 2 for a, b in zip(rad1, rad1[1:])])

    return rad1, rad_annulus, masks, masks_annulus


def gal_key_function(k):
    gal_str = []
    for letter in k:
        gal_str.append(letter)
    gal_key = gal_str[4] + gal_str[5] + gal_str[6] + gal_str[7] + gal_str[8]
    return gal_key

################# for input photometry ##############


def photometry(input_hot, input_cold, params, temp_hot, temp_cold, name):
    work_dir = params["work_dir"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    t = 0
    if input_cold != []:
        for f in input_cold:
            data_cold = fits.getdata(work_dir + f)
            aper_cold = [abs(np.sum(data_cold[masks[k]])) for k in range(len(rad1))]
            gal_key = gal_key_function(f)

            ax1.plot(rad1, aper_cold, color='b', alpha=(t + 1) / (len(input_cold)), label='cold T=%s, %s' % (temp_cold[t], gal_key))
            t = t + 1

            ax1.legend()
    t = 0
    if input_hot != []:
        for f in input_hot:
            data_hot = fits.getdata(work_dir + f)
            aper_hot = [abs(np.sum(data_hot[masks[k]])) for k in range(len(rad1))]
            gal_key = gal_key_function(f)

            ax1.plot(rad1, aper_hot, color='r', alpha=(t + 1) / (len(input_hot)), label='hot T=%s, %s' % (temp_hot[t], gal_key))
            t = t + 1
            ax1.legend()
    # dark_radii = float(params["mask_radii"])

    ax1.set_xlabel('pixels')
    ax1.set_ylabel('cumulative summation over circular apertures')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

    data = fits.getdata(work_dir + input_hot[0])
    show2 = ax2.imshow(data, vmin=0.0, vmax=0.06, origin='lower')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="8%", pad=0.0)
    cbar = plt.colorbar(show2, cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    ax2.set_title('dark frame %s' % (input_hot[0].split('/')[-1]), y=1.15)
    ax2.plot(cent_x, cent_y, '+', color='r', markersize=10)
    circle(cent_x, cent_y, dark_radii, 'r', ax2)
    plt.suptitle('Input photometry', fontsize=20)
    fig.savefig('%s%s_input_photometry.png' % (DIR_PNG, name), dvi=400)
    if show_plot == True:
        plt.show()


def dark_sort(dark_FLT):

    temp_FLT = np.zeros(len(dark_FLT))
    for i in range(len(dark_FLT)):
        fits_dark_FLT = fits.open(dark_FLT[i])
        temp_FLT[i] = (float(fits_dark_FLT[1].header["MDECODT1"]) +
                       float(fits_dark_FLT[1].header["MDECODT2"])) / 2.

    arg = np.argsort(temp_FLT)
    dark_FLT = list(dark_FLT[u] for u in arg)
    temp_FLT = list(temp_FLT[u] for u in arg)
    return dark_FLT, temp_FLT


def basic_params(configfile, section):

    config = configparser.ConfigParser(interpolation=ExtendedInterpolation())
    config.read(configfile)
    options = config.options(section)
    params = {}
    for option in options:
        params[option] = config.get(section, option)
    return params


############ function for dark subtraction |G(r)| method################

def function_dark_sub(a, k, aper_diff_gal, aper_diff_dark, dark_radii, galaxy, dark):

    aper_diff_ann = np.array(aper_diff_gal) - a * np.array(aper_diff_dark) - k
    s1 = np.sum(abs(aper_diff_ann[dark_radii:len(rad1) - 1]))

    return s1

################ function for chi square minimization ##############


def dark_sub_chisq(aper_diff_gal, aper_diff_dark, dark_radii, rad1):
    sci = np.array(aper_diff_gal[dark_radii:len(rad1) - 1])
    drk = np.array(aper_diff_dark[dark_radii:len(rad1) - 1])
    err = sp.ones_like(sci)
    # err = np.sqrt(sci)
    u = sp.sum(sci * drk / err)
    v = sp.sum(drk**2 / err)
    w = sp.sum(drk / err)
    x = sp.sum(sci / err)
    y = sp.sum(1. / err)

    A_best = (u / v - w * x / v / y) / (1. - w**2 / v / y)
    K_best = (x - A_best * w) / y

    mod_best = drk * A_best + K_best

    chi2_fun = ((sci - mod_best)**2 / (sci * (len(sci) - 2)))

    cond = np.isinf(chi2_fun)
    chi2_fun[cond] = 0.0
    chi2 = np.nansum(chi2_fun)

    return A_best, K_best, chi2


def verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, minimum_var, A, K, ind, label, DQ):
    print ("6) plotting the minimization values now for <<<<<%s >>>> method\n" % (label))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(temp_FLT, minimum_var, "o", markersize=8)
    ax.set_xlabel(r" Dark Temp[$^o$ C]")
    ax.set_ylabel("Minimizer %s [counts]" % (label))
    ax.axvline(x=temp_gal, color="g", label="galaxy temperature")
    ax.axvline(x=temp_FLT[ind], color="r", label="minimum value A= %.1e\n, K = %.1e,\n\
        index = %.1e\n" % (A, K, ind))
    ax.legend(loc='lower left')
    minimizer_png = file_name_no_dir.replace('.fits', '')
    minimizer_png = minimizer_png + "_minimizer_%s_dr_%s.png" % (label, dark_radii)
    ax.set_title('minimizer values %s method' % (label))
    fig.savefig("%s/INTERMEDIATE_PNG/%s" % (work_dir, minimizer_png), dvi=400, bbox_inches='tight')
    if show_plot == True:
        plt.show()

    print ("7) plotting the difference image now  for <<<<<%s>>>>> method..... with galaxy\n" % (label))

    hdulist = fits.open(file_name_flt)
    exp_time = hdulist[0].header["EXPTIME"]
    data_gal = hdulist[1].data

    hdu_dark_selected = fits.open(dark_FLT[ind])
    data_dark = (A * hdu_dark_selected[1].data * exp_time / float(params["exp_dark"]) + K) / exp_time  # exp time for darks is 1000 secs
    dark_selected = work_dir + file_name_no_dir.replace("flt", "dark_%s_%s_dr_%s" % (ind + 1, label, dark_radii))

    data_gal[DQ != 0] = 0.0
    data_dark[DQ != 0] = 0.0

    data_sub = data_gal - data_dark * exp_time
    data_sub[DQ != 0] = 0.0

    hdulist[1].data = data_sub

    hdu_dark_selected[1].data = data_dark

    hdu_dark_selected.writeto(dark_selected, overwrite=True)
    hdu_dark_selected.close()

    sub_name = file_name_flt.replace('flt', 'drk_flt_%s_dr_%s' % (label, dark_radii))

    hdulist.writeto(sub_name, overwrite=True, output_verify="ignore")
    hdulist.close()
    aper_before = [np.mean(data_gal[masks_annulus[k]]) for k in range(len(rad1) - 1)]
    aper_dark = [np.mean(data_dark[masks_annulus[k]] * exp_time) for k in range(len(rad1) - 1)]
    aper_subtracted = [np.mean(data_sub[masks_annulus[k]]) for k in range(len(rad1) - 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(rad_annulus, aper_before, color="orange", label="before")
    ax1.plot(rad_annulus, aper_dark, color="green", label="dark")
    ax1.plot(rad_annulus, aper_subtracted, color="blue", label="subtracted")
    ax1.set_xlabel("pixels")
    ax1.set_ylabel("annuli mean counts")
    ax1.set_title(" Removed Mask for exposure %s" % (file_name_no_dir), fontsize=16)
    ax1.axhline(y=0, color='k')
    y1 = 8.11e-6 * exp_time
    ax1.axhline(y=y1, linestyle='--', color="k", label="constant dark ISR y = %.1e" % (y1))
    ax1.legend()

    aper_before = [abs(np.sum(data_gal[masks[k]])) for k in range(len(rad1))]
    aper_subtracted = [abs(np.sum(data_sub[masks[k]])) for k in range(len(rad1))]
    ax2.plot(rad1, aper_before, color="orange", label="before")

    ax2.plot(rad1, aper_subtracted, color="blue", label="subtracted")
    ax2.set_xlabel("pixels")
    ax2.set_ylabel("cumulative counts")
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

    ax2.legend()

    verification_png = file_name_no_dir.replace(".fits", "")
    verification_png = verification_png + "_verification_%s_dr_%s.png" % (label, dark_radii)
    plt.suptitle('radial profiles %s method' % (label), fontsize=20)
    fig.savefig("%s/INTERMEDIATE_PNG/%s" % (work_dir, verification_png), dvi=400, bbox_inches='tight')
    if show_plot == True:
        plt.show()
    plt.close(fig)


def dark_sub(file_name_flt, dark_radii, dark_FLT, temp, params, rad1, rad_annulus, masks, masks_annulus, table_dark, chisq):
    file_name_no_dir = file_name_flt.split("/")
    file_name_no_dir = file_name_no_dir[-1]

    table_name_dark = '%sINTERMEDIATE_TXT/FLT_drk_%s_dr_%s.txt' %\
        (work_dir, params['name'], dark_radii)

    print ('Table for dark subtraction information\n', table_name_dark)

    hdulist = fits.open(file_name_flt)
    exp_time = hdulist[0].header["EXPTIME"]
    data = hdulist[1].data
    temp_gal = (float(hdulist[1].header["MDECODT1"]) + float(hdulist[1].header["MDECODT2"])) / 2.
    print ('\n\n\n')
    print ('6) Performing dark subtraction\n\n')

    A_lim = float(params["a_lim"])
    K_lim = float(params["k_lim"])
    del_A = float(params["del_a"])
    del_K = float(params["del_k"])

    A = np.arange(0, A_lim, del_A)
    K = np.arange(0, K_lim, del_K)
    minimum_var = np.zeros(len(dark_FLT))
    A_min = np.zeros(len(dark_FLT))
    K_min = np.zeros(len(dark_FLT))
    diff_ar = np.zeros((len(A), len(K)))

    if chisq == True:
        chi2 = np.zeros(len(dark_FLT))
        A_best = np.zeros(len(dark_FLT))
        K_best = np.zeros(len(dark_FLT))
    if dark_radii == 0:
        file_mask = file_name_flt.replace("flt.fits", "flt_mask.fits")
        fits_mask = fits.open(file_mask)
        data_mask = fits_mask[0].data
        mask = np.where(data_mask != 0)
        data[mask] = 0.0

        hdu = fits.PrimaryHDU(data=data)
        header = hdu.header
        header.add_history("masked image with the mask created by sextractor usng file %s" % (file_mask))
        hdu.writeto(file_name_flt.replace("flt", "flt_masked"), overwrite=True)

    for i in range(len(dark_FLT)):
        fits_dark_FLT = fits.open(dark_FLT[i])
        data_dark = fits_dark_FLT[1].data * exp_time / float(params["exp_dark"])  # exp time for darks is 1000 secs

        ### looking for DQ array from dark FLT files ###

        fits_dark_FLT = fits.open(dark_FLT[i])
        data_dark_DQ = fits_dark_FLT[3].data

        DQ = fits_dark_FLT[3].data
        DQ[int(hot_pix_x[0]), int(hot_pix_y[0])] = 1000
        DQ[int(hot_pix_x[1]), int(hot_pix_y[1])] = 1000
        # print ('DQ!=0 pixels  replaced by 0')
        # print ('hot pixels replaced by 1000')

        data_dark[DQ != 0] = 0.0
        data[DQ != 0] = 0.0
        if dark_radii == 0:
            data_dark[mask] = 0.0

        aper_diff_gal = [(np.mean(data[masks_annulus[k]])) for k in range(len(rad1) - 1)]

        print ("performing minimization  for dark = %s" % (i + 1))

        aper_diff_dark = [(np.mean(data_dark[masks_annulus[k]])) for k in range(len(rad1) - 1)]

        diff_ar = [function_dark_sub(a, k, aper_diff_gal, aper_diff_dark, dark_radii, data, data_dark)
                   for a in A for k in K]
        c1 = [(a, k) for a in A for k in K]

        c = (np.unravel_index(np.array(diff_ar).argmin(), np.array(diff_ar).shape))
        par = (np.array(c1)[c])
        scale_factor = par[0]
        sky = par[1]
        dark_final = scale_factor * data_dark + sky
        fits_dark_name = file_name_no_dir.replace(".fits", "dark_%s_%s_dr_%s.fits" % (i + 1, 'Gr', dark_radii))

        hdu = fits.PrimaryHDU(data=dark_final)
        header = hdu.header
        header["RAWNAME"] = (file_name_flt, "FLT file for dark subtraction")
        # header["MASK"] = (file_mask, "mask file used for galaxy")
        header["AMIN"] = (par[0], " A value that minimizes the spatial variation")
        header["KMIN"] = (par[1], "K value that minimizes the spatial variation")
        header["DARKFILE"] = (dark_FLT[i], "dark file")
        header["ALIM"] = A_lim
        header["KLIM"] = K_lim
        header["ADEL"] = del_A
        header["KDEL"] = del_K
        header["MINVAR"] = np.min(diff_ar)
        header.add_history(" dark file dark subtraction method using G_subtracted = Galaxy - A*Dark - K. THis file has (A*Dark+K) ")
        hdu.writeto("%sINTERMEDIATE_FITS/%s" % (work_dir, fits_dark_name), overwrite=True)

        print ("minimization done")
        c2 = np.reshape(diff_ar, (len(A), len(K)))

        fits_variance_name = file_name_no_dir.replace(".fits", "dark_%s_diff_%s_dr_%s.fits" % (i + 1, 'Gr', dark_radii))
        fits.writeto("%sINTERMEDIATE_FITS/%s" % (work_dir, fits_variance_name), data=c2, header=header, overwrite=True)

        minimum_var[i] = np.min(diff_ar)
        A_min[i] = par[0]  # A[c[0]]
        K_min[i] = par[1]  # K[c[1]]
        if chisq == True:

            A_best[i], K_best[i], chi2[i] = dark_sub_chisq(aper_diff_gal, aper_diff_dark, dark_radii, rad1)
        print ("dark %s  done" % (i + 1))
    ind = np.argmin(minimum_var)
    if chisq == True:
        ind_new = np.argmin(chi2)
    # print(chi2)
    print ("dark minimum index for |G[r]| minimizer", ind)
    print ("\nA = ", A[ind])
    print ("\nK = ", K[ind])
    if chisq == True:
        print ("dark minimum index for chisquare minimizer", ind_new)
        print ("\nA = ", A_best[ind_new])
        print ("\nK = ", K_best[ind_new])

        table_dark.add_row((file_name_no_dir, dark_radii, ind_new, A_best[ind_new], K_best[ind_new], np.min(chi2), exp_time))

    table_dark.add_row((file_name_no_dir, dark_radii, ind, A_min[ind], K_min[ind], np.min(diff_ar), exp_time))
    table_dark.write(table_name_dark, format='ascii', overwrite=True)
    # ind_gal, ind_dark, temp, minimum_var, A_min, K_min):
    verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, minimum_var, A_min[ind], K_min[ind], ind, 'Gr', DQ)
    if chisq == True:
        verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, chi2, A_best[ind_new], K_best[ind_new], ind_new, 'chisq', DQ)


def galaxy_sky_sub(file_name, params, table_sky, name):
    work_dir = params["work_dir"]

    hdulist = fits.open(work_dir + file_name)
    DQ = hdulist[3].data
    DQ[int(hot_pix_x[0]), int(hot_pix_y[0])] = float(params['dq'])
    DQ[int(hot_pix_x[1]), int(hot_pix_y[1])] = float(params['dq'])
    data = hdulist[1].data
    data[DQ != 0] = float(params['dq_bad'])

    x_filter = hdulist[0].header["FILTER1"]
    aper_annulus = [np.mean(data[masks_annulus[k1]]) for k1 in range(len(rad1) - 1)]
    exp_time = hdulist[0].header["EXPTIME"]
    sky_value = np.mean(aper_annulus[int(params['sky_min']):int(params['sky_max'])])

    print ("output sky value of exposure with index = %s filter = %s sky_value = %s" % (gal_key, x_filter, sky_value))

    hdulist.close()

    hdulist = fits.open(work_dir + file_name)
    data = hdulist[1].data
    data[DQ != 0] = float(params['dq_bad'])
    hdulist[1].data = data - sky_value
    hdulist[1].data[DQ != 0] = float(params['dq_bad'])

    table_sky.add_row((file_name.replace(work_dir, ""), sky_value, exp_time, x_filter, name))
    table_name = '%sINTERMEDIATE_TXT/FLT_sky_%s.txt' % (work_dir, name)
    table_sky.write(table_name, format='ascii', overwrite=True)

    sky_sub_name = work_dir + file_name.replace("flt.fits", "sky_flt.fits")
    hdulist.writeto(sky_sub_name, overwrite=True, output_verify="ignore")
    hdulist.close()



# change this if you are intersted in intermediate plots
show_plot = False


if __name__ == '__main__':

    global hot_pix_x, hot_pix_y, rad1, rad_annulus, masks_annulus,\
        masks, dark_dir, dark_radii, DIR_PNG, DIR_FITS, DIR_TXT
    params = basic_params('dark_sub.cfg', 'basic')
    gal_name = params['name']
    dark_radii = int(params["mask_radii"])

    work_dir = params['work_dir']
    DIR_TXT = '%sINTERMEDIATE_TXT/' % (work_dir)
    DIR_PNG = '%sINTERMEDIATE_PNG/' % (work_dir)
    DIR_FITS = '%sINTERMEDIATE_FITS/' % (work_dir)
    DIR_SEG = '%sINTERMEDIATE_SEG/' % (work_dir)

    print ('1) creating directories for intermediate files ... \n')
    directory_check(DIR_TXT)
    directory_check(DIR_PNG)
    directory_check(DIR_FITS)
    directory_check(DIR_SEG)

    print ('Following directories have been created:-\n')
    print (DIR_TXT)
    print (DIR_PNG)
    print (DIR_FITS)
    print (DIR_SEG)

    print ('\n\n2) selecting the hot and cold frames from the list of FLT files .. \n')

    input_frames = params['input_frames']
    input_frames = input_frames.split(',')
    print ('Total number of SBC frames\n', len(input_frames))
    input_hot = []
    input_cold = []
    temp_cold = []
    temp_hot = []
    for j in input_frames:
        hdu = fits.open(work_dir + j)

        temp = (float(hdu[1].header["MDECODT1"]) +
                float(hdu[1].header["MDECODT2"])) / 2.
        if temp > float(params['temp_threshold']):
            input_hot.append(j)
            temp_hot.append((float(hdu[1].header["MDECODT1"]) +
                             float(hdu[1].header["MDECODT2"])) / 2.)
        else:
            input_cold.append(j)
            temp_cold.append((float(hdu[1].header["MDECODT1"]) +
                              float(hdu[1].header["MDECODT2"])) / 2.)
    print ("\ncold frames :- \n", input_cold)
    print ("\nhot frames:- \n", input_hot)
    print ('\n 3) creating apertures for circular annuli\n')

    cent_x = int(params['cent_x'])
    cent_y = int(params['cent_y'])
    width = int(params['width'])
    aper_lim = int(params['aper_lim'])
    nx = int(params['nx'])
    ny = int(params['ny'])
    chisq = params['chisq']
    rad1, rad_annulus, masks, masks_annulus = masks_circular(cent_x, cent_y, width, aper_lim, nx, ny)

    print ('4) performing photometry on input cold and hot frames\n')
    photometry(input_hot, input_cold, params, temp_hot, temp_cold, gal_name)

    '''additional hot pixels in data'''
    hot_pix_x = np.array(params['hot_pix_x'].split(','))
    hot_pix_y = np.array(params['hot_pix_y'].split(','))

    taget_name = params["name"]
    print ('5) creating table to write the output for dark and sky subtraction process')

    table_dark = Table(names=('file_name', 'dark_radii', 'ind', 'A_min', 'K_min', 'diff_ar', 'exp_time'), dtype=('S100', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    table_dark.meta['comments'] = ['galaxy name %s Output drk values calculated on FLT images for future preference \
     !Remember the corrresponding plots have sky value per exposure time ' % (gal_name)]
    print ('\n\n\n')

    table_dark_seg = Table(names=('file_name', 'dark_radii', 'ind', 'A_min', 'K_min', 'diff_ar', 'exp_time'), dtype=('S100', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    table_dark_seg.meta['comments'] = ['galaxy name %s Output drk values calculated on FLT images for future preference \
     !Remember the corrresponding plots have sky value per exposure time ' % (gal_name)]
    print ('\n\n\n')

    table_sky = Table(names=('file_name', 'sky_value', 'exp_time', 'filter', 'name'),
                      dtype=('S100', 'f4', 'f4', 'S4', 'S100'))
    table_sky.meta['comments'] = ['galaxy  with name %s Output sky values calculated on FLT images for future preference \
    !Remember the corrresponding plots have sky value per exposure time ' % (gal_name)]

    for file_name in input_frames:

        gal_str = []
        for letter in file_name:
            gal_str.append(letter)
        # !!!! remember this is very specfic to file names of the form 'jcmc41eeq'... \
        # I am finding just extracting 41eeq part here
        gal_key = gal_str[4] + gal_str[5] + gal_str[6] + gal_str[7] + gal_str[8]
        if file_name in input_cold:
            print ("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<< sky subtraction for %s>>>>>>>>>>>>>>>>>>>> \n" % (gal_key))

            galaxy_sky_sub(file_name, params, table_sky, gal_name)

            print ("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<< moving to next cold frame>>>>>>>>>>>>>>>>>>>> \n")

        elif file_name in input_hot:
            print ("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<< dark subtraction for %s>>>>>>>>>>>>>>>>>>>> \n" % (gal_key))

            dark_dir = params['dark_dir']
            dark_FLT = glob.glob("%s*flt.fits" % (dark_dir))
            dark_FLT, temp_FLT = dark_sort(dark_FLT)

            print ('\n\n\n')
            if dark_radii == 0:
                print ("<<<<< smoothing FLT for sextractor segmentation maps >>>>")
                smooth_scale = float(params['smooth_scale'])
                print (smooth_scale)
                file_smooth = smooth_gal(work_dir + file_name, gal_key, smooth_scale, params)
                print ("<<<<< Getting SEXTRACTOR for finding segmentaion maps >>>>")
                sex_config = (params['sex_config'])

                seg_map = sextractor_seg(file_smooth, sex_config, params)
                print ("<<<<<  Getting masked image for galaxy >>>>")

                file_mask = galaxy_mask(file_smooth, gal_key, smooth_scale, seg_map, sex_config)
                dark_sub(work_dir + file_name, dark_radii, dark_FLT, temp, params,
                         rad1, rad_annulus, masks, masks_annulus, table_dark_seg, chisq=True)

            else:

                dark_sub(work_dir + file_name, dark_radii, dark_FLT, temp, params,
                         rad1, rad_annulus, masks, masks_annulus, table_dark, chisq=True)
            print ("dark subtraction done for ", file_name)
            print ("HAPPY!!! \n")

            print ("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<< moving to next dark frame>>>>>>>>>>>>>>>>>>>> \n")

    types = ('*catalog.cat*', '*mask*', '*seg_map*', '*smooth*')  # the tuple of file types
    file1 = []
    for files in types:
        file1.extend(glob.glob(work_dir + files))
    print ("Copying intermediate Sextractor files now")
    for i in range(len(file1)):
        dest1 = DIR_SEG + file1[i].split('/')[-1]
        if os.path.exists(dest1):
            os.remove(dest1)
        shutil.move(file1[i], DIR_SEG)

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


################# for input photometry ##############
def photometry(file_hot, file_cold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    data = fits.getdata(file_hot)
    aper = [abs(np.sum(data[masks[k]])) for k in range(len(rad1))]
    ax1.plot(rad1, aper, color='r', label='hot exposure %s' % (file_hot))
    if file_cold != '1':
        data_cold = fits.getdata(file_cold)
        aper_cold = [abs(np.sum(data_cold[masks[k]])) for k in range(len(rad1))]
        ax1.plot(rad1, aper_cold, color='b', label='cold exposure %s' % (file_cold))
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('cumulative summation over circular apertures')
    ax1.legend()
    show2 = ax2.imshow(data, vmin=0.0, vmax=0.06, origin='lower')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="8%", pad=0.0)
    cbar = plt.colorbar(show2, cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    ax2.set_title('dark frame %s' % (file_hot.split('/')[-1]), y=1.15)
    ax2.plot(cent_x, cent_y, '+', color='r', markersize=10)
    circle(cent_x, cent_y, dark_radii, 'r', ax2)
    print (file_hot.split('.')[0], DIR_PNG)
    plt.suptitle('Input photometry', fontsize=20)
    fig.savefig('%s%s_input_photometry.png' % (DIR_PNG, file_hot.split('.')[0]), dvi=400)
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

def function_dark_sub(a, k,
                      aper_diff_gal, aper_diff_dark, dark_radii, galaxy, dark):

    aper_diff_ann = np.array(aper_diff_gal) - a * np.array(aper_diff_dark) - k
    s1 = np.sum(abs(aper_diff_ann[dark_radii:len(rad1) - 1]))

    return s1

################ function for chi square minimization ##############


def dark_sub_chisq(aper_diff_gal, aper_diff_dark, dark_radii, rad1):
    sci = np.array(aper_diff_gal[dark_radii:len(rad1) - 1])
    drk = np.array(aper_diff_dark[dark_radii:len(rad1) - 1])
    err = sp.ones_like(sci)
    #err = np.sqrt(sci)
    u = sp.sum(sci * drk / err)
    v = sp.sum(drk**2 / err)
    w = sp.sum(drk / err)
    x = sp.sum(sci / err)
    y = sp.sum(1. / err)

    A_best = (u / v - w * x / v / y) / (1. - w**2 / v / y)
    K_best = (x - A_best * w) / y

    mod_best = drk * A_best + K_best

    chi2 = sp.sum((sci - mod_best)**2 / (sci * (len(sci) - 2)))
    return A_best, K_best, chi2


def verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, minimum_var, A, K, ind, label, DQ):
    print ("6) plotting the minimization values now\n")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(temp_FLT, minimum_var, "o", markersize=8)
    ax.set_xlabel(r" Dark Temp[$^o$ C]")
    ax.set_ylabel("Minimizer %s [counts]" % (label))
    ax.axvline(x=temp_gal, color="g", label="galaxy temperature")
    ax.axvline(x=temp_FLT[ind], color="r", label="minimum value A= %.1e\n, K = %.1e,\n\
        index = %.1e\n" % (A, K, ind))
    ax.legend(loc='lower left')
    minimizer_png = file_name_no_dir.replace('.fits', '')
    minimizer_png = minimizer_png + "_minimizer_%s.png" % (label)
    ax.set_title('minimizer values %s method' % (label))
    fig.savefig("%s/INTERMEDIATE_PNG/%s" % (work_dir, minimizer_png), dvi=400, bbox_inches='tight')
    plt.show()

    print ("7) plotting the difference image now ..... with galaxy\n")

    hdulist = fits.open(file_name_flt)
    exp_time = hdulist[0].header["EXPTIME"]
    data_gal = hdulist[1].data

    hdu_dark_selected = fits.open(dark_FLT[ind])
    data_dark = (A * hdu_dark_selected[1].data * exp_time / float(params["exp_dark"]) + K) / exp_time  # exp time for darks is 1000 secs
    dark_selected = work_dir + file_name_no_dir.replace("flt", "dark_%s" % (ind + 1))

    data_gal[DQ != 0] = 0.0
    data_dark[DQ != 0] = 0.0

    data_sub = data_gal - data_dark * exp_time
    data_sub[DQ != 0] = 0.0

    hdulist[1].data = data_sub

    hdu_dark_selected[1].data = data_dark

    hdu_dark_selected.writeto(dark_selected, overwrite=True)
    hdu_dark_selected.close()
    sub_name = file_name_flt.replace("flt", "drk_flt")

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
    verification_png = verification_png + "_verification%s.png" % (label)
    plt.suptitle('radial profiles %s method' % (label), fontsize=20)
    fig.savefig("%s/INTERMEDIATE_PNG/%s" % (work_dir, verification_png), dvi=400, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def dark_sub(file_name_flt, dark_flt, params, chisq):
    file_name_no_dir = file_name_flt.split("/")
    file_name_no_dir = file_name_no_dir[-1]

    table_name_dark = '%sINTERMEDIATE_TXT/FLT_drk_gal_%s.txt' %\
        (work_dir, file_name_no_dir.split('.')[0])

    print ('Table for dark subtraction information', table_name_dark)

    hdulist = fits.open(file_name_flt)
    exp_time = hdulist[0].header["EXPTIME"]
    data = hdulist[1].data
    temp_gal = (float(hdulist[1].header["MDECODT1"]) + float(hdulist[1].header["MDECODT2"])) / 2.
    print ('\n\n\n')
    print ('5) Performing dark subtraction\n\n')

    A_lim = float(params["a_lim"])
    K_lim = float(params["k_lim"])
    del_A = float(params["del_a"])
    del_K = float(params["del_k"])
    dark_radii = int(params["mask_radii"])

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

    for i in range(len(dark_FLT)):
        fits_dark_FLT = fits.open(dark_FLT[i])
        data_dark = fits_dark_FLT[1].data * exp_time / float(params["exp_dark"])  # exp time for darks is 1000 secs

        ### looking for DQ array from dark FLT files ###

        fits_dark_FLT = fits.open(dark_FLT[i])
        data_dark_DQ = fits_dark_FLT[3].data

        DQ = fits_dark_FLT[3].data
        print ('hot pixels replaced by 1000')
        DQ[int(hot_pix_x[0]), int(hot_pix_y[0])] = 1000
        DQ[int(hot_pix_x[1]), int(hot_pix_y[1])] = 1000
        print ('DQ!=0 pixels  replaced by 0')

        data_dark[DQ != 0] = 0.0
        data[DQ != 0] = 0.0
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
        fits_dark_name = file_name_no_dir.replace(".fits", "dark_%s_%s.fits" % (i + 1, '|G(r)|'))

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

        fits_variance_name = file_name_no_dir.replace(".fits", "dark_%s_diff_%s.fits" % (i + 1, '|G[r]|'))
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
    print(chi2)
    print ("dark minimum index for G[r] minimizer", ind)
    if chisq == True:
        print ("dark minimum index for chisquare minimizer", ind_new)
        table_dark.add_row((file_name_no_dir, dark_radii, ind_new, A_best[ind_new], K_best[ind_new], np.min(chi2), exp_time))
    #
    table_dark.add_row((file_name_no_dir, dark_radii, ind, A_min[ind], K_min[ind], np.min(diff_ar), exp_time))
    table_dark.write(table_name_dark, format='ascii', overwrite=True)
    # ind_gal, ind_dark, temp, minimum_var, A_min, K_min):
    verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, minimum_var, A_min[ind], K_min[ind], ind, '|G[r]|', DQ)
    if chisq == True:
        verify_plot(params, file_name_no_dir, file_name_flt, dark_FLT, temp_FLT, temp_gal, minimum_var, A_best[ind_new], K_best[ind_new], ind_new, 'chisq', DQ)


from astropy.table import Table

if __name__ == '__main__':

    global hot_pix_x, hot_pix_y, rad1, rad_annulus, masks_annulus,\
        masks, dark_dir, dark_radii, DIR_PNG, DIR_FITS, DIR_TXT
    params = basic_params('dark_sub.cfg', 'basic')
    work_dir = params['work_dir']
    DIR_TXT = '%s/INTERMEDIATE_TXT/' % (work_dir)
    DIR_PNG = '%s/INTERMEDIATE_PNG/' % (work_dir)
    DIR_FITS = '%s/INTERMEDIATE_FITS/' % (work_dir)

    print ('1) creating directories for intermediate files ... ')
    directory_check(DIR_TXT)
    directory_check(DIR_PNG)
    directory_check(DIR_FITS)
    print ('following directories have been created:-\n')
    print (DIR_TXT)
    print (DIR_PNG)
    print (DIR_FITS)
    input_hot = params['input_hot']
    input_cold = params['input_cold']
    '''additional hot pixels in data'''
    hot_pix_x = np.array(params['hot_pix_x'].split(','))
    hot_pix_y = np.array(params['hot_pix_y'].split(','))
    dark_radii = int(params['mask_radii'])
    dark_dir = params['dark_dir']
    dark_FLT = glob.glob("%s*flt.fits" % (dark_dir))
    dark_FLT, temp_FLT = dark_sort(dark_FLT)
    cent_x = int(params['cent_x'])
    cent_y = int(params['cent_y'])
    width = int(params['width'])
    aper_lim = int(params['aper_lim'])
    nx = int(params['nx'])
    ny = int(params['ny'])
    chisq = params('chisq')
    print ('\n\n\n')

    print ('2) creating apertures for circular annuli\n')
    rad1, rad_annulus, masks, masks_annulus = masks_circular(cent_x, cent_y, width, aper_lim, nx, ny)
    print ('\n\n\n')
    print ('3) performing photometry on input cold and hot frames\n')
    photometry(input_hot, input_cold)
    dark_radii = int(params['mask_radii'])
    print ('\n\n\n')

    print ('4) creating table to write the output for dark subtraction process')

    table_dark = Table(names=('file_name', 'dark_radii', 'ind', 'A_min', 'K_min', 'diff_ar', 'exp_time'), dtype=('S100', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    table_dark.meta['comments'] = ['galaxy name %s Output sky values calculated on FLT images for future preference \
    \n !Remember the corrresponding plots have sky value per exposure time ' % (input_hot)]
    print ('\n\n\n')
    dark_sub(input_hot, dark_FLT, params, chisq=True)

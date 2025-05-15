"""
This script prepares a masterflat for reducing HeI observations by doing the following: 
0) Initializes necessary directories
1) 

Notes: 
- The script is designed to be run from the command line with the following arguments: 
    $python prep_fits_from_spectrum.py masterdark.fits updated_flats_list

- Note that the calibrations directory is specified in the code, so no path for the masterdark is 
requied.

"""
#%%
import sys
import astropy.io.fits as fits
import numpy as np
import glob
import os
import scipy.signal
import matplotlib.pyplot as plt
from helper_functions import trace_edge, interpolate_missing,subtract_crosstalk, subtract_overscan
import importlib
import NIRSPEC_CONSTANTS

# Import Constants
N = NIRSPEC_CONSTANTS.N
RIGHT_MARGIN = NIRSPEC_CONSTANTS.RIGHT_MARGIN
LEFT_MARGIN = NIRSPEC_CONSTANTS.LEFT_MARGIN 
FLAT_MIN = NIRSPEC_CONSTANTS.FLAT_MIN
FLAT_MAX = NIRSPEC_CONSTANTS.FLAT_MAX 
MIN_ORDER_SEPARATION = NIRSPEC_CONSTANTS.MIN_ORDER_SEPARATION
OVERSCAN_WIDTH = NIRSPEC_CONSTANTS.OVERSCAN_WIDTH
FILTER = NIRSPEC_CONSTANTS.NIRSPEC1_70

# Spcifiy dark parth 
dark_path = 'NIRSPEC_1/221228/masterdark.fits'

def combine_flats(filenames, masterdark):

    print("Combining flats...")
    data = []
    medians = []
    for filename in filenames:
        with fits.open(filename) as hdulist:
            flat = np.rot90(hdulist[0].data, 3) / hdulist[0].header["COADDS"]
            plt.imshow(flat, origin='lower')
            flat = subtract_crosstalk(flat)
            flat = subtract_overscan(flat)
            flat -= masterdark
            medians.append(np.median(flat))
            flat /= np.median(flat)
            data.append(flat)

    data = np.array(data) * np.median(medians)
    plt.imshow(data[0], origin='lower')
    
    masterflat = np.median(data, axis=0)
    std = np.std(data, axis=0)
    return masterflat, std


def fit_and_divide(masterflat, top_edges, bot_edges, X_ORD=5, Y_ORD=3):
    print('Fitting and dividing...')
    normalization = np.ones(masterflat.shape) * np.inf
    for o in range(len(top_edges)):
        top_edge = top_edges[o]
        bot_edge = bot_edges[o]
        #if o == 0: top_edge += 10 #Hack
                    
        all_scaled_xs = []
        all_scaled_ys = []
        all_values = []
        all_xs = []
        all_ys = []

        for c in range(LEFT_MARGIN, masterflat.shape[1] - RIGHT_MARGIN):
            y_min = int(round(top_edge[c])) + 1
            y_max = int(round(bot_edge[c])) - 1
            rows = np.arange(y_min, y_max)
            scaled_ys = (rows - y_min) / (y_max - y_min)
            all_scaled_xs += [c / masterflat.shape[1]] * (y_max - y_min)
            all_scaled_ys += list(scaled_ys)
            all_values += list(masterflat[y_min : y_max, c])
            all_xs += [c] * (y_max - y_min)
            all_ys += list(rows)

        all_scaled_xs = np.array(all_scaled_xs)
        all_scaled_ys = np.array(all_scaled_ys)
        all_values = np.array(all_values)
        all_xs = np.array(all_xs)
        all_ys = np.array(all_ys)
        
        A = []
        for i in range(X_ORD):
            for j in range(Y_ORD):
                A.append(all_scaled_xs**i * all_scaled_ys**j)
        A = np.array(A).T
        coeffs, _, _, _ = scipy.linalg.lstsq(A, all_values)
        predicted = A.dot(coeffs)
        normalization[all_ys, all_xs] = predicted
        #plt.scatter(all_scaled_xs, all_values)
        #plt.plot(all_scaled_xs, predicted, color='r')
        #plt.show()
        
            
    return masterflat/normalization, normalization


def identify_anomalous_gains(masterflat, bad_pixel_map, top_edges, bottom_edges, margin=1):
    print('Identifying anomalous gains...')
    bad_pixel_map = np.copy(bad_pixel_map)
    for o in range(len(top_edges)):
        for c in range(N):
            min_y = int(np.round(top_edges[o][c])) + margin
            max_y = int(np.round(bottom_edges[o][c])) - margin
            bad_gains = np.logical_or(masterflat[min_y : max_y, c] < FLAT_MIN,
                                      masterflat[min_y : max_y, c] > FLAT_MAX)
            bad_pixel_map[min_y : max_y, c][bad_gains] = True
    return bad_pixel_map

def trace_edges(masterflat, FILTER, window=10, percentile=98):
    print('Tracing edges...')
    if filter is None:
        edges = scipy.ndimage.sobel(masterflat, axis=0)
        edges[0:OVERSCAN_WIDTH + 2] = 0

        midsection = np.median(edges[:, int(N/2 - window) : int(N/2 + window)], axis=1)
        plt.imshow(masterflat)
        plt.figure()
        plt.plot(midsection)
        plt.show()
        
        upper_positions = scipy.signal.find_peaks(midsection,
                                        height=np.percentile(midsection, percentile),
                                        distance=MIN_ORDER_SEPARATION)[0]
        lower_positions = scipy.signal.find_peaks(-midsection,
                                                  height=np.percentile(-midsection, percentile),
                                                  distance=MIN_ORDER_SEPARATION)[0]
    else:
        # Get filter bounds
        bounds = FILTER
        if len(bounds) == 2:
            upper_positions = np.array([bounds[0]])
            lower_positions = np.array([bounds[1]])
               
    num_orders = min(len(upper_positions), len(lower_positions))
    upper_edges = []
    lower_edges = []

    for o in range(num_orders):
        #print(o)
        upper_edges.append(trace_edge(masterflat, upper_positions[o], False))
        lower_edges.append(trace_edge(masterflat, lower_positions[o], True))

    return np.array(upper_edges), np.array(lower_edges)
#%%

if __name__ == "__main__":

    #   Read in the master dark path 
    print(f"Using existing master dark from: {dark_path}")
    with fits.open(dark_path) as hdul:
        masterdark = hdul[0].data
        std = hdul["STD"].data if "STD" in hdul else np.zeros_like(masterdark)
        mask = hdul["MASK"].data.astype(bool) if "MASK" in hdul else np.zeros_like(masterdark, dtype=bool)

    data_dir = 'NIRSPEC_1/221228/fits/'
    list_of_flats = 'NIRSPEC_1/221228/flats_list.txt'

    filenames = [line.strip() for line in open(list_of_flats)]
    # prepend the directory to each filename
    full_filenames = [os.path.join(data_dir, filename) for filename in filenames]

    print(full_filenames)

    #raw_masterflat, raw_std = combine_flats(filenames, masterdark_hdul[0].data)
    raw_masterflat, raw_std = combine_flats(full_filenames, masterdark)
    #combine_flats(full_filenames, masterdark)
    top_edges, bottom_edges = trace_edges(raw_masterflat, FILTER)

    #dark_bad_pixels = np.array(masterdark_hdul["MASK"].data, dtype=bool)
    dark_bad_pixels = np.array(mask, dtype=bool)
    is_bad_pixel = dark_bad_pixels
    is_bad_pixel[:, -RIGHT_MARGIN:] = False
    masterflat = interpolate_missing(raw_masterflat, is_bad_pixel)

    # Divide out smooth variations in spectrum
    masterflat, normalization = fit_and_divide(masterflat, top_edges, bottom_edges)

    is_bad_pixel = identify_anomalous_gains(masterflat, is_bad_pixel, top_edges, bottom_edges)

    image_hdu = fits.PrimaryHDU(masterflat)
    bad_pixels_hdu = fits.ImageHDU(np.array(is_bad_pixel, dtype=int), name="BADPIX")

    #Create HDU of edge traces
    cols = fits.ColDefs([
        fits.Column(name='Top edges', format='{}D'.format(N), array=np.array(top_edges)),
        fits.Column(name='Bottom edges', format='{}D'.format(N), array=np.array(bottom_edges))
    ])
    edges_hdu = fits.BinTableHDU.from_columns(cols, name="EDGES")

    #Create final mask
    mask = np.zeros(masterflat.shape, dtype=bool)
    mask[masterflat < FLAT_MIN] = True
    mask[masterflat > FLAT_MAX] = True
    mask[:, -RIGHT_MARGIN:] = True
    mask = np.logical_or(is_bad_pixel, mask)
    mask_hdu = fits.ImageHDU(np.array(mask, dtype=int), name="MASK")

    hdul = fits.HDUList([image_hdu, bad_pixels_hdu, edges_hdu, mask_hdu,
                        fits.ImageHDU(raw_masterflat, name="RAW"),
                        fits.ImageHDU(raw_std, name="STD"),
                        fits.ImageHDU(normalization, name="NORMALIZATION")])

    hdul.writeto(os.path.join('./NIRSPEC_HeI'+ 'masterflat.fits'), overwrite=True)

    plt.imshow(masterflat, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f"Master Flat for {FILTER}")
    for o in range(len(hdul["EDGES"].data)):
        upper_edge, lower_edge = hdul["EDGES"].data[o]
        plt.plot(upper_edge, color='r')
        plt.plot(lower_edge, color='black')
    plt.show()    

#%%

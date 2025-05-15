import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import sys
import scipy.stats
import pdb

def noise_rs(im):
    im = np.copy(im).flatten()
    imsort = im[np.argsort(im)]
    npix = imsort.shape[0]
    num = int(2*npix/3 + 1)
    noise = min(imsort[num-1:npix-1]+imsort[num-2:npix-2]-imsort[0:npix-num]-imsort[1:npix-num+1])/2.0
    noise = noise / (1.9348+(float(num-2)/float(npix) - 0.66667)/.249891)
    if npix > 100:
        noise = noise/(1.0-0.6141*np.exp(-0.5471*np.log(npix)))
    else:
        noise = noise/(1.0-0.2223*np.exp(-0.3294*np.log(npix)))

    return noise

def med_list(list_of_arrays):    
    flat_list = [item for sublist in list_of_arrays for item in sublist]
    return np.median(flat_list)

def fixpix(im, iternum=30, quiet=False):
    print("Starting fixpix")
    medimg = np.median(im)
    imnew = np.copy(im)
    maskarr = np.zeros(im.shape, dtype=bool)
    if iternum is None:
        iternum = 100
    mparm = 1

    #DETERMINE THE AVERAGE NOISE IN THE ENTIRE IMAGE

    #scan through 5x5 boxes to determine some average estimate of the
    #image noise

    xs = im.shape[1] 
    ys = im.shape[0] 
    sigma = np.zeros(int(xs/5*ys/5))  # Corrected by SSKIM

    n=0
    for i in range(2, ys-2, 5):
        for j in range(2, xs-2, 5):
            tmp=im[i-2:i+3,j-2:j+3].flatten()
            srt=tmp[np.argsort(tmp)]
            sigma[n] = (min(srt[16:24]+srt[15:23]-srt[0:8]-srt[1:9])/2.0)/1.540
            n = n+1

    # define the median value of the sigmas, and the sigma of the sigmas

    medsig = np.median(sigma)
    sigsig = noise_rs(sigma)  # Modified by SSKIM

    # BEGIN SCANNING FOR HOT & COLD PIXELS
    # start at (4,4) (so that any pixel in the box can have a 5x5 
    # square centered on it.  find the hottest and coldest pixels
    # loop through several iterations of replactments

    for iter in range(1, iternum + 1):
        hotcnt = 0 # variables to count the replacements of hot and cold pixels
        coldcnt = 0
        addon = ((iter+1) % 2) *2

        for i in range(4 + addon, xs-4, 5):
            for j in range(4 + addon, ys-4, 5):
                box = imnew[j-2:j+3,i-2:i+3]

                hotval = np.max(box)
                hoti = np.argmax(box)

                hotx = hoti % 5 + i - 2
                hoty = int(hoti / 5) + j - 2
                
                coldval = np.min(box)
                coldi = np.argmin(box)
                coldx = coldi % 5 + i-2
                coldy = int(coldi / 5) + j-2

                # begin the decision process for the hottest pixel
                hot = imnew[hoty-2:hoty+3,hotx-2:hotx+3]
                med8 = med_list([hot[1,1:4], hot[3,1:4], [hot[2,1], hot[2,3]]])
                med16 = med_list([hot[0,0:5], hot[4,0:5], hot[1:4,0], hot[1:4,4]])
                srt = hot.flatten()[np.argsort(hot.flatten())]
                sig = (np.min(srt[16:25]+srt[15:24]-srt[0:9]-srt[1:10])/2.0)/1.540

                # decide from the noise in the box if we 
                # are on a feature or a gaussian background               
                if sig > (medsig + 2.0*sigsig):
                  sig = np.max([medsig+2.0*sigsig, np.sqrt(5.0+.210*np.abs(med8)/mparm)*mparm])

                # decide whether to replace pixel
                if ((med8+2.0*med16)/3.0 - medimg) > 2.0*sig:
                    if (imnew[hoty,hotx] > (2.0*med8-medimg+3.0*sig)):
                        imnew[hoty,hotx] = med8
                        maskarr[hoty,hotx]=True
                        hotcnt=hotcnt+1
                else:
                    if ((imnew[hoty,hotx] - (med8+2.0*med16)/3) > 5.0*sig):
                        imnew[hoty,hotx] = med8
                        maskarr[hoty,hotx]=True
                        hotcnt=hotcnt+1

                # begin the decision process for the coldest pixel
                cld = imnew[coldy-2:coldy+3,coldx-2:coldx+3]
                med8 = med_list([cld[1, 1:4], cld[3, 1:4], [cld[2,1],cld[2,3]]])
                med16 = med_list([cld[0,0:5],cld[4,0:5], cld[1:4,0], cld[1:4,4]])
                srt = cld.flatten()[np.argsort(cld.flatten())]
                sig = (np.min(srt[16:25]+srt[15:24]-srt[0:9]-srt[1:10])/2.0)/1.540

                # decide from the noise in the box if we 
                # are on a feature or a gaussian background
                if sig > (medsig + 2.0*sigsig):
                  sig = np.max([medsig+2.0*sigsig, np.sqrt(5.0+.210*np.abs(med8))])

                # decide whether to replace pixel
                if ((med8+2.0*med16)/3.0 - medimg) < -2.0*sig:
                    if (imnew[coldy,coldx] < (2.0*med8-medimg-3.0*sig)):
                        imnew[coldy,coldx] = med8
                        maskarr[coldy,coldx]=True
                        coldcnt=coldcnt+1
                else:
                    if ((imnew[coldy,coldx]-(med8+2.0*med16)/3) < -5.0*sig):
                        imnew[coldy,coldx] = med8
                        maskarr[coldy,coldx]=True
                        coldcnt = coldcnt+1

        print("hot and cold pixels in iter.",hotcnt,coldcnt,iter)
        if hotcnt == 0 and coldcnt == 0:
            break
    return imnew, maskarr

if __name__ == "__main__":
    image = astropy.io.fits.open("cal_nspec190403_1000.fits")[0].data[1000:1100, 1000:1100]
    fixpix(image)

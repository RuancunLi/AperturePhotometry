
# coding: utf-8

# In[1]:

import numpy as np
import warnings
import matplotlib.pyplot as plt
from astropy.table import hstack, Table, Column
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import detect_threshold
from photutils.background import BkgZoomInterpolator
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources
from astropy.visualization import  LogStretch,AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d
from pprint import pprint
import pickle as pickle
from matplotlib.patches import Ellipse
from scipy.optimize import brentq
import sys


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def maskPSFGeneratorplus(image, sources, psfFWHM=1, f0=1., QuietMode=True, dtype=bool, sbp = None       ,sourcecleaned=None,merger=None,imagepath=None,correctionindex=1.,calibration=None):

    #merger={
    #'sample':table of each component,
    #'distance':distance between each component,
    #'height':surface bright of mask edge
    #    }
    #correctionindex: masksize correction for brilliant star

    ny, nx = image.shape
    positions = (sources['xcentroid'], sources['ycentroid'])
    msList = []
    sys.setrecursionlimit(200000)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(image, sigma=3.0, iters=5)
    if(dtype == bool):
        mask = np.zeros_like(image, dtype=bool)
    if (dtype == int) :
        mask = np.zeros_like(image, dtype=int)
    if (merger is not None) :
        centX1 = merger['sample']['xcentroid'][loop]
        centY1 = merger['sample']['ycentroid'][loop]
        peak1  = merger['sample']['peak'][loop]
        xaxis = np.arange(nx)
        yaxis = np.arange(ny)
        xmesh, ymesh = np.meshgrid(xaxis, yaxis)
        srcRad = np.sqrt( (xmesh - centX1)**2. + (ymesh - centY1)**2. )
        sbps=lambda x : sbp(x)*peak1-merger['height']*peak1
        zeropoint=brentq(sbps,a=0.,b=70)
        maskSize=np.min([merger['distance']/2,zeropoint])
        if (dtype == bool) :
            mask[srcRad < maskSize] = True
            mask[np.isnan(image)] = True
            mask[np.isinf(image)] = True
        if (dtype == int) :
            mask[srcRad < maskSize] = 1
            mask[np.isnan(image)] = 1
            mask[np.isinf(image)] = 1
    for loop in range(len(sources)):
        centX = sources['xcentroid'][loop]
        centY = sources['ycentroid'][loop]
        flux  = sources['flux'][loop]
        peak  = sources['peak'][loop]
        if peak < 1.5*sky_std :
            msList.append(0)
            continue
        if calibration is not None:
            if peak < calibration:
                msList.append(0)
                continue
        xaxis = np.arange(nx)
        yaxis = np.arange(ny)
        xmesh, ymesh = np.meshgrid(xaxis, yaxis)
        srcRad = np.sqrt( (xmesh - centX)**2. + (ymesh - centY)**2. )
        if(sbp is None):
            maskSize = 2*psfFWHM/2. * (np.max([np.log10(flux/f0), 0]) + 1.5)
        else:
            if (peak<10*sky_std):
                sbps=lambda x : sbp(x)*peak-sky_std
            else:
                sbps=lambda x : sbp(x)*peak-3.*sky_std
            if (sbps(0.1)*sbps(69.9)) > 0.:
                maskSize=70.
            else:
                zeropoint=brentq(sbps,a=0.,b=70)
                maskSize=zeropoint
        if  sourcecleaned is not None  :
            for loop7 in range(len(sourcecleaned)):
                if sources['id'][loop] == sourcecleaned['id'][loop7]:
                    maskSize=maskSize*0.5
        maskSize=pow(maskSize,correctionindex)
        msList.append(maskSize)
        if (dtype == bool) :
            mask[srcRad < maskSize] = True
            #Also mask the position with bad pixels
            mask[np.isnan(image)] = True
            mask[np.isinf(image)] = True
        if (dtype == int) :
            mask[srcRad < maskSize] = 1
            #Also mask the position with bad pixels
            mask[np.isnan(image)] = 1
            mask[np.isinf(image)] = 1
        #print('loading   {0}% '.format(loop*100./(len(sources))))
    if not QuietMode:
        fig = plt.figure(figsize=(10, 20))
        norm = ImageNormalize(stretch=AsinhStretch())
        plt.imshow(image, cmap='Greys', origin='lower', norm=norm,vmin=(sky_median-sky_std),vmax=(sky_median+3*sky_std))
        ax = plt.gca()
        for loop in range(len(sources)):
            centX = sources['xcentroid'][loop]
            centY = sources['ycentroid'][loop]
            cir = plt.Circle((centX, centY), radius=msList[loop], color='gold', fill=False)
            ax.add_patch(cir)
        if imagepath is None :
            plt.show()
        else:
            plt.savefig(imagepath['savepath'] + "{0}.png".format(imagepath['name']), dpi=200)
        plt.close()
    maskResults = {
        'mask':mask,
        'msList':msList,
    }
    return maskResults

def Maskellipse (mask,dtype,posi,a,ellipticity,PA,antimask=False):
    #PA from x axis anti-clockwise
    masky=mask.copy()
    ny, nx = mask.shape
    xm=int(posi[0])
    ym=int(posi[1])
    b=a*(1.-ellipticity)
    theta=PA*np.pi/180.
    maskpix=[]
    aint=int(a)+1
    xmin=np.max([-aint-1,0-xm])
    xmax=np.min([aint+1,nx-xm-1])
    ymin=np.max([-aint-1,0-ym])
    ymax=np.min([aint+1,ny-1-ym])
   # print [xmax,ymax]
    x=xmin
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    while (x<(xmax+1)):
        y=ymin
        while(y<(ymax+1)):
            d=A*x**2+B*x*y+C*y**2-(a**2)*(b**2)
            if(d <= 0):
                maskpix.append((xm+x,ym+y))
                if(dtype == int):
                    masky[ym+y][xm+x]=1
                    if antimask:
                        masky[ym+y][xm+x]=0
                if(dtype == bool):
                    masky[ym+y][xm+x]=True
                    if antimask:
                        masky[ym+y][xm+x]=False
            y +=1
        x +=1
    result = {
        'mask':masky,
        'maskpix':maskpix
    }
    return result


def getsky (data,psfFWHMpix,snr=3.0,dtype=bool) :
    threshold = detect_threshold(data, snr=snr)
    sigma = psfFWHMpix * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(data, threshold, npixels=5, filter_kernel=kernel)
    ny, nx =data.shape
    if(dtype == bool):
        mask = np.zeros_like(data, dtype=bool)
        mask[np.isnan(data)] = True
        mask[np.isinf(data)] = True
    if (dtype == int) :
        mask = np.zeros_like(data, dtype=int)
        mask[np.isnan(data)] = 1
        mask[np.isinf(data)] = 1
    for loopx in range(nx) :
        for loopy in range(ny):
            if segm.data[loopy][loopx] > 0:
                if(dtype == bool):
                    mask[loopy][loopx]=True
                if (dtype == int) :
                    mask[loopy][loopx]=1
    result = {
        'mask':mask,
        'segmantation':segm
    }
    return result



def masksegm (mask):
    mask=mask.astype(float)
    threshold = detect_threshold(mask, snr=1.)
    segm= detect_sources(mask,threshold,npixels=5)
    return segm



def polynomialfit(data,mask,order=3):
    i, j = np.mgrid[:data.shape[0], :data.shape[1]]
    i=data.shape[0]-i
    imask=i[~mask.astype(bool)]
    jmask=j[~mask.astype(bool)]
    datamask=data[~mask.astype(bool)]
    p_init = models.Polynomial2D(degree=order)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fit_p(p_init, imask, jmask, datamask)
    background=p(i,j)
    datap=data-background
    result={
        'bkgfunc':p,
        'bkg':background,
        'residual':datap
    }
    return result

def polynomialinterpld(data,mask,order=3,boxsize=60)  :
    sigma_clip = SigmaClip(sigma=3., iters=5)
    bkg_estimator = MedianBackground()
    inter=BkgZoomInterpolator(order=order)
    bkg = Background2D(data, (boxsize, boxsize), filter_size=(3, 3),mask=mask.astype(bool),sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, edge_method="pad",interpolator=inter)
    background=bkg.background
    datap=data-background
    result={
        'bkg':background,
        'residual':datap
    }
    return result

def addpsf(data,xy,psf,peak):
    ny,nx=psf.shape
    ny0,nx0=data.shape
    peak0=np.max(psf)
    intx=int(xy[0])
    inty=int(xy[1])
    py,px = np.unravel_index(psf.argmax(),psf.shape)
    psfrescale=psf*peak/peak0
    minx=np.max([0,intx-px])
    maxx=np.min([nx0-1,intx-px+nx-1])
    miny=np.max([0,inty-py])
    maxy=np.min([ny0-1,inty-py+ny-1])
    x=minx
    datafake=data.copy()
    while x <= maxx:
        y=miny
        while y<= maxy:
            datafake[(y,x)]+=psfrescale[(y-inty+py,x-intx+px)]
            y+=1
        x+=1
    return datafake

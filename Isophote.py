
# coding: utf-8


from pyraf import iraf
from astropy.table import Table, Column
import numpy as np
import os
import sys

def Stdout2(new_out):
    """
    Redirect the stdout.
    """
    old_out, sys.stdout = sys.stdout, new_out # replace sys.stdout
    return sys.stdout, old_out

tmp_file = "ellipse.tmp"
nout = open(tmp_file, "w")
nout, oout = Stdout2(nout)

imco=iraf.imcopy
dump=iraf.tdump
iraf.stsdas()
iraf.analysis()
iraf.isophote()
ellipse = iraf.ellipse
samp=iraf.samplepar
nout.close()
Stdout2(oout)
os.remove(tmp_file)



def runellipse(filepath,talname,fitsname,sources,maxsma,band,label='F',step=0.1,linear=False,psfFWHM=5,f0=1.,shape=None,fflag=0.5):
    imco.mode='h'
    ellipse.unlearn()
    ellipse.mode='h'
    samp.mode='h'
    ellipse.interactive=False
    #imco.interactive=False
    nout = open(tmp_file, "w")
    nout, oout = Stdout2(nout)
    imco.input=filepath+fitsname+'.fits'
    name=filepath+talname+'{0}{1}'.format(label,band)+'.fit'
    if os.path.exists(name):
        os.remove(name)
    imco.output=name
    imco.run()
    imco.input=filepath+talname+'{0}{1}'.format(label,band)+'mask.fits'
    imco.output=filepath+talname+'{0}{1}'.format(label,band)+'.pl'
    imco.run()
    ellipse.input=filepath+talname+'{0}{1}'.format(label,band)+'.fit'
    ellipse.linear=linear
    ellipse.step=step
    ellipse.maxsma=maxsma
    #print ('maxa1:{0}'.format(maxsma))
    ellipse.conver=0.05
    ellipse.minit=10
    ellipse.maxit=50
    ellipse.soft=False
    ellipse.region=False
    ellipse.wander=2.
    sourcenumber=len(sources)
    ellipse.hellip=False
    ellipse.hpa=False
    ellipse.maxgerr=0.5
    ellipse.usclip=3.0
    ellipse.lsclip=3.0
    ellipse.nclip=0
    ellipse.fflag=fflag
    ellipse.region = False
    ellipse.memory = True
    ellipse.verbose = True
    #ellipse.sdevice=None
    #ellipse.tsample=None
    #ellipse.absangle=True
    #ellipse.harmonics=None
    if(sourcenumber == 1):
        ellipse.hcenter=False
        ellipse.x0 = sources['xcentroid'][0]
        ellipse.y0 = sources['ycentroid'][0]
        ellipse.minsma=0
        ellipse.ellip0=0.2
        ellipse.pa0=0
        ellipse.recenter=True
        ellipse.xylearn=True
        ellipse.olthresh=0.
        ellipse.sma0=5.
    if(sourcenumber == 2):
        x1 = sources['xcentroid'][0]
        y1 = sources['ycentroid'][0]
        x2 = sources['xcentroid'][1]
        y2 = sources['ycentroid'][1]
        X0=0.5*(x1+x2)
        Y0=0.5*(y1+y2)
        ellipse.hcenter=True
        ellipse.hellip=True
        ellipse.hpa=True
        c=0.5*np.sqrt((x1 - x2)**2. + (y1 - y2)**2.)
        ellipse.maxsma=np.max([2*c,maxsma])
        ellipse.x0=X0
        ellipse.y0=Y0
        ellipse.recenter=False
        ellipse.xylearn=False
        ellipse.minsma=0.625*np.sqrt((x1 - x2)**2. + (y1 - y2)**2.)
        a=c+15
        ellipse.sma0=np.max([ellipse.minsma,5.0])
        the=180/np.pi*np.arctan((y1 - y2)/(x1 - x2))
        if the < 0 :
            the += 180
        ellipse.pa0=theta=-90+the
        ellipse.ellip0=np.max( [0.05,1-np.sqrt(1. - (c/a)**2.)])
        ellipse.olthresh=0.
    ellipse.output=filepath+talname+'{0}{1}'.format(label,band)+'.tab'
   # print ('maxa2:{0}'.format(ellipse.maxsma))
    if shape is not None:
        ellipse.hcenter=True
        ellipse.hellip=True
        ellipse.hpa=True
        if shape['PA'] < -90.:
            shape['PA'] += 180.
        ellipse.pa0=shape['PA']
        ellipse.ellip0=np.max([shape['ellipticity'],0.05])
        ellipse.x0=shape['posi'][0]
        ellipse.y0=shape['posi'][1]

    ellipse.run()
    dump.unlearn()
    dump.mode='h'
    #dump.interactive=False
    dump.table=filepath+talname+'{0}{1}'.format(label,band)+'.tab'
    dump.datafile=filepath+talname+'{0}{1}'.format(label,band)+'.txt'
    dump.run()
    nout.close()
    Stdout2(oout)
    os.remove(tmp_file)

    if os.path.exists(name):
        os.remove(name)
    name=filepath+talname+'{0}{1}'.format(label,band)+'.pl'
    if os.path.exists(name):
        os.remove(name)
    name=filepath+talname+'{0}{1}'.format(label,band)+'.tab'
    if os.path.exists(name):
        os.remove(name)


def meanintensity(data,aperture,mask=None):
    '''
    aperture={
        'posi': center position of elliptical aperture[x,y],
        'sma' : semi-major axis,
        'ellipticity': ellipticity,
        'PA': position angle in degree, anti-clockwise from y axis
        }
    '''
    if mask is not None:
        mask=mask.astype(bool)
    else:
        mask=np.zeros_like(data,dtype=bool)
    a=aperture['sma']
    b=a*(1-aperture['ellipticity'])
    theta=(90.+aperture['PA'])*np.pi/180.
    centx=aperture['posi'][0]
    centy=aperture['posi'][1]
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    D=-2*A*centx-B*centy
    E=-B*centx-2*C*centy
    F=A*(centx**2)+B*centx*centy+C*(centy**2)-(a**2)*(b**2)
    ny,nx=data.shape
    xmin=np.max([0,int(centx-a-2)])
    xmax=np.min([nx-1,int(centx+a+2)])
    ymin=np.max([0,int(centy-a-2)])
    ymax=np.min([ny-1,int(centy+a+2)])
    pix=[]
    x=xmin
    while x <= xmax:
        ea=C
        eb=E+B*float(x)
        ec=A*float(x**2)+D*float(x)+F
        delta=eb**2-4.*ea*ec
        if delta>0.:
            y1=(-eb+np.sqrt(delta))/(2*ea)
            y2=(-eb-np.sqrt(delta))/(2*ea)
            iy1=int(y1)
            iy2=int(y2)
            if (iy1 >= ymin)&(iy1 <= ymax):
                if not mask[iy1,int(x)]:
                    pix.append(data[iy1,int(x)])
            if iy1==iy2:
                x=int(x)
                x+=1
                continue
            if (iy2 >= ymin)&(iy2 <= ymax):
                if not mask[iy2,int(x)]:
                    pix.append(data[iy2,int(x)])
        x=int(x)
        x+=1
    mean=np.nanmean(pix)
    median=np.median(pix)
    std=np.std(pix)
    result={
        'mean':mean,
        'median':median,
        'std':std
    }
    return result


def spikePA(data,posi,length,PA0=0.1,deltaPA=0.1):
    step=90./deltaPA
    x0=posi[0]
    y0=posi[1]
    mean=[]
    APA=[]
    intx=int(x0)
    inty=int(y0)
    radiint=int(length)+2
    ny,nx=data.shape
    minx=np.max([intx-radiint,0])
    miny=np.max([inty-radiint,0])
    maxx=np.min([intx+radiint,nx-1])
    maxy=np.min([inty+radiint,ny-1])
    for loop in range(int(step)):
        PA=PA0+deltaPA*loop
        if (PA==90.)|(PA==0.):
            PA+=0.01
        APA.append(PA)
        k1=np.tan(PA*np.pi/180.)
        pixel=[]
        if k1 < 0. :
            k2=k1
            k1=-1./k2
        else:
            k2=-1./k1
        x=minx
        while x <= maxx:
            y=miny
            while y <=maxy:
                d1=y+1.-y0-k1*(x-x0)
                d2=y-y0-k1*(x+1.-x0)
                d3=y+1.-y0-k2*(x+1.-x0)
                d4=y-y0-k2*(x-x0)
                if (d1*d2 < 0.)|(d3*d4 <0.):
                    distan=np.sqrt((x+0.5-x0)**2+(y+0.5-y0)**2)
                    if distan < length:
                        pixel.append((y,x))
                y+=1
            x+=1
        mean.append(np.mean(data[pixel]))
    nmax=np.argmax(mean)
    return APA[nmax]


# coding: utf-8



import numpy as np
from astropy.table import Table, Column, hstack
from scipy.optimize import brentq
from astropy.stats import sigma_clipped_stats
import random as rand


def cleanseg(sourcecleaned,segm,threshold=0.):
    cleanlist=[]
    for loop in range(sourcecleaned):
        intx=int(sourcecleaned['xcentroid'][loop])
        inty=int(sourcecleaned['ycentroid'][loop])
        peak=sourcecleaned['peak'][loop]
        if segm[inty][intx] is not 0:
            cleanlist.append(segm[inty][intx])
    selection={
        'cleanlist':cleanlist,
        'segm':segm
    }
    return selection

def maskclean(data,mask,isophote,aperture,sky_std=None,expend=1.0,selection=None):
    '''
    aperture={
        'posi': center position of elliptical aperture[x,y],
        'sma' : semi-major axis,
        'ellipticity': ellipticity,
        'PA': position angle in degree, anti-clockwise from y axis
        }
    when the center of pixel in aperture, it will be cleaned
    selection: a segmentation tell othersources
    selection = {
        'cleanlist':a list of interger,
        'segm'     :othersource segmentation
        }
    '''
    if sky_std is None:
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    ellip=isophote['col6']
    ex_0=isophote['col10']
    ey_0=isophote['col12']
    ePA=isophote['col8']
    eSMA=isophote['col1']
    intensity=isophote['col2']
    datafake=data.copy()
    cleansize=int(expend*aperture['sma'])
    ny,nx=data.shape
    cleanpix=[]
    intx=int(aperture['posi'][0])
    inty=int(aperture['posi'][1])
    xmin=np.max([-cleansize-1+intx,0])
    xmax=np.min([cleansize+1+intx,nx-1])
    ymin=np.max([-cleansize-1+inty,0])
    ymax=np.min([cleansize+1+inty,ny-1])
    x=xmin
    a=expend*aperture['sma']
    b=a*(1-aperture['ellipticity'])
    theta=(aperture['PA']+90.)*np.pi/180.
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    while (x<(xmax+1)):
        y=ymin
        while(y<(ymax+1)):
            dx=x+0.5-aperture['posi'][0]
            dy=y+0.5-aperture['posi'][1]
            d=A*dx**2+B*dx*dy+C*dy**2-(a**2)*(b**2)
            if((d <= 0)&mask[y][x]):
                if selection is not None:
                    sel=selection['segm'][y][x]
                    for loops in range(selection['cleanlist']):
                        if sel == selection['cleanlist'][loops]:
                            cleanpix.append((y,x))
                            break
                else:
                    cleanpix.append((y,x))
            y +=1
        x +=1
    for loop2 in range(len(cleanpix)):
        inten=[]
        c=rand.uniform(-sky_std,sky_std)
        Find=False
        for loop in range(len(eSMA)):
            a=eSMA[loop]
            b=a*( 1 - (ellip[loop]) )
            psfdx=cleanpix[loop2][1]-ex_0[loop]
            psfdy=cleanpix[loop2][0]-ey_0[loop]
            psfdx1=psfdx+1
            psfdy1=psfdy+1
            psfdx2=psfdx+1
            psfdy2=psfdy
            psfdx3=psfdx
            psfdy3=psfdy+1
            theta=(ePA[loop]+90)*np.pi/180
            A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
            B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
            C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
            distan=A*psfdx**2+B*psfdx*psfdy+C*psfdy**2-(a**2)*(b**2)
            distan1=A*psfdx1**2+B*psfdx1*psfdy1+C*psfdy1**2-(a**2)*(b**2)
            distan2=A*psfdx2**2+B*psfdx2*psfdy2+C*psfdy2**2-(a**2)*(b**2)
            distan3=A*psfdx3**2+B*psfdx3*psfdy3+C*psfdy3**2-(a**2)*(b**2)
            IS=(((distan*distan1) < 0.)|((distan2*distan3) < 0.))
            if IS:
                inten.append(intensity[loop])
                Find=True
            if (not IS)&Find :
                break
        if len(inten) > 0 :
            datafake[cleanpix[loop2]]=np.mean(inten)+c

    return  datafake

def substitution(sourcecleaned,data,isophote,sky_std,func_sbp=None,psfFWHM=None) :
    sky_mean, sky_median, sky_std1 = sigma_clipped_stats(data, sigma=3.0, iters=5)
    ellip=isophote['col6']
    ex_0=isophote['col10']
    ey_0=isophote['col12']
    ePA=isophote['col8']
    eSMA=isophote['col1']
    intensity=isophote['col2']
    datafake=data.copy()
    for sourceid in range(len(sourcecleaned)):
        meanint=[]
        psfx=sourcecleaned['xcentroid'][sourceid]
        psfy=sourcecleaned['ycentroid'][sourceid]
        psfix=int(psfx)
        psfiy=int(psfy)
        for loop in range(len(eSMA)):
            a=eSMA[loop]
            b=a*( 1 - (ellip[loop]) )
            posi = [ex_0[loop], ey_0[loop]]
            psfdx=psfix-posi[0]
            psfdy=psfiy-posi[1]
            psfdx1=psfdx+1
            psfdy1=psfdy+1
            psfdx2=psfdx+1
            psfdy2=psfdy
            psfdx3=psfdx
            psfdy3=psfdy+1
            theta=(ePA[loop]+90)*np.pi/180
            A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
            B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
            C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
            distan=A*psfdx**2+B*psfdx*psfdy+C*psfdy**2-(a**2)*(b**2)
            distan1=A*psfdx1**2+B*psfdx1*psfdy1+C*psfdy1**2-(a**2)*(b**2)
            distan2=A*psfdx2**2+B*psfdx2*psfdy2+C*psfdy2**2-(a**2)*(b**2)
            distan3=A*psfdx3**2+B*psfdx3*psfdy3+C*psfdy3**2-(a**2)*(b**2)
            if (((distan*distan1) < 0.)|((distan2*distan3 ) <0.) ):
                meanint.append(intensity[loop])
        meanintensity=np.mean(meanint)
        peak =  sourcecleaned['peak'][sourceid]
        flux =  sourcecleaned['flux'][sourceid]
        if func_sbp is not None :
            sbps=lambda x : (func_sbp(x)*(peak+1.5*sky_median)-meanintensity)
            zeropoint=brentq(sbps,a=0.,b=50)
        if psfFWHM is not None :
            zeropoint = psfFWHM/2. * (np.max([np.log10(flux), 0]) + 1.5)
        cleansize=int(2*zeropoint)
        print('sourceid:{0}  cleansize:{1}'.format(sourcecleaned['id'][sourceid] ,cleansize ))
        if(cleansize == 0):
            continue
        print('cleansize{0}'.format(cleansize) )
        cleanpix=[]
        xmin=-cleansize-1
        xmax=cleansize+1
        ymin=-cleansize-1
        ymax=cleansize+1
        x=xmin
        while (x<(xmax+1)):
            y=ymin
            while(y<(ymax+1)):
                d=x**2+y**2-cleansize**2
                if(d <= 0):
                    cleanpix.append((x,y))
                y +=1
            x +=1
        for loop2 in range(len(cleanpix)):
            inten=[]
            x=psfix-1+cleanpix[loop2][0]
            y=psfiy+cleanpix[loop2][1]
            c=rand.uniform(-sky_std,sky_std)
            for loop in range(len(eSMA)):
                a=eSMA[loop]
                b=a*( 1 - (ellip[loop]) )
                posi = [ex_0[loop], ey_0[loop]]
                psfdx=psfix-1-posi[0]+cleanpix[loop2][0]
                psfdy=psfiy-posi[1]+cleanpix[loop2][1]
                psfdx1=psfdx+1
                psfdy1=psfdy+1
                psfdx2=psfdx+1
                psfdy2=psfdy
                psfdx3=psfdx
                psfdy3=psfdy+1
                theta=(ePA[loop]+90)*np.pi/180
                A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
                B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
                C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
                distan=A*psfdx**2+B*psfdx*psfdy+C*psfdy**2-(a**2)*(b**2)
                distan1=A*psfdx1**2+B*psfdx1*psfdy1+C*psfdy1**2-(a**2)*(b**2)
                distan2=A*psfdx2**2+B*psfdx2*psfdy2+C*psfdy2**2-(a**2)*(b**2)
                distan3=A*psfdx3**2+B*psfdx3*psfdy3+C*psfdy3**2-(a**2)*(b**2)
                if (((distan*distan1) < 0.)|((distan2*distan3) < 0.)):
                    inten.append(intensity[loop])
            if len(inten) > 0 :
                datafake[y][x]=np.mean(inten)+c
    return datafake


def mergermaskclean(data,mask,isophote1,isophote2,aperture,sky_std=None,expend=1.0,selection=None):
    '''
    both of isophote must hold the center

    '''
    if sky_std is None:
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    posi1=[isophote1['col10'][1],isophote1['col12'][1]]
    posi2=[isophote2['col10'][1],isophote2['col12'][1]]
    datafake=data.copy()
    cleansize=int(expend*aperture['sma'])
    ny,nx=data.shape
    cleanpix=[]
    intx=int(aperture['posi'][0])
    inty=int(aperture['posi'][1])
    xmin=np.max([-cleansize-1+intx,0])
    xmax=np.min([cleansize+1+intx,nx-1])
    ymin=np.max([-cleansize-1+inty,0])
    ymax=np.min([cleansize+1+inty,ny-1])
    x=xmin
    a=expend*aperture['sma']
    b=a*(1-aperture['ellipticity'])
    theta=(aperture['PA']+90.)*np.pi/180.
    A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
    B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
    C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
    while (x<(xmax+1)):
        y=ymin
        while(y<(ymax+1)):
            dx=x+0.5-aperture['posi'][0]
            dy=y+0.5-aperture['posi'][1]
            d=A*dx**2+B*dx*dy+C*dy**2-(a**2)*(b**2)
            if((d <= 0)&mask[y][x]):
                if selection is not None:
                    sel=selection['segm'][y][x]
                    for loops in range(selection['cleanlist']):
                        if sel == selection['cleanlist'][loops]:
                            cleanpix.append((y,x))
                            break
                else:
                    cleanpix.append((y,x))
            y +=1
        x +=1
    for loop2 in range(len(cleanpix)):
        close1=(float(cleanpix[loop2][1])+0.5-posi1[0])**2 + (float(cleanpix[loop2][0])+0.5-posi1[1])**2
        close2=(float(cleanpix[loop2][1])+0.5-posi2[0])**2 + (float(cleanpix[loop2][0])+0.5-posi2[1])**2
        isophote=isophote1
        if close2 < close1:
            isophote=isophote2
        ellip=isophote['col6']
        ex_0=isophote['col10']
        ey_0=isophote['col12']
        ePA=isophote['col8']
        eSMA=isophote['col1']
        intensity=isophote['col2']
        inten=[]
        c=rand.uniform(-sky_std,sky_std)
        Find=False
        for loop in range(len(eSMA)):
            a=eSMA[loop]
            b=a*( 1 - (ellip[loop]) )
            psfdx=cleanpix[loop2][1]-ex_0[loop]
            psfdy=cleanpix[loop2][0]-ey_0[loop]
            psfdx1=psfdx+1
            psfdy1=psfdy+1
            psfdx2=psfdx+1
            psfdy2=psfdy
            psfdx3=psfdx
            psfdy3=psfdy+1
            theta=(ePA[loop]+90)*np.pi/180
            A=(a**2)*(np.sin(theta))**2+(b**2)*(np.cos(theta))**2
            B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
            C=(a**2)*(np.cos(theta))**2+(b**2)*(np.sin(theta))**2
            distan=A*psfdx**2+B*psfdx*psfdy+C*psfdy**2-(a**2)*(b**2)
            distan1=A*psfdx1**2+B*psfdx1*psfdy1+C*psfdy1**2-(a**2)*(b**2)
            distan2=A*psfdx2**2+B*psfdx2*psfdy2+C*psfdy2**2-(a**2)*(b**2)
            distan3=A*psfdx3**2+B*psfdx3*psfdy3+C*psfdy3**2-(a**2)*(b**2)
            IS=(((distan*distan1) < 0.)|((distan2*distan3) < 0.))
            if IS:
                inten.append(intensity[loop])
                Find=True
            if (not IS)&Find :
                break
        if len(inten) > 0 :
            datafake[cleanpix[loop2]]=np.mean(inten)+c
        if len(inten) == 0 :
            datafake[cleanpix[loop2]]=sky_median+c
    return  datafake


# coding: utf-8

# In[ ]:




# In[ ]:
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils import aperture_photometry, CircularAperture, CircularAnnulus,EllipticalAperture,EllipticalAnnulus
from astropy.table import Table, Column
from scipy.interpolate import interp1d
from astropy.visualization import LogStretch
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.visualization import SqrtStretch, LogStretch
from astropy.modeling.models import Sersic2D,Sersic1D
import random as rand
# In[ ]:

class Trial():
    posi=[0,0]
    PA=0.
    def __init__(self,posi,PA):
        self.posi=posi
        self.PA=PA

# ----------------------
# DN-to-Jy conv.
# ----------------------
C = {
    'w1': 1.9350e-06,
    'w2': 2.7048e-06,
    'w3': 1.8326e-06,
    'w4': 5.2269e-05
}
# ----------------------
# Aperture corrections
# ----------------------
Deltam = {
    'w1': 0.0,
    'w2': 0.0,
    'w3': 0.0,
    'w4': 0.0
}

F_apcor = {
    'w1': 1.0,
    'w2': 1.0,
    'w3': 1.0,
    'w4': 1.0
}

sig_magzp = {
    'w1': 0.006,
    'w2': 0.007,
    'w3': 0.012,
    'w4': 0.012
}

def get_elliptical_aperture(data,isophote,mask=None,skyp=2):
    lenofell=len(isophote)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    datap=data-sky_median
    sky=sky_median
    DeltaSKY=0
    ellip=isophote['col6']
    ex_0=isophote['col10']
    ey_0=isophote['col12']
    ePA=isophote['col8']
    eSMA=isophote['col1']
    fluxList=[]
    aperarea=[]
    for loop in range(lenofell):
        a=eSMA[loop]
        b=a*( 1 - (ellip[loop]) )
        posi = [ex_0[loop], ey_0[loop]]
        apertures = EllipticalAperture(positions=posi,a=a,b=b,theta=(ePA[loop]+90))
        rawflux_table = aperture_photometry(datap, apertures, mask=mask)
        aperturesMaskedArea = aperture_photometry(mask.astype(int), apertures)['aperture_sum']
        aperarea.append((apertures.area()-aperturesMaskedArea[0]))
        fluxList.append(rawflux_table['aperture_sum'][0])
        print ("{0} percent done".format(100*loop/lenofell) )
    for loop in range(skyp) :
        antieerFunc=interp1d(fluxList,eSMA)
        aper1=1.4*antieerFunc(0.9*fluxList[lenofell-1])
        deltaSKY=0
        mass=0
        if(aper1 < 0.5*eSMA[lenofell-1]):
            aper1=0.5*eSMA[lenofell-1]
        if(aper1 > 0.75*eSMA[lenofell-1]):
            aper1=0.5*eSMA[lenofell-1]
        bignumber=0
        for loop2 in range(lenofell) :
            if eSMA[loop2] > aper1 :
                bignumber=loop2
                break
        loop3=bignumber+1
        while (loop3 < lenofell) :
            deltaSKy=(fluxList[loop3]-fluxList[bignumber])/(aperarea[loop3]-aperarea[bignumber])
            deltaSKY += deltaSKy*(pow(loop3,3))
            loop3+=1
            mass += pow(loop3,3)
        Dsky=deltaSKY/mass
        DeltaSKY += 0.3*Dsky
        for loop3 in range(lenofell) :
            fluxList[loop3] += (-0.3)*Dsky*(aperarea[loop3])
        print ("{0} percent done".format(100*loop/lenofell) )
        fig = plt.figure(figsize=(8, 6))
        plt.plot(eSMA, fluxList, label='Curve of Growth')
        plt.axvline(x=1.4*antieerFunc(0.95*fluxList[lenofell-1]), linestyle='--', color='b',   label='standard aperture')
        plt.xlabel('Semi-major axis length [pixel]', fontsize=16)
        plt.legend(loc='lower right')
        plt.show()
    sky += DeltaSKY
    antieerFunc=interp1d(fluxList,eSMA)
    for loop in range(lenofell) :
        if eSMA[loop] > (1.4*antieerFunc(0.95*fluxList[lenofell-1])) :
            std2=loop
            break
    a2=eSMA[std2]
    b2=a2*( 1 - (ellip[std2]) )
    posi2 = [ex_0[std2], ey_0[std2]]
    aperturestd = EllipticalAperture(positions=posi2,a=a2,b=b2,theta=(ePA[std2]+90))
    stdflux=fluxList[std2]
    plt.figure(figsize=(10, 10))
    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
    ax = plt.gca()
    ellipseAp = patches.Ellipse(xy=posi2,width=2*a2,height=2*b2,angle=(ePA[std2]+90),color='r',fill=False)
    ax.add_patch(ellipseAp)
    plt.show()
    result = {
        'aperture': aperturestd ,
        'flux' : stdflux ,
        'sky': sky,
        'fluxList': fluxList,
        'stdn':std2
    }
    return result

#def get_galaxysize()

def get_aperture(data,isophote,gain,sky_std=None,sky_median=None,mask=None,sigma=3,CGSmodel=False,imagepath=None,QuietMode=True,hold=None,holdsky=None,withmask=False,instrument='2MASS',usemedian=False):
    #imagepath=(name(targname_band),savepath)
    #hold=(posi,sma,PA,ellipticity)
    eta = 1
    N_depth = 1
    kappa = 1
    sky_mean, sky_median1, sky_std1 = sigma_clipped_stats(data, sigma=3.0, iters=5)
    if(sky_std == None):
        sky_std = sky_std1
    if(sky_median == None):
        sky_median = sky_median1
    if hold is None:
        inten=isophote['col2']
        for loop in range(len(isophote)):
            if inten[loop] < (sigma*sky_std+sky_median) :
                std2=loop
                break
        if CGSmodel :
            loop2=std2-1
            while loop2 < len(isophote):
                if (inten[loop2] - inten[loop2+1]) < -0.005:
                    std2=loop2
                    break
                loop2 +=1
        ellip=isophote['col6']
        ex_0=isophote['col10']
        ey_0=isophote['col12']
        ePA=isophote['col8']
        eSMA=isophote['col1']
        posi=[ex_0[std2],ey_0[std2]]
        a=eSMA[std2]
        theta=ePA[std2]+90.
        ellipticity=ellip[std2]
    if hold is not None :
        posi=hold['posi']
        theta=hold['PA']+90.
        ellipticity=hold['ellipticity']
        a=hold['sma']
        std2=0
    b=a*(1-ellipticity)
    if mask is None :
        mask = np.zeros_like(data, dtype=bool)
    annulus = np.zeros_like(data, dtype=bool)
    thetarad=(theta)*np.pi/180.
    radiint=int(1.801*a)
    intx=int(posi[0])
    inty=int(posi[1])
    radiint=int(2*a)
    nny, nnx = data.shape
    boundary=[0,nnx-1,0,nny-1]
    minx=np.max([intx-radiint,boundary[0]])
    miny=np.max([inty-radiint,boundary[2]])
    maxx=np.min([intx+radiint,boundary[1]])
    maxy=np.min([inty+radiint,boundary[3]])
    a1=a*1.250
    b1=b*1.250
    a2=a*1.601
    b2=b*1.601
    A=(a**2)*(np.sin(thetarad))**2+(b**2)*(np.cos(thetarad))**2
    B=2*(b**2-a**2)*np.sin(thetarad)*np.cos(thetarad)
    C=(a**2)*(np.cos(thetarad))**2+(b**2)*(np.sin(thetarad))**2
    A1=(a1**2)*(np.sin(thetarad))**2+(b1**2)*(np.cos(thetarad))**2
    B1=2*(b1**2-a1**2)*np.sin(thetarad)*np.cos(thetarad)
    C1=(a1**2)*(np.cos(thetarad))**2+(b1**2)*(np.sin(thetarad))**2
    A2=(a2**2)*(np.sin(thetarad))**2+(b2**2)*(np.cos(thetarad))**2
    B2=2*(b2**2-a2**2)*np.sin(thetarad)*np.cos(thetarad)
    C2=(a2**2)*(np.cos(thetarad))**2+(b2**2)*(np.sin(thetarad))**2
    loopx=minx
    masknan=np.zeros_like(data, dtype=bool)
    masknan[np.isnan(data)] = True
    masknan[np.isinf(data)] = True
    while loopx < (maxx+1) :
        loopy=miny
        while loopy < (maxy+1) :
            dx1=loopx+0.5-posi[0]
            dy1=loopy+0.5-posi[1]
            distan1=A1*dx1**2+B1*dx1*dy1+C1*dy1**2-(a1**2)*(b1**2)
            distan2=A2*dx1**2+B2*dx1*dy1+C2*dy1**2-(a2**2)*(b2**2)
            inner=distan1*distan2
            if (inner < 0.)&(not mask[loopy][loopx]) :
                annulus[loopy][loopx] = True
            loopy += 1
        loopx += 1
    new_image = data[annulus]
    temp1 = mask.astype(int)
    temp2 = temp1[annulus]
    new_mask = temp2.astype(bool)
    annulus_int = annulus.astype(int)
    sky_area = np.sum(annulus_int) - np.sum(temp2)
    sky_final=new_image[~new_mask]
    sky_meanF=np.nanmean(sky_final)
    sky_medianF=np.nanmedian(sky_final)
    sky_stdF=np.nanstd(sky_final)
    sky =  sky_meanF
    if holdsky is not None :
        sky_meanF=holdsky
    if usemedian:
        sky_meanF=sky_medianF
    datap=data-sky_meanF
    fluxList=[]
    aperturestd = EllipticalAperture(positions=posi,a=a,b=b,theta=thetarad)
    if  withmask :
        rawflux_table0 = aperture_photometry(datap, aperturestd, mask=mask)
    else :
        rawflux_table0 = aperture_photometry(datap, aperturestd, mask=masknan)
    area=aperturestd.area()
    stdflux=rawflux_table0['aperture_sum'][0]
    flux_err = (stdflux / (eta * gain * N_depth) + kappa * (area * sky_stdF)**2
             / sky_area + 1.7**2 * 4 * area * sky_stdF**2)**0.5
    if hold is None :
        for loop in range(len(isophote)):
            a111=eSMA[loop]
            b222=a111*( 1 - (ellip[loop]) )
            posi111 = [ex_0[loop], ey_0[loop]]
            aperture111 = EllipticalAperture(positions=posi111,a=a111,b=b222,theta=((ePA[loop]+90)*np.pi/180.))
            rawflux_table111 = aperture_photometry(datap, aperture111, mask=mask)
            fluxList.append(rawflux_table111['aperture_sum'][0])
        antieerFunc=interp1d(fluxList,eSMA)
        if (np.min(fluxList) < 0.3*stdflux)&(np.max(fluxList) > 0.9*stdflux) :
            HLR=antieerFunc(0.5*stdflux)
            concentration=5*np.log(antieerFunc(0.7*stdflux)/antieerFunc(0.3*stdflux))
        else:
            HLR=np.nan
            concentration=np.nan
    else:
        HLR=np.nan
        concentration=np.nan
    if (not QuietMode)&(hold is None) :
        lenofe=len(eSMA)
        amax=eSMA[lenofe-1]
        bmax=amax*(1-ellip[lenofe-1])
        fig = plt.figure(figsize=(8, 6))
        plt.plot(eSMA, fluxList, label='Curve of Growth')
        plt.axvline(x=a, linestyle='--', color='b',   label='standard aperture')
        plt.axvline(x=1.602*a, linestyle='--', color='r',   label='outer ellipse')
        plt.axvline(x=amax, linestyle='--', color='k',   label='biggest ellipse')
        plt.xlabel('Semi-major axis length [pixel]', fontsize=16)
        plt.title('{0} Curve of growth'.format(imagepath['name']), fontsize=16)
        plt.ylabel('Total Flux [DN]', fontsize=16)
        plt.legend(loc='lower right')
        if imagepath is None :
            plt.show()
        else:
            plt.savefig(imagepath['savepath'] + "{0}cog.png".format(imagepath['name']), dpi=200)
        plt.close()
        if instrument is '2MASS':
            plt.figure(figsize=(10, 20))
        else:
            plt.figure(figsize=(20, 20))
        norm = ImageNormalize(stretch=LogStretch())
        plt.imshow(data, cmap='Greys', origin='lower', norm=norm)
        ax = plt.gca()
        ellipseAp2 = patches.Ellipse(xy=posi,width=2.5*a,height=2.5*b,angle=theta,color='c',fill=False)
        ellipseAp3 = patches.Ellipse(xy=posi,width=2*1.601*a,height=2*1.601*b,angle=theta,color='r',fill=False)
        ellipseAp1 = patches.Ellipse(xy=posi,width=2*a,height=2*b,angle=theta,color='b',fill=False)
        ellipseAp4 = patches.Ellipse(xy=posi,width=2*amax,height=2*bmax,angle=ePA[lenofe-1]+90.,color='k',fill=False)
        ax.add_patch(ellipseAp4)
        ax.add_patch(ellipseAp1)
        ax.add_patch(ellipseAp2)
        ax.add_patch(ellipseAp3)
        if imagepath is None :
            plt.show()
        else:
            plt.savefig(imagepath['savepath'] + "{0}sky.png".format(imagepath['name']), dpi=200)
        plt.close()
    result = {
        'sma': a ,
        'b':b,
        'PA': theta ,
        'paY' :theta-90.,
        'ellipticity': ellipticity ,
        'posi':posi,
        'flux' : stdflux ,
        'flux_err' :flux_err,
        'fluxList':fluxList,
        'sky': sky_meanF,
        'sky_std':sky_stdF,
        'stdn':std2,
        'concentration':concentration,
        'HLR':HLR,
    }
    return result

def profile_analyse(x,func):
    flux=[]
    for loop in range(len(x)):
        x_1=x[0:loop+1]
        f=2*np.pi*x_1*func(x_1)
        flux.append(np.trapz(f,x=x_1))
    stdflux=np.nanmax(flux)
    antieerFunc=interp1d(flux,x)
    HLR=antieerFunc(0.5*stdflux)
    concentration=5*np.log(antieerFunc(0.7*stdflux)/antieerFunc(0.3*stdflux))
    result = {
        'concentration':concentration,
        'HLR':HLR,
    }
    return result

def logSersic1D(x,amplitude,n,r_eff):
    model=Sersic1D(amplitude=amplitude,n=n,r_eff=r_eff)
    return np.log(model(x))

def profile_analyseplus(x,data,aperture,mask=None):
    flux=[]
    sma=aperture['sma']
    epsilon=aperture['ellipticity']
    theta=(aperture['PA']+90.)*np.pi/180.
    aperturestd = EllipticalAperture(positions=aperture['posi'],a=sma,b=sma*(1-epsilon),theta=theta)
    rawflux_table = aperture_photometry(data, aperturestd, mask=mask)
    stdflux=rawflux_table['aperture_sum'][0]
    for x1 in x:
        apertures = EllipticalAperture(positions=aperture['posi'],a=x1,b=x1*(1-epsilon),theta=theta)
        rawflux_table = aperture_photometry(data, apertures, mask=mask)
        flux.append(rawflux_table['aperture_sum'][0])
    antieerFunc=interp1d(flux,x)
    HLR=antieerFunc(0.5*stdflux)
    concentration=5*np.log(antieerFunc(0.7*stdflux)/antieerFunc(0.3*stdflux))
    result = {
        'concentration':concentration,
        'HLR':HLR,
    }
    return result

def WISEuncertainty(data,band,mask,sigmamap,aperture,sample,Fcorr,flux,magzp):
    a=aperture['sma']
    k=np.pi/2.
    b=a*(1.-aperture['ellipticity'])
    theta0=(aperture['PA']+90.)*np.pi/180.
    aperture0 = EllipticalAperture(positions=aperture['posi'],a=a,b=b,theta=theta0)
    maskfloat=mask.astype(float)
    maskbool=mask.astype(bool)
    sigmasqur=sigmamap**2
    onedata=np.ones_like(data,dtype=float)
    rawflux_tableS = aperture_photometry(onedata, aperture0, mask=None)
    rawflux_tablesig = aperture_photometry(sigmasqur, aperture0, mask=None)
    f_apcor=F_apcor[band]
    N_A=rawflux_tableS['aperture_sum'][0]
    bck=[]
    for samp in sample:
        theta=(samp.PA+90.)*np.pi/180.
        aperturetri = EllipticalAperture(positions=samp.posi,a=a,b=b,theta=theta)
        rawflux_table = aperture_photometry(data, aperturetri, mask=maskbool)
        rawflux_table1 = aperture_photometry(maskfloat, aperturetri, mask=None)
        rawflux_table2 = aperture_photometry(onedata, aperturetri, mask=None)
        cc=rawflux_table['aperture_sum'][0]*rawflux_table2['aperture_sum'][0]/(rawflux_table2['aperture_sum'][0]-rawflux_table1['aperture_sum'][0])
        bck.append(cc)
    sigma_apbck=(np.std(bck))**2
    sigma_conf=sigma_apbck-Fcorr*N_A*np.sum(sigmasqur[~maskbool])/np.sum(onedata[~maskbool])
    ny,nx=data.shape
    xaxis = np.arange(nx)
    yaxis = np.arange(ny)
    xmesh, ymesh = np.meshgrid(xaxis, yaxis)
    a1=1.25*a
    b1=1.25*a
    a2=1.601*a
    b2=1.601*b
    A1=(a1*np.sin(theta0))**2+(b1*np.cos(theta0))**2
    B1=2*(b1**2-a1**2)*np.sin(theta0)*np.cos(theta0)
    C1=(a1*np.cos(theta0))**2+(b1*np.sin(theta0))**2
    D1=-2.*A1*aperture['posi'][0]-B1*aperture['posi'][1]
    E1=-1.*B1*aperture['posi'][0]-2.*C1*aperture['posi'][1]
    F1=A1*(aperture['posi'][0]**2)+B1*aperture['posi'][0]*aperture['posi'][1]+C1*(aperture['posi'][1]**2)-(a1**2)*(b1**2)
    A2=(a2*np.sin(theta0))**2+(b2*np.cos(theta0))**2
    B2=2*(b2**2-a2**2)*np.sin(theta0)*np.cos(theta0)
    C2=(a2*np.cos(theta0))**2+(b2*np.sin(theta0))**2
    D2=-2.*A2*aperture['posi'][0]-B2*aperture['posi'][1]
    E2=-1.*B2*aperture['posi'][0]-2.*C2*aperture['posi'][1]
    F2=A2*(aperture['posi'][0]**2)+B2*aperture['posi'][0]*aperture['posi'][1]+C2*(aperture['posi'][1]**2)-(a2**2)*(b2**2)
    Delta=(A1*(xmesh**2)+B1*xmesh*ymesh+C1*(ymesh**2)+D1*xmesh+E1*ymesh+F1)*(A2*(xmesh**2)+B2*xmesh*ymesh+C2*(ymesh**2)+D2*xmesh+E2*ymesh+F2)
    sigfilter=sigmasqur[~maskbool]
    Delfilter=Delta[~maskbool]
    squmedian=np.nanmedian(sigfilter[Delfilter < 0.])
    sig_pixel=(f_apcor**2)*Fcorr*(rawflux_tablesig['aperture_sum'][0])
    sig_sky=(f_apcor**2)*Fcorr*( k*(N_A**2)*squmedian/(len(sigfilter[Delfilter < 0.])) )
    sigma_src=sig_pixel+ sig_sky + sigma_conf
    mag=magzp-2.5*np.log10(flux)-Deltam[band]
    sig_mag=(sig_magzp[band]**2)+1.179*sigma_src/((flux*f_apcor)**2)
    result={
        'flux'   :flux,
        'sig_src' :np.sqrt(sigma_src),
        'mag'     :mag,
        'sig_mag' :np.sqrt(sig_mag),
        'sig_pixel':sig_pixel,
        'sig_sky':sig_sky,
        'sig_conf':sigma_conf
    }
    return result



def sampler(mask,aperture,threshold=0.8,overlap=0.2,sample=0.5):
    '''
    aperture={
        'posi': center position of elliptical aperture[x,y],
        'sma' : semi-major axis,
        'ellipticity': ellipticity,
        'PA': position angle in degree, anti-clockwise from y axis
        }
    sample
        parameter to end the loop
        sample fraction to nonmasked pixels
    '''
    maskbo=mask.astype(bool)
    maskbool=maskbo.copy()
    mask=mask.astype(float)
    sampleimage=np.zeros_like(mask,dtype=float)
    emmm=1.-sampleimage
    Ssample=np.sum(sampleimage)
    totalnmp=np.sum(1.-mask)
    ny,nx=mask.shape
    a=aperture['sma']
    ellipticity=aperture['ellipticity']
    b=a*(1-ellipticity)
    area=np.pi*a*b
    APE=[]
    posi=aperture['posi']
    PA=aperture['PA']
    k=4.
    likehoodfunc=lambda x : np.exp(k*(-x))
    while Ssample < totalnmp*sample:
        samp=False
        while not samp:
            step=rand.uniform(0.8*a,2.5*a)
            theta=(PA+90.)*np.pi/180.
            aperturetri = EllipticalAperture(positions=posi,a=a,b=b,theta=theta)
            rawflux_table = aperture_photometry(sampleimage, aperturetri, mask=None)
            cc0=rawflux_table['aperture_sum'][0]/area
            x=posi[0]+step*np.cos(theta)
            if x > nx:
                x -= nx
            if x < 0.:
                x += nx
            y=posi[1]+step*np.sin(theta)
            if y > ny:
                y -= ny
            if y < 0.:
                y += ny
            traPA=rand.uniform(-90.,90.)
            theta=(90.+traPA)*np.pi/180.
            aperturetri = EllipticalAperture(positions=[x,y],a=a,b=b,theta=theta)
            rawflux_table = aperture_photometry(sampleimage, aperturetri, mask=None)
            cc=rawflux_table['aperture_sum'][0]/area
            rawflux_table1 = aperture_photometry(mask, aperturetri, mask=None)
            rawflux_table2 = aperture_photometry(emmm, aperturetri, mask=None)
            dd=(rawflux_table2['aperture_sum'][0]-rawflux_table1['aperture_sum'][0])/area
            if (cc < cc0):
                    posi=[x,y]
                    PA=traPA
            else:
                ran=np.random.random()
                if ran < likehoodfunc(cc-cc0):
                    posi=[x,y]
                    PA=traPA
            if (dd > threshold)&(cc < overlap):
                trial=Trial(posi=[x,y],PA=traPA)
                APE.append(trial)
                samp=True
                xaxis = np.arange(nx)
                yaxis = np.arange(ny)
                xmesh, ymesh = np.meshgrid(xaxis, yaxis)
                A=(a*np.sin(theta))**2+(b*np.cos(theta))**2
                B=2*(b**2-a**2)*np.sin(theta)*np.cos(theta)
                C=(a*np.cos(theta))**2+(b*np.sin(theta))**2
                D=-2.*A*x-B*y
                E=-1.*B*x-2.*C*y
                F=A*(x**2)+B*x*y+C*(y**2)-(a**2)*(b**2)
                Delta=A*(xmesh**2)+B*xmesh*ymesh+C*(ymesh**2)+D*xmesh+E*ymesh+F
                sampleimage[Delta < 0.] = 1.
                Ssample=np.sum(sampleimage[~maskbool])
    return APE


def calculate_master_aperture(a, b, bandlist,PSFList):
	'''

	:param a: list of semi-major axis
	:param b: list of semi-minor axis list
	:param PA: list of band sequence (e.g. j, h, k, w1, w2, w3, w4)
	:return: dict of master aperture size, semi-major axis and semi-minor axis in input sequence
	'''

	# list of master aperture of each band
	a_list = []
	b_list = []
	# list of master aperture after psf correction
	ma = []
	mb = []
	for i in range(len(bandlist)):
		psf = PSFList[bandlist[i]]
		a_temp = (a[i]*a[i] - psf*psf)**0.5
		b_temp = (b[i]*b[i] - psf*psf)**0.5
		a_list.append(a_temp)
		b_list.append(b_temp)

	master_a = max(a_list)
	master_b = max(b_list)

	for i in range(len(bandlist)):
		psf = PSFList[bandlist[i]]
		a_temp = (master_a*master_a + psf*psf)**0.5
		b_temp = (master_b*master_b + psf*psf)**0.5
		ma.append(a_temp)
		mb.append(b_temp)

	return {
		"master aperture": [master_a, master_b],
		"semi-major axis": ma,
		"semi-minor axis": mb
	}

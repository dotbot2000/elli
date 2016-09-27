from __future__ import print_function
from matplotlib.mlab import prctile
from numpy.random import rand, seed
from scipy.interpolate import interp1d 

Isochrones = '/home/dotter/lib/python/Isochrones.py'
emcee='/home/jlin/emcee/dfm-emcee-6779c01/emcee'
import os
import sys
sys.path.append(os.path.dirname(os.path.expanduser(Isochrones)))
sys.path.append(os.path.dirname(os.path.expanduser(emcee)))
import Isochrones
import emcee
from Isochrones import DSED_Isochrones
from numpy import *
import scipy
from scipy.spatial.distance import cdist
import csv
from astropy import units as u


include_teff=1
include_logg=1
include_kmag=0
include_feh=1
mask=array([include_teff,include_logg,include_kmag,include_feh])

#constants
twopi4=pow(2*pi,4) #(2*pi)^4

iso_dir='/priv/coala/jlin/mcmc/iso2'
iso_list=['fehm25afep6.UBVRIJHKsKp',
          'fehm20afep4.UBVRIJHKsKp',
          'fehm15afep4.UBVRIJHKsKp',
          'fehm10afep4.UBVRIJHKsKp',
          'fehm05afep2.UBVRIJHKsKp',          
          'fehp00afep0.UBVRIJHKsKp',
          'fehp02afep0.UBVRIJHKsKp',
          'fehp03afep0.UBVRIJHKsKp',
          'fehp05afep0.UBVRIJHKsKp']
y=[]
for iso in iso_list:
    y.append(DSED_Isochrones(iso_dir+'/'+iso))

feh=[i.FeH for i in y]

def age_mass_guess(iso_set, value, sigma):
    def lnProb_guess(value,sigma,model):
        norm_guess=log(sqrt( twopi4 * prod(pow(sigma[nonzero(mask)],2))))
        return -sum(mask*  pow( (value-model)/sigma, 2)) - norm_guess

    #empty storage
    lnP_guess=[]
    for isoc in iso_set.data:
        p=[]
        #for each point in the isochrone, calculate ln(P)
        for star in isoc:
            model=array([pow(10,star['LogTeff']), star['LogG'], star['Ks      '], iso_set.FeH ])
            p.append(lnProb_guess(value,sigma,model))
        lnP_guess.append(p)

    lnP_guess=array(lnP_guess)
    mass_guess=0.0
    age_guess=0.0
    norm_guess=0.0

    #now do a weighted sum of mass and age
    for i in range(len(iso_set.data)):
        iso=iso_set.data[i]
        iso_age=iso_set.ages[i]
        p=exp(lnP_guess[i])
        for j in range(len(iso)):
            age_guess+=iso_age*p[j]
            mass_guess+=iso['M_Mo'][j]*p[j]
            norm_guess+=p[j]
    mass_guess = mass_guess/norm_guess
    age_guess = age_guess/norm_guess
    return age_guess, mass_guess

def do_age_mass_guess(value,sigma,y=y):
    FeH=value[3]
    if FeH>=feh[-1]:
        print('metallicity outside the isochrone range, returning 5Gyr & 1M_o ')
        return(5,1)
    if FeH<=feh[0]:
        print('metallicity outside the isochrone range, returning 5Gyr & 1M_o ')
        return(5,1)
    if y[0].FeH <= FeH <= y[-1].FeH:
        for i in range(len(y)):
            if y[i].FeH <= FeH < y[i+1].FeH: 
                iy=i #index of the iso set closest in feh
                break
        ageg0=[]; massg0=[]
        for x in y[iy:iy+2]:
            ageg1,massg1=age_mass_guess(x, value, sigma)
            if isnan(ageg1) ==False and isnan(massg1)==False:
                ageg0.append(ageg1)
                massg0.append(massg1)
        if len(ageg0)>1:
            ageg=interp([y[iy].FeH,y[iy+1].FeH],[ageg0[0],ageg0[1]])(value[3])
            massg=interp([y[iy].FeH,y[iy+1].FeH],[massg0[0],massg0[1]])(value[3])
            return(ageg,massg)
        if len(ageg0)==0:
            print('no starting guesses found, returning 5Gyr & 1M_o')
            return(5,1)
        else:
            return(ageg0[0],massg0[0])


def parallax_distance(parallax,err_parallax): #both in mas
    if include_kmag==0:
        return 1,1
    d=(parallax*u.mas).to(u.parsec, equivalencies=u.parallax())
    d=d.value 
    sigma_d=abs(d*err_parallax/float(parallax))
    print('distance: '+str(d)+'+/-'+str(sigma_d)+' pc')
    return d,sigma_d #both in pc

def Kmag_from_distance(kmag,err_kmag,d,err_d):
    Kmag= kmag-5*(log10(d)-1 )
    sigma_Kmag=sqrt( pow(err_kmag,2)+ 25*pow( err_d/ (float(d)*log(10) ),2)  )
    print('abs K mag: '+ str(Kmag)+'+/-'+str(sigma_Kmag))
    return Kmag, sigma_Kmag #absolute kmag

def interp(x,y):
    return interp1d(x=x,y=y,kind='linear')

def get_one_star(age,mass,x):
    ok=False
    params=empty(3)
    if x.ages[0] <= age <= x.ages[-1]:
        for i in range(len(x.ages)-1):
            if x.ages[i] <= age <= x.ages[i+1]: 
                i0=i
                i1=i+1
                break
        m0=x.data[i0]['M_Mo'] #x.data[i0,i1] are the closest isos in age
        m1=x.data[i1]['M_Mo']
        if m0[0] <= mass <= m0[-1] and m1[0] <= mass <= m1[-1]:
            T0=interp(m0,x.data[i0]['LogTeff'])(mass) 
            T1=interp(m1,x.data[i1]['LogTeff'])(mass)

            g0=interp(m0,x.data[i0]['LogG'])(mass)
            g1=interp(m1,x.data[i1]['LogG'])(mass)

            K0=interp(m0,x.data[i0]['Ks      '])(mass)
            K1=interp(m1,x.data[i1]['Ks      '])(mass)

            alfa=(age-x.ages[i0])/(x.ages[i1]-x.ages[i0])
            beta=1.0-alfa

            Teff=alfa*pow(10,T1) + beta*pow(10,T0)
            logg=alfa*g1 + beta*g0
            Kmag=alfa*K1 + beta*K0
            params=array([Teff,logg,Kmag])  #CCCCCHANGED!
            #params=array([Teff,Kmag]) 

            ok=True
    return ok,params

def get_star(age,mass,FeH,y):
    ok=False
    params=[]
    #params=[0,0,0]
    if y[0].FeH <= FeH <= y[-1].FeH:
        met=[]; Teff=[];  Kmag=[] ; logg=[] #CCCCCHANGED!
        for i in range(len(y)):
            if y[i].FeH <= FeH < y[i+1].FeH: 
                iy=i #index of the iso set closest in feh
                break
        for x in y[iy:iy+2]:
            ok,params=get_one_star(age,mass,x)
            if ok:
                met.append(x.FeH)
                Teff.append(params[0])
                logg.append(params[1])
                Kmag.append(params[2]) #CCCCCHANGED!
        #now we have two lists, 
        #one filled with mets and the other with results
        if len(met)>1 and met[0] <= FeH <= met[-1]:
            params[0] = interp(met,Teff)(FeH)
            params[1] = interp(met,logg)(FeH)
            params[2] = interp(met,Kmag)(FeH) #CCCCCHANGED!
        params=append(params,FeH)
    return ok, array(params)

def lnP(model,value,sigma):
    alpha=-2.35 #Salpeter IMF slope
    m0=0.1
    norm=-(alpha+1)/pow(m0,alpha+1.)
    age =model[0]  # Gyr
    mass=model[1] # Msun
    feh =model[2] 
    return lnProb(value,sigma,age,mass,feh)


def lnPrior(m):
    alpha=-2.35 #Salpeter IMF slope                                                                        
    m0=0.1
    norm=-(alpha+1.)/pow(m0,alpha+1.)
    return log(norm*pow(m,alpha))

def lnProb(value,sigma,age,mass,feh):
    ok,model=get_star(age,mass,feh,y)
    if ok:
        norm=log(sqrt( twopi4 * prod( pow(sigma[nonzero(mask)] ,2))))
        probb=lnPrior(mass) -0.5* sum( mask * pow( (value-model)/sigma, 2) ) + norm
        return probb, model
    else:
        return -inf, array([0,99.99,99.99,99.99])


def run_emcee(FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,star_data): #kmagS,ekmagS=app kmag
    print(star_data[ID])
    seed() #each thread has an independent random number sequence
    FeH=star_data[FehS]

    if type(eparallaxS)==str:
        distance,err_distance=parallax_distance(star_data[parallaxS],star_data[eparallaxS])
    else:
        distance,err_distance=parallax_distance(star_data[parallaxS],eparallaxS)

    if type(ekmagS)==str:
        kmag_S,ekmag_S=Kmag_from_distance(star_data[kmagS],star_data[ekmagS],distance,err_distance)
    else:
        kmag_S,ekmag_S=Kmag_from_distance(star_data[kmagS],ekmagS,distance,err_distance)

    value=array([star_data[TeffS],star_data[loggS],kmag_S,star_data[FehS]])
 

    if type(eTeffS)==str:
        sigma=array([star_data[eTeffS],star_data[eloggS],ekmag_S,star_data[eFehS]]) 
    else:
        sigma=array([eTeffS,eloggS,ekmag_S,eFehS])
    print(value)
    print(sigma)
    ageg,massg=do_age_mass_guess(value,sigma)
    sampler=None

    #for a, typically 2 is a good number; must be > 1
    nwalkers=200
    nburn=200
    nrun=500
    ndim=3
    my_a=2 #2.

    guess=array([ageg,massg,star_data[FehS]])
    print(' guess = ', guess)

    if guess[-1]==0:
        guess=array([guess[0],guess[1],guess[2]+0.001])
    p0=[guess*(1-0.4*(0.5-rand(ndim))) for i in range(nwalkers)]

    #create an instance 
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnP,args=[value,sigma],a=my_a)

    #burn-in: save end state and reset
    pos,prob,state,blob=sampler.run_mcmc(p0,nburn)
    sampler.reset()

    #main run
    sampler.run_mcmc(pos,nrun)
    lnp_flat=sampler.flatlnprobability

    Teff_dist=[]
    logg_dist=[]
    kmag_dist=[]
    for j in range(nwalkers):
        for i in range(nrun):
            Teff_dist.append(sampler.blobs[i][j][0])
            logg_dist.append(sampler.blobs[i][j][1])
            kmag_dist.append(sampler.blobs[i][j][2])

    logg_dist=array(logg_dist)
    Teff_dist=array(Teff_dist)
    kmag_dist=array(kmag_dist)


    #print basic results
    print()
    print("Result: ", star_data[ID])
    print("Mean acceptance fraction: {0:10.4g}".format(mean(sampler.acceptance_fraction)))

    result=[]

    age_dist=sampler.flatchain[:,0]
    print("len(age_dist)=", len(age_dist))
    print(min(age_dist))
    print(max(age_dist))
    print("len(age_dist)=", len(age_dist))

    mass_dist=sampler.flatchain[:,1]
    print("len(mass_dist)=", len(mass_dist))
    print(min(mass_dist))
    print(max(mass_dist))
    print("len(mass_dist)=", len(mass_dist))

    feh_dist = sampler.flatchain[:,2]
    print("len(feh_dist)=", len(feh_dist))
    print(min(feh_dist))
    print(max(feh_dist))
    print("len(feh_dist)=", len(feh_dist))

    print("len(kmag_dist)=", len(kmag_dist))
    print(min(kmag_dist))
    print(max(kmag_dist))
    print("len(kmag_dist)=", len(kmag_dist))

    good=where( (lnp_flat>-15)& (kmag_dist!=99.99) & (logg_dist!=99.99) & (Teff_dist!=0) )

    if len(good)==0:
        logg_dist=logg_dist[0:0]
        Teff_dist=Teff_dist[0:0]
        kmag_dist=kmag_dist[0:0]
        age_dist=age_dist[0:0]
        mass_dist=mass_dist[0:0]
        feh_dist=feh_dist[0:0]
        lnp_flat=lnp_flat[0:0]
    else:
        logg_dist=logg_dist[good]
        Teff_dist=Teff_dist[good]
        kmag_dist=kmag_dist[good]
        age_dist=age_dist[good]
        mass_dist=mass_dist[good]
        feh_dist=feh_dist[good]
        lnp_flat=lnp_flat[good]

    result.append(ageg)
    if len(age_dist)>0: 
        pct=prctile(age_dist,p=[2.5,50,97.5])
        result.append(mean(age_dist))
        result.append(std(age_dist))
        result.append(pct[0]) #2.5%
        result.append(pct[1]) #50%
        result.append(pct[2]) #97.5%
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    result.append(massg)
    if len(mass_dist)>0:
        pct=prctile(mass_dist,p=[2.5,50,97.5])
        result.append(mean(mass_dist))
        result.append(std(mass_dist))
        result.append(pct[0]) #2.5th percentile
        result.append(pct[1]) #50th percentile==median
        result.append(pct[2]) #97.5%
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    if len(Teff_dist) > 0:
        pct=prctile(Teff_dist,p=[2.5,50,97.5])
        result.append(mean(Teff_dist))
        result.append(std(Teff_dist))
        result.append(pct[0])
        result.append(pct[1])
        result.append(pct[2])
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    if len(logg_dist) > 0:
        pct=prctile(logg_dist,p=[2.5,50,97.5])
        result.append(mean(logg_dist))
        result.append(std(logg_dist))
        result.append(pct[0])
        result.append(pct[1])
        result.append(pct[2])
    else:
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)

    if len(kmag_dist)>0: 
        pct=prctile(kmag_dist,p=[2.5,50,97.5])
        print('mean kmag')
        print(mean(kmag_dist))
        result.append(mean(kmag_dist))
        result.append(std(kmag_dist))
        result.append(pct[0]) #2.5%
        result.append(pct[1]) #50%
        result.append(pct[2]) #97.5%
    else:
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)        

    if len(feh_dist)>0:
        mean_feh = mean(feh_dist)
        std_feh = std(feh_dist)
        pct=prctile(feh_dist,p=[2.5,50,97.5])
        result.append(mean_feh)
        result.append(std_feh)
        result.append(pct[0]) #5th percentile
        result.append(pct[1]) #50th percentile==median
        result.append(pct[2]) #95%
        
    else:
        mean_feh = 0.0
        std_feh = 0.0
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
    result.append(mean(sampler.acceptance_fraction))
    return array(result)

def do_run_emcee(data,out_dir,out_prefix,FehS,eFehS,TeffS,loggS,eTeffS,eloggS ,kmagS,ekmagS,parallaxS,eparallaxS,ID,my_thread=1,max_thread=1):
    max_stars = len(data)
    begin,end,increment = my_thread,max_stars,max_thread 

    filename = out_dir + '/' + out_prefix+'_'+str(begin)+'_'+str(increment)+'.dat'
    f=open(filename.strip(),'w')
    f.write("{0:>5s}".format('HIP'))

    f.write("{0:>13s}{1:>13s}{2:>13s}{3:>13s}{4:>13s}{5:>13s}{6:>13s}{7:>13s}{8:>13s}{9:>13s}{10:>13s}{11:>13s}{12:>13s}{13:>13s}{14:>13s}{15:>13s}{16:>13s}{17:>13s}{18:>13s}{19:>13s}{20:>13s}{21:>13s}{22:>13s}{23:>13s}{24:>13s}{25:>13s}{26:>13s}{27:>13s}{28:>13s}{29:>13s}{30:>13s}{31:>13s}{32:>13s}{33:>13s}{34:>13s}{35:>13s}{36:>13s}{37:>13s}".format(
         'age_guess',  'age_mean',  'age_sigma',  'age_05th',  'age_median',  'age_95th',
         'mass_guess', 'mass_mean', 'mass_sigma', 'mass_05th', 'mass_median', 'mass_95th',
                       'teff_mean','teff_sigma','teff_05th','teff_median','teff_95th',
                       'logg_mean','logg_sigma','logg_05th','logg_median','logg_95th',
                       'kmag_mean','kmag_sigma','kmag_05th','kmag_median','kmag_95th','feh_mean','feh_sigma','feh_05th','feh_median','feh_95th','acc_frac',
         'teffS','loggS','FehS','kmagS','parallaxS'))
    f.write('\n')
    for i in range(begin,end,increment):
        result = run_emcee(FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,data[i])
        #write mags and errors
        f.write("{0:>10s}".format(str(data[ID][i]) ))
        print(data[ID][i])
        #write results from emcee
        for j in range(len(result)):
            f.write(" {0:12.4e}".format(result[j]))
        #done with line
        f.write(' {0:>12.4e} {1:>12.4e} {2:>12.4e} {3:>12.4e} {4:>12.4e} '.format(data[TeffS][i],data[loggS][i],data[FehS][i],data[kmagS][i],data[parallaxS][i]))
        f.write("\n")
    #done writing
    f.close()

def run_emcee_full(FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,star_data): #kmagS,ekmagS=app kmag

    print(star_data[ID])
    seed() #each thread has an independent random number sequence
    FeH=star_data[FehS]

    if type(eparallaxS)==str:
        distance,err_distance=parallax_distance(star_data[parallaxS],star_data[eparallaxS])
    else:
        distance,err_distance=parallax_distance(star_data[parallaxS],eparallaxS)

    if type(ekmagS)==str:
        kmag_S,ekmag_S=Kmag_from_distance(star_data[kmagS],star_data[ekmagS],distance,err_distance)
    else:
        kmag_S,ekmag_S=Kmag_from_distance(star_data[kmagS],ekmagS,distance,err_distance)

    value=array([star_data[TeffS],star_data[loggS],kmag_S,star_data[FehS]]) 
    print(value)
    if type(eTeffS)==str:
        sigma=array([star_data[eTeffS],star_data[eloggS],ekmag_S,star_data[eFehS]]) 

    else:
        sigma=array([eTeffS,eloggS,ekmag_S,eFehS]) 
    print(sigma)

    ageg,massg=do_age_mass_guess(value,sigma)

    sampler=None

    #for a, typically 2 is a good number; must be > 1
    nwalkers=200
    nburn=200
    nrun=500
    ndim=3
    my_a=2

    guess=array([ageg,massg,star_data[FehS]])
    print(' guess = ', guess)
    if guess[-1]==0:
        guess=array([guess[0],guess[1],guess[2]+0.001])
    p0=[guess*(1-0.4*(0.5-rand(ndim))) for i in range(nwalkers)]

    #create an instance 
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnP,args=[value,sigma],a=my_a)

    #burn-in: save end state and reset
    pos,prob,state,blob=sampler.run_mcmc(p0,nburn)
    sampler.reset()

    #main run
    sampler.run_mcmc(pos,nrun)
    lnp=sampler.lnprobability
    chain=sampler.chain
    min_w,min_s=unravel_index(lnp.argmin(),lnp.shape) #min lnp step+walker
    mode=chain[min_w,min_s] #min lnp for age,mass,feh
    lnp_flat=sampler.flatlnprobability
    
    Teff_dist=[]
    logg_dist=[]
    kmag_dist=[]
    for j in range(nwalkers):
        for i in range(nrun):
            logg_dist.append(sampler.blobs[i][j][1])
            Teff_dist.append(sampler.blobs[i][j][0])
            kmag_dist.append(sampler.blobs[i][j][2])

    logg_dist=array(logg_dist)
    Teff_dist=array(Teff_dist)
    kmag_dist=array(kmag_dist)
#    import pickle
#    pickle.dump((sampler.chain,sampler.blobs,sampler.lnprobability,sampler.flatlnprobability,sampler.flatchain,array([Teff_dist,logg_dist,kmag_dist]),p0),open('test_34028.p','wb'))

    print('acor:')
    print (sampler.acor)
 
    #print basic results
    print()
    print("Result: ", star_data[ID])
    print("Mean acceptance fraction: {0:10.4g}".format(
        mean(sampler.acceptance_fraction)))

    result=[]

    age_dist=sampler.flatchain[:,0]
    print("len(age_dist)=", len(age_dist))
    print(min(age_dist),max(age_dist),mean(age_dist))
    print("len(age_dist)=", len(age_dist))

    mass_dist=sampler.flatchain[:,1]
    print("len(mass_dist)=", len(mass_dist))
    print(min(mass_dist),max(mass_dist),mean(mass_dist))
    print("len(mass_dist)=", len(mass_dist))

    feh_dist = sampler.flatchain[:,2]
    print("len(feh_dist)=", len(feh_dist))
    print(min(feh_dist),max(feh_dist),mean(feh_dist))
    print("len(feh_dist)=", len(feh_dist))

    print("len(kmag_dist)=", len(kmag_dist))
    print(min(kmag_dist),max(kmag_dist),mean(kmag_dist))
    print("len(kmag_dist)=", len(kmag_dist))

    good=where( (lnp_flat>-15)& (kmag_dist!=99.99) & (logg_dist!=99.99) & (Teff_dist!=0) )
    if len(good)==0:
        logg_dist=logg_dist[0:0]
        Teff_dist=Teff_dist[0:0]
        kmag_dist=kmag_dist[0:0]
        age_dist=age_dist[0:0]
        mass_dist=mass_dist[0:0]
        feh_dist=feh_dist[0:0]
        lnp_flat=lnp_flat[0:0]
    else:
        logg_dist=logg_dist[good]
        Teff_dist=Teff_dist[good]
        kmag_dist=kmag_dist[good]
        age_dist=age_dist[good]
        mass_dist=mass_dist[good]
        feh_dist=feh_dist[good]
        lnp_flat=lnp_flat[good]

    result.append(ageg)

    if len(age_dist)>0:
        pct=prctile(age_dist,p=[2.5,50,97.5])
        result.append(mean(age_dist))
        result.append(std(age_dist))
        result.append(pct[0]) #2.5%
        result.append(pct[1]) #50%
        result.append(pct[2]) #97.5%
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    result.append(massg)

    if len(mass_dist)>0:
        pct=prctile(mass_dist,p=[2.5,50,97.5])
        result.append(mean(mass_dist))
        result.append(std(mass_dist))
        result.append(pct[0]) #2.5th percentile
        result.append(pct[1]) #50th percentile==median
        result.append(pct[2]) #97.5%
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    if len(Teff_dist) > 0:
        pct=prctile(Teff_dist,p=[2.5,50,97.5])
        result.append(mean(Teff_dist))
        result.append(std(Teff_dist))
        result.append(pct[0])
        result.append(pct[1])
        result.append(pct[2])
    else:
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    if len(logg_dist) > 0:
        pct=prctile(logg_dist,p=[2.5,50,97.5])
        result.append(mean(logg_dist))
        result.append(std(logg_dist))
        result.append(pct[0])
        result.append(pct[1])
        result.append(pct[2])
    else:
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)

    if len(kmag_dist) > 0:
        pct=prctile(kmag_dist,p=[2.5,50,97.5])
        result.append(mean(kmag_dist))
        result.append(std(kmag_dist))
        result.append(pct[0])
        result.append(pct[1])
        result.append(pct[2])
    else:
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)
        result.append(99.0)

    if len(feh_dist)>0:
        mean_feh = mean(feh_dist)
        std_feh = std(feh_dist)
        pct=prctile(feh_dist,p=[2.5,50,97.5])
        result.append(mean_feh)
        result.append(std_feh)
        result.append(pct[0]) #5th percentile
        result.append(pct[1]) #50th percentile==median
        result.append(pct[2]) #95%
        
    else:
        mean_feh = 0.0
        std_feh = 0.0
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)
        result.append(0.0)

    result.append(mode[0])#mode age
    result.append(mode[1])#mass
    result.append(mode[2])#feh
    result.append(mean(sampler.acceptance_fraction))

    return(array(result),age_dist,mass_dist,Teff_dist,logg_dist,kmag_dist,feh_dist,lnp_flat)

def do_run_emcee_full(data,out_dir,out_prefix,FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,my_thread=1,max_thread=1):

    max_stars = len(data)
    begin,end,increment = my_thread,max_stars,max_thread 

    filename = out_dir + '/' + out_prefix+'_'+str(begin)+'_'+str(increment)+'.dat'
    f=open(filename.strip(),'w')
    f.write("{0:>5s}".format('HIP'))
    f.write("{0:>13s}{1:>13s}{2:>13s}{3:>13s}{4:>13s}{5:>13s}{6:>13s}{7:>13s}{8:>13s}{9:>13s}{10:>13s}{11:>13s}{12:>13s}{13:>13s}{14:>13s}{15:>13s}{16:>13s}{17:>13s}{18:>13s}{19:>13s}{20:>13s}".format('age_mean',  'age_sigma',  'mass_mean', 'mass_sigma',  'teff_mean','teff_sigma','logg_mean','logg_sigma', 'kmag_mean','kmag_sigma','FeH_mean', 'FeH_sigma','age_mode','mass_mode','feh_mode','acc_frac','teffS','loggS','FehS','kmagS','parallaxS'))
    f.write('\n')
    for i in range(begin,end,increment):
        results = run_emcee_full(FehS,eFehS,TeffS,loggS,eTeffS,eloggS,kmagS,ekmagS,parallaxS,eparallaxS,ID,data[i])

        result=[results[0][1],results[0][2],results[0][7],results[0][8],results[0][12],results[0][13],results[0][17],results[0][18],results[0][22],results[0][23],results[0][27],results[0][28],results[0][32],results[0][33],results[0][34],results[0][35],data[TeffS][i],data[loggS][i],data[FehS][i],data[kmagS][i],data[parallaxS][i]] #all mean and sigma vals only
        gg=open(out_dir+ '/' + str(data[ID][i]),'w')
        gg.write("# {0:>10s}\n".format(str(data[ID][i])  ))
        names=['# age_guess' , '# age_mean',  '# age_sigma',  '# age_05th',  '# age_median',  '# age_95th',
         '# mass_guess', '# mass_mean', '# mass_sigma', '# mass_05th', '# mass_median', '# mass_95th',
                       '# teff_mean','# teff_sigma','# teff_05th','# teff_median','# teff_95th',
                       '# logg_mean','# logg_sigma','# logg_05th','# logg_median','# logg_95th',
                       '# kmag_mean','# kmag_sigma','# kmag_05th','# kmag_median','# kmag_95th',
                       '# feh_mean','# feh_sigma','# feh_05th','# feh_median','# feh_95th', '# age_mode',
               '# mass_mode','# feh_mode ','# acc_frac', '# teffS','# loggS','# FehS','# kmagS','# parallaxS']
        gg.close()
        with open(out_dir+ '/' +str(data[ID][i]) ,'a') as gg:
            writer = csv.writer(gg, delimiter='\t')
            writer.writerows(zip(names,list(results[0])+[data[TeffS][i],data[loggS][i],data[FehS][i],data[kmagS][i],data[parallaxS][i]]))

        with open(out_dir+ '/' +str(data[ID][i]) ,'a') as gg:
            gg.write('# age     mass      teff     logg    kmag     feh       lnP\n')
            writer = csv.writer(gg, delimiter=',')
            writer.writerows(zip(results[1],results[2],results[3],results[4],results[5],results[6],results[7]))
        
        #write mags and errors
        f.write("{0:>10s}".format(str(data[ID][i]) ))
        #write results from emcee
        for j in range(len(result)):
            f.write(" {0:12.4e}".format(result[j]))
        f.write("\n")
    #done writing
    f.close()
